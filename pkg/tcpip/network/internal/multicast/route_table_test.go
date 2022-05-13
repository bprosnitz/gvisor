// Copyright 2022 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package multicast

import (
	"os"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"gvisor.dev/gvisor/pkg/refs"
	"gvisor.dev/gvisor/pkg/refsvfs2"
	"gvisor.dev/gvisor/pkg/tcpip"
	"gvisor.dev/gvisor/pkg/tcpip/buffer"
	"gvisor.dev/gvisor/pkg/tcpip/faketime"
	"gvisor.dev/gvisor/pkg/tcpip/stack"
	"gvisor.dev/gvisor/pkg/tcpip/testutil"
)

const (
	defaultMinTTL             = 10
	inputNICID    tcpip.NICID = 1
	outgoingNICID tcpip.NICID = 2
	defaultNICID  tcpip.NICID = 3
)

var (
	defaultAddress            = testutil.MustParse4("192.168.1.1")
	defaultRouteKey           = RouteKey{UnicastSource: defaultAddress, MulticastDestination: defaultAddress}
	defaultOutgoingInterfaces = []OutgoingInterface{{ID: outgoingNICID, MinTTL: defaultMinTTL}}
)

func newPacketBuffer(body string) *stack.PacketBuffer {
	return stack.NewPacketBuffer(stack.PacketBufferOptions{
		Data: buffer.View(body).ToVectorisedView(),
	})
}

type configOption func(*Config)

func withMaxPendingQueueSize(size uint8) configOption {
	return func(c *Config) {
		c.MaxPendingQueueSize = size
	}
}

func withClock(clock tcpip.Clock) configOption {
	return func(c *Config) {
		c.Clock = clock
	}
}

func defaultConfig(opts ...configOption) Config {
	c := &Config{
		MaxPendingQueueSize: DefaultMaxPendingQueueSize,
		Clock:               faketime.NewManualClock(),
	}

	for _, opt := range opts {
		opt(c)
	}

	return *c
}

func installedRouteComparer(a *InstalledRoute, b *InstalledRoute) bool {
	if !cmp.Equal(a.OutgoingInterfaces(), b.OutgoingInterfaces()) {
		return false
	}

	if a.ExpectedInputInterface() != b.ExpectedInputInterface() {
		return false
	}

	return a.LastUsedTimestamp() == b.LastUsedTimestamp()
}

func TestInit(t *testing.T) {
	tests := []struct {
		name        string
		config      Config
		invokeTwice bool
		wantErr     error
	}{
		{
			name:        "MissingClock",
			config:      defaultConfig(withClock(nil)),
			invokeTwice: false,
			wantErr:     ErrMissingClock,
		},
		{
			name:        "AlreadyInitialized",
			config:      defaultConfig(),
			invokeTwice: true,
			wantErr:     ErrAlreadyInitialized,
		},
		{
			name:        "ValidConfig",
			config:      defaultConfig(),
			invokeTwice: false,
			wantErr:     nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			table := RouteTable{}
			defer table.Close()
			err := table.Init(tc.config)

			if tc.invokeTwice {
				err = table.Init(tc.config)
			}

			if !cmp.Equal(err, tc.wantErr, cmpopts.EquateErrors()) {
				t.Errorf("table.Init(%#v) = %s, want %s", tc.config, err, tc.wantErr)
			}
		})
	}
}

func TestNewInstalledRoute(t *testing.T) {
	table := RouteTable{}
	defer table.Close()
	clock := faketime.NewManualClock()
	clock.Advance(5 * time.Second)

	config := defaultConfig(withClock(clock))
	if err := table.Init(config); err != nil {
		t.Fatalf("table.Init(%#v): %s", config, err)
	}

	route := table.NewInstalledRoute(inputNICID, defaultOutgoingInterfaces)
	expectedRoute := &InstalledRoute{expectedInputInterface: inputNICID, outgoingInterfaces: defaultOutgoingInterfaces, lastUsedTimestamp: clock.NowMonotonic()}

	if diff := cmp.Diff(expectedRoute, route, cmp.Comparer(installedRouteComparer)); diff != "" {
		t.Errorf("Installed route mismatch (-want +got):\n%s", diff)
	}
}

func TestPendingRouteStates(t *testing.T) {
	table := RouteTable{}
	defer table.Close()
	config := defaultConfig(withMaxPendingQueueSize(2))
	if err := table.Init(config); err != nil {
		t.Fatalf("table.Init(%#v): %s", config, err)
	}

	pkt := newPacketBuffer("hello")
	defer pkt.DecRef()
	// Queue two pending packets for the same route. The PendingRouteState should
	// transition from PendingRouteStateInstalled to PendingRouteStateAppended.
	for _, wantPendingRouteState := range []PendingRouteState{PendingRouteStateInstalled, PendingRouteStateAppended} {
		routeResult, err := table.GetRouteOrInsertPending(defaultRouteKey, pkt)

		if err != nil {
			t.Errorf("table.GetRouteOrInsertPending(%#v, %#v) = (_, %v), want = (_, nil)", defaultRouteKey, pkt, err)
		}

		expectedResult := GetRouteResult{PendingRouteState: wantPendingRouteState}
		if diff := cmp.Diff(expectedResult, routeResult); diff != "" {
			t.Errorf("table.GetRouteOrInsertPending(%#v, %#v) GetRouteResult mismatch (-want +got):\n%s", defaultRouteKey, pkt, diff)
		}
	}

	// Queuing a third packet should yield an error since the pending queue is
	// already at max capacity.
	if _, err := table.GetRouteOrInsertPending(defaultRouteKey, pkt); err != ErrNoBufferSpace {
		t.Errorf("table.GetRouteOrInsertPending(%#v, %#v) = (_, %v), want = (_, ErrNoBufferSpace)", defaultRouteKey, pkt, err)
	}
}

func TestPendingRouteExpiration(t *testing.T) {
	pkt := newPacketBuffer("foo")
	defer pkt.DecRef()

	testCases := []struct {
		name                string
		advanceBeforeInsert time.Duration
		advanceAfterInsert  time.Duration
		wantPendingRoute    bool
	}{
		{
			name:                "not expired",
			advanceBeforeInsert: DefaultCleanupInterval / 2,
			// The time is advanced far enough to run the cleanup routine, but not
			// far enough to expire the route.
			advanceAfterInsert: DefaultCleanupInterval,
			wantPendingRoute:   true,
		},
		{
			name: "expired",
			// The cleanup routine will be run twice. The second invocation will
			// remove the expired route.
			advanceBeforeInsert: DefaultCleanupInterval / 2,
			advanceAfterInsert:  DefaultCleanupInterval * 2,
			wantPendingRoute:    false,
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			clock := faketime.NewManualClock()

			table := RouteTable{}
			defer table.Close()
			config := defaultConfig(withClock(clock))

			if err := table.Init(config); err != nil {
				t.Fatalf("table.Init(%#v): %s", config, err)
			}

			clock.Advance(test.advanceBeforeInsert)

			if _, err := table.GetRouteOrInsertPending(defaultRouteKey, pkt); err != nil {
				t.Fatalf("table.GetRouteOrInsertPending(%#v, %#v): %v", defaultRouteKey, pkt, err)
			}

			clock.Advance(test.advanceAfterInsert)

			table.pendingMu.RLock()
			_, ok := table.pendingRoutes[defaultRouteKey]

			if table.isCleanupRoutineRunning != test.wantPendingRoute {
				t.Errorf("table.isCleanupRoutineRunning = %t, want = %t", table.isCleanupRoutineRunning, test.wantPendingRoute)
			}
			table.pendingMu.RUnlock()

			if test.wantPendingRoute != ok {
				t.Errorf("table.pendingRoutes[%#v] = (_, %t), want = (_, %t)", defaultRouteKey, ok, test.wantPendingRoute)
			}
		})
	}
}

func TestAddInstalledRouteWithPending(t *testing.T) {
	pkt := newPacketBuffer("foo")
	defer pkt.DecRef()

	cmpOpts := []cmp.Option{
		cmp.Transformer("AsViews", func(pkt *stack.PacketBuffer) []buffer.View {
			return pkt.Views()
		}),
		cmp.Comparer(func(a []buffer.View, b []buffer.View) bool {
			return cmp.Equal(a, b)
		}),
	}

	testCases := []struct {
		name    string
		advance time.Duration
		want    []*stack.PacketBuffer
	}{
		{
			name:    "not expired",
			advance: DefaultPendingRouteExpiration,
			want:    []*stack.PacketBuffer{pkt},
		},
		{
			name:    "expired",
			advance: DefaultPendingRouteExpiration + 1,
			want:    nil,
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			clock := faketime.NewManualClock()

			table := RouteTable{}
			defer table.Close()
			config := defaultConfig(withClock(clock))

			if err := table.Init(config); err != nil {
				t.Fatalf("table.Init(%#v): %s", config, err)
			}

			if _, err := table.GetRouteOrInsertPending(defaultRouteKey, pkt); err != nil {
				t.Fatalf("table.GetRouteOrInsertPending(%#v, %#v): %v", defaultRouteKey, pkt, err)
			}

			// Disable the cleanup routine.
			table.cleanupPendingRoutesTimer.Stop()

			clock.Advance(test.advance)

			route := table.NewInstalledRoute(inputNICID, defaultOutgoingInterfaces)
			pendingPackets := table.AddInstalledRoute(defaultRouteKey, route)

			if diff := cmp.Diff(test.want, pendingPackets, cmpOpts...); diff != "" {
				t.Errorf("table.AddInstalledRoute(%#v, %#v) mismatch (-want +got):\n%s", defaultRouteKey, route, diff)
			}

			for _, pendingPkt := range pendingPackets {
				pendingPkt.DecRef()
			}

			// Verify that the pending route is actually deleted.
			table.pendingMu.RLock()
			if pendingRoute, ok := table.pendingRoutes[defaultRouteKey]; ok {
				t.Errorf("table.pendingRoutes[%#v] = (%#v, true), want (_, false)", defaultRouteKey, pendingRoute)
			}
			table.pendingMu.RUnlock()
		})
	}
}

func TestAddInstalledRouteWithNoPending(t *testing.T) {
	table := RouteTable{}
	defer table.Close()
	config := defaultConfig()
	if err := table.Init(config); err != nil {
		t.Fatalf("table.Init(%#v): %s", config, err)
	}

	firstRoute := table.NewInstalledRoute(inputNICID, defaultOutgoingInterfaces)
	secondRoute := table.NewInstalledRoute(defaultNICID, defaultOutgoingInterfaces)

	pkt := newPacketBuffer("hello")
	defer pkt.DecRef()
	for _, route := range [...]*InstalledRoute{firstRoute, secondRoute} {
		if pendingPackets := table.AddInstalledRoute(defaultRouteKey, route); pendingPackets != nil {
			t.Errorf("got table.AddInstalledRoute(%#v, %#v) = %#v, want = false", defaultRouteKey, route, pendingPackets)
		}

		// AddInstalledRoute is invoked for the same routeKey two times. Verify
		// that the fetched InstalledRoute reflects the most recent invocation of
		// AddInstalledRoute.
		routeResult, err := table.GetRouteOrInsertPending(defaultRouteKey, pkt)

		if err != nil {
			t.Fatalf("table.GetRouteOrInsertPending(%#v, %#v): %v", defaultRouteKey, pkt, err)
		}

		if routeResult.PendingRouteState != PendingRouteStateNone {
			t.Errorf("routeResult.PendingRouteState = %s, want = PendingRouteStateNone", routeResult.PendingRouteState)
		}

		if diff := cmp.Diff(route, routeResult.InstalledRoute, cmp.Comparer(installedRouteComparer)); diff != "" {
			t.Errorf("route.InstalledRoute mismatch (-want +got):\n%s", diff)
		}
	}
}

func TestRemoveInstalledRoute(t *testing.T) {
	table := RouteTable{}
	defer table.Close()
	config := defaultConfig()
	if err := table.Init(config); err != nil {
		t.Fatalf("table.Init(%#v): %s", config, err)
	}

	route := table.NewInstalledRoute(inputNICID, defaultOutgoingInterfaces)

	table.AddInstalledRoute(defaultRouteKey, route)

	if removed := table.RemoveInstalledRoute(defaultRouteKey); !removed {
		t.Errorf("table.RemoveInstalledRoute(%#v) = false, want = true", defaultRouteKey)
	}

	pkt := newPacketBuffer("hello")
	defer pkt.DecRef()

	result, err := table.GetRouteOrInsertPending(defaultRouteKey, pkt)

	if err != nil {
		t.Fatalf("table.GetRouteOrInsertPending(%#v, %#v): %v", defaultRouteKey, pkt, err)
	}

	if result.InstalledRoute != nil {
		t.Errorf("result.InstalledRoute = %v, want = nil", result.InstalledRoute)
	}
}

func TestRemoveInstalledRouteWithNoMatchingRoute(t *testing.T) {
	table := RouteTable{}
	defer table.Close()
	config := defaultConfig()
	if err := table.Init(config); err != nil {
		t.Fatalf("table.Init(%#v): %s", config, err)
	}

	if removed := table.RemoveInstalledRoute(defaultRouteKey); removed {
		t.Errorf("table.RemoveInstalledRoute(%#v) = true, want = false", defaultRouteKey)
	}
}

func TestGetLastUsedTimestampWithNoMatchingRoute(t *testing.T) {
	table := RouteTable{}
	defer table.Close()
	config := defaultConfig()
	if err := table.Init(config); err != nil {
		t.Fatalf("table.Init(%#v): %s", config, err)
	}

	if _, found := table.GetLastUsedTimestamp(defaultRouteKey); found {
		t.Errorf("table.GetLastUsedTimetsamp(%#v) = (_, true), want = (_, false)", defaultRouteKey)
	}
}

func TestSetLastUsedTimestamp(t *testing.T) {
	clock := faketime.NewManualClock()
	clock.Advance(10 * time.Second)

	currentTime := clock.NowMonotonic()
	validLastUsedTime := currentTime.Add(10 * time.Second)

	tests := []struct {
		name             string
		lastUsedTime     tcpip.MonotonicTime
		wantLastUsedTime tcpip.MonotonicTime
	}{
		{
			name:             "valid timestamp",
			lastUsedTime:     validLastUsedTime,
			wantLastUsedTime: validLastUsedTime,
		},
		{
			name:             "timestamp before",
			lastUsedTime:     currentTime.Add(-5 * time.Second),
			wantLastUsedTime: currentTime,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			table := RouteTable{}
			defer table.Close()
			config := defaultConfig(withClock(clock))
			if err := table.Init(config); err != nil {
				t.Fatalf("table.Init(%#v): %s", config, err)
			}

			route := table.NewInstalledRoute(inputNICID, defaultOutgoingInterfaces)

			table.AddInstalledRoute(defaultRouteKey, route)

			route.SetLastUsedTimestamp(test.lastUsedTime)

			// Verify that the updated timestamp is actually reflected in the RouteTable.
			timestamp, found := table.GetLastUsedTimestamp(defaultRouteKey)

			if !found {
				t.Fatalf("table.GetLastUsedTimestamp(%#v) = (_, false_), want = (_, true)", defaultRouteKey)
			}

			if timestamp != test.wantLastUsedTime {
				t.Errorf("table.GetLastUsedTimestamp(%#v) = (%s, _), want = (%s, _)", defaultRouteKey, timestamp, test.wantLastUsedTime)
			}
		})
	}
}

func TestMain(m *testing.M) {
	refs.SetLeakMode(refs.LeaksPanic)
	code := m.Run()
	refsvfs2.DoLeakCheck()
	os.Exit(code)
}
