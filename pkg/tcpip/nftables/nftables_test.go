// Copyright 2024 The gVisor Authors.
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

package nftables

import (
	"encoding/binary"
	"fmt"
	"reflect"
	"testing"

	"gvisor.dev/gvisor/pkg/abi/linux"
	"gvisor.dev/gvisor/pkg/buffer"
	"gvisor.dev/gvisor/pkg/tcpip"
	"gvisor.dev/gvisor/pkg/tcpip/header"
	"gvisor.dev/gvisor/pkg/tcpip/stack"
)

// Table Constants.
const (
	arbitraryTargetChain         string        = "target_chain"
	arbitraryHook                Hook          = Prerouting
	arbitraryFamily              AddressFamily = Inet
	arbitraryReservedHeaderBytes int           = 16
)

var (
	arbitraryPriority Priority = func() Priority {
		priority, err := NewStandardPriority("filter", arbitraryFamily, arbitraryHook)
		if err != nil {
			panic(fmt.Sprintf("unexpected error for NewStandardPriority: %v", err))
		}
		return priority
	}()

	arbitraryInfoPolicyAccept *BaseChainInfo = &BaseChainInfo{
		BcType:   BaseChainTypeFilter,
		Hook:     arbitraryHook,
		Priority: arbitraryPriority,
	}
)

// Packet Constants.
const (
	transportProtocol = tcpip.TransportProtocolNumber(6)

	arbitraryHeaderID   = 3
	arbitraryTimeToLive = 64

	// TODO(b/345684870): Use constants defined in the pkg/tcpip/header package.
	// Ethernet Offsets and Lengths.
	ethDstAddrOffset = 0
	ethDstAddrLen    = 6
	ethSrcAddrOffset = 6
	ethSrcAddrLen    = 6
	ethTypeOffset    = 12
	ethTypeLen       = 2

	// IPv4 Offsets and Lengths.
	ipv4LengthOffset   = 2
	ipv4LengthLen      = 2
	ipv4IDOffset       = 4
	ipv4IDLen          = 2
	ipv4FragOffOffset  = 6
	ipv4FragOffLen     = 2
	ipv4TTLOffset      = 8
	ipv4TTLLen         = 1
	ipv4ProtocolOffset = 9
	ipv4ProtocolLen    = 1
	ipv4ChecksumOffset = 10
	ipv4ChecksumLen    = 2
	ipv4SrcAddrOffset  = 12
	ipv4SrcAddrLen     = 4
	ipv4DstAddrOffset  = 16
	ipv4DstAddrLen     = 4

	// IPv6 Offsets and Lengths.
	ipv6LengthOffset   = 4
	ipv6LengthLen      = 2
	ipv6NextHdrOffset  = 6
	ipv6NextHdrLen     = 1
	ipv6HopLimitOffset = 7
	ipv6HopLimitLen    = 1
	ipv6SrcAddrOffset  = 8
	ipv6SrcAddrLen     = 16
	ipv6DstAddrOffset  = 24
	ipv6DstAddrLen     = 16

	// TCP Offsets and Lengths.
	tcpSrcPortOffset  = 0
	tcpSrcPortLen     = 2
	tcpDstPortOffset  = 2
	tcpDstPortLen     = 2
	tcpSeqNumOffset   = 4
	tcpSeqNumLen      = 4
	tcpAckNumOffset   = 8
	tcpAckNumLen      = 4
	tcpWindowOffset   = 14
	tcpWindowLen      = 2
	tcpChecksumOffset = 16
	tcpChecksumLen    = 2
	tcpUrgPtrOffset   = 18
	tcpUrgPtrLen      = 2
)

var (
	arbitraryLinkAddr     = tcpip.LinkAddress("\x02\x02\x03\x04\x05\x06")
	arbitraryLinkAddr2    = tcpip.LinkAddress("\x02\x02\x03\x04\x05\x07")
	arbitraryLinkAddrB    = [6]byte{0x02, 0x02, 0x03, 0x04, 0x05, 0x06}
	arbitraryLinkAddrB2   = [6]byte{0x02, 0x02, 0x03, 0x04, 0x05, 0x07}
	arbitraryEthernetType = header.IPv4ProtocolNumber

	arbitraryIPv4AddrB  = [4]byte{192, 168, 1, 1}
	arbitraryIPv4AddrB2 = [4]byte{192, 168, 1, 9}
	ipv4MinTotalLength  = header.IPv4MinimumSize

	arbitraryIPv6AddrB   = [16]byte{0x20, 0x01, 0x0d, 0xb8, 0x85, 0xa3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xaa}
	arbitraryIPv6AddrB2  = [16]byte{0x20, 0x01, 0x0d, 0xb8, 0x85, 0xa3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xbb}
	ipv6MinPayloadLength = 0

	arbitraryPort    = 12345
	arbitraryPort2   = 80
	tcpSeqNum        = 32
	tcpAckNum        = 165
	tcpWinSize       = 65535
	tcpUrgentPointer = 0

	arbitraryNonZeroFragmentOffset = 16
)

// makeArbitraryGeneralPacket creates an arbitrary packet for testing.
func makeArbitraryGeneralPacket(reserved int) *stack.PacketBuffer {
	return stack.NewPacketBuffer(stack.PacketBufferOptions{
		ReserveHeaderBytes: reserved,
		Payload:            buffer.MakeWithData([]byte{0, 2, 4, 8, 16, 32, 64, 128}),
	})
}

// makeArbitraryEtherPacket creates a packet with an arbitrary ethernet header.
func makeArbitraryEtherPacket(reserved int) *stack.PacketBuffer {
	eth := make([]byte, header.EthernetMinimumSize)
	header.Ethernet(eth).Encode(&header.EthernetFields{
		SrcAddr: arbitraryLinkAddr,
		DstAddr: arbitraryLinkAddr2,
		Type:    arbitraryEthernetType,
	})
	pkt := stack.NewPacketBuffer(stack.PacketBufferOptions{
		ReserveHeaderBytes: reserved,
		Payload:            buffer.MakeWithData(eth),
	})
	pkt.LinkHeader().Consume(header.EthernetMinimumSize)
	return pkt
}

// makeArbitraryIPv4Packet creates a packet with an arbitrary IPv4 header.
func makeArbitraryIPv4Packet(reserved int) *stack.PacketBuffer {
	// Creates a new PacketBuffer with enough space for the IPv4 header.
	pkt := stack.NewPacketBuffer(stack.PacketBufferOptions{
		ReserveHeaderBytes: reserved,
	})

	// Prepends the IPv4 header to the packet buffer.
	ipv4Hdr := header.IPv4(pkt.NetworkHeader().Push(header.IPv4MinimumSize))

	// Initializes the IPv4 header with fields.
	ipv4Hdr.Encode(&header.IPv4Fields{
		TOS:            0,
		TotalLength:    uint16(ipv4MinTotalLength),
		ID:             arbitraryHeaderID,
		FragmentOffset: 0,
		TTL:            arbitraryTimeToLive,
		Protocol:       uint8(transportProtocol),
		Checksum:       0,
		SrcAddr:        tcpip.AddrFrom4(arbitraryIPv4AddrB),
		DstAddr:        tcpip.AddrFrom4(arbitraryIPv4AddrB2),
		Options:        nil,
	})

	// Calculates and sets the checksum.
	ipv4Hdr.SetChecksum(^ipv4Hdr.CalculateChecksum())

	// Sets the network protocol number.
	pkt.NetworkProtocolNumber = header.IPv4ProtocolNumber

	return pkt
}

// makeFragmentedIPv4Packet creates a packet with an arbitrary IPv4 header that
// is fragmented (FragmentOffset != 0).
func makeFragmentedIPv4Packet(reserved int) *stack.PacketBuffer {
	pkt := stack.NewPacketBuffer(stack.PacketBufferOptions{
		ReserveHeaderBytes: reserved,
	})
	ipv4Hdr := header.IPv4(pkt.NetworkHeader().Push(header.IPv4MinimumSize))
	ipv4Hdr.Encode(&header.IPv4Fields{
		TOS:            0,
		TotalLength:    uint16(ipv4MinTotalLength),
		ID:             arbitraryHeaderID,
		FragmentOffset: uint16(arbitraryNonZeroFragmentOffset),
		TTL:            arbitraryTimeToLive,
		Protocol:       uint8(transportProtocol),
		Checksum:       0,
		SrcAddr:        tcpip.AddrFrom4(arbitraryIPv4AddrB),
		DstAddr:        tcpip.AddrFrom4(arbitraryIPv4AddrB2),
		Options:        nil,
	})
	ipv4Hdr.SetChecksum(^ipv4Hdr.CalculateChecksum())
	pkt.NetworkProtocolNumber = header.IPv4ProtocolNumber
	return pkt
}

// makeArbitraryIPv6Packet creates a packet with an arbitrary IPv6 header.
func makeArbitraryIPv6Packet(reserved int) *stack.PacketBuffer {
	// Creates a new PacketBuffer with enough space for the IPv4 header.
	pkt := stack.NewPacketBuffer(stack.PacketBufferOptions{
		ReserveHeaderBytes: reserved,
	})

	// Prepends the IPv6 header to the packet buffer.
	ipv6Hdr := header.IPv6(pkt.NetworkHeader().Push(header.IPv6MinimumSize))

	// Initializes the IPv6 header with fields.
	ipv6Hdr.Encode(&header.IPv6Fields{
		TrafficClass:      0,
		FlowLabel:         0,
		PayloadLength:     uint16(ipv6MinPayloadLength),
		TransportProtocol: transportProtocol,
		HopLimit:          arbitraryTimeToLive,
		SrcAddr:           tcpip.AddrFrom16(arbitraryIPv6AddrB),
		DstAddr:           tcpip.AddrFrom16(arbitraryIPv6AddrB2),
	})

	// Sets the network protocol number.
	pkt.NetworkProtocolNumber = header.IPv6ProtocolNumber

	return pkt
}

// makeArbitraryIPv4TCPPacket creates a packet with an arbitrary IPv4 and TCP
// header.
func makeArbitraryIPv4TCPPacket(reserved int) *stack.PacketBuffer {
	pkt := makeArbitraryIPv4Packet(reserved)

	// Prepends the TCP header to the packet buffer.
	tcpHdr := header.TCP(pkt.TransportHeader().Push(header.TCPMinimumSize))

	// Initializes the TCP header with fields.
	tcpHdr.Encode(&header.TCPFields{
		SrcPort:       uint16(arbitraryPort),
		DstPort:       uint16(arbitraryPort2),
		SeqNum:        uint32(tcpSeqNum),
		AckNum:        uint32(tcpAckNum),
		DataOffset:    header.TCPMinimumSize,
		WindowSize:    uint16(tcpWinSize),
		Checksum:      0,
		UrgentPointer: uint16(tcpUrgentPointer),
	})

	// Calculates the TCP checksum using the pseudo-header and set it in the TCP header.
	tcpHdr.SetChecksum(tcpHdr.CalculateChecksum(header.PseudoHeaderChecksum(
		header.TCPProtocolNumber,
		tcpip.AddrFrom4(arbitraryIPv4AddrB),
		tcpip.AddrFrom4(arbitraryIPv4AddrB2),
		header.TCPMinimumSize,
	)))

	// Sets the transport protocol number.
	pkt.TransportProtocolNumber = header.TCPProtocolNumber

	return pkt
}

// TestUnsupportedAddressFamily tests that an empty NFTables object returns an
// error when evaluating a packet for an unsupported address family.
func TestUnsupportedAddressFamily(t *testing.T) {
	// Makes arbitrary packet for comparison (to check for no changes).
	cmpPkt := makeArbitraryGeneralPacket(arbitraryReservedHeaderBytes)
	nf := NewNFTables()
	for _, unsupportedFamily := range []AddressFamily{AddressFamily(NumAFs), AddressFamily(-1)} {
		// Note: the Prerouting hook is arbitrary (any hook would work).
		pkt := makeArbitraryGeneralPacket(arbitraryReservedHeaderBytes)
		v, err := nf.EvaluateHook(unsupportedFamily, arbitraryHook, pkt)
		if err == nil {
			t.Fatalf("expecting error for EvaluateHook with unsupported address family %d; got %v verdict, %s packet, and error %v",
				int(unsupportedFamily),
				v, packetResultString(cmpPkt, pkt), err)
		}
	}
}

// TestAcceptAll tests that an empty NFTables object accepts all packets for
// supported hooks and errors for unsupported hooks for all address families
// when evaluating packets at the hook-level.
func TestAcceptAllForSupportedHooks(t *testing.T) {
	// Makes arbitrary packet for comparison (to check for no changes).
	cmpPkt := makeArbitraryGeneralPacket(arbitraryReservedHeaderBytes)
	for _, family := range []AddressFamily{IP, IP6, Inet, Arp, Bridge, Netdev} {
		t.Run(family.String()+" address family", func(t *testing.T) {
			nf := NewNFTables()
			for _, hook := range []Hook{Prerouting, Input, Forward, Output, Postrouting, Ingress, Egress} {
				pkt := makeArbitraryGeneralPacket(arbitraryReservedHeaderBytes)
				v, err := nf.EvaluateHook(family, hook, pkt)

				supported := false
				for _, h := range supportedHooks[family] {
					if h == hook {
						supported = true
						break
					}
				}

				if supported {
					if err != nil || v.Code != VC(linux.NF_ACCEPT) {
						t.Fatalf("expecting accept verdict for EvaluateHook with supported hook %v for family %v; got %v verdict, %s packet, and error %v",
							hook, family,
							v, packetResultString(cmpPkt, pkt), err)
					}
				} else {
					if err == nil {
						t.Fatalf("expecting error for EvaluateHook with unsupported hook %v for family %v; got %v verdict, %s packet, and error %v",
							hook, family,
							v, packetResultString(cmpPkt, pkt), err)
					}
				}
			}
		})
	}
}

// TestEvaluateImmediateVerdict tests that the Immediate operation correctly sets the
// register value and behaves as expected during evaluation.
func TestEvaluateImmediateVerdict(t *testing.T) {
	for _, test := range []struct {
		tname    string
		baseOp1  operation // will be nil if unused
		baseOp2  operation // will be nil if unused
		targetOp operation // will be nil if unused
		verdict  Verdict
	}{
		{
			tname:   "no operations",
			verdict: Verdict{Code: VC(linux.NF_ACCEPT)}, // from base chain policy
		},
		{
			tname:   "immediately accept",
			baseOp1: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_ACCEPT)})),
			verdict: Verdict{Code: VC(linux.NF_ACCEPT)},
		},
		{
			tname:   "immediately drop",
			baseOp1: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})),
			verdict: Verdict{Code: VC(linux.NF_DROP)},
		},
		{
			tname:   "immediately continue with base chain policy accept",
			baseOp1: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_CONTINUE)})),
			verdict: Verdict{Code: VC(linux.NF_ACCEPT)}, // from base chain policy
		},
		{
			tname:   "immediately return with base chain policy accept",
			baseOp1: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_RETURN)})),
			verdict: Verdict{Code: VC(linux.NF_ACCEPT)}, // from base chain policy
		},
		{
			tname:    "immediately jump to target chain that accepts",
			baseOp1:  mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: arbitraryTargetChain})),
			targetOp: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_ACCEPT)})),
			verdict:  Verdict{Code: VC(linux.NF_ACCEPT)},
		},
		{
			tname:    "immediately jump to target chain that drops",
			baseOp1:  mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: arbitraryTargetChain})),
			targetOp: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})),
			verdict:  Verdict{Code: VC(linux.NF_DROP)},
		},
		{
			tname:    "immediately jump to target chain that continues with second rule that accepts",
			baseOp1:  mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: arbitraryTargetChain})),
			targetOp: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_CONTINUE)})),
			baseOp2:  mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_ACCEPT)})),
			verdict:  Verdict{Code: VC(linux.NF_ACCEPT)},
		},
		{
			tname:    "immediately jump to target chain that continues with second rule that drops",
			baseOp1:  mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: arbitraryTargetChain})),
			targetOp: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_CONTINUE)})),
			baseOp2:  mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})),
			verdict:  Verdict{Code: VC(linux.NF_DROP)},
		},
		{
			tname:    "immediately goto to target chain that accepts",
			baseOp1:  mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: arbitraryTargetChain})),
			targetOp: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_ACCEPT)})),
			verdict:  Verdict{Code: VC(linux.NF_ACCEPT)},
		},
		{
			tname:    "immediately goto to target chain that drops",
			baseOp1:  mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: arbitraryTargetChain})),
			targetOp: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})),
			verdict:  Verdict{Code: VC(linux.NF_DROP)},
		},
		{
			tname:    "immediately goto to target chain that continues with second rule that accepts",
			baseOp1:  mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: arbitraryTargetChain})),
			targetOp: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_CONTINUE)})),
			baseOp2:  mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_ACCEPT)})),
			verdict:  Verdict{Code: VC(linux.NF_ACCEPT)}, // from base chain policy
		},
		{
			tname:    "immediately goto to target chain that continues with second rule that drops",
			baseOp1:  mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: arbitraryTargetChain})),
			targetOp: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_CONTINUE)})),
			baseOp2:  mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})),
			verdict:  Verdict{Code: VC(linux.NF_ACCEPT)}, // from base chain policy
		},
		{
			tname:   "add data to register then accept",
			baseOp1: mustCreateImmediate(t, linux.NFT_REG32_13, newBytesData([]byte{0, 1, 2, 3})),
			baseOp2: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_ACCEPT)})),
			verdict: Verdict{Code: VC(linux.NF_ACCEPT)},
		},
		{
			tname:   "add data to register then drop",
			baseOp1: mustCreateImmediate(t, linux.NFT_REG32_15, newBytesData([]byte{0, 1, 2, 3})),
			baseOp2: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})),
			verdict: Verdict{Code: VC(linux.NF_DROP)},
		},
		{
			tname:   "add data to register then continue",
			baseOp1: mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{0, 1, 2, 3})),
			baseOp2: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_CONTINUE)})),
			verdict: Verdict{Code: VC(linux.NF_ACCEPT)}, // from base chain policy
		},
		{
			tname:   "multiple accepts",
			baseOp1: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_ACCEPT)})),
			baseOp2: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_ACCEPT)})),
			verdict: Verdict{Code: VC(linux.NF_ACCEPT)},
		},
		{
			tname:   "multiple drops",
			baseOp1: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})),
			baseOp2: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})),
			verdict: Verdict{Code: VC(linux.NF_DROP)},
		},
		{
			tname:   "immediately accept then drop",
			baseOp1: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_ACCEPT)})),
			baseOp2: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})),
			verdict: Verdict{Code: VC(linux.NF_ACCEPT)},
		},
		{
			tname:   "immediately drop then accept",
			baseOp1: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})),
			baseOp2: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_ACCEPT)})),
			verdict: Verdict{Code: VC(linux.NF_DROP)},
		},
		{
			tname:   "immediate load register",
			baseOp1: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})),
			baseOp2: mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_ACCEPT)})),
			verdict: Verdict{Code: VC(linux.NF_DROP)},
		},
	} {
		t.Run(test.tname, func(t *testing.T) {
			// Sets up an NFTables object with a base chain (for 2 rules) and another
			// target chain (for 1 rule).
			nf := NewNFTables()
			tab, err := nf.AddTable(arbitraryFamily, "test", "test table", false)
			if err != nil {
				t.Fatalf("unexpected error for AddTable: %v", err)
			}
			bc, err := tab.AddChain("base_chain", nil, "test chain", false)
			if err != nil {
				t.Fatalf("unexpected error for AddChain: %v", err)
			}
			bc.SetBaseChainInfo(arbitraryInfoPolicyAccept)
			tc, err := tab.AddChain(arbitraryTargetChain, nil, "test chain", false)
			if err != nil {
				t.Fatalf("unexpected error for AddChain: %v", err)
			}

			// Adds testing rules and operations.
			if test.baseOp1 != nil {
				rule1 := &Rule{}
				rule1.addOperation(test.baseOp1)
				if err := bc.RegisterRule(rule1, -1); err != nil {
					t.Fatalf("unexpected error for RegisterRule for the first operation: %v", err)
				}
			}
			if test.baseOp2 != nil {
				rule2 := &Rule{}
				rule2.addOperation(test.baseOp2)
				if err := bc.RegisterRule(rule2, -1); err != nil {
					t.Fatalf("unexpected error for RegisterRule for the second operation: %v", err)
				}
			}
			if test.targetOp != nil {
				ruleTarget := &Rule{}
				ruleTarget.addOperation(test.targetOp)
				if err := tc.RegisterRule(ruleTarget, -1); err != nil {
					t.Fatalf("unexpected error for RegisterRule for the target operation: %v", err)
				}
			}

			// Runs evaluation and checks verdict.
			pkt := makeArbitraryGeneralPacket(arbitraryReservedHeaderBytes)
			v, err := nf.EvaluateHook(arbitraryFamily, arbitraryHook, pkt)

			if err != nil {
				t.Fatalf("unexpected error for EvaluateHook: %v", err)
			}
			if v.Code != test.verdict.Code {
				t.Fatalf("expected verdict %v, got %v", test.verdict, v)
			}
		})
	}
}

// TestEvaluateImmediateVerdict tests that the Immediate operation correctly
// loads bytes data of all lengths into all supported registers.
func TestEvaluateImmediateBytesData(t *testing.T) {
	bytes := []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10}
	for blen := 1; blen <= len(bytes); blen++ {
		for _, registerSize := range []int{linux.NFT_REG32_SIZE, linux.NFT_REG_SIZE} {
			if blen > registerSize {
				continue
			}
			tname := fmt.Sprintf("immediately load %d bytes into %d-byte registers", blen, registerSize)
			t.Run(tname, func(t *testing.T) {
				// Sets up an NFTables object with a base chain with policy accept.
				nf := NewNFTables()
				tab, err := nf.AddTable(arbitraryFamily, "test", "test table", false)
				if err != nil {
					t.Fatalf("unexpected error for AddTable: %v", err)
				}
				bc, err := tab.AddChain("base_chain", nil, "test chain", false)
				if err != nil {
					t.Fatalf("unexpected error for AddChain: %v", err)
				}
				bc.SetBaseChainInfo(arbitraryInfoPolicyAccept)

				// Adds a rule and immediate operation per register of registerSize.
				switch registerSize {
				case linux.NFT_REG32_SIZE:
					for reg := linux.NFT_REG32_00; reg <= linux.NFT_REG32_15; reg++ {
						rule := &Rule{}
						rule.addOperation(mustCreateImmediate(t, uint8(reg), newBytesData(bytes[:blen])))
						if err := bc.RegisterRule(rule, -1); err != nil {
							t.Fatalf("unexpected error for RegisterRule for rule %d: %v", reg-linux.NFT_REG32_00, err)
						}
					}
				case linux.NFT_REG_SIZE:
					for reg := linux.NFT_REG_1; reg <= linux.NFT_REG_4; reg++ {
						rule := &Rule{}
						rule.addOperation(mustCreateImmediate(t, uint8(reg), newBytesData(bytes[:blen])))
						if err := bc.RegisterRule(rule, -1); err != nil {
							t.Fatalf("unexpected error for RegisterRule for rule %d: %v", reg-linux.NFT_REG_1, err)
						}
					}
				}
				// Runs evaluation and checks for default policy verdict accept
				pkt := makeArbitraryGeneralPacket(arbitraryReservedHeaderBytes)
				v, err := nf.EvaluateHook(arbitraryFamily, arbitraryHook, pkt)
				if err != nil {
					t.Fatalf("unexpected error for EvaluateHook: %v", err)
				}
				if v.Code != linux.NF_ACCEPT {
					t.Fatalf("expected default policy verdict accept, got %v", v)
				}
			})
		}
	}
}

// TestEvaluateComparison tests that the Comparison operation correctly compares
// the data in the source register to the given data.
// Note: Relies on expected behavior of the Immediate operation.
func TestEvaluateComparison(t *testing.T) {
	for _, test := range []struct {
		tname string
		op1   operation // will be nil if unused
		op2   operation // will be nil if unused
		res   bool      // should be true if we reach end of the rule (no breaks)
	}{
		// 4-byte data comparisons, alternates between 4-byte and 16-byte registers.
		{
			tname: "compare register == 4-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData([]byte{0, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare register == 4-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG32_11, newBytesData([]byte{1, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG32_11, linux.NFT_CMP_EQ, newBytesData([]byte{0, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare register != 4-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG32_03, newBytesData([]byte{1, 7, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG32_03, linux.NFT_CMP_NEQ, newBytesData([]byte{1, 98, 0, 56})),
			res:   true,
		},
		{
			tname: "compare register != 4-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{1, 98, 0, 56})),
			op2:   mustCreateComparison(t, linux.NFT_REG_3, linux.NFT_CMP_NEQ, newBytesData([]byte{1, 98, 0, 56})),
			res:   false,
		},
		{
			tname: "compare register < 4-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{29, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_LT, newBytesData([]byte{100, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare register < 4-byte data, false eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG32_04, newBytesData([]byte{100, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG32_04, linux.NFT_CMP_LT, newBytesData([]byte{100, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare register < 4-byte data, false gt",
			op1:   mustCreateImmediate(t, linux.NFT_REG32_14, newBytesData([]byte{200, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG32_14, linux.NFT_CMP_LT, newBytesData([]byte{100, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare register > 4-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG32_15, newBytesData([]byte{29, 76, 230, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG32_15, linux.NFT_CMP_GT, newBytesData([]byte{0, 0, 0, 1})),
			res:   true,
		},
		{
			tname: "compare register > 4-byte data, false eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG32_07, newBytesData([]byte{29, 76, 230, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG32_07, linux.NFT_CMP_GT, newBytesData([]byte{29, 76, 230, 0})),
			res:   false,
		},
		{
			tname: "compare register > 4-byte data, false lt",
			op1:   mustCreateImmediate(t, linux.NFT_REG32_05, newBytesData([]byte{28, 76, 230, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG32_05, linux.NFT_CMP_GT, newBytesData([]byte{29, 76, 230, 0})),
			res:   false,
		},
		{
			tname: "compare register <= 4-byte data, true lt",
			op1:   mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{29, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_2, linux.NFT_CMP_LTE, newBytesData([]byte{100, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare register <= 4-byte data, true eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG32_09, newBytesData([]byte{100, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG32_09, linux.NFT_CMP_LTE, newBytesData([]byte{100, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare register <= 4-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG32_06, newBytesData([]byte{200, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG32_06, linux.NFT_CMP_LTE, newBytesData([]byte{100, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare register >= 4-byte data, true gt",
			op1:   mustCreateImmediate(t, linux.NFT_REG32_12, newBytesData([]byte{29, 76, 230, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG32_12, linux.NFT_CMP_GTE, newBytesData([]byte{0, 0, 0, 1})),
			res:   true,
		},
		{
			tname: "compare register >= 4-byte data, true eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{29, 76, 230, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_GTE, newBytesData([]byte{29, 76, 230, 0})),
			res:   true,
		},
		{
			tname: "compare register >= 4-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{28, 76, 230, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_3, linux.NFT_CMP_GTE, newBytesData([]byte{29, 76, 230, 0})),
			res:   false,
		},
		// 8-byte data comparisons.
		{
			tname: "compare register == 8-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{0, 0, 0, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData([]byte{0, 0, 0, 0, 0, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare register == 8-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{1, 0, 0, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_2, linux.NFT_CMP_EQ, newBytesData([]byte{0, 0, 0, 0, 0, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare register != 8-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{1, 7, 0, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_3, linux.NFT_CMP_NEQ, newBytesData([]byte{1, 98, 0, 56, 0, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare register != 8-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{1, 98, 0, 56, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_NEQ, newBytesData([]byte{1, 98, 0, 56, 0, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare register < 8-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{29, 0, 0, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_LT, newBytesData([]byte{100, 0, 0, 0, 0, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare register < 8-byte data, false eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{100, 0, 0, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_2, linux.NFT_CMP_LT, newBytesData([]byte{100, 0, 0, 0, 0, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare register < 8-byte data, false gt",
			op1:   mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{200, 0, 0, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_3, linux.NFT_CMP_LT, newBytesData([]byte{100, 0, 0, 0, 0, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare register > 8-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{29, 76, 230, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_GT, newBytesData([]byte{0, 0, 0, 1, 0, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare register > 8-byte data, false eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{29, 76, 230, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_GT, newBytesData([]byte{29, 76, 230, 0, 0, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare register > 8-byte data, false lt",
			op1:   mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{28, 76, 230, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_2, linux.NFT_CMP_GT, newBytesData([]byte{29, 76, 230, 0, 0, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare register <= 8-byte data, true lt",
			op1:   mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{29, 0, 0, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_3, linux.NFT_CMP_LTE, newBytesData([]byte{100, 0, 0, 0, 0, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare register <= 8-byte data, true eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{100, 0, 0, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_LTE, newBytesData([]byte{100, 0, 0, 0, 0, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare register <= 8-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{200, 0, 0, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_LTE, newBytesData([]byte{100, 0, 0, 0, 0, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare register >= 8-byte data, true gt",
			op1:   mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{30, 0, 0, 1, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_2, linux.NFT_CMP_GTE, newBytesData([]byte{29, 76, 230, 0, 0, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare register >= 8-byte data, true eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{29, 76, 230, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_3, linux.NFT_CMP_GTE, newBytesData([]byte{29, 76, 230, 0, 0, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare register >= 8-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{28, 76, 230, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_GTE, newBytesData([]byte{29, 76, 230, 0, 0, 0, 0, 0})),
			res:   false,
		},
		// 12-byte data comparisons.
		{
			tname: "compare register == 12-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData([]byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare register == 12-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1})),
			op2:   mustCreateComparison(t, linux.NFT_REG_2, linux.NFT_CMP_EQ, newBytesData([]byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare register != 12-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})),
			op2:   mustCreateComparison(t, linux.NFT_REG_3, linux.NFT_CMP_NEQ, newBytesData([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})),
			res:   true,
		},
		{
			tname: "compare register != 12-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})),
			op2:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_NEQ, newBytesData([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})),
			res:   false,
		},
		{
			tname: "compare register < 12-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{0x0a, 0x00, 0x01, 0x1f, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_LT, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			res:   true,
		},
		{
			tname: "compare register < 12-byte data, false eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			op2:   mustCreateComparison(t, linux.NFT_REG_2, linux.NFT_CMP_LT, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			res:   false,
		},
		{
			tname: "compare register < 12-byte data, false gt",
			op1:   mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{0x0a, 0x00, 0x01, 0x21, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			op2:   mustCreateComparison(t, linux.NFT_REG_3, linux.NFT_CMP_LT, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			res:   false,
		},
		{
			tname: "compare register > 12-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{0x0a, 0x00, 0x01, 0x21, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			op2:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_GT, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			res:   true,
		},
		{
			tname: "compare register > 12-byte data, false eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_GT, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			res:   false,
		},
		{
			tname: "compare register > 12-byte data, false lt",
			op1:   mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{0x0a, 0x00, 0x01, 0x1f, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			op2:   mustCreateComparison(t, linux.NFT_REG_2, linux.NFT_CMP_GT, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			res:   false,
		},
		{
			tname: "compare register <= 12-byte data, true lt",
			op1:   mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			op2:   mustCreateComparison(t, linux.NFT_REG_3, linux.NFT_CMP_LTE, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			res:   true,
		},
		{
			tname: "compare register <= 12-byte data, true eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			op2:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_LTE, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			res:   true,
		},
		{
			tname: "compare register <= 12-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{0xaa, 0xaa, 0xaa, 0x20, 0xaa, 0xaa, 0xaa, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_LTE, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			res:   false,
		},
		{
			tname: "compare register >= 12-byte data, true gt",
			op1:   mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{0xaa, 0xaa, 0xaa, 0x20, 0xaa, 0xaa, 0xaa, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			op2:   mustCreateComparison(t, linux.NFT_REG_2, linux.NFT_CMP_GTE, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			res:   true,
		},
		{
			tname: "compare register >= 12-byte data, true eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{0xab, 0xbc, 0xcd, 0xde, 0xef, 0x00, 0x01, 0x12, 0x23, 0x34, 0x45, 0x56})),
			op2:   mustCreateComparison(t, linux.NFT_REG_3, linux.NFT_CMP_GTE, newBytesData([]byte{0xab, 0xbc, 0xcd, 0xde, 0xef, 0x00, 0x01, 0x12, 0x23, 0x34, 0x45, 0x56})),
			res:   true,
		},
		{
			tname: "compare register >= 12-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{0x0a, 0x00, 0x01, 0x19, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			op2:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_GTE, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00})),
			res:   false,
		},
		// 16-byte data comparisons.
		{
			tname: "compare register == 16-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData([]byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare register == 16-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
			op2:   mustCreateComparison(t, linux.NFT_REG_2, linux.NFT_CMP_EQ, newBytesData([]byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1})),
			res:   false,
		},
		{
			tname: "compare register != 16-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})),
			op2:   mustCreateComparison(t, linux.NFT_REG_3, linux.NFT_CMP_NEQ, newBytesData([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})),
			res:   true,
		},
		{
			tname: "compare register != 16-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})),
			op2:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_NEQ, newBytesData([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})),
			res:   false,
		},
		{
			tname: "compare register < 16-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{0x0a, 0x00, 0x01, 0x1f, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0xaa})),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_LT, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			res:   true,
		},
		{
			tname: "compare register < 16-byte data, false eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			op2:   mustCreateComparison(t, linux.NFT_REG_2, linux.NFT_CMP_LT, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			res:   false,
		},
		{
			tname: "compare register < 16-byte data, false gt",
			op1:   mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{0x0a, 0x00, 0x01, 0x21, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0xaa})),
			op2:   mustCreateComparison(t, linux.NFT_REG_3, linux.NFT_CMP_LT, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			res:   false,
		},
		{
			tname: "compare register > 16-byte data, true",
			op1:   mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{0x0a, 0x00, 0x01, 0x21, 0xaa, 0xaa, 0xaa, 0xaa, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			op2:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_GT, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0xcc, 0xcc, 0xcc, 0xcc, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			res:   true,
		},
		{
			tname: "compare register > 16-byte data, false eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_GT, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			res:   false,
		},
		{
			tname: "compare register > 16-byte data, false lt",
			op1:   mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{0x0a, 0x00, 0x01, 0x1f, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x90})),
			op2:   mustCreateComparison(t, linux.NFT_REG_2, linux.NFT_CMP_GT, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			res:   false,
		},
		{
			tname: "compare register <= 16-byte data, true lt",
			op1:   mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x86})),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_LTE, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			res:   true,
		},
		{
			tname: "compare register <= 16-byte data, true eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			op2:   mustCreateComparison(t, linux.NFT_REG_2, linux.NFT_CMP_LTE, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			res:   true,
		},
		{
			tname: "compare register <= 16-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0xaa, 0x00, 0x0b, 0x13, 0x6a, 0x88})),
			op2:   mustCreateComparison(t, linux.NFT_REG_3, linux.NFT_CMP_LTE, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			res:   false,
		},
		{
			tname: "compare register >= 16-byte data, true gt",
			op1:   mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{0xaa, 0xaa, 0xaa, 0x20, 0xaa, 0xaa, 0xaa, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			op2:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_GTE, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			res:   true,
		},
		{
			tname: "compare register >= 16-byte data, true eq",
			op1:   mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{0xab, 0xbc, 0xcd, 0xde, 0xef, 0x00, 0x01, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78, 0x89, 0x90})),
			op2:   mustCreateComparison(t, linux.NFT_REG_3, linux.NFT_CMP_GTE, newBytesData([]byte{0xab, 0xbc, 0xcd, 0xde, 0xef, 0x00, 0x01, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78, 0x89, 0x90})),
			res:   true,
		},
		{
			tname: "compare register >= 16-byte data, false",
			op1:   mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0a, 0x13, 0x6a, 0x85})),
			op2:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_GTE, newBytesData([]byte{0x0a, 0x00, 0x01, 0x20, 0x00, 0x00, 0x0f, 0x13, 0xc0, 0x09, 0x00, 0x00, 0x0b, 0x13, 0x6a, 0x87})),
			res:   false,
		},
		// Empty register comparisons.
		{
			tname: "compare empty 4-byte register, true",
			op1:   mustCreateComparison(t, linux.NFT_REG32_10, linux.NFT_CMP_EQ, newBytesData([]byte{0, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare empty 4-byte register, false",
			op1:   mustCreateComparison(t, linux.NFT_REG32_11, linux.NFT_CMP_EQ, newBytesData([]byte{1, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare empty 8-byte register, true",
			op1:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_NEQ, newBytesData([]byte{1, 1, 1, 1, 0, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare empty 8-byte register, false",
			op1:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_GT, newBytesData([]byte{1, 1, 1, 1, 0, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare empty 12-byte register, true",
			op1:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_LTE, newBytesData([]byte{1, 1, 1, 1, 0, 0, 0, 0, 8, 9, 10, 11})),
			res:   true,
		},
		{
			tname: "compare empty 12-byte register, false",
			op1:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_NEQ, newBytesData([]byte{0, 0, 0, 0, 0, 0, 0, 0})),
			res:   false,
		},
		{
			tname: "compare empty 16-byte register, true",
			op1:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_LT, newBytesData([]byte{1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
			res:   true,
		},
		{
			tname: "compare empty 16-byte register, false",
			op1:   mustCreateComparison(t, linux.NFT_REG_4, linux.NFT_CMP_GTE, newBytesData([]byte{1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})),
			res:   false,
		},
	} {
		t.Run(test.tname, func(t *testing.T) {
			// Sets up an NFTables object with a single table, chain, and rule.
			nf := NewNFTables()
			tab, err := nf.AddTable(arbitraryFamily, "test", "test table", false)
			if err != nil {
				t.Fatalf("unexpected error for AddTable: %v", err)
			}
			bc, err := tab.AddChain("base_chain", nil, "test chain", false)
			if err != nil {
				t.Fatalf("unexpected error for AddChain: %v", err)
			}
			bc.SetBaseChainInfo(arbitraryInfoPolicyAccept)
			rule := &Rule{}

			// Adds testing operations.
			if test.op1 != nil {
				rule.addOperation(test.op1)
			}
			if test.op2 != nil {
				rule.addOperation(test.op2)
			}

			// Add an operation that drops. This is what the final verdict should be
			// if all the comparisons are true (res = true).
			rule.addOperation(mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})))

			// Registers the rule to the base chain.
			if err := bc.RegisterRule(rule, -1); err != nil {
				t.Fatalf("unexpected error for RegisterRule: %v", err)
			}

			// Runs evaluation and checks verdict.
			pkt := makeArbitraryGeneralPacket(arbitraryReservedHeaderBytes)
			v, err := nf.EvaluateHook(arbitraryFamily, arbitraryHook, pkt)
			if err != nil {
				t.Fatalf("unexpected error for EvaluateHook: %v", err)
			}
			// If all comparisons are true, the packet will get to the end of the rule
			// and the last operation above will set the final verdict to oppose the
			// base chain policy. If any comparison is false, the comparison operation
			// will break from the rule and the final verdict will default to the base
			// chain policy.
			if test.res {
				if v.Code != VC(linux.NF_DROP) {
					t.Fatalf("expected verdict Drop for %t result, got %v", test.res, v)
				}
			} else {
				if v.Code != VC(linux.NF_ACCEPT) {
					t.Fatalf("expected base chain policy verdict Accept for %t result, got %v", test.res, v)
				}
			}
		})
	}
}

// TestEvaluatePayloadLoad tests that the Payload Load operation correctly loads
// the specified payload into the destination register.
// The nft binary commands used to generate these are stated above each test.
// All commands should be preceded by nft --debug=netlink.
// Note: Relies on expected behavior of the Comparison operation.
// TODO(b/339691111): Add tests for VLAN, ARP, ICMP, ICMPv6, IGMP, UDP headers.
func TestEvaluatePayloadLoad(t *testing.T) {
	// Sets testing packets.
	ethernetPacket := makeArbitraryEtherPacket(0)
	ipv4Packet := makeArbitraryIPv4Packet(header.IPv4MinimumSize)
	ipv6Packet := makeArbitraryIPv6Packet(header.IPv6MinimumSize)
	tcpPacket := makeArbitraryIPv4TCPPacket(header.IPv4MinimumSize + header.TCPMinimumSize)

	for _, test := range []struct {
		tname string
		pkt   *stack.PacketBuffer
		op1   operation // Payload Load operation to test.
		op2   operation // Comparison operation to check resulting data in register,
		// nil if expecting a break during evaluation.
	}{
		// Ethernet header expression commands.
		{ // cmd: add rule ip tab ch ether saddr 02:02:03:04:05:06
			tname: "load ethernet header source address",
			pkt:   ethernetPacket,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_LL_HEADER, ethSrcAddrOffset, ethSrcAddrLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(arbitraryLinkAddrB[:])),
		},
		{ // cmd: add rule ip tab ch ether daddr 02:02:03:04:05:07
			tname: "load ethernet header destination address",
			pkt:   ethernetPacket,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_LL_HEADER, ethDstAddrOffset, ethDstAddrLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(arbitraryLinkAddrB2[:])),
		},
		{ // cmd: add rule ip tab ch ether type ip
			tname: "load ethernet header type",
			pkt:   ethernetPacket,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_LL_HEADER, ethTypeOffset, ethTypeLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(numToBE(int(arbitraryEthernetType), ethTypeLen))),
		},

		// IPv4 header expression commands.
		{ // cmd: add rule ip tab ch ip length 20
			tname: "load ipv4 header length",
			pkt:   ipv4Packet,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_NETWORK_HEADER, ipv4LengthOffset, ipv4LengthLen, linux.NFT_REG32_01),
			op2:   mustCreateComparison(t, linux.NFT_REG32_01, linux.NFT_CMP_EQ, newBytesData(numToBE(header.IPv4MinimumSize, ipv4LengthLen))),
		},
		{ // cmd: add rule ip tab ch ip id 3
			tname: "load ipv4 header ip id",
			pkt:   ipv4Packet,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_NETWORK_HEADER, ipv4IDOffset, ipv4IDLen, linux.NFT_REG32_01),
			op2:   mustCreateComparison(t, linux.NFT_REG32_01, linux.NFT_CMP_EQ, newBytesData(numToBE(arbitraryHeaderID, ipv4IDLen))),
		},
		{ // cmd: add rule ip tab ch ip frag-off 0
			tname: "load ipv4 header fragment offset",
			pkt:   ipv4Packet,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_NETWORK_HEADER, ipv4FragOffOffset, ipv4FragOffLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(numToBE(0, ipv4FragOffLen))),
		},
		// Though the packet is fragmented, there should be no issue because we are
		// changing data within the network header.
		{ // cmd: add rule ip tab ch ip frag-off 1
			tname: "load ipv4 header fragment offset non zero for fragmented packet",
			pkt:   makeFragmentedIPv4Packet(header.IPv4MinimumSize),
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_NETWORK_HEADER, ipv4FragOffOffset, ipv4FragOffLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(numToBE(arbitraryNonZeroFragmentOffset/8, ipv4FragOffLen))),
			// we divide by 8 because the fragment offset is in units of 8 bytes,
			// which is encoded into the packet in IPv4.Encode()
		},
		{ // cmd: add rule ip tab ch ip ttl 64
			tname: "load ipv4 header time to live",
			pkt:   ipv4Packet,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_NETWORK_HEADER, ipv4TTLOffset, ipv4TTLLen, linux.NFT_REG32_01),
			op2:   mustCreateComparison(t, linux.NFT_REG32_01, linux.NFT_CMP_EQ, newBytesData(numToBE(arbitraryTimeToLive, ipv4TTLLen))),
		},
		{ // cmd: add rule ip tab ch tcp
			tname: "load ipv4 header protocol",
			pkt:   ipv4Packet,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_NETWORK_HEADER, ipv4ProtocolOffset, ipv4ProtocolLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(numToBE(int(transportProtocol), ipv4ProtocolLen))),
		},
		{ // cmd: add rule ip tab ch ip saddr 192.168.1.1
			tname: "load ipv4 header source address",
			pkt:   ipv4Packet,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_NETWORK_HEADER, ipv4SrcAddrOffset, ipv4SrcAddrLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(arbitraryIPv4AddrB[:])),
		},
		{ // cmd: add rule ip tab ch ip daddr 192.168.1.9
			tname: "load ipv4 header destination address",
			pkt:   ipv4Packet,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_NETWORK_HEADER, ipv4DstAddrOffset, ipv4DstAddrLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(arbitraryIPv4AddrB2[:])),
		},
		{ // cmd: add rule ip tab ch ip checksum __
			tname: "load ipv4 header checksum",
			pkt:   ipv4Packet,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_NETWORK_HEADER, ipv4ChecksumOffset, ipv4ChecksumLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(numToBE(int(header.IPv4(ipv4Packet.NetworkHeader().Slice()).Checksum()), ipv4ChecksumLen))),
		},

		// IPv6 header expression commands.
		{ // cmd: add rule ip6 tab ch ip6 length 0
			tname: "load ipv6 header length",
			pkt:   ipv6Packet,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_NETWORK_HEADER, ipv6LengthOffset, ipv6LengthLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(numToBE(0, ipv6LengthLen))),
		},
		{ // cmd: add rule ip6 tab ch ip6 nexthdr tcp
			tname: "load ipv6 header next header",
			pkt:   ipv6Packet,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_NETWORK_HEADER, ipv6NextHdrOffset, ipv6NextHdrLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(numToBE(int(transportProtocol), ipv6NextHdrLen))),
		},
		{ // cmd: add rule ip6 tab ch ip6 hoplimit 64
			tname: "load ipv6 header hop limit",
			pkt:   ipv6Packet,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_NETWORK_HEADER, ipv6HopLimitOffset, ipv6HopLimitLen, linux.NFT_REG32_01),
			op2:   mustCreateComparison(t, linux.NFT_REG32_01, linux.NFT_CMP_EQ, newBytesData(numToBE(arbitraryTimeToLive, ipv6HopLimitLen))),
		},
		{ // cmd: add rule ip6 tab ch ip6 saddr 2001:db8:85a3::aa
			tname: "load ipv6 header source address",
			pkt:   ipv6Packet,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_NETWORK_HEADER, ipv6SrcAddrOffset, ipv6SrcAddrLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(arbitraryIPv6AddrB[:])),
		},
		{ // cmd: add rule ip6 tab ch ip6 saddr 2001:db8:85a3::bb
			tname: "load ipv6 header destination address",
			pkt:   ipv6Packet,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_NETWORK_HEADER, ipv6DstAddrOffset, ipv6DstAddrLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(arbitraryIPv6AddrB2[:])),
		},

		// TCP header expression commands.
		// Since we are changing data within the transport header with a fragmented
		// IPv4 packet, this can be problematic, so the evaluation should break.
		{
			tname: "load for transport header with a fragmented ipv4 packet",
			pkt: func() *stack.PacketBuffer {
				p := makeFragmentedIPv4Packet(header.IPv4MinimumSize + header.TCPMinimumSize)
				tcpHdr := header.TCP(p.TransportHeader().Push(header.TCPMinimumSize))
				tcpHdr.Encode(&header.TCPFields{
					SrcPort:       uint16(arbitraryPort),
					DstPort:       uint16(arbitraryPort2),
					SeqNum:        uint32(tcpSeqNum),
					AckNum:        uint32(tcpAckNum),
					DataOffset:    header.TCPMinimumSize,
					WindowSize:    uint16(tcpWinSize),
					Checksum:      0,
					UrgentPointer: uint16(tcpUrgentPointer),
				})
				tcpHdr.SetChecksum(tcpHdr.CalculateChecksum(header.PseudoHeaderChecksum(
					header.TCPProtocolNumber,
					tcpip.AddrFrom4(arbitraryIPv4AddrB),
					tcpip.AddrFrom4(arbitraryIPv4AddrB2),
					header.TCPMinimumSize,
				)))
				p.TransportProtocolNumber = header.TCPProtocolNumber
				return p
			}(),
			op1: mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_TRANSPORT_HEADER, tcpSrcPortOffset, tcpSrcPortLen, linux.NFT_REG_1),
			op2: nil,
		},
		{ // cmd: add rule ip tab ch tcp sport 12345
			tname: "load tcp header source port",
			pkt:   tcpPacket,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_TRANSPORT_HEADER, tcpSrcPortOffset, tcpSrcPortLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(numToBE(arbitraryPort, tcpSrcPortLen))),
		},
		{ // cmd: add rule ip tab ch tcp dport 80
			tname: "load tcp header destination port",
			pkt:   tcpPacket,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_TRANSPORT_HEADER, tcpDstPortOffset, tcpDstPortLen, linux.NFT_REG32_01),
			op2:   mustCreateComparison(t, linux.NFT_REG32_01, linux.NFT_CMP_EQ, newBytesData(numToBE(arbitraryPort2, tcpDstPortLen))),
		},
		{
			// cmd: add rule ip tab ch tcp sequence 32
			tname: "load tcp header sequence number",
			pkt:   tcpPacket,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_TRANSPORT_HEADER, tcpSeqNumOffset, tcpSeqNumLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(numToBE(tcpSeqNum, tcpSeqNumLen))),
		},
		{ // cmd: add rule ip tab ch tcp ackseq 165
			tname: "load tcp header acknowledgement sequence number",
			pkt:   tcpPacket,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_TRANSPORT_HEADER, tcpAckNumOffset, tcpAckNumLen, linux.NFT_REG32_01),
			op2:   mustCreateComparison(t, linux.NFT_REG32_01, linux.NFT_CMP_EQ, newBytesData(numToBE(tcpAckNum, tcpAckNumLen))),
		},
		{ // cmd: add rule ip tab ch tcp window 65535
			tname: "load tcp header window",
			pkt:   tcpPacket,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_TRANSPORT_HEADER, tcpWindowOffset, tcpWindowLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(numToBE(tcpWinSize, tcpWindowLen))),
		},
		{ // cmd: add rule ip tab ch checksum __
			tname: "load tcp header checksum",
			pkt:   tcpPacket,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_TRANSPORT_HEADER, tcpChecksumOffset, tcpChecksumLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(numToBE(int(header.TCP(tcpPacket.TransportHeader().Slice()).Checksum()), tcpChecksumLen))),
		},
		{ // cmd: add rule ip tab ch urgptr 0
			tname: "load tcp header urgent pointer",
			pkt:   tcpPacket,
			op1:   mustCreatePayloadLoad(t, linux.NFT_PAYLOAD_TRANSPORT_HEADER, tcpUrgPtrOffset, tcpUrgPtrLen, linux.NFT_REG_1),
			op2:   mustCreateComparison(t, linux.NFT_REG_1, linux.NFT_CMP_EQ, newBytesData(numToBE(tcpUrgentPointer, tcpUrgPtrLen))),
		},
	} {
		t.Run(test.tname, func(t *testing.T) {
			// Sets up an NFTables object with a single table, chain, and rule.
			nf := NewNFTables()
			tab, err := nf.AddTable(arbitraryFamily, "test", "test table", false)
			if err != nil {
				t.Fatalf("unexpected error for AddTable: %v", err)
			}
			bc, err := tab.AddChain("base_chain", nil, "test chain", false)
			if err != nil {
				t.Fatalf("unexpected error for AddChain: %v", err)
			}
			bc.SetBaseChainInfo(arbitraryInfoPolicyAccept)
			rule := &Rule{}

			// Adds testing operations.
			if test.op1 != nil {
				rule.addOperation(test.op1)
			}
			if test.op2 != nil {
				rule.addOperation(test.op2)
			}

			// Adds drop operation. Will be final verdict if all comparisons are true.
			rule.addOperation(mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})))

			// Registers the rule to the base chain.
			if err := bc.RegisterRule(rule, -1); err != nil {
				t.Fatalf("unexpected error for RegisterRule: %v", err)
			}

			// Runs evaluation.
			v, err := nf.EvaluateHook(arbitraryFamily, arbitraryHook, test.pkt)
			if err != nil {
				t.Fatalf("unexpected error for EvaluateHook: %v", err)
			}

			// Checks for final verdict.
			if test.op2 == nil {
				// If no comparison operation is set, then payload load should break,
				// resulting in Accept as the default policy verdict.
				if v.Code != VC(linux.NF_ACCEPT) {
					t.Fatalf("expected verdict Accept for break during evaluation, got %v", v)
				}
			} else {
				// If a comparison operation is set, both payload load and comparison
				// should succeed, resulting in Drop as the final verdict.
				if v.Code != VC(linux.NF_DROP) {
					t.Fatalf("expected verdict Drop for true comparison, got %v", v)
				}
			}
		})
	}
}

// TestLoopCheckOnRegisterAndUnregister tests the loop checking and accompanying
// logic on registering and unregistering rules.
func TestLoopCheckOnRegisterAndUnregister(t *testing.T) {
	for _, test := range []struct {
		tname     string
		chains    map[string]*Chain
		verdict   Verdict
		shouldErr bool
	}{
		{
			tname: "jump to non-existent chain",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "non_existent_chain"}))},
					}},
				},
			},
			shouldErr: true,
		},
		{
			tname: "goto to non-existent chain",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "non_existent_chain"}))},
					}},
				},
			},
			shouldErr: true,
		},
		{
			tname: "jump to itself",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "base_chain"}))},
					}},
				},
			},
			shouldErr: true,
		},
		{
			tname: "goto to itself",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "base_chain"}))},
					}},
				},
			},
			shouldErr: true,
		},
		{
			tname: "simple 2-chain loop",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain"}))},
					}},
				},
				"aux_chain": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "base_chain"}))},
					}},
				},
			},
			shouldErr: true,
		},
		{
			tname: "2-chain loop with entry point outside loop",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain"}))},
					}},
				},
				"aux_chain": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "aux_chain2"}))},
					}},
				},
				"aux_chain2": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "aux_chain"}))},
					}},
				},
			},
			shouldErr: true,
		},
		{
			tname: "simple 3-chain loop",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain"}))},
					}},
				},
				"aux_chain": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain2"}))},
					}},
				},
				"aux_chain2": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "base_chain"}))},
					}},
				},
			},
			shouldErr: true,
		},
		{
			tname: "3-chain loop with entry point 2 points outside loop",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain"}))},
					}},
				},
				"aux_chain": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "aux_chain2"}))},
					}},
				},
				"aux_chain2": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain3"}))},
					}},
				},
				"aux_chain3": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "aux_chain4"}))},
					}},
				},
				"aux_chain4": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain2"}))},
					}},
				},
			},
			shouldErr: true,
		},
		{
			tname: "simple 4-chain loop",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain"}))},
					}},
				},
				"aux_chain": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "aux_chain2"}))},
					}},
				},
				"aux_chain2": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain3"}))},
					}},
				},
				"aux_chain3": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "base_chain"}))},
					}},
				},
			},
			shouldErr: true,
		},
		{
			tname: "simple 5-chain loop",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain"}))},
					}},
				},
				"aux_chain": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "aux_chain2"}))},
					}},
				},
				"aux_chain2": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain3"}))},
					}},
				},
				"aux_chain3": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "base_chain"}))},
					}},
				},
			},
			shouldErr: true,
		},
		{
			//     0
			//  	/ \
			//   v   v
			//   1 <- 2 <-> 3
			tname: "complex 2-3 loop",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{&Rule{
						ops: []operation{
							mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain"})),
							mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain2"})),
						},
					}},
				},
				"aux_chain": &Chain{
					comment: "strictly target",
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)}))},
					}},
				},
				"aux_chain2": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{
							mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain"})),
							mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain3"})),
						},
					}},
				},
				"aux_chain3": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "aux_chain2"}))},
					}},
				},
			},
			shouldErr: true,
		},
		{
			tname: "simple loop amongst other rules and operations",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{
						&Rule{ops: []operation{mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{0, 1, 2, 3}))}},
						&Rule{ops: []operation{mustCreateImmediate(t, linux.NFT_REG32_14, newBytesData([]byte{0, 1, 2, 3}))}},
						&Rule{ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain"}))}},
					},
				},
				"aux_chain": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{
							mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})),
							mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "aux_chain2"})),
						},
					}},
				},
				"aux_chain2": &Chain{
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain3"}))},
					}},
				},
				"aux_chain3": &Chain{
					rules: []*Rule{
						&Rule{ops: []operation{mustCreateImmediate(t, linux.NFT_REG_1, newBytesData([]byte{0, 1, 2, 3}))}},
						&Rule{ops: []operation{mustCreateImmediate(t, linux.NFT_REG32_14, newBytesData([]byte{0, 1, 2, 3}))}},
						&Rule{ops: []operation{
							mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})),
							mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_GOTO), ChainName: "aux_chain"})),
							mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})),
						}},
					},
				},
			},
			shouldErr: true,
		},
		{
			tname: "base chain jump to 3 other chains",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{
						&Rule{
							ops: []operation{
								mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain"})),
								mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain2"})),
							},
						},
						&Rule{ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain3"}))}},
					},
				},
				"aux_chain": &Chain{
					comment: "strictly target",
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}))},
					}},
				},
				"aux_chain2": &Chain{
					comment: "strictly target",
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}))},
					}},
				},
				"aux_chain3": &Chain{
					comment: "strictly target",
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}))},
					}},
				},
			},
			verdict: Verdict{Code: VC(linux.NF_ACCEPT)}, // from base chain policy
		},
		{
			tname: "base chain jump to 3 other chains with last chain dropping",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{
						&Rule{
							ops: []operation{
								mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain"})),
								mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain2"})),
							},
						},
						&Rule{ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain3"}))}},
					},
				},
				"aux_chain": &Chain{
					comment: "strictly target",
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}))},
					}},
				},
				"aux_chain2": &Chain{
					comment: "strictly target",
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}))},
					}},
				},
				"aux_chain3": &Chain{
					comment: "strictly target",
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)}))},
					}},
				},
			},
			verdict: Verdict{Code: VC(linux.NF_DROP)}, // from last chain
		},
		{
			tname: "base chain jump to 3 other chains with last rule in base chain dropping",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{
						&Rule{
							ops: []operation{
								mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain"})),
								mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain2"})),
							},
						},
						&Rule{ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain3"}))}},
						&Rule{ops: []operation{mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)}))}},
					},
				},
				"aux_chain": &Chain{
					comment: "strictly target",
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_2, newBytesData([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}))},
					}},
				},
				"aux_chain2": &Chain{
					comment: "strictly target",
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_3, newBytesData([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}))},
					}},
				},
				"aux_chain3": &Chain{
					comment: "strictly target",
					rules: []*Rule{&Rule{
						ops: []operation{mustCreateImmediate(t, linux.NFT_REG_4, newBytesData([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}))},
					}},
				},
			},
			verdict: Verdict{Code: VC(linux.NF_DROP)}, // from last rule in base chain
		},
		{
			tname: "jump to the same chain",
			chains: map[string]*Chain{
				"base_chain": &Chain{
					baseChainInfo: arbitraryInfoPolicyAccept,
					rules: []*Rule{
						&Rule{
							ops: []operation{
								mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain"})),
								mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NFT_JUMP), ChainName: "aux_chain"})),
							},
						},
					},
				},
				"aux_chain": &Chain{
					comment: "strictly target",
					rules:   []*Rule{&Rule{}},
				},
			},
			verdict: Verdict{Code: VC(linux.NF_ACCEPT)}, // from base chain policy
		},
	} {
		t.Run(test.tname, func(t *testing.T) {
			// Sets up an NFTables object based on test struct.
			nf := NewNFTables()
			tab, err := nf.AddTable(arbitraryFamily, "test", "test table", false)
			if err != nil {
				t.Fatalf("unexpected error for AddTable: %v", err)
			}
			// Creates all chains in the test struct first. This is necessary so the
			// loop checking sees the target chains exist (otherwise it would error).
			for chainName, chainInit := range test.chains {
				tab.AddChain(chainName, chainInit.GetBaseChainInfo(), chainInit.GetComment(), false)
			}
			if len(test.chains) != tab.ChainCount() {
				t.Fatalf("not all chains added to table")
			}
			// Registers all rules to all chains in the test struct.
			for chainName, chainInit := range test.chains {
				chain, err := nf.GetChain(tab.GetAddressFamily(), tab.GetName(), chainName)
				if err != nil {
					t.Fatalf("unexpected error for GetChain: %v", err)
				}
				for _, rule := range chainInit.rules {
					// Note: this is where the loop checking is triggered.
					if err := chain.RegisterRule(rule, -1); err != nil {
						if !test.shouldErr {
							t.Fatalf("unexpected error for RegisterRule: %v", err)
						}
						return
					}
					// Checks that the chain was assigned to the rule.
					if rule.chain == nil {
						t.Fatalf("chain is not assigned to rule after RegisterRule")
					}
				}
				if chainInit.RuleCount() != chain.RuleCount() {
					t.Fatalf("not all rules added to chain")
				}
			}

			// Runs evaluation and checks verdict.
			pkt := makeArbitraryGeneralPacket(arbitraryReservedHeaderBytes)
			v, err := nf.EvaluateHook(arbitraryFamily, arbitraryHook, pkt)
			if err != nil {
				if test.verdict.ChainName != "error" {
					t.Fatalf("unexpected error for EvaluateHook: %v", err)
				}
			}
			if v.Code != test.verdict.Code {
				t.Fatalf("expected verdict %v, got %v", test.verdict, v)
			}

			// Unregisters all rules from all chains and checks that the chain is
			// unassigned from the rule.
			for chainName, chainInit := range test.chains {
				chain, err := nf.GetChain(tab.GetAddressFamily(), tab.GetName(), chainName)
				if err != nil {
					t.Fatalf("unexpected error for GetChain: %v", err)
				}
				for rIdx := chainInit.RuleCount() - 1; rIdx >= 0; rIdx-- {
					rule, err := chain.UnregisterRule(rIdx)
					if err != nil {
						t.Fatalf("unexpected error for UnregisterRule: %v", err)
					}
					if rule != chainInit.rules[rIdx] {
						t.Fatalf("rule returned by UnregisterRule does not match previously registered rule")
					}
					if rule.chain != nil {
						t.Fatalf("chain is not unassigned from rule after UnregisterRule")
					}
				}
				if chain.RuleCount() != 0 {
					t.Fatalf("not all rules removed from chain")
				}
			}
		})
	}
}

// TestMaxNestedJumps tests the limit on nested jumps (no limit for gotos).
func TestMaxNestedJumps(t *testing.T) {
	for _, test := range []struct {
		tname         string
		useJumpOp     bool
		numberOfJumps int
		verdict       Verdict // ChainName is set to "error" if an error is expected
	}{
		{
			tname:         "nested jump limit reached with jumps",
			useJumpOp:     true,
			numberOfJumps: nestedJumpLimit,
			verdict:       Verdict{Code: VC(linux.NF_DROP)},
		},
		{
			tname:         "nested jump limit reached with gotos",
			useJumpOp:     false,
			numberOfJumps: nestedJumpLimit,
			verdict:       Verdict{Code: VC(linux.NF_DROP)},
		},
		{
			tname:         "nested jump limit exceeded with jumps",
			useJumpOp:     true,
			numberOfJumps: nestedJumpLimit + 1,
			verdict:       Verdict{ChainName: "error"},
		},
		{
			tname:         "nested jump limit exceeded with gotos",
			useJumpOp:     false,
			numberOfJumps: nestedJumpLimit + 1,
			verdict:       Verdict{Code: VC(linux.NF_DROP)}, // limit only for jumps
		},
	} {
		t.Run(test.tname, func(t *testing.T) {
			// Sets up chains of nested jumps or gotos.
			nf := NewNFTables()
			tab, err := nf.AddTable(arbitraryFamily, "test", "test table", false)
			if err != nil {
				t.Fatalf("unexpected error for AddTable: %v", err)
			}
			for i := test.numberOfJumps - 1; i >= 0; i-- {
				name := fmt.Sprintf("chain %d", i)
				c, err := tab.AddChain(name, nil, "test chain", false)
				if i == 0 {
					c.SetBaseChainInfo(arbitraryInfoPolicyAccept)
				}
				if err != nil {
					t.Fatalf("unexpected error for AddChain: %v", err)
				}
				r := &Rule{}
				if i == test.numberOfJumps-1 {
					err = r.addOperation(mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: VC(linux.NF_DROP)})))
				} else {
					targetName := fmt.Sprintf("chain %d", i+1)
					code := VC(linux.NFT_JUMP)
					if !test.useJumpOp {
						code = VC(linux.NFT_GOTO)
					}
					err = r.addOperation(mustCreateImmediate(t, linux.NFT_REG_VERDICT, newVerdictData(Verdict{Code: code, ChainName: targetName})))
				}
				if err != nil {
					t.Fatalf("unexpected error for AddOperation: %v", err)
				}
				if err := c.RegisterRule(r, -1); err != nil {
					t.Fatalf("unexpected error for RegisterRule: %v", err)
				}
			}

			// Runs evaluation and checks verdict.
			pkt := makeArbitraryGeneralPacket(arbitraryReservedHeaderBytes)
			v, err := nf.EvaluateHook(arbitraryFamily, arbitraryHook, pkt)
			if err != nil {
				if test.verdict.ChainName != "error" {
					t.Fatalf("unexpected error for EvaluateHook: %v", err)
				}
			}
			if v.Code != test.verdict.Code {
				t.Fatalf("expected verdict %v, got %v", test.verdict, v)
			}
		})
	}
}

// numToBE converts an n-byte int to Big Endian where n is in [1, 8].
// Assumes the given number can be represented in n bytes.
func numToBE(v int, n int) []byte {
	if n > 8 {
		panic("cannot support more than 8 bytes")
	}
	// Gets 8-byte slice Big Endian representation of the number.
	be64 := binary.BigEndian.AppendUint64(nil, uint64(v))
	// Returns last n bytes as the n-byte Big Endian representation.
	return be64[8-n:]
}

// packetResultString compares 2 packets by equality and returns a string
// representation.
func packetResultString(initial, final *stack.PacketBuffer) string {
	if final == nil {
		return "nil"
	}
	if reflect.DeepEqual(final, initial) {
		return "unmodified"
	}
	return "modified"
}

// mustCreateImmediate wraps the NewImmediate function for brevity.
func mustCreateImmediate(t *testing.T, dreg uint8, data registerData) *immediate {
	imm, err := newImmediate(dreg, data)
	if err != nil {
		t.Fatalf("failed to create immediate: %v", err)
	}
	return imm
}

// mustCreateComparison wraps the NewComparison function for brevity.
func mustCreateComparison(t *testing.T, sreg uint8, cop int, data registerData) *comparison {
	cmp, err := newComparison(sreg, cop, data)
	if err != nil {
		t.Fatalf("failed to create comparison: %v", err)
	}
	return cmp
}

// mustCreatePayloadLoad wraps the NewPayloadLoad function for brevity.
func mustCreatePayloadLoad(t *testing.T, base payloadBase, offset, len, dreg uint8) *payloadLoad {
	pdload, err := newPayloadLoad(base, offset, len, dreg)
	if err != nil {
		t.Fatalf("failed to create payload load: %v", err)
	}
	return pdload
}