// automatically generated by stateify.

package dev

import (
	"gvisor.dev/gvisor/pkg/state"
)

func (f *filesystem) StateTypeName() string {
	return "pkg/sentry/fs/dev.filesystem"
}

func (f *filesystem) StateFields() []string {
	return []string{}
}

func (f *filesystem) beforeSave() {}

func (f *filesystem) StateSave(stateSinkObject state.Sink) {
	f.beforeSave()
}

func (f *filesystem) afterLoad() {}

func (f *filesystem) StateLoad(stateSourceObject state.Source) {
}

func (f *fullDevice) StateTypeName() string {
	return "pkg/sentry/fs/dev.fullDevice"
}

func (f *fullDevice) StateFields() []string {
	return []string{
		"InodeSimpleAttributes",
	}
}

func (f *fullDevice) beforeSave() {}

func (f *fullDevice) StateSave(stateSinkObject state.Sink) {
	f.beforeSave()
	stateSinkObject.Save(0, &f.InodeSimpleAttributes)
}

func (f *fullDevice) afterLoad() {}

func (f *fullDevice) StateLoad(stateSourceObject state.Source) {
	stateSourceObject.Load(0, &f.InodeSimpleAttributes)
}

func (f *fullFileOperations) StateTypeName() string {
	return "pkg/sentry/fs/dev.fullFileOperations"
}

func (f *fullFileOperations) StateFields() []string {
	return []string{}
}

func (f *fullFileOperations) beforeSave() {}

func (f *fullFileOperations) StateSave(stateSinkObject state.Sink) {
	f.beforeSave()
}

func (f *fullFileOperations) afterLoad() {}

func (f *fullFileOperations) StateLoad(stateSourceObject state.Source) {
}

func (n *netTunInodeOperations) StateTypeName() string {
	return "pkg/sentry/fs/dev.netTunInodeOperations"
}

func (n *netTunInodeOperations) StateFields() []string {
	return []string{
		"InodeSimpleAttributes",
	}
}

func (n *netTunInodeOperations) beforeSave() {}

func (n *netTunInodeOperations) StateSave(stateSinkObject state.Sink) {
	n.beforeSave()
	stateSinkObject.Save(0, &n.InodeSimpleAttributes)
}

func (n *netTunInodeOperations) afterLoad() {}

func (n *netTunInodeOperations) StateLoad(stateSourceObject state.Source) {
	stateSourceObject.Load(0, &n.InodeSimpleAttributes)
}

func (n *netTunFileOperations) StateTypeName() string {
	return "pkg/sentry/fs/dev.netTunFileOperations"
}

func (n *netTunFileOperations) StateFields() []string {
	return []string{
		"device",
	}
}

func (n *netTunFileOperations) beforeSave() {}

func (n *netTunFileOperations) StateSave(stateSinkObject state.Sink) {
	n.beforeSave()
	stateSinkObject.Save(0, &n.device)
}

func (n *netTunFileOperations) afterLoad() {}

func (n *netTunFileOperations) StateLoad(stateSourceObject state.Source) {
	stateSourceObject.Load(0, &n.device)
}

func (n *nullDevice) StateTypeName() string {
	return "pkg/sentry/fs/dev.nullDevice"
}

func (n *nullDevice) StateFields() []string {
	return []string{
		"InodeSimpleAttributes",
	}
}

func (n *nullDevice) beforeSave() {}

func (n *nullDevice) StateSave(stateSinkObject state.Sink) {
	n.beforeSave()
	stateSinkObject.Save(0, &n.InodeSimpleAttributes)
}

func (n *nullDevice) afterLoad() {}

func (n *nullDevice) StateLoad(stateSourceObject state.Source) {
	stateSourceObject.Load(0, &n.InodeSimpleAttributes)
}

func (n *nullFileOperations) StateTypeName() string {
	return "pkg/sentry/fs/dev.nullFileOperations"
}

func (n *nullFileOperations) StateFields() []string {
	return []string{}
}

func (n *nullFileOperations) beforeSave() {}

func (n *nullFileOperations) StateSave(stateSinkObject state.Sink) {
	n.beforeSave()
}

func (n *nullFileOperations) afterLoad() {}

func (n *nullFileOperations) StateLoad(stateSourceObject state.Source) {
}

func (z *zeroDevice) StateTypeName() string {
	return "pkg/sentry/fs/dev.zeroDevice"
}

func (z *zeroDevice) StateFields() []string {
	return []string{
		"nullDevice",
	}
}

func (z *zeroDevice) beforeSave() {}

func (z *zeroDevice) StateSave(stateSinkObject state.Sink) {
	z.beforeSave()
	stateSinkObject.Save(0, &z.nullDevice)
}

func (z *zeroDevice) afterLoad() {}

func (z *zeroDevice) StateLoad(stateSourceObject state.Source) {
	stateSourceObject.Load(0, &z.nullDevice)
}

func (z *zeroFileOperations) StateTypeName() string {
	return "pkg/sentry/fs/dev.zeroFileOperations"
}

func (z *zeroFileOperations) StateFields() []string {
	return []string{}
}

func (z *zeroFileOperations) beforeSave() {}

func (z *zeroFileOperations) StateSave(stateSinkObject state.Sink) {
	z.beforeSave()
}

func (z *zeroFileOperations) afterLoad() {}

func (z *zeroFileOperations) StateLoad(stateSourceObject state.Source) {
}

func (r *randomDevice) StateTypeName() string {
	return "pkg/sentry/fs/dev.randomDevice"
}

func (r *randomDevice) StateFields() []string {
	return []string{
		"InodeSimpleAttributes",
	}
}

func (r *randomDevice) beforeSave() {}

func (r *randomDevice) StateSave(stateSinkObject state.Sink) {
	r.beforeSave()
	stateSinkObject.Save(0, &r.InodeSimpleAttributes)
}

func (r *randomDevice) afterLoad() {}

func (r *randomDevice) StateLoad(stateSourceObject state.Source) {
	stateSourceObject.Load(0, &r.InodeSimpleAttributes)
}

func (r *randomFileOperations) StateTypeName() string {
	return "pkg/sentry/fs/dev.randomFileOperations"
}

func (r *randomFileOperations) StateFields() []string {
	return []string{}
}

func (r *randomFileOperations) beforeSave() {}

func (r *randomFileOperations) StateSave(stateSinkObject state.Sink) {
	r.beforeSave()
}

func (r *randomFileOperations) afterLoad() {}

func (r *randomFileOperations) StateLoad(stateSourceObject state.Source) {
}

func (t *ttyInodeOperations) StateTypeName() string {
	return "pkg/sentry/fs/dev.ttyInodeOperations"
}

func (t *ttyInodeOperations) StateFields() []string {
	return []string{
		"InodeSimpleAttributes",
	}
}

func (t *ttyInodeOperations) beforeSave() {}

func (t *ttyInodeOperations) StateSave(stateSinkObject state.Sink) {
	t.beforeSave()
	stateSinkObject.Save(0, &t.InodeSimpleAttributes)
}

func (t *ttyInodeOperations) afterLoad() {}

func (t *ttyInodeOperations) StateLoad(stateSourceObject state.Source) {
	stateSourceObject.Load(0, &t.InodeSimpleAttributes)
}

func (t *ttyFileOperations) StateTypeName() string {
	return "pkg/sentry/fs/dev.ttyFileOperations"
}

func (t *ttyFileOperations) StateFields() []string {
	return []string{}
}

func (t *ttyFileOperations) beforeSave() {}

func (t *ttyFileOperations) StateSave(stateSinkObject state.Sink) {
	t.beforeSave()
}

func (t *ttyFileOperations) afterLoad() {}

func (t *ttyFileOperations) StateLoad(stateSourceObject state.Source) {
}

func init() {
	state.Register((*filesystem)(nil))
	state.Register((*fullDevice)(nil))
	state.Register((*fullFileOperations)(nil))
	state.Register((*netTunInodeOperations)(nil))
	state.Register((*netTunFileOperations)(nil))
	state.Register((*nullDevice)(nil))
	state.Register((*nullFileOperations)(nil))
	state.Register((*zeroDevice)(nil))
	state.Register((*zeroFileOperations)(nil))
	state.Register((*randomDevice)(nil))
	state.Register((*randomFileOperations)(nil))
	state.Register((*ttyInodeOperations)(nil))
	state.Register((*ttyFileOperations)(nil))
}
