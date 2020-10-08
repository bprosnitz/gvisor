// automatically generated by stateify.

package memdev

import (
	"gvisor.dev/gvisor/pkg/state"
)

func (f *fullDevice) StateTypeName() string {
	return "pkg/sentry/devices/memdev.fullDevice"
}

func (f *fullDevice) StateFields() []string {
	return []string{}
}

func (f *fullDevice) beforeSave() {}

func (f *fullDevice) StateSave(stateSinkObject state.Sink) {
	f.beforeSave()
}

func (f *fullDevice) afterLoad() {}

func (f *fullDevice) StateLoad(stateSourceObject state.Source) {
}

func (f *fullFD) StateTypeName() string {
	return "pkg/sentry/devices/memdev.fullFD"
}

func (f *fullFD) StateFields() []string {
	return []string{
		"vfsfd",
		"FileDescriptionDefaultImpl",
		"DentryMetadataFileDescriptionImpl",
		"NoLockFD",
	}
}

func (f *fullFD) beforeSave() {}

func (f *fullFD) StateSave(stateSinkObject state.Sink) {
	f.beforeSave()
	stateSinkObject.Save(0, &f.vfsfd)
	stateSinkObject.Save(1, &f.FileDescriptionDefaultImpl)
	stateSinkObject.Save(2, &f.DentryMetadataFileDescriptionImpl)
	stateSinkObject.Save(3, &f.NoLockFD)
}

func (f *fullFD) afterLoad() {}

func (f *fullFD) StateLoad(stateSourceObject state.Source) {
	stateSourceObject.Load(0, &f.vfsfd)
	stateSourceObject.Load(1, &f.FileDescriptionDefaultImpl)
	stateSourceObject.Load(2, &f.DentryMetadataFileDescriptionImpl)
	stateSourceObject.Load(3, &f.NoLockFD)
}

func (n *nullDevice) StateTypeName() string {
	return "pkg/sentry/devices/memdev.nullDevice"
}

func (n *nullDevice) StateFields() []string {
	return []string{}
}

func (n *nullDevice) beforeSave() {}

func (n *nullDevice) StateSave(stateSinkObject state.Sink) {
	n.beforeSave()
}

func (n *nullDevice) afterLoad() {}

func (n *nullDevice) StateLoad(stateSourceObject state.Source) {
}

func (n *nullFD) StateTypeName() string {
	return "pkg/sentry/devices/memdev.nullFD"
}

func (n *nullFD) StateFields() []string {
	return []string{
		"vfsfd",
		"FileDescriptionDefaultImpl",
		"DentryMetadataFileDescriptionImpl",
		"NoLockFD",
	}
}

func (n *nullFD) beforeSave() {}

func (n *nullFD) StateSave(stateSinkObject state.Sink) {
	n.beforeSave()
	stateSinkObject.Save(0, &n.vfsfd)
	stateSinkObject.Save(1, &n.FileDescriptionDefaultImpl)
	stateSinkObject.Save(2, &n.DentryMetadataFileDescriptionImpl)
	stateSinkObject.Save(3, &n.NoLockFD)
}

func (n *nullFD) afterLoad() {}

func (n *nullFD) StateLoad(stateSourceObject state.Source) {
	stateSourceObject.Load(0, &n.vfsfd)
	stateSourceObject.Load(1, &n.FileDescriptionDefaultImpl)
	stateSourceObject.Load(2, &n.DentryMetadataFileDescriptionImpl)
	stateSourceObject.Load(3, &n.NoLockFD)
}

func (r *randomDevice) StateTypeName() string {
	return "pkg/sentry/devices/memdev.randomDevice"
}

func (r *randomDevice) StateFields() []string {
	return []string{}
}

func (r *randomDevice) beforeSave() {}

func (r *randomDevice) StateSave(stateSinkObject state.Sink) {
	r.beforeSave()
}

func (r *randomDevice) afterLoad() {}

func (r *randomDevice) StateLoad(stateSourceObject state.Source) {
}

func (r *randomFD) StateTypeName() string {
	return "pkg/sentry/devices/memdev.randomFD"
}

func (r *randomFD) StateFields() []string {
	return []string{
		"vfsfd",
		"FileDescriptionDefaultImpl",
		"DentryMetadataFileDescriptionImpl",
		"NoLockFD",
		"off",
	}
}

func (r *randomFD) beforeSave() {}

func (r *randomFD) StateSave(stateSinkObject state.Sink) {
	r.beforeSave()
	stateSinkObject.Save(0, &r.vfsfd)
	stateSinkObject.Save(1, &r.FileDescriptionDefaultImpl)
	stateSinkObject.Save(2, &r.DentryMetadataFileDescriptionImpl)
	stateSinkObject.Save(3, &r.NoLockFD)
	stateSinkObject.Save(4, &r.off)
}

func (r *randomFD) afterLoad() {}

func (r *randomFD) StateLoad(stateSourceObject state.Source) {
	stateSourceObject.Load(0, &r.vfsfd)
	stateSourceObject.Load(1, &r.FileDescriptionDefaultImpl)
	stateSourceObject.Load(2, &r.DentryMetadataFileDescriptionImpl)
	stateSourceObject.Load(3, &r.NoLockFD)
	stateSourceObject.Load(4, &r.off)
}

func (z *zeroDevice) StateTypeName() string {
	return "pkg/sentry/devices/memdev.zeroDevice"
}

func (z *zeroDevice) StateFields() []string {
	return []string{}
}

func (z *zeroDevice) beforeSave() {}

func (z *zeroDevice) StateSave(stateSinkObject state.Sink) {
	z.beforeSave()
}

func (z *zeroDevice) afterLoad() {}

func (z *zeroDevice) StateLoad(stateSourceObject state.Source) {
}

func (z *zeroFD) StateTypeName() string {
	return "pkg/sentry/devices/memdev.zeroFD"
}

func (z *zeroFD) StateFields() []string {
	return []string{
		"vfsfd",
		"FileDescriptionDefaultImpl",
		"DentryMetadataFileDescriptionImpl",
		"NoLockFD",
	}
}

func (z *zeroFD) beforeSave() {}

func (z *zeroFD) StateSave(stateSinkObject state.Sink) {
	z.beforeSave()
	stateSinkObject.Save(0, &z.vfsfd)
	stateSinkObject.Save(1, &z.FileDescriptionDefaultImpl)
	stateSinkObject.Save(2, &z.DentryMetadataFileDescriptionImpl)
	stateSinkObject.Save(3, &z.NoLockFD)
}

func (z *zeroFD) afterLoad() {}

func (z *zeroFD) StateLoad(stateSourceObject state.Source) {
	stateSourceObject.Load(0, &z.vfsfd)
	stateSourceObject.Load(1, &z.FileDescriptionDefaultImpl)
	stateSourceObject.Load(2, &z.DentryMetadataFileDescriptionImpl)
	stateSourceObject.Load(3, &z.NoLockFD)
}

func init() {
	state.Register((*fullDevice)(nil))
	state.Register((*fullFD)(nil))
	state.Register((*nullDevice)(nil))
	state.Register((*nullFD)(nil))
	state.Register((*randomDevice)(nil))
	state.Register((*randomFD)(nil))
	state.Register((*zeroDevice)(nil))
	state.Register((*zeroFD)(nil))
}
