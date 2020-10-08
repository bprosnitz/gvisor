// automatically generated by stateify.

package seqfile

import (
	"gvisor.dev/gvisor/pkg/state"
)

func (s *SeqData) StateTypeName() string {
	return "pkg/sentry/fs/proc/seqfile.SeqData"
}

func (s *SeqData) StateFields() []string {
	return []string{
		"Buf",
		"Handle",
	}
}

func (s *SeqData) beforeSave() {}

func (s *SeqData) StateSave(stateSinkObject state.Sink) {
	s.beforeSave()
	stateSinkObject.Save(0, &s.Buf)
	stateSinkObject.Save(1, &s.Handle)
}

func (s *SeqData) afterLoad() {}

func (s *SeqData) StateLoad(stateSourceObject state.Source) {
	stateSourceObject.Load(0, &s.Buf)
	stateSourceObject.Load(1, &s.Handle)
}

func (s *SeqFile) StateTypeName() string {
	return "pkg/sentry/fs/proc/seqfile.SeqFile"
}

func (s *SeqFile) StateFields() []string {
	return []string{
		"InodeSimpleExtendedAttributes",
		"InodeSimpleAttributes",
		"SeqSource",
		"source",
		"generation",
		"lastRead",
	}
}

func (s *SeqFile) beforeSave() {}

func (s *SeqFile) StateSave(stateSinkObject state.Sink) {
	s.beforeSave()
	stateSinkObject.Save(0, &s.InodeSimpleExtendedAttributes)
	stateSinkObject.Save(1, &s.InodeSimpleAttributes)
	stateSinkObject.Save(2, &s.SeqSource)
	stateSinkObject.Save(3, &s.source)
	stateSinkObject.Save(4, &s.generation)
	stateSinkObject.Save(5, &s.lastRead)
}

func (s *SeqFile) afterLoad() {}

func (s *SeqFile) StateLoad(stateSourceObject state.Source) {
	stateSourceObject.Load(0, &s.InodeSimpleExtendedAttributes)
	stateSourceObject.Load(1, &s.InodeSimpleAttributes)
	stateSourceObject.Load(2, &s.SeqSource)
	stateSourceObject.Load(3, &s.source)
	stateSourceObject.Load(4, &s.generation)
	stateSourceObject.Load(5, &s.lastRead)
}

func (s *seqFileOperations) StateTypeName() string {
	return "pkg/sentry/fs/proc/seqfile.seqFileOperations"
}

func (s *seqFileOperations) StateFields() []string {
	return []string{
		"seqFile",
	}
}

func (s *seqFileOperations) beforeSave() {}

func (s *seqFileOperations) StateSave(stateSinkObject state.Sink) {
	s.beforeSave()
	stateSinkObject.Save(0, &s.seqFile)
}

func (s *seqFileOperations) afterLoad() {}

func (s *seqFileOperations) StateLoad(stateSourceObject state.Source) {
	stateSourceObject.Load(0, &s.seqFile)
}

func init() {
	state.Register((*SeqData)(nil))
	state.Register((*SeqFile)(nil))
	state.Register((*seqFileOperations)(nil))
}
