import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from emg.compact_model import CompactFilter,read_recording,make_dataset,compact_split,file_hashes
from emg.dataset_v2 import DatasetV2Protocol,DatasetV2Runner,DatasetV2SyntheticSource,V2_COMPACT_GESTURES,V2_ROW_FIELDS,v2_metadata
from emg.recording import SessionRecorder
from emg.features import extract_time_domain_features
from emg.model import _new_pipeline
from emg.evaluation.benchmark import DEFAULT_CONFIG,evaluate_model


def fixture(root,name='chunk',posture='reference',purpose='training'):
    protocol=DatasetV2Protocol(gestures=V2_COMPACT_GESTURES,postures=(posture,),posture_chunk=True,
                               repetitions=3,baseline_seconds=5.,gesture_seconds=3.,
                               preparation_seconds=.016,rest_seconds=.016,independent_rest_seconds=1.2,
                               posture_change_seconds=.016)
    source=DatasetV2SyntheticSource();path=root/name
    metadata=v2_metadata(name,'S01','P2',protocol,{'collection_purpose':purpose},'synthetic',source.unit)
    with SessionRecorder(path,metadata,condition_fields=V2_ROW_FIELDS) as recorder:
        DatasetV2Runner(source,recorder,protocol,announce=lambda _:None,wait_for_prompts=False).run()
        recorder.mark_complete()
    return path


class CompactTests(unittest.TestCase):
    def test_batch_and_live_filters_match_with_and_without_notch(self):
        values=np.random.default_rng(7).normal(size=(1500,8))+1000
        for notch in (None,50.):
            offline=CompactFilter(notch).process(values)
            live_filter=CompactFilter(notch)
            live=np.asarray([live_filter.update(row) for row in values])
            np.testing.assert_allclose(offline,live,rtol=1e-11,atol=1e-11)

    def test_notch_attenuates_50_hz_without_changing_shape(self):
        time=np.arange(2500)/125
        values=np.tile((np.sin(2*np.pi*50*time)+.1*np.sin(2*np.pi*20*time))[:,None],(1,8))
        ordinary=CompactFilter().process(values)[1000:]
        notched=CompactFilter(50).process(values)[1000:]
        self.assertEqual(ordinary.shape,notched.shape)
        self.assertLess(np.mean(notched**2),.05*np.mean(ordinary**2))

    def test_evaluation_purpose_is_rejected_from_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=fixture(Path(tmp),purpose='held_out_evaluation')
            with self.assertRaisesRegex(ValueError,'must never train'):
                read_recording(path)

    def test_corrections_are_applied_in_memory_and_checked_against_original_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=fixture(Path(tmp),posture='elbow_extended')
            hashes=file_hashes(path)
            with (path/'samples.csv').open(newline='') as stream:
                rows=list(csv.DictReader(stream))
            baseline=[i for i,r in enumerate(rows) if r['phase']=='baseline']
            correction={'session_id':'chunk','source_sha256':hashes,'actual_baseline_posture':'elbow_extended',
                        'sample_condition_overrides':[{'start_sample':baseline[0],'end_sample':baseline[-1]+1,
                            'phase':'baseline','recorded_arm_posture':'reference','actual_arm_posture':'elbow_extended'}]}
            (path/'condition_corrections.json').write_text(json.dumps(correction))
            recording=read_recording(path)
            self.assertEqual(recording.rows[baseline[0]]['arm_posture'],'elbow_extended')
            self.assertEqual(file_hashes(path),hashes)
            correction['source_sha256']['samples.csv']='bad'
            (path/'condition_corrections.json').write_text(json.dumps(correction))
            with self.assertRaisesRegex(ValueError,'provenance'):
                read_recording(path)

    def test_audited_trial_exclusion_preserves_raw_files_and_omits_only_named_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=fixture(Path(tmp))
            hashes=file_hashes(path)
            metadata=json.loads((path/'metadata.json').read_text())
            trial_id=metadata['trials'][0]['trial_id']
            exclusion={'schema_version':1,'session_id':'chunk','source_sha256':hashes,
                       'excluded_trials':[{'trial_id':trial_id,'reason':'participant timing report'}]}
            (path/'trial_exclusions.json').write_text(json.dumps(exclusion))
            recording=read_recording(path)
            self.assertEqual(recording.audit['selected_trials'],len(metadata['trials'])-1)
            self.assertEqual(recording.audit['excluded_trials'],
                             [{'trial_id':trial_id,'reason':'participant timing report'}])
            self.assertNotIn(trial_id,{trial['trial_id'] for trial in recording.trials})
            self.assertEqual(file_hashes(path),hashes)
            exclusion['source_sha256']['samples.csv']='bad'
            (path/'trial_exclusions.json').write_text(json.dumps(exclusion))
            with self.assertRaisesRegex(ValueError,'provenance'):
                read_recording(path)

    def test_partial_reference_reconnect_exclusion_and_whole_trial_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reference=fixture(root,'reference');chunk=fixture(root,'extended','elbow_extended')
            m=json.loads((reference/'metadata.json').read_text())
            m['session_complete']=False
            (reference/'metadata.json').write_text(json.dumps(m))
            trial=next(t for t in m['trials'] if t['gesture']=='rest')
            with (reference/'events.csv').open('a') as stream:
                stream.write(f"{trial['start_sample']+100},stream_reconnect,,\n")
            index={'reference_source_sha256':file_hashes(reference),'completed_reference_trials':m['trials']}
            recordings=[read_recording(reference,reference_index=index),read_recording(chunk)]
            self.assertEqual(recordings[0].audit['excluded_trials'][0]['trial_id'],trial['trial_id'])
            dataset,conditions,scale=make_dataset(recordings,normalization='fixed_reference')
            self.assertEqual(scale.shape,(8,));self.assertTrue(np.all(np.isfinite(dataset.data.features)))
            self.assertNotIn(trial['trial_id'],dataset.data.groups)
            train,test=compact_split(dataset,7)
            self.assertFalse(set(dataset.data.groups[train])&set(dataset.data.groups[test]))
            self.assertEqual(len(set(dataset.data.groups[test])),14)
            self.assertEqual(set(dataset.data.labels[train]),set(V2_COMPACT_GESTURES))
            self.assertEqual(set(dataset.data.labels[test]),set(V2_COMPACT_GESTURES))
            model=_new_pipeline(7).fit(dataset.data.features[train],dataset.data.labels[train])
            result=evaluate_model(model,dataset,test,DEFAULT_CONFIG)
            self.assertTrue(np.all(np.isfinite(result.raw_probabilities)))

            aligned,_,_,signals=make_dataset(
                recordings,normalization='fixed_reference',include_signal_windows=True
            )
            self.assertEqual(signals.shape,(len(aligned.windows),8,25))
            np.testing.assert_allclose(
                extract_time_domain_features(signals[0].T),aligned.data.features[0],rtol=1e-6,atol=1e-6
            )


if __name__=='__main__':
    unittest.main()
