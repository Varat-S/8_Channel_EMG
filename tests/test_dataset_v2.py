import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np

from emg.dataset_v2 import (DatasetV2Protocol, DatasetV2Runner, DatasetV2SyntheticSource,
                           FIVE_CORE_GESTURES, V2_GESTURES, V2_COMPACT_GESTURES, V2_POSTURES, V2_ROW_FIELDS, parse_v2_conditions,
                           v2_metadata, validate_v2_recording)
from emg.evaluation.dataset import select_sessions, write_manifest
from emg.evaluation.dataset_v2 import (calibration_split, independent_rest_metrics,
                                     load_v2_dataset, run_dataset_v2_benchmark, trial_split)
from emg.recording import SessionRecorder
from prepare_compact_recordings import audit_reference, prepare


def record(root, name="a", role="A", placement="P2", date="2026-10-06", gestures=V2_GESTURES, gesture_seconds=1.2):
    protocol = DatasetV2Protocol(session_role=role, gestures=gestures, repetitions=6,
                                baseline_seconds=1.2, preparation_seconds=.016,
                                gesture_seconds=gesture_seconds, rest_seconds=.016,
                                independent_rest_seconds=1.2, posture_change_seconds=.016)
    directory = root/"recordings"/"S01"/name
    source = DatasetV2SyntheticSource()
    metadata = v2_metadata(name, "S01", placement, protocol, {"recording_date": date}, "synthetic", source.unit)
    with SessionRecorder(directory, metadata, condition_fields=V2_ROW_FIELDS) as recorder:
        DatasetV2Runner(source, recorder, protocol, announce=lambda _: None, wait_for_prompts=False).run()
        recorder.mark_complete()
    return directory


def args(root, **overrides):
    return argparse.Namespace(**{**dict(manifest=root/"data/manifest.csv", output=root/"results",
                                       session_a="a", session_b="b", session_c="c", gestures=list(V2_GESTURES),
                                       experiment="all", training_postures=list(V2_POSTURES[:3]),
                                       calibration_reps=[0,1,2,3], choices=2, calibration_training="new-only",
                                       calibration_target="B", include_baseline_rest=False, validate_only=False, seed=7), **overrides})


class ProtocolTests(unittest.TestCase):
    def test_posture_chunk_records_only_selected_trials_but_reference_baseline_and_rest(self):
        protocol = DatasetV2Protocol(gestures=V2_COMPACT_GESTURES,
                                    postures=("elbow_extended",), posture_chunk=True, repetitions=3,
                                    baseline_seconds=1.2, preparation_seconds=.016,
                                    gesture_seconds=1.2, rest_seconds=.016,
                                    independent_rest_seconds=1.2, posture_change_seconds=.016)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "recordings/S01/chunk"
            source = DatasetV2SyntheticSource()
            metadata = v2_metadata("chunk", "S01", "P2", protocol,
                                   {"collection_purpose": "held_out_evaluation"}, "synthetic", source.unit)
            with SessionRecorder(directory, metadata, condition_fields=V2_ROW_FIELDS) as recorder:
                DatasetV2Runner(source, recorder, protocol, announce=lambda _: None, wait_for_prompts=False).run()
                recorder.mark_complete()
            self.assertEqual(validate_v2_recording(directory)["trial_count"], 21)
            with (directory / "samples.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r["arm_posture"] for r in rows if r["phase"] == "gesture"}, {"elbow_extended"})
            self.assertEqual({r["arm_posture"] for r in rows if r["phase"] in ("baseline", "rest_pre", "rest_post")}, {"reference"})
            self.assertFalse({"radial_deviation", "ulnar_deviation"} & {r["gesture"] for r in rows})
            manifest = root / "manifest.csv"
            write_manifest(root / "recordings", manifest)
            selected, _ = select_sessions(manifest, tuple(sorted(V2_COMPACT_GESTURES)), ["chunk"])
            dataset, conditions, trials, _ = load_v2_dataset(selected)
            self.assertEqual(len(trials), 21)
            self.assertEqual(set(dataset.data.labels), set(V2_COMPACT_GESTURES))
            self.assertTrue(all(c["arm_posture"] == "elbow_extended" for w, c in zip(dataset.windows, conditions) if w.phase == "gesture"))
            metadata = json.loads((directory / "metadata.json").read_text())
            metadata["posture_chunk"] = False
            (directory / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "invalid gesture/posture"):
                validate_v2_recording(directory)

    def test_collection_purpose_is_explicit_and_validated(self):
        conditions, _ = parse_v2_conditions({"collection_group_id": "compact_A", "collection_purpose": "held_out_evaluation"})
        self.assertEqual(conditions["collection_purpose"], "held_out_evaluation")
        with self.assertRaises(ValueError):
            parse_v2_conditions({"collection_purpose": "unknown"})

    def test_recording_with_unapplied_physical_condition_corrections_is_not_evaluated_silently(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = record(Path(temporary), gestures=("rest", "fist"))
            (directory / "condition_corrections.json").write_text(json.dumps({"actual_baseline_posture": "elbow_extended"}))
            with self.assertRaisesRegex(ValueError, "apply them explicitly"):
                validate_v2_recording(directory)

    def test_default_session_a_schedule_is_balanced_deterministic_and_interleaved(self):
        protocol = DatasetV2Protocol()
        postures, schedule = protocol.schedule("v2A")
        self.assertEqual(postures, V2_POSTURES)
        self.assertEqual(len(schedule), 216)
        self.assertEqual(len({t["trial_id"] for t in schedule}), 216)
        self.assertEqual(protocol.schedule("v2A"), (postures, schedule))
        for offset in range(0,216,9):
            self.assertEqual({t["gesture"] for t in schedule[offset:offset+9]}, set(V2_GESTURES))
            self.assertEqual(len({t["repetition_index"] for t in schedule[offset:offset+9]}),1)
        self.assertNotEqual(protocol.schedule("v2A"), replace(protocol, seed=11).schedule("v2A"))

    def test_b_c_have_reference_only_and_posture_order_can_be_randomized(self):
        for role in ("B","C"):
            posture, trials = DatasetV2Protocol(session_role=role).schedule(role)
            self.assertEqual(posture, ("reference",)); self.assertEqual(len(trials),54)
        order,_ = DatasetV2Protocol(randomize_postures=True).schedule("a")
        self.assertEqual(order[0],"reference"); self.assertEqual(set(order),set(V2_POSTURES))

    def test_conditions_accept_identity_and_blocks_but_reject_dynamic_or_unknown_values(self):
        conditions, blocks = parse_v2_conditions({"session_id":"a","subject_id":"S01",
                                                  "gesture_set":"dataset_v2_9class",
                                                  "block_conditions":{"elbow_flexed":{"elbow_position":"approximately_90_deg"}}})
        self.assertEqual(conditions["session_id"],"a"); self.assertIn("elbow_flexed",blocks)
        for values in ({"dynamic_arm_motion":True},{"arm_posture":"elbow_flexed"},{"unexpected":1}):
            with self.assertRaises(ValueError): parse_v2_conditions(values)
        with self.assertRaises(ValueError): DatasetV2Protocol(postures=("elbow_flexed","reference"))
        with self.assertRaises(ValueError): DatasetV2Protocol(block_conditions={"reference":{"electrode_placement_id":"P3"}})

    def test_protocol_cli_plan_and_expanded_conditions_do_not_open_hardware(self):
        project=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,str(project/"calibrate.py"),"--protocol","dataset_v2","--plan",
                               "--source","serial","--conditions-json",str(project/"data/dataset_v2_session_A.example.json")],
                              capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        metadata=json.loads(result.stdout)
        self.assertEqual(len(metadata["planned_trial_order"]),216)
        self.assertEqual(metadata["electrode_placement_id"],"P2")

    def test_five_core_conditions_select_exact_reapplication_vocabulary(self):
        project=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,str(project/"calibrate.py"),"--protocol","dataset_v2","--plan",
                               "--source","serial","--session-role","B","--conditions-json",
                               str(project/"data/reapplication_five_core_P3.example.json")],
                              capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        metadata=json.loads(result.stdout)
        self.assertEqual(tuple(metadata["gestures"]),FIVE_CORE_GESTURES)
        self.assertEqual(len(metadata["planned_trial_order"]),30)
        self.assertEqual(metadata["collection_purpose"],"held_out_evaluation")


class RecordingAndEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temporary.name)
        cls.a=record(cls.root)
        cls.b=record(cls.root,"b","B","P3")
        cls.c=record(cls.root,"c","C","P4","2026-10-07")
        cls.manifest=cls.root/"data/manifest.csv"
        write_manifest(cls.root/"recordings",cls.manifest)
        cls.sessions,_=select_sessions(cls.manifest,tuple(sorted(V2_GESTURES)),["a","b","c"])
        cls.dataset,cls.conditions,cls.trials,cls.validations=load_v2_dataset(cls.sessions)

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def reference(self, session="a"):
        return np.array([i for i,w in enumerate(self.dataset.windows) if w.phase=="gesture"
                         and w.session_id==session and self.conditions[i]["arm_posture"]=="reference"])

    def test_recorded_v2_metadata_trial_rows_and_independent_rest(self):
        validation=validate_v2_recording(self.a)
        self.assertEqual(validation["trial_count"],216)
        metadata=json.loads((self.a/"metadata.json").read_text())
        self.assertEqual(metadata["gesture_set"],"dataset_v2_9class")
        self.assertEqual(len(metadata["rest_segments"]),2)
        with (self.a/"samples.csv").open(newline='',encoding='utf-8') as file:
            rows=list(csv.DictReader(file))
        self.assertTrue(all(row["electrode_placement_id"]=="P2" for row in rows))
        self.assertTrue(all(row["dynamic_arm_motion"]=="false" for row in rows))
        self.assertEqual({row["phase"] for row in rows if row["phase"].startswith("rest_")},{"rest_pre","rest_post"})

    def test_compact_preparation_preserves_source_and_marks_missing_event_evidence(self):
        import hashlib
        source_hashes = {name: hashlib.sha256((self.a/name).read_bytes()).hexdigest()
                         for name in ("samples.csv", "events.csv", "metadata.json")}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "compact"
            prepare(self.a, output)
            collection = json.loads((output / "collection.json").read_text())
            self.assertEqual(collection["completed_reference_trial_count"], 42)
            self.assertEqual([c["trials"] for c in collection["chunks"]], [21,21,21,14])
            self.assertEqual(collection["reference_source_sha256"], source_hashes)
            self.assertEqual(collection["chunks"][-1]["purpose"], "held_out_evaluation")
        self.assertEqual(source_hashes, {name: hashlib.sha256((self.a/name).read_bytes()).hexdigest()
                                       for name in source_hashes})
        with tempfile.TemporaryDirectory() as temporary:
            import shutil
            source = Path(temporary) / "source"
            shutil.copytree(self.a, source)
            metadata, trials = audit_reference(source)
            selected_id = trials[0]["trial_id"]
            event_path = source / "events.csv"
            with event_path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                events = [e for e in reader if not (e["trial"] == selected_id and e["event"].startswith("gesture_"))]
            with event_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader(); writer.writerows(events)
            with self.assertRaisesRegex(ValueError, "boundary mismatch"):
                audit_reference(source)
            metadata["session_complete"] = False
            (source / "metadata.json").write_text(json.dumps(metadata))
            _, trials = audit_reference(source)
            self.assertEqual(next(t for t in trials if t["trial_id"] == selected_id)["boundary_evidence"],
                             "metadata_and_samples_only")
        self.assertIsNone(metadata.get("elbow_position"))
        with self.manifest.open(newline='',encoding='utf-8') as file:
            inventory=list(csv.DictReader(file))
        self.assertEqual(inventory[0]["protocol"],"dataset_v2")
        self.assertEqual(inventory[0]["posture_set"],"|".join(V2_POSTURES))

    def test_trial_grouping_and_sample_ranges_are_disjoint(self):
        train,test=trial_split(self.dataset,self.reference(),7)
        self.assertFalse(set(self.dataset.data.groups[train])&set(self.dataset.data.groups[test]))
        self.assertTrue(all("_reference_" in group for group in self.dataset.data.groups[train]))
        train_samples={(self.dataset.windows[i].session_id,s) for i in train
                       for s in range(self.dataset.windows[i].start_sample,self.dataset.windows[i].end_sample)}
        test_samples={(self.dataset.windows[i].session_id,s) for i in test
                      for s in range(self.dataset.windows[i].start_sample,self.dataset.windows[i].end_sample)}
        self.assertFalse(train_samples&test_samples)

    def test_exact_recalibration_counts_disjointness_nested_choices_and_determinism(self):
        previous=set()
        for reps in (0,1,2,3,4):
            calibration,test=calibration_split(self.dataset,self.reference("b"),reps,7,0)
            actual=set(self.dataset.data.groups[calibration])
            self.assertTrue(previous.issubset(actual)); previous=actual
            self.assertFalse(actual&set(self.dataset.data.groups[test]))
            for gesture in V2_GESTURES:
                self.assertEqual(sum(self.trials[g]["gesture"]==gesture for g in actual),reps)
            again=calibration_split(self.dataset,self.reference("b"),reps,7,0)
            np.testing.assert_array_equal(calibration,again[0]); np.testing.assert_array_equal(test,again[1])
        first=calibration_split(self.dataset,self.reference("b"),2,7,0)[0]
        other=calibration_split(self.dataset,self.reference("b"),2,7,1)[0]
        self.assertNotEqual(set(self.dataset.data.groups[first]),set(self.dataset.data.groups[other]))
        with self.assertRaises(ValueError): calibration_split(self.dataset,self.reference("b"),6)

    def test_reconnect_subsegments_of_a_trial_share_one_splitting_group(self):
        original=(self.a/"events.csv").read_bytes()
        trial=next(t for t in self.trials.values() if t["session_id"]=="a")
        split=trial["start_sample"]+75
        try:
            with (self.a/"events.csv").open("a") as file: file.write(f"{split},stream_reconnect,,\n")
            with self.assertRaisesRegex(ValueError,"no usable guarded windows"):
                load_v2_dataset(self.sessions)
        finally: (self.a/"events.csv").write_bytes(original)
        # Test a long enough trial separately so both reconnect pieces survive guards.
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            directory=record(root,gestures=("rest","fist"),gesture_seconds=3.0)
            metadata=json.loads((directory/"metadata.json").read_text())
            trial=metadata["trials"][0]
            # Both surviving segments of this trial must stay in one split group.
            with (directory/"events.csv").open("a") as file: file.write(f"{trial['start_sample']+188},stream_reconnect,,\n")
            manifest=root/"manifest.csv"; write_manifest(root/"recordings",manifest)
            selected,_=select_sessions(manifest,("rest","fist"),["a"])
            grouped,_,_,_=load_v2_dataset(selected)
            selected_groups={grouped.data.groups[i] for i,w in enumerate(grouped.windows) if w.trial_id==trial["trial_id"]}
            self.assertEqual(selected_groups,{trial["trial_id"]})
            segments={w.group_id for w in grouped.windows if w.trial_id==trial["trial_id"]}
            self.assertEqual(len(segments),2)

    def test_malformed_missing_timing_overlap_and_condition_metadata_rejected(self):
        path=self.a/"metadata.json"; original=path.read_bytes()
        mutations=[lambda m:m["trials"].pop(),
                   lambda m:m["trials"][0].update(end_sample=m["trials"][0]["start_sample"]),
                   lambda m:m["trials"][1].update(start_sample=m["trials"][0]["start_sample"]),
                   lambda m:m.update(dynamic_arm_motion=True),
                   lambda m:m.update(session_complete=False)]
        try:
            for mutate in mutations:
                changed=json.loads(original); mutate(changed); path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): validate_v2_recording(self.a)
        finally: path.write_bytes(original)

    def test_all_modes_reports_splits_rest_and_deterministic_predictions(self):
        first=args(self.root,output=self.root/"integration1")
        result=run_dataset_v2_benchmark(first)
        self.assertEqual({r["experiment"] for r in result["summary"]},
                         {"reference_baseline","cross_posture","posture_inclusive","reapplication_zero_shot","recalibration","cross_day"})
        for row in result["summary"]:
            path=first.output/row["result_id"]
            for name in ("metrics.json","summary.csv","confusion_matrix.csv","predictions.csv","run_config.json"):
                self.assertTrue((path/name).is_file(),name)
            config=json.loads((path/"run_config.json").read_text())
            self.assertFalse(set(config["train_trial_ids"])&set(config["test_trial_ids"]))
            self.assertTrue(all(self.dataset.windows[i].phase in ("rest_pre","rest_post") for i in config["rest_window_indices"]))
            self.assertFalse(any(self.dataset.windows[i].phase=="baseline" for i in config["train_window_indices"]+config["test_window_indices"]))
            if row["experiment"]=="cross_posture": self.assertNotEqual(row["train_posture"],row["test_posture"])
            if row["experiment"]=="reapplication_zero_shot": self.assertNotEqual(row["train_placement"],row["test_placement"])
            if row["experiment"]=="recalibration" and row["calibration_reps_per_gesture"]:
                self.assertEqual(row["train_session"],"b")
                self.assertEqual(len(config["calibration_trial_ids"]),9*row["calibration_reps_per_gesture"])
                self.assertFalse(set(config["calibration_trial_ids"])&set(config["test_trial_ids"]))
        second=args(self.root,output=self.root/"integration2")
        again=run_dataset_v2_benchmark(second)
        self.assertEqual(result["summary"],again["summary"])
        for row in result["summary"]:
            self.assertEqual((first.output/row["result_id"]/"predictions.csv").read_bytes(),
                             (second.output/row["result_id"]/"predictions.csv").read_bytes())
        for name in ("cross_posture.png","recalibration_curve.png","posture_recall_heatmap.png","reapplication_comparison.png"):
            self.assertTrue((first.output/"plots"/name).is_file())

    def test_combined_calibration_and_cross_day_calibration_reuse(self):
        result=run_dataset_v2_benchmark(args(self.root,output=self.root/"combined",experiment="recalibration",
                                             calibration_training="combined",calibration_target="C",calibration_reps=[0,1],choices=1))
        row=next(r for r in result["summary"] if r["experiment"]=="recalibration" and r["calibration_reps_per_gesture"]==1)
        self.assertEqual(row["train_session"],"a|c"); self.assertEqual(row["test_session"],"c")

    def test_wrong_placement_or_same_cross_day_date_rejected_before_output(self):
        path=self.c/"metadata.json"; original=path.read_bytes()
        try:
            data=json.loads(original); data["recording_date"]="2026-10-06"; path.write_text(json.dumps(data))
            manifest=self.root/"data/invalid.csv"; write_manifest(self.root/"recordings",manifest)
            with self.assertRaisesRegex(ValueError,"different recording date"):
                run_dataset_v2_benchmark(args(self.root,manifest=manifest,output=self.root/"invalid_output"))
            self.assertFalse((self.root/"invalid_output").exists())
        finally: path.write_bytes(original)

        path=self.b/"metadata.json"; original=path.read_bytes()
        try:
            data=json.loads(original); data["electrode_placement_id"]="P2"; path.write_text(json.dumps(data))
            manifest=self.root/"data/same_placement.csv"; write_manifest(self.root/"recordings",manifest)
            with self.assertRaisesRegex(ValueError,"new placement ID"):
                run_dataset_v2_benchmark(args(self.root,manifest=manifest,output=self.root/"invalid_placement_output"))
            self.assertFalse((self.root/"invalid_placement_output").exists())
        finally: path.write_bytes(original)

    def test_validation_only_accepts_a_without_optional_b_c_and_writes_no_results(self):
        output=self.root/"validation_only"
        result=run_dataset_v2_benchmark(args(self.root,session_b=None,session_c=None,validate_only=True,output=output))
        self.assertEqual(result["validation"][0]["trial_count"],216)
        self.assertFalse(output.exists())

    def test_invalid_calibration_size_is_rejected_before_results(self):
        output=self.root/"invalid_size"
        with self.assertRaisesRegex(ValueError,"no held-out trial"):
            run_dataset_v2_benchmark(args(self.root,output=output,calibration_reps=[0,6]))
        self.assertFalse(output.exists())

    def test_recording_and_evaluation_cli_end_to_end(self):
        project=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            conditions=root/"conditions.json"
            conditions.write_text(json.dumps({"session_id":"cli_A","subject_id":"S01","electrode_placement_id":"P2",
                                              "block_conditions":{"elbow_flexed":{"elbow_position":"approximately_90_deg"}}}))
            acquisition=subprocess.run([sys.executable,str(project/"calibrate.py"),"--protocol","dataset_v2",
                                         "--source","synthetic","--fast","--conditions-json",str(conditions),
                                         "--output-root",str(root/"recordings"),"--baseline-seconds","1.2",
                                         "--gesture-seconds","1.2","--rest-seconds","0.016",
                                         "--preparation-seconds","0.016","--posture-change-seconds","0.016",
                                         "--independent-rest-seconds","1.2"],capture_output=True,text=True)
            self.assertEqual(acquisition.returncode,0,acquisition.stderr)
            directory=root/"recordings/S01/cli_A"
            self.assertEqual(validate_v2_recording(directory)["trial_count"],216)
            metadata=json.loads((directory/"metadata.json").read_text())
            trial=next(t for t in metadata["trials"] if t["arm_posture"]=="elbow_flexed")
            self.assertEqual(trial["elbow_position"],"approximately_90_deg")
            manifest=root/"manifest.csv"; write_manifest(root/"recordings",manifest)
            evaluation=subprocess.run([sys.executable,str(project/"dataset_v2_benchmark.py"),"--manifest",str(manifest),
                                        "--session-a","cli_A","--experiment","reference-baseline",
                                        "--output",str(root/"results")],capture_output=True,text=True)
            self.assertEqual(evaluation.returncode,0,evaluation.stderr)
            config=json.loads((root/"results/reference_baseline/reference/run_config.json").read_text())
            self.assertEqual(config["latency"]["window_duration_ms"],200)
            self.assertFalse(config["is_real_data"])


class IndependentRestTests(unittest.TestCase):
    def test_episode_durations_active_label_changes_and_boundary_censoring(self):
        from emg.evaluation.dataset import WindowInfo
        windows=[WindowInfo("s","rest_pre","segment", "rest_pre",i,5*i,5*i+25,i*.04,i*.04+.2) for i in range(6)]
        metrics=independent_rest_metrics(["rest","fist","point","rest","fist","fist"],windows,.04)
        self.assertEqual(metrics["false_activation_episodes"],2)
        self.assertAlmostEqual(metrics["false_active_window_fraction"],4/6)
        self.assertAlmostEqual(metrics["mean_false_activation_duration_seconds"],.08)
        self.assertAlmostEqual(metrics["max_false_activation_duration_seconds"],.08)
        self.assertAlmostEqual(metrics["false_activation_episodes_per_minute"],500)
        self.assertEqual(metrics["right_censored_episodes"],1)
        self.assertEqual(metrics["left_censored_episodes"],0)
        self.assertEqual(metrics["switch_count"],4)

    def test_rest_segments_and_gaps_do_not_bridge_episodes_or_switches(self):
        from emg.evaluation.dataset import WindowInfo
        windows=[WindowInfo("s",str(i),str(i),"rest_pre",i,100*i,100*i+25,i*.8,i*.8+.2) for i in range(2)]
        metrics=independent_rest_metrics(["fist","point"],windows,.04)
        self.assertEqual(metrics["false_activation_episodes"],2)
        self.assertEqual(metrics["left_censored_episodes"],2)
        self.assertIsNone(metrics["switches_per_second"])

    def test_fully_inactive_rest_has_no_false_activation_duration(self):
        from emg.evaluation.dataset import WindowInfo
        window=WindowInfo("s","rest","rest","rest_pre",0,0,25,0,.2)
        metrics=independent_rest_metrics(["rest"],[window],.04)
        self.assertEqual(metrics["false_activation_episodes"],0)
        self.assertEqual(metrics["mean_false_activation_duration_seconds"],0)
        self.assertEqual(metrics["max_false_activation_duration_seconds"],0)


if __name__=="__main__": unittest.main()
