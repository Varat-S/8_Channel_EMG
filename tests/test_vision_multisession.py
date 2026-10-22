from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from emg.cnn import SmallEmgCnn
from experiment_multisession_vision import VARIANTS, fold_masks, predict, train_model
from evaluate_p5_candidates import load_candidate, verify_manifest
from emg.vision_reapplication import sha256


class MultiSessionVisionTests(unittest.TestCase):
    @staticmethod
    def sample_data():
        rng = np.random.default_rng(23)
        labels = ("fist", "peace", "pinch", "point", "rest", "splayed", "thumbs_up")
        rows = [(session, gesture, repetition)
                for session in ("P3", "P4") for gesture in labels for repetition in range(1, 7)]
        return {
            "windows": rng.normal(size=(len(rows), 8, 25)).astype(np.float32),
            "labels": np.asarray([row[1] for row in rows]),
            "repetitions": np.asarray([row[2] for row in rows]),
            "groups": np.asarray([f"{session}:{gesture}:{repetition}" for session, gesture, repetition in rows]),
            "labels_order": labels,
        }

    def test_paired_folds_hold_out_every_whole_trial_once(self):
        data = self.sample_data()
        held_out = set()
        for train, test in fold_masks(data):
            self.assertFalse(set(data["groups"][train]) & set(data["groups"][test]))
            self.assertEqual(int(test.sum()), 28)
            held_out.update(data["groups"][test])
        self.assertEqual(held_out, set(data["groups"]))

    def test_each_augmentation_variant_trains_and_predicts(self):
        data = self.sample_data()
        train, test = fold_masks(data)[0]
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model, mean, std = train_model(data, train, variant, epochs=1, seed=7)
                probabilities = predict(model, mean, std, data["windows"][test])
                self.assertEqual(probabilities.shape, (28, 7))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)

    def test_p5_candidate_requires_expected_source_sessions(self):
        labels = self.sample_data()["labels_order"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.pt"
            torch.save({"state_dict": SmallEmgCnn(7).state_dict(),
                        "classes": labels, "training_sessions": ("P3", "P4")}, path)
            model, _ = load_candidate(path, ("P3", "P4"), labels)
            self.assertIsInstance(model, SmallEmgCnn)
            with self.assertRaisesRegex(ValueError, "training sessions"):
                load_candidate(path, ("P3", "P5"), labels)

    def test_frozen_manifest_rejects_model_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            hashes = {}
            for variant in VARIANTS:
                model = models / f"cnn_{variant}.pt"
                model.write_bytes(variant.encode())
                hashes[variant] = sha256(model)
            manifest = {"status": "frozen before P5 collection",
                        "primary_variant": "plain",
                        "secondary_variants": ["gain", "gain_dropout"],
                        "p5_protocol_sha256": "protocol-hash",
                        "model_sha256": hashes}
            (models / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(verify_manifest(models, {"_sha256": "protocol-hash"}), manifest)
            (models / "cnn_plain.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "frozen model hash mismatch"):
                verify_manifest(models, {"_sha256": "protocol-hash"})


if __name__ == "__main__":
    unittest.main()
