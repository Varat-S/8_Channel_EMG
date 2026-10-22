import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg.activation import StableGestureDetector, WakeSequenceGate


class StableGestureDetectorTests(unittest.TestCase):
    def test_emits_only_once_per_continuous_hold(self):
        detector = StableGestureDetector(hold_seconds=0.4, confidence_threshold=0.6)
        self.assertIsNone(detector.update("fist", 0.8, 0.0))
        self.assertIsNone(detector.update("fist", 0.8, 0.39))
        self.assertEqual(detector.update("fist", 0.8, 0.4), "fist")
        self.assertIsNone(detector.update("fist", 0.9, 1.0))
        self.assertIsNone(detector.update("peace", 0.9, 1.1))
        self.assertEqual(detector.update("peace", 0.9, 1.51), "peace")

    def test_low_confidence_cancels_hold(self):
        detector = StableGestureDetector(hold_seconds=0.4, confidence_threshold=0.6)
        detector.update("fist", 0.8, 0.0)
        detector.update("fist", 0.5, 0.3)
        self.assertIsNone(detector.update("fist", 0.8, 0.5))
        self.assertEqual(detector.update("fist", 0.8, 0.9), "fist")


class WakeSequenceGateTests(unittest.TestCase):
    def test_arms_after_complete_sequence(self):
        gate = WakeSequenceGate(("fist", "peace", "fist", "peace"))
        self.assertFalse(gate.accept("fist", 0.0))
        self.assertFalse(gate.accept("peace", 0.5))
        self.assertFalse(gate.accept("fist", 1.0))
        self.assertTrue(gate.accept("peace", 1.5))
        self.assertTrue(gate.armed)

    def test_wrong_or_late_step_resets_progress(self):
        gate = WakeSequenceGate(("fist", "peace"), max_step_gap_seconds=1.0)
        gate.accept("fist", 0.0)
        self.assertFalse(gate.accept("thumbs_up", 0.5))
        self.assertEqual(gate.progress, 0)
        gate.accept("fist", 1.0)
        self.assertFalse(gate.accept("peace", 2.1))
        self.assertEqual(gate.progress, 0)

    def test_rest_is_ignored_without_erasing_wake_progress(self):
        gate = WakeSequenceGate(("fist", "splayed"))
        self.assertFalse(gate.accept("fist", 0.0))
        self.assertEqual(gate.progress, 1)
        self.assertFalse(gate.accept("rest", 0.4))
        self.assertEqual(gate.progress, 1)
        self.assertEqual(gate.expected, "splayed")
        self.assertTrue(gate.accept("splayed", 0.8))


if __name__ == "__main__":
    unittest.main()
