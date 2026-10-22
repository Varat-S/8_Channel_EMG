import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compact_webapp import CompactController
from emg.activation import StableGestureDetector, WakeSequenceGate
from webapp import EventHub


class CompactWebappTests(unittest.TestCase):
    def controller(self):
        controller = CompactController.__new__(CompactController)
        controller.hub = EventHub()
        controller.detector = StableGestureDetector(.4, .55)
        controller.gate = WakeSequenceGate(("splayed",))
        controller.gate.armed = True
        controller.output_threshold = .82
        controller.output_policy = "hold"
        controller._last_accepted_output = "fist"
        return controller

    def test_low_confidence_prediction_holds_previous_output(self):
        controller = self.controller()
        controller._publish_prediction("wrist_up", .70)
        self.assertEqual(controller.hub.last_event["rawGesture"], "wrist_up")
        self.assertEqual(controller.hub.last_event["gesture"], "fist")
        self.assertTrue(controller.hub.last_event["rejected"])

    def test_confident_prediction_updates_output(self):
        controller = self.controller()
        controller._publish_prediction("wrist_up", .90)
        self.assertEqual(controller.hub.last_event["gesture"], "wrist_up")
        self.assertFalse(controller.hub.last_event["rejected"])
        self.assertEqual(controller._last_accepted_output, "wrist_up")


if __name__ == "__main__":
    unittest.main()
