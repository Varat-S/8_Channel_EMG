import math
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import unittest
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmark_vision_offline import trial_bootstrap
from emg.types import Sample
from vision_live import DEFAULT_MODEL, EventHub, LivePredictor, make_handler, load_original_baseline_scale


class FiniteSource:
    def __init__(self, count=110):
        self.count = count
        self.index = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read_sample(self):
        if self.index == self.count:
            raise EOFError("end of synthetic test")
        index = self.index
        self.index += 1
        values = tuple((1 + channel / 8) * math.sin(
            2 * math.pi * (15 + channel * 2) * index / 125 + channel / 3)
            for channel in range(8))
        return Sample(index=index, values=values)


class VisionOfflineTests(unittest.TestCase):
    def test_frozen_live_mode_predicts_without_collecting_a_baseline(self):
        events = []
        scale = load_original_baseline_scale()
        predictor = LivePredictor(DEFAULT_MODEL, FiniteSource, baseline_mode="original",
                                  original_scale=scale, publish=events.append)
        predictor._run()
        self.assertEqual(scale.shape, (8,))
        self.assertFalse(any(event.get("phase") == "baseline" for event in events))
        self.assertTrue(any(event["type"] == "prediction" for event in events))
        self.assertEqual(next(event for event in events if event["type"] == "signal_quality")["baseline_origin"], "training_session")

    def test_live_predictor_collects_baseline_then_emits_probabilities(self):
        events = []
        predictor = LivePredictor(DEFAULT_MODEL, FiniteSource, baseline_seconds=.2,
                                  publish=events.append)
        predictor._run()
        phases = [event.get("phase") for event in events]
        self.assertLess(phases.index("baseline"), phases.index("live"))
        quality = next(event for event in events if event["type"] == "signal_quality")
        self.assertEqual(len(quality["baseline_rms_by_channel"]), 8)
        predictions = [event for event in events if event["type"] == "prediction"]
        self.assertTrue(predictions)
        self.assertEqual(set(predictions[0]["probabilities"]), set(predictor.labels))
        self.assertAlmostEqual(sum(predictions[0]["probabilities"].values()), 1, places=3)

    def test_trial_bootstrap_resamples_trials_with_fixed_seed(self):
        truth = np.array(["a"] * 8 + ["b"] * 8)
        predicted = np.array(["a"] * 4 + ["b"] * 4 + ["b"] * 8)
        groups = np.repeat(["a1", "a2", "b1", "b2"], 4)
        first = trial_bootstrap(truth, predicted, groups, ("a", "b"), draws=100, seed=11)
        second = trial_bootstrap(truth, predicted, groups, ("a", "b"), draws=100, seed=11)
        self.assertEqual(first, second)
        self.assertLessEqual(first["lower_95"], first["point"])
        self.assertLessEqual(first["point"], first["upper_95"])

    def test_live_site_serves_page_and_status_without_serial(self):
        hub = EventHub()
        predictor = LivePredictor(DEFAULT_MODEL, FiniteSource, publish=hub.publish)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(predictor, hub, "synthetic"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(base + "/vision_live.html") as response:
                self.assertIn(b"Live EMG inference", response.read())
            with urlopen(base + "/api/live/config") as response:
                self.assertIn(b'"source": "synthetic"', response.read())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
