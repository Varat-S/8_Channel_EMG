from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from emg.preprocessing import CausalBandpass, filter_recording


class PreprocessingTests(unittest.TestCase):
    def test_offline_and_sample_by_sample_filters_match(self) -> None:
        rng = np.random.default_rng(4)
        values = rng.normal(size=(200, 8))
        offline = filter_recording(
            values, sample_rate_hz=125, low_hz=10, high_hz=55
        )
        live_filter = CausalBandpass(
            sample_rate_hz=125, low_hz=10, high_hz=55
        )
        live = np.vstack([live_filter.update(sample) for sample in values])
        np.testing.assert_allclose(live, offline, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

