from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from emg.features import extract_time_domain_features, feature_names


class FeatureTests(unittest.TestCase):
    def test_constant_channels_have_only_mav(self) -> None:
        window = np.tile(np.arange(1.0, 9.0), (5, 1))
        features = extract_time_domain_features(window).reshape(8, 4)
        np.testing.assert_allclose(features[:, 0], np.arange(1.0, 9.0))
        np.testing.assert_allclose(features[:, 1:], 0.0)

    def test_alternating_signal_counts_crossings_and_slope_changes(self) -> None:
        signal = np.asarray([-2.0, 2.0, -2.0, 2.0, -2.0])
        window = np.tile(signal[:, None], (1, 8))
        features = extract_time_domain_features(window).reshape(8, 4)
        np.testing.assert_allclose(features[:, 0], 2.0)
        np.testing.assert_allclose(features[:, 1], 16.0)
        np.testing.assert_allclose(features[:, 2], 4.0)
        np.testing.assert_allclose(features[:, 3], 3.0)

    def test_threshold_suppresses_small_changes(self) -> None:
        signal = np.asarray([-1.0, 1.0, -1.0, 1.0])
        window = np.tile(signal[:, None], (1, 8))
        features = extract_time_domain_features(window, thresholds=3.0).reshape(8, 4)
        np.testing.assert_allclose(features[:, 2:], 0.0)

    def test_names_match_feature_vector(self) -> None:
        self.assertEqual(len(feature_names()), 32)
        self.assertEqual(feature_names()[0], "CH0_mav")
        self.assertEqual(feature_names()[-1], "CH7_ssc")


if __name__ == "__main__":
    unittest.main()

