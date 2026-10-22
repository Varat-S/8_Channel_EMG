import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg.rejection import apply_confidence_rejection


class ConfidenceRejectionTests(unittest.TestCase):
    def test_hold_keeps_last_accepted_label_and_resets_at_sequence_boundary(self):
        probabilities = np.asarray(
            [[.1, .9], [.45, .55], [.2, .8], [.45, .55], [.9, .1]]
        )
        output, rejected = apply_confidence_rejection(
            probabilities,
            np.asarray(["rest", "fist"]),
            np.asarray(["a", "a", "a", "b", "b"]),
            .7,
            policy="hold",
        )
        np.testing.assert_array_equal(output, ["fist", "fist", "fist", "rest", "rest"])
        np.testing.assert_array_equal(rejected, [False, True, False, True, False])

    def test_rest_policy_maps_only_low_confidence_rows_to_rest(self):
        output, rejected = apply_confidence_rejection(
            [[.2, .8], [.45, .55]], ["rest", "fist"], ["a", "a"], .7, policy="rest"
        )
        np.testing.assert_array_equal(output, ["fist", "rest"])
        np.testing.assert_array_equal(rejected, [False, True])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "rest class"):
            apply_confidence_rejection([[.5, .5]], ["fist", "splayed"], ["a"], .5, policy="hold")
        with self.assertRaisesRegex(ValueError, "policy"):
            apply_confidence_rejection([[.5, .5]], ["rest", "fist"], ["a"], .5, policy="unknown")


if __name__ == "__main__":
    unittest.main()
