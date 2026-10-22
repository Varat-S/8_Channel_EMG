from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from emg.cnn import SmallEmgCnn


class CnnTests(unittest.TestCase):
    def test_output_shape_for_eight_by_twenty_five_windows(self) -> None:
        model = SmallEmgCnn(class_count=5)
        output = model(torch.zeros(4, 8, 25))
        self.assertEqual(tuple(output.shape), (4, 5))


if __name__ == "__main__":
    unittest.main()

