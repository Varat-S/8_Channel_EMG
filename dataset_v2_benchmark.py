"""Evaluate recorded Dataset v2 sessions without changing the decoder."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from emg.evaluation.dataset_v2 import main

if __name__ == "__main__":
    raise SystemExit(main())
