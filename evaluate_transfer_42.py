"""Exploratory P3-to-P4 transfer on a corrected 42-trial reapplication session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch

from emg.cnn import SmallEmgCnn
from evaluate_vision_posture import load_session, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = load_session(args.source)
    target = load_session(args.target)
    artifact = torch.load(args.model, map_location="cpu", weights_only=False)
    labels = tuple(artifact["classes"])
    model = SmallEmgCnn(len(labels))
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    mean = np.asarray(artifact["channel_mean"], dtype=np.float32)
    std = np.asarray(artifact["channel_std"], dtype=np.float32)
    results = {}
    for name, windows in (
        ("strict_P3_baseline", target["windows"] *
         (target["baseline_scale"] / source["baseline_scale"])[None, :, None]),
        ("fresh_P4_baseline", target["windows"]),
    ):
        probabilities = []
        with torch.no_grad():
            for batch in np.array_split(windows, max(1, (len(windows) + 127) // 128)):
                inputs = torch.tensor((batch - mean) / std, dtype=torch.float32)
                probabilities.append(torch.softmax(model(inputs), dim=1).numpy())
        results[name] = score(labels, target["labels"], target["groups"], np.vstack(probabilities))
    report = {
        "scope": "exploratory 42-trial reapplication; deviates from frozen 84-trial protocol",
        "source_session": source["metadata"]["session_id"],
        "target_session": target["metadata"]["session_id"],
        "target_model_training": "none; the P3 model was frozen before P4 collection",
        "posture_correction": target["protocol"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"accuracy": row["smoothed"]["accuracy"],
                             "macro_f1": row["smoothed"]["macro_f1"]}
                      for name, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
