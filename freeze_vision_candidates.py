"""Write a non-overwriting manifest for P3/P4 models before P5 collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emg.vision_reapplication import load_protocol, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models/vision_multisession_P3_P4_v1"))
    parser.add_argument("--results", type=Path, default=Path("results/vision_multisession_P3_P4_v1/summary.json"))
    parser.add_argument("--protocol", type=Path, default=Path("data/vision_reapplication_protocol_v1.json"))
    args = parser.parse_args()
    destination = args.models / "manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    protocol = load_protocol(args.protocol, Path(__file__).resolve().parent)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    if results.get("sessions") != ["vision_20261016_192456", "vision_20261022_202644"]:
        raise ValueError("unexpected development sessions")
    variants = ("plain", "gain", "gain_dropout")
    hashes = {variant: sha256(args.models / f"cnn_{variant}.pt") for variant in variants}
    manifest = {
        "status": "frozen before P5 collection",
        "primary_variant": "plain",
        "secondary_variants": ["gain", "gain_dropout"],
        "training_sessions": results["sessions"],
        "development_results_sha256": sha256(args.results),
        "p5_protocol_sha256": protocol["_sha256"],
        "model_sha256": hashes,
        "selection_rule": "plain selected using P3/P4 whole-trial grouped folds; P5 primary score is plain zero-shot smoothed macro F1",
    }
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
