"""Show live LDA predictions as gesture-specific images in a Tk window."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from PIL import Image, ImageOps, ImageTk

from emg.features import extract_time_domain_features
from emg.model import load_model
from emg.preprocessing import CausalBandpass, apply_spatial_reference
from emg.sources import SerialFrameSource


IMAGE_FILES = {
    "peace": "peace.png",
    "rest": "rest.png",
    "fist": "fist.png",
    "thumbs_up": "thumbs_up.png",
    "index_thumb": "index_thumb.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("models/s01_v2_seated_lda.joblib")
    )
    parser.add_argument("--assets", type=Path, default=Path("assets/gestures"))
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--baud", type=int, default=921_600)
    parser.add_argument("--baseline-seconds", type=float, default=5.0)
    parser.add_argument(
        "--smoothing-windows",
        type=int,
        default=7,
        help="average this many successive probability vectors",
    )
    args = parser.parse_args()
    if args.baseline_seconds <= 0 or args.smoothing_windows <= 0:
        parser.error("baseline-seconds and smoothing-windows must be positive")
    return args


class PredictionWindow:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.images = self._load_images(args.assets)

        root.title("Live EMG Gesture")
        root.geometry("720x720")
        root.minsize(480, 520)
        root.configure(bg="#f5f1ea")
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.title = tk.StringVar(value="REST")
        self.score = tk.StringVar(value="Waiting for baseline")
        self.status = tk.StringVar(value="Starting...")

        ttk.Label(root, textvariable=self.title, font=("Segoe UI", 28, "bold")).pack(
            pady=(20, 4)
        )
        ttk.Label(root, textvariable=self.score, font=("Segoe UI", 13)).pack()
        self.image_label = ttk.Label(root, anchor="center")
        self.image_label.pack(fill="both", expand=True, padx=24, pady=18)
        ttk.Label(
            root,
            textvariable=self.status,
            font=("Segoe UI", 11),
            anchor="center",
            wraplength=650,
        ).pack(fill="x", padx=20, pady=(0, 20))

        self._show_gesture("rest")
        self.worker = threading.Thread(target=self._prediction_worker, daemon=True)
        self.worker.start()
        self.root.after(50, self._poll_messages)

    def _load_images(self, directory: Path) -> dict[str, Image.Image]:
        images: dict[str, Image.Image] = {}
        for gesture, filename in IMAGE_FILES.items():
            path = directory / filename
            if path.exists():
                with Image.open(path) as image:
                    images[gesture] = image.convert("RGB").copy()
        return images

    def _show_gesture(self, gesture: str) -> None:
        image = self.images.get(gesture)
        if image is None:
            expected = self.args.assets / IMAGE_FILES.get(gesture, f"{gesture}.png")
            self.image_label.configure(
                image="",
                text=f"Image not found\n{expected}",
                font=("Segoe UI", 16),
            )
            self.image_label.image = None
            return

        available_width = max(400, self.root.winfo_width() - 60)
        available_height = max(350, self.root.winfo_height() - 190)
        fitted = ImageOps.contain(image, (available_width, available_height))
        photo = ImageTk.PhotoImage(fitted)
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo

    def _poll_messages(self) -> None:
        try:
            while True:
                message_type, payload = self.messages.get_nowait()
                if message_type == "status":
                    self.status.set(str(payload))
                elif message_type == "prediction":
                    gesture, confidence = payload
                    self.title.set(str(gesture).replace("_", " ").upper())
                    self.score.set(f"Smoothed score: {float(confidence):.1%}")
                    self.status.set("Live prediction active — keep the seated pose")
                    self._show_gesture(str(gesture))
                elif message_type == "error":
                    self.status.set(f"Error: {payload}")
                    self.score.set("Prediction stopped")
        except queue.Empty:
            pass
        if not self.stop_event.is_set():
            self.root.after(50, self._poll_messages)

    def _prediction_worker(self) -> None:
        try:
            artifact = load_model(self.args.model)
            model = artifact["model"]
            window_size = int(artifact["window_size"])
            hop_size = int(artifact["hop_size"])
            thresholds = artifact["thresholds"]
            spatial_reference = artifact.get("spatial_reference", "none")
            bandpass_hz = artifact.get("bandpass_hz")
            sample_rate_hz = float(artifact["sample_rate_hz"])
            use_baseline = bool(artifact.get("baseline_normalize", False))
            signal_filter = (
                CausalBandpass(
                    sample_rate_hz=sample_rate_hz,
                    low_hz=float(bandpass_hz[0]),
                    high_hz=float(bandpass_hz[1]),
                )
                if bandpass_hz is not None
                else None
            )

            with SerialFrameSource(port=self.args.port, baud=self.args.baud) as source:
                baseline_scale = np.ones(8, dtype=np.float64)
                if use_baseline:
                    baseline_scale = self._collect_baseline(
                        source, signal_filter, spatial_reference, sample_rate_hz
                    )
                self.messages.put(("status", "Baseline complete; predictions starting"))

                sample_buffer: deque[np.ndarray] = deque(maxlen=window_size)
                probability_history: deque[np.ndarray] = deque(
                    maxlen=self.args.smoothing_windows
                )
                samples_after_full_window = 0
                while not self.stop_event.is_set():
                    sample = source.read_sample()
                    referenced = apply_spatial_reference(
                        sample.values, spatial_reference
                    )
                    processed = (
                        signal_filter.update(referenced)
                        if signal_filter is not None
                        else referenced
                    )
                    sample_buffer.append(np.asarray(processed) / baseline_scale)
                    if len(sample_buffer) < window_size:
                        continue
                    prediction_due = samples_after_full_window % hop_size == 0
                    samples_after_full_window += 1
                    if not prediction_due:
                        continue

                    window = np.asarray(sample_buffer)
                    features = extract_time_domain_features(window, thresholds)
                    probabilities = model.predict_proba(features.reshape(1, -1))[0]
                    probability_history.append(probabilities)
                    smoothed = np.mean(probability_history, axis=0)
                    best = int(np.argmax(smoothed))
                    self.messages.put(
                        (
                            "prediction",
                            (str(model.classes_[best]), float(smoothed[best])),
                        )
                    )
        except TimeoutError:
            self.messages.put(
                ("error", "controller stream paused; close this window and reopen it")
            )
        except Exception as error:
            self.messages.put(("error", f"{type(error).__name__}: {error}"))

    def _collect_baseline(
        self,
        source: SerialFrameSource,
        signal_filter: CausalBandpass | None,
        spatial_reference: str,
        sample_rate_hz: float,
    ) -> np.ndarray:
        count = round(self.args.baseline_seconds * sample_rate_hz)
        values = []
        previous_second = None
        for index in range(count):
            if self.stop_event.is_set():
                raise RuntimeError("window closed")
            seconds_left = int(np.ceil((count - index) / sample_rate_hz))
            if seconds_left != previous_second:
                self.messages.put(
                    ("status", f"Relax completely for baseline: {seconds_left}")
                )
                previous_second = seconds_left
            sample = source.read_sample()
            referenced = apply_spatial_reference(sample.values, spatial_reference)
            processed = (
                signal_filter.update(referenced)
                if signal_filter is not None
                else referenced
            )
            values.append(processed)
        scale = np.sqrt(np.mean(np.asarray(values) ** 2, axis=0))
        return np.maximum(scale, 1e-8)

    def close(self) -> None:
        self.stop_event.set()
        self.root.destroy()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    PredictionWindow(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
