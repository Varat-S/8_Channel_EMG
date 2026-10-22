"""Local live seven-gesture CNN inference without saving a recording."""

from __future__ import annotations

import argparse
from collections import deque
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import sys
import threading
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch

from emg.cnn import SmallEmgCnn
from emg.preprocessing import CausalBandpass, apply_spatial_reference
from emg.sources import SerialFrameSource
from emg.types import SAMPLE_RATE_HZ
from emg.vision_reapplication import load_protocol
from vision_ground_truth import PacedSyntheticSource
from evaluate_vision_posture import load_session


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models/vision_posture_20261016_192456/cnn_all_postures.pt"
FROZEN_PROTOCOL = ROOT / "data/vision_reapplication_protocol_v1.json"


def load_original_baseline_scale() -> np.ndarray:
    """Verify frozen inputs and recover the P3 training-session baseline RMS."""
    protocol = load_protocol(FROZEN_PROTOCOL, ROOT)
    source = ROOT / "recordings/S01" / protocol["source_session"]
    scale = np.asarray(load_session(source)["baseline_scale"], dtype=np.float64)
    if scale.shape != (8,) or not np.all(np.isfinite(scale)) or np.any(scale < 1e-8):
        raise ValueError("invalid frozen source baseline scale")
    return scale


def load_recorded_baseline_scale(session: Path) -> np.ndarray:
    """Recover the baseline used to train a session-specific deployment model."""
    scale = np.asarray(load_session(session)["baseline_scale"], dtype=np.float64)
    if scale.shape != (8,) or not np.all(np.isfinite(scale)) or np.any(scale < 1e-8):
        raise ValueError("invalid recorded baseline scale")
    return scale


class LivePredictor:
    def __init__(self, artifact_path: Path, source_factory, *, baseline_seconds: float = 5.0,
                 confidence_floor: float = 0.55, publish=None,
                 baseline_mode: str = "fresh", original_scale: np.ndarray | None = None) -> None:
        if baseline_seconds <= 0 or not 0 <= confidence_floor <= 1:
            raise ValueError("invalid baseline duration or confidence floor")
        if baseline_mode not in {"fresh", "original"}:
            raise ValueError("baseline mode must be fresh or original")
        if baseline_mode == "original":
            if original_scale is None:
                raise ValueError("original baseline mode requires the frozen source scale")
            original_scale = np.asarray(original_scale, dtype=np.float64)
            if original_scale.shape != (8,) or not np.all(np.isfinite(original_scale)) or np.any(original_scale < 1e-8):
                raise ValueError("invalid frozen source baseline scale")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        required = {"classes", "state_dict", "channel_mean", "channel_std", "window_size",
                    "hop_size", "bandpass_hz", "spatial_reference", "baseline_normalize"}
        if not required <= artifact.keys() or not artifact["baseline_normalize"]:
            raise ValueError("incompatible CNN artifact")
        self.labels = tuple(str(label) for label in artifact["classes"])
        self.training_session = str(artifact.get("training_session", ""))
        self.training_sessions = tuple(str(session) for session in artifact.get("training_sessions", ()))
        self.model_name = f"{artifact_path.parent.name}/{artifact_path.name}"
        self.baseline_session = "vision_20261016_192456" if not self.training_sessions else ""
        self.model = SmallEmgCnn(len(self.labels))
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()
        self.mean = np.asarray(artifact["channel_mean"], dtype=np.float32)
        self.std = np.asarray(artifact["channel_std"], dtype=np.float32)
        self.window_size = int(artifact["window_size"])
        self.hop_size = int(artifact["hop_size"])
        self.bandpass_hz = tuple(artifact["bandpass_hz"])
        self.spatial_reference = str(artifact["spatial_reference"])
        if self.mean.shape != (1, 8, 1) or self.std.shape != (1, 8, 1) or np.any(self.std <= 0):
            raise ValueError("invalid CNN normalization")
        self.source_factory = source_factory
        self.baseline_samples = round(baseline_seconds * SAMPLE_RATE_HZ)
        self.baseline_mode = baseline_mode
        self.original_scale = original_scale
        self.confidence_floor = confidence_floor
        self.publish = publish or (lambda event: None)
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self.status = {"type": "status", "phase": "idle", "message": "Ready to connect"}

    def _emit(self, event: dict) -> None:
        with self._lock:
            self.status = event
        self.publish(event)

    def start(self) -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            self._emit({"type": "status", "phase": "connecting", "message": "Opening EMG stream"})
            with self.source_factory() as source:
                filt = CausalBandpass(sample_rate_hz=SAMPLE_RATE_HZ,
                                      low_hz=self.bandpass_hz[0], high_hz=self.bandpass_hz[1])
                retries = 0
                if self.baseline_mode == "original":
                    scale = self.original_scale
                    self._emit({"type": "signal_quality", "phase": "live",
                                "message": "Using recorded training-session baseline; no new baseline collected",
                                "baseline_origin": "training_session",
                                "baseline_rms_by_channel": [round(float(value), 2) for value in scale]})
                else:
                    baseline = []
                    self._emit({"type": "status", "phase": "baseline", "message": "Relax your hand for the baseline"})
                    while len(baseline) < self.baseline_samples and not self._stop.is_set():
                        try:
                            sample = source.read_sample()
                            retries = 0
                        except TimeoutError:
                            retries += 1
                            if retries > 3:
                                raise TimeoutError("baseline stream did not resume after three reconnect attempts")
                            source.reconnect()
                            filt.reset()
                            baseline.clear()
                            continue
                        baseline.append(filt.update(apply_spatial_reference(sample.values, self.spatial_reference)))
                    if self._stop.is_set():
                        return
                    scale = np.sqrt(np.mean(np.square(baseline), axis=0))
                    if not np.all(np.isfinite(scale)) or np.any(scale < 1e-8):
                        raise ValueError("flat or invalid baseline channel; check electrode contact")
                    self._emit({"type": "signal_quality", "phase": "baseline",
                                "message": "Baseline complete; inspect all channel amplitudes, especially CH3",
                                "baseline_origin": "fresh",
                                "baseline_rms_by_channel": [round(float(value), 2) for value in scale]})
                window = deque(maxlen=self.window_size)
                history = deque(maxlen=7)
                samples_after_window = 0
                self._emit({"type": "status", "phase": "live", "message": "Live inference running"})
                while not self._stop.is_set():
                    try:
                        sample = source.read_sample()
                        retries = 0
                    except TimeoutError:
                        retries += 1
                        if retries > 3:
                            raise TimeoutError("live stream did not resume after three reconnect attempts")
                        self._emit({"type": "status", "phase": "reconnecting", "message": "Receiver paused; reconnecting"})
                        source.reconnect()
                        filt.reset()
                        window.clear()
                        history.clear()
                        samples_after_window = 0
                        self._emit({"type": "status", "phase": "live", "message": "Live inference running"})
                        continue
                    processed = filt.update(apply_spatial_reference(sample.values, self.spatial_reference))
                    window.append(processed / scale)
                    if len(window) < self.window_size:
                        continue
                    due = samples_after_window % self.hop_size == 0
                    samples_after_window += 1
                    if not due:
                        continue
                    values = np.asarray(window, dtype=np.float32).T[None]
                    with torch.no_grad():
                        probabilities = torch.softmax(
                            self.model(torch.from_numpy((values - self.mean) / self.std)), dim=1
                        ).numpy()[0]
                    history.append(probabilities)
                    averaged = np.mean(history, axis=0)
                    best = int(np.argmax(averaged))
                    confidence = float(averaged[best])
                    self._emit({"type": "prediction", "phase": "live", "label": self.labels[best]
                                if confidence >= self.confidence_floor else "uncertain",
                                "confidence": round(confidence, 4),
                                "live_rms_relative_to_baseline": [round(float(value), 2) for value in
                                    np.sqrt(np.mean(np.square(np.asarray(window)), axis=0))],
                                "probabilities": {label: round(float(p), 4)
                                                  for label, p in zip(self.labels, averaged)}})
        except Exception as error:
            self._emit({"type": "status", "phase": "error", "message": f"{type(error).__name__}: {error}"})
        finally:
            if self._stop.is_set():
                self._emit({"type": "status", "phase": "idle", "message": "EMG stream stopped"})


class EventHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.subscribers = set()

    def publish(self, event):
        with self.lock:
            subscribers = tuple(self.subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                pass


def make_handler(predictor: LivePredictor, hub: EventHub, source_name: str = "serial"):
    class Handler(SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT / "webapp"), **kwargs)

        def do_GET(self):
            if self.path == "/api/live/config":
                self._json({"source": source_name, "labels": predictor.labels,
                            "baselineMode": predictor.baseline_mode,
                            "baselineSession": predictor.baseline_session,
                            "modelName": predictor.model_name, "savesData": False})
            elif self.path == "/api/live/status":
                self._json(predictor.status)
            elif self.path == "/api/live/events":
                subscriber = queue.Queue(maxsize=32)
                with hub.lock:
                    hub.subscribers.add(subscriber)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.close_connection = True
                try:
                    while True:
                        try:
                            event = subscriber.get(timeout=10)
                            content = f"data: {json.dumps(event)}\n\n".encode()
                        except queue.Empty:
                            content = b": keep-alive\n\n"
                        self.wfile.write(content)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                finally:
                    with hub.lock:
                        hub.subscribers.discard(subscriber)
            else:
                super().do_GET()

        def do_POST(self):
            if self.path == "/api/live/start":
                self._json({"ok": predictor.start()})
            elif self.path == "/api/live/stop":
                predictor.stop()
                self._json({"ok": True})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _json(self, value):
            body = json.dumps(value).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", choices=("serial", "synthetic"), default="serial")
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8768)
    parser.add_argument("--baseline-seconds", type=float, default=5.0)
    parser.add_argument("--baseline-mode", choices=("original", "fresh"), default="original")
    parser.add_argument("--baseline-session", type=Path,
                        help="training session whose recorded baseline scale is reused without collecting a new one")
    parser.add_argument("--confidence-floor", type=float, default=0.55)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    hub = EventHub()
    original_scale = None
    if args.baseline_mode == "original":
        if args.baseline_session:
            original_scale = load_recorded_baseline_scale(args.baseline_session)
        elif args.model.resolve() == DEFAULT_MODEL.resolve():
            original_scale = load_original_baseline_scale()
        else:
            raise ValueError("a custom model in original mode requires --baseline-session")
    source_factory = PacedSyntheticSource if args.source == "synthetic" else lambda: SerialFrameSource(args.port, args.baud)
    predictor = LivePredictor(args.model, source_factory,
                              baseline_seconds=args.baseline_seconds,
                              confidence_floor=args.confidence_floor, publish=hub.publish,
                              baseline_mode=args.baseline_mode, original_scale=original_scale)
    if args.baseline_session:
        baseline_metadata = json.loads((args.baseline_session / "metadata.json").read_text(encoding="utf-8"))
        if (predictor.training_session != baseline_metadata.get("session_id")
                and baseline_metadata.get("session_id") not in predictor.training_sessions):
            raise ValueError("model and baseline session do not match")
        predictor.baseline_session = str(baseline_metadata["session_id"])
    server = ThreadingHTTPServer((args.host, args.http_port), make_handler(predictor, hub, args.source))
    url = f"http://{args.host}:{args.http_port}/vision_live.html"
    print(f"Seven-gesture live inference available at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        predictor.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
