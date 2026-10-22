"""Serve the local 3D hand UI and stream live LDA predictions to it."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import re
import sys
import threading
import time
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from emg.activation import StableGestureDetector, WakeSequenceGate
from emg.calibration import CalibrationConfig, CalibrationRunner
from emg.dataset import load_feature_dataset
from emg.features import extract_time_domain_features
from emg.model import load_model, save_model, train_and_evaluate
from emg.preprocessing import CausalBandpass, apply_spatial_reference
from emg.recording import SessionRecorder
from emg.sources import SerialFrameSource
from emg.types import SAMPLE_RATE_HZ


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "webapp"
CALIBRATION_GESTURES = ("splayed", "fist", "wrist_up", "wrist_down")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--baud", type=int, default=921_600)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--baseline-seconds", type=float, default=5.0)
    parser.add_argument("--calibration-trials", type=int, default=3)
    parser.add_argument("--smoothing-windows", type=int, default=7)
    parser.add_argument("--hold-seconds", type=float, default=0.4)
    parser.add_argument("--minimum-confidence", type=float, default=0.55)
    parser.add_argument("--step-timeout", type=float, default=4.0)
    parser.add_argument(
        "--wake-sequence",
        default="splayed,fist,splayed",
        help="comma-separated trained gesture labels",
    )
    parser.add_argument("--demo", action="store_true", help="animate without opening the EMG receiver")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.model is None:
        latest_web_model = Path("models/web_latest_lda.joblib")
        args.model = latest_web_model if latest_web_model.exists() else Path("models/s01_v2_seated_lda.joblib")
    args.wake_sequence = tuple(part.strip() for part in args.wake_sequence.split(",") if part.strip())
    if not args.wake_sequence:
        parser.error("wake-sequence cannot be empty")
    if args.baseline_seconds <= 0 or args.smoothing_windows <= 0 or args.hold_seconds <= 0:
        parser.error("baseline-seconds, smoothing-windows, and hold-seconds must be positive")
    if args.calibration_trials < 2:
        parser.error("calibration-trials must be at least 2")
    if not 0.0 <= args.minimum_confidence <= 1.0:
        parser.error("minimum-confidence must be between 0 and 1")
    return args


class EventHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue[dict[str, object]]] = set()
        self.last_event: dict[str, object] = {"type": "status", "phase": "idle", "message": "Ready to connect"}

    def publish(self, event: dict[str, object]) -> None:
        self.last_event = event
        with self._lock:
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(event)
                except queue.Empty:
                    pass

    def subscribe(self) -> queue.Queue[dict[str, object]]:
        subscriber: queue.Queue[dict[str, object]] = queue.Queue(maxsize=32)
        subscriber.put(self.last_event)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, object]]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)


class PredictionController:
    def __init__(self, args: argparse.Namespace, hub: EventHub) -> None:
        self.args = args
        self.hub = hub
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self.active_model_path = args.model
        self._calibration_step = 0
        self._calibration_prompt_gesture = "rest"
        self.gate = WakeSequenceGate(args.wake_sequence, args.step_timeout)
        self.detector = StableGestureDetector(args.hold_seconds, args.minimum_confidence)

    @property
    def running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            if not self.args.demo and not self._model_compatible():
                self.hub.publish(
                    {
                        "type": "error",
                        "message": (
                            "The saved model uses the previous gesture set. "
                            "Run Fresh calibration for the new five-pose gesture set first."
                        ),
                    }
                )
                return False
            self.stop_event.clear()
            self.gate.disarm()
            self.detector.reset()
            target = self._demo_worker if self.args.demo else self._serial_worker
            self._worker = threading.Thread(
                target=self._worker_entry,
                args=(target,),
                name="emg-prediction",
                daemon=True,
            )
            self._worker.start()
            return True

    def start_calibration(self) -> bool:
        with self._lock:
            if self.running:
                return False
            if self.args.demo:
                self.hub.publish({"type": "error", "message": "Fresh calibration requires COM10 mode."})
                return False
            self.stop_event.clear()
            self.gate.disarm()
            self.detector.reset()
            self._calibration_step = 0
            self._calibration_prompt_gesture = "rest"
            self._worker = threading.Thread(
                target=self._worker_entry,
                args=(self._calibration_worker,),
                name="emg-calibration",
                daemon=True,
            )
            self._worker.start()
            return True

    def _worker_entry(self, target) -> None:
        try:
            target()
        finally:
            with self._lock:
                if self._worker is threading.current_thread():
                    self._worker = None
            if self.stop_event.is_set():
                self.hub.publish(
                    {
                        "type": "status",
                        "phase": "idle",
                        "message": "Connection reset. Ready for calibration or saved-model testing.",
                    }
                )

    def reset(self) -> bool:
        with self._lock:
            was_running = self.running
            self.stop_event.set()
        self.hub.publish(
            {
                "type": "status",
                "phase": "stopping" if was_running else "idle",
                "message": (
                    "Stopping the previous serial attempt..."
                    if was_running
                    else "Ready for calibration or saved-model testing."
                ),
            }
        )
        return was_running

    def disarm(self) -> None:
        self.gate.disarm()
        self.detector.reset()
        self._publish_prediction("rest", 1.0)

    def stop(self) -> None:
        self.stop_event.set()

    def config_event(self) -> dict[str, object]:
        model_labels = self._model_labels()
        return {
            "type": "config",
            "mode": "demo" if self.args.demo else "COM10" if self.args.port == "COM10" else self.args.port,
            "wakeSequence": list(self.args.wake_sequence),
            "holdSeconds": self.args.hold_seconds,
            "minimumConfidence": self.args.minimum_confidence,
            "calibrationGestures": list(CALIBRATION_GESTURES),
            "calibrationTrials": self.args.calibration_trials,
            "modelLabels": sorted(model_labels),
            "modelCompatible": self.args.demo or model_labels == self._expected_labels(),
        }

    @staticmethod
    def _expected_labels() -> set[str]:
        return {"rest", *CALIBRATION_GESTURES}

    def _model_labels(self) -> set[str]:
        try:
            artifact = load_model(self.active_model_path)
            return {str(label) for label in artifact["model"].classes_}
        except Exception:
            return set()

    def _model_compatible(self) -> bool:
        return self._model_labels() == self._expected_labels()

    def _publish_prediction(self, gesture: str, confidence: float) -> None:
        now = time.monotonic()
        stable = self.detector.update(gesture, confidence, now)
        just_armed = self.gate.accept(stable, now) if stable is not None else False
        self.hub.publish(
            {
                "type": "prediction",
                "rawGesture": gesture,
                "gesture": gesture if self.gate.armed else "rest",
                "confidence": round(confidence, 4),
                "armed": self.gate.armed,
                "justArmed": just_armed,
                "progress": self.gate.progress,
                "expected": self.gate.expected,
                "wakeSequence": list(self.gate.sequence),
            }
        )

    def _demo_worker(self) -> None:
        self.hub.publish({"type": "status", "phase": "demo", "message": "Demo running: performing wake sequence"})
        sequence = list(self.args.wake_sequence) + [
            "rest",
            "splayed",
            "fist",
            "wrist_up",
            "wrist_down",
            "rest",
        ]
        try:
            for gesture in sequence:
                until = time.monotonic() + max(0.85, self.args.hold_seconds + 0.25)
                while time.monotonic() < until and not self.stop_event.is_set():
                    self._publish_prediction(gesture, 0.92)
                    self.stop_event.wait(0.05)
            while not self.stop_event.wait(0.05):
                self._publish_prediction("rest", 0.94)
        finally:
            self.hub.publish({"type": "status", "phase": "idle", "message": "Demo stopped"})

    def _new_filter(self, artifact: dict[str, object]) -> CausalBandpass | None:
        bandpass_hz = artifact.get("bandpass_hz")
        if bandpass_hz is None:
            return None
        return CausalBandpass(
            sample_rate_hz=float(artifact["sample_rate_hz"]),
            low_hz=float(bandpass_hz[0]),
            high_hz=float(bandpass_hz[1]),
        )

    def _calibration_worker(self) -> None:
        session_id = datetime.now().strftime("web_%Y%m%d_%H%M%S")
        session_directory = ROOT / "recordings" / "WEB" / session_id
        model_path = ROOT / "models" / "web_latest_lda.joblib"
        config = CalibrationConfig(
            gestures=CALIBRATION_GESTURES,
            trials_per_gesture=self.args.calibration_trials,
            baseline_seconds=self.args.baseline_seconds,
            preparation_seconds=3.0,
            gesture_seconds=2.0,
            rest_seconds=1.0,
        )
        metadata = {
            "subject_id": "WEB",
            "session_id": session_id,
            "posture": "seated_forearm_perpendicular",
            "source": "serial",
            "value_unit": "unscaled_adc_count",
            "gestures": list(config.gestures),
            "trials_per_gesture": config.trials_per_gesture,
            "baseline_seconds": config.baseline_seconds,
            "preparation_seconds": config.preparation_seconds,
            "gesture_seconds": config.gesture_seconds,
            "rest_seconds": config.rest_seconds,
            "protocol_interpretation": "38-byte provisional big-endian decoder",
            "label_timing_valid": True,
            "created_by": "local_webapp",
        }
        try:
            self.hub.publish(
                {
                    "type": "calibration",
                    "phase": "connecting",
                    "message": f"Opening {self.args.port} for fresh calibration...",
                    "promptGesture": "rest",
                    "step": 0,
                    "totalSteps": len(CALIBRATION_GESTURES) * self.args.calibration_trials,
                }
            )
            with SerialFrameSource(port=self.args.port, baud=self.args.baud) as source:
                with SessionRecorder(session_directory, metadata) as recorder:
                    runner = CalibrationRunner(
                        source,
                        recorder,
                        config,
                        announce=self._publish_calibration_message,
                        wait_for_prompts=True,
                    )
                    runner.run()
                    recorder.mark_complete()

            self.hub.publish(
                {
                    "type": "calibration",
                    "phase": "training",
                    "message": "Recording complete. Training and checking the new LDA model...",
                    "promptGesture": "rest",
                    "step": self._calibration_step,
                    "totalSteps": self._calibration_step,
                }
            )
            dataset = load_feature_dataset(
                [session_directory],
                window_size=25,
                hop_size=5,
                thresholds=0.0,
                bandpass_hz=(10.0, 55.0),
                guard_samples=round(0.3 * SAMPLE_RATE_HZ),
                spatial_reference="common_average",
                baseline_normalize=True,
            )
            model, evaluation = train_and_evaluate(dataset)
            save_model(
                model_path,
                model,
                window_size=25,
                hop_size=5,
                thresholds=0.0,
                sessions=[str(session_directory)],
                bandpass_hz=(10.0, 55.0),
                sample_rate_hz=float(SAMPLE_RATE_HZ),
                spatial_reference="common_average",
                baseline_normalize=True,
            )
            self.active_model_path = model_path
            self.hub.publish(
                {
                    "type": "calibration",
                    "phase": "complete",
                    "message": (
                        "Fresh model ready. "
                        f"Internal grouped balanced accuracy: {evaluation.balanced_accuracy:.1%}. "
                        "Stay relaxed for the live normalization baseline."
                    ),
                    "promptGesture": "rest",
                    "step": self._calibration_step,
                    "totalSteps": self._calibration_step,
                    "balancedAccuracy": round(evaluation.balanced_accuracy, 4),
                    "model": str(model_path),
                }
            )
            if not self.stop_event.is_set():
                self.gate.disarm()
                self.detector.reset()
                self._serial_worker()
        except Exception as error:
            self.hub.publish({"type": "error", "message": f"Calibration failed — {type(error).__name__}: {error}"})

    def _publish_calibration_message(self, message: str) -> None:
        phase = "calibration"
        prompt_gesture = "rest"
        display = message
        seconds_left: int | None = None
        hold = re.match(r"HOLD ([a-z_]+) \(trial (\d+)\)", message)
        if message == "Starting rest baseline.":
            phase = "baseline"
            display = f"RELAX completely for {self.args.baseline_seconds:g} seconds"
        elif message.startswith("Prepare: "):
            phase = "prepare"
            prompt_gesture = message.removeprefix("Prepare: ")
            self._calibration_prompt_gesture = prompt_gesture
            display = f"GET READY — {prompt_gesture.replace('_', ' ').upper()}"
        elif message.isdigit():
            phase = "countdown"
            prompt_gesture = self._calibration_prompt_gesture
            seconds_left = int(message)
            display = f"Starting in {message}..."
        elif hold:
            phase = "hold"
            prompt_gesture = hold.group(1)
            self._calibration_prompt_gesture = prompt_gesture
            self._calibration_step += 1
            display = f"HOLD — {prompt_gesture.replace('_', ' ').upper()}"
        elif message == "RELAX":
            phase = "rest"
            self._calibration_prompt_gesture = "rest"
            display = "RELAX"
        elif message.startswith("WARNING:") or message.startswith("Reopening"):
            phase = "reconnecting"
        elif message.startswith("Stream resumed"):
            phase = "resumed"
        elif message.startswith("Calibration complete"):
            phase = "recorded"
        self.hub.publish(
            {
                "type": "calibration",
                "phase": phase,
                "message": display,
                "promptGesture": prompt_gesture,
                "secondsLeft": seconds_left,
                "step": self._calibration_step,
                "totalSteps": len(CALIBRATION_GESTURES) * self.args.calibration_trials,
            }
        )

    def _serial_worker(self) -> None:
        try:
            artifact = load_model(self.active_model_path)
            model = artifact["model"]
            labels = {str(label) for label in model.classes_}
            unknown = set(self.args.wake_sequence) - labels
            if unknown:
                raise ValueError(f"wake sequence contains labels absent from model: {sorted(unknown)}")
            window_size = int(artifact["window_size"])
            hop_size = int(artifact["hop_size"])
            thresholds = artifact["thresholds"]
            spatial_reference = str(artifact.get("spatial_reference", "none"))
            sample_rate_hz = float(artifact["sample_rate_hz"])
            signal_filter = self._new_filter(artifact)

            self.hub.publish({"type": "status", "phase": "connecting", "message": f"Opening {self.args.port} at {self.args.baud} baud"})
            with SerialFrameSource(port=self.args.port, baud=self.args.baud) as source:
                baseline_scale, signal_filter = self._collect_baseline(
                    source, artifact, signal_filter, spatial_reference, sample_rate_hz
                )
                sample_buffer: deque[np.ndarray] = deque(maxlen=window_size)
                probability_history: deque[np.ndarray] = deque(maxlen=self.args.smoothing_windows)
                samples_after_full_window = 0
                retries = 0
                self.hub.publish({"type": "status", "phase": "live", "message": "Live EMG connected; complete the wake sequence"})

                while not self.stop_event.is_set():
                    try:
                        sample = source.read_sample()
                        retries = 0
                    except TimeoutError:
                        if self.stop_event.is_set():
                            break
                        retries += 1
                        if retries > 3:
                            raise TimeoutError("stream did not resume after three reconnect attempts")
                        self.hub.publish({"type": "status", "phase": "reconnecting", "message": f"Stream paused; reconnecting ({retries}/3)"})
                        source.reconnect()
                        signal_filter = self._new_filter(artifact)
                        sample_buffer.clear()
                        probability_history.clear()
                        samples_after_full_window = 0
                        continue

                    referenced = apply_spatial_reference(sample.values, spatial_reference)
                    processed = signal_filter.update(referenced) if signal_filter is not None else referenced
                    sample_buffer.append(np.asarray(processed) / baseline_scale)
                    if len(sample_buffer) < window_size:
                        continue
                    due = samples_after_full_window % hop_size == 0
                    samples_after_full_window += 1
                    if not due:
                        continue
                    features = extract_time_domain_features(np.asarray(sample_buffer), thresholds)
                    probabilities = model.predict_proba(features.reshape(1, -1))[0]
                    probability_history.append(probabilities)
                    smoothed = np.mean(probability_history, axis=0)
                    best = int(np.argmax(smoothed))
                    self._publish_prediction(str(model.classes_[best]), float(smoothed[best]))
        except Exception as error:
            self.hub.publish({"type": "error", "message": f"{type(error).__name__}: {error}"})

    def _collect_baseline(
        self,
        source: SerialFrameSource,
        artifact: dict[str, object],
        signal_filter: CausalBandpass | None,
        spatial_reference: str,
        sample_rate_hz: float,
    ) -> tuple[np.ndarray, CausalBandpass | None]:
        if not bool(artifact.get("baseline_normalize", False)):
            return np.ones(8), signal_filter
        count = round(self.args.baseline_seconds * sample_rate_hz)
        attempts = 0
        while attempts < 4:
            values: list[np.ndarray] = []
            previous_second = None
            try:
                for index in range(count):
                    if self.stop_event.is_set():
                        raise RuntimeError("connection cancelled")
                    seconds_left = int(np.ceil((count - index) / sample_rate_hz))
                    if seconds_left != previous_second:
                        self.hub.publish({"type": "status", "phase": "baseline", "message": f"Relax completely: baseline starts in {seconds_left}", "secondsLeft": seconds_left})
                        previous_second = seconds_left
                    sample = source.read_sample()
                    referenced = apply_spatial_reference(sample.values, spatial_reference)
                    processed = signal_filter.update(referenced) if signal_filter is not None else referenced
                    values.append(np.asarray(processed))
                scale = np.sqrt(np.mean(np.asarray(values) ** 2, axis=0))
                return np.maximum(scale, 1e-8), signal_filter
            except TimeoutError:
                attempts += 1
                if attempts >= 4:
                    raise
                self.hub.publish({"type": "status", "phase": "reconnecting", "message": f"Baseline stream paused; reconnecting ({attempts}/3)"})
                source.reconnect()
                signal_filter = self._new_filter(artifact)
        raise RuntimeError("baseline failed")


def make_handler(controller: PredictionController, hub: EventHub):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

        def log_message(self, format: str, *args: object) -> None:
            if self.path != "/api/events":
                super().log_message(format, *args)

        def do_GET(self) -> None:
            if self.path == "/api/events":
                self._serve_events()
                return
            if self.path == "/api/config":
                self._json(controller.config_event())
                return
            super().do_GET()

        def do_POST(self) -> None:
            if self.path == "/api/start":
                started = controller.start()
                self._json({"ok": True, "started": started})
            elif self.path == "/api/calibrate":
                started = controller.start_calibration()
                self._json({"ok": started, "started": started})
            elif self.path == "/api/disarm":
                controller.disarm()
                self._json({"ok": True})
            elif self.path == "/api/reset":
                was_running = controller.reset()
                self._json({"ok": True, "wasRunning": was_running})
            elif self.path == "/api/stop":
                controller.stop()
                self._json({"ok": True})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _json(self, value: dict[str, object]) -> None:
            body = json.dumps(value).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_events(self) -> None:
            subscriber = hub.subscribe()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(f"data: {json.dumps(controller.config_event())}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        event = subscriber.get(timeout=10.0)
                        payload = f"data: {json.dumps(event)}\n\n".encode("utf-8")
                    except queue.Empty:
                        payload = b": keep-alive\n\n"
                    self.wfile.write(payload)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            finally:
                hub.unsubscribe(subscriber)

    return Handler


def main() -> int:
    args = parse_args()
    browser_modules = (WEB_ROOT / "vendor" / "three.module.js", WEB_ROOT / "vendor" / "three.core.js")
    if not all(path.exists() for path in browser_modules):
        print("Missing local Three.js browser modules. Run: npm.cmd install", file=sys.stderr)
        return 1
    hub = EventHub()
    controller = PredictionController(args, hub)
    server = ThreadingHTTPServer((args.host, args.http_port), make_handler(controller, hub))
    url = f"http://{args.host}:{args.http_port}"
    print(f"3D EMG hand available at {url}")
    print("Press Ctrl+C to stop. Close the proprietary EMG GUI before connecting COM10.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web app.")
    finally:
        controller.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
