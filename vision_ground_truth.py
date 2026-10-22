"""Run the local camera-assisted ground-truth recorder for synchronized EMG."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import sys
import threading
import time
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emg.sources import SerialFrameSource, SyntheticSource
from emg.types import SAMPLE_RATE_HZ
from emg.vision_capture import VISION_GESTURES, VisionCaptureController
from emg.vision_reapplication import load_protocol, schedule


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "webapp"


def load_latest_vision_calibration(recordings_root: Path) -> tuple[dict | None, str | None]:
    """Recover the newest complete template set after a local server restart."""
    root = Path(recordings_root)
    if not root.exists():
        return None, None
    try:
        candidates = sorted(
            root.rglob("metadata.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None, None
    for path in candidates:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            calibration = metadata.get("vision_calibration")
            templates = calibration.get("pose_templates") if isinstance(calibration, dict) else None
            if not isinstance(templates, dict):
                continue
            if not all(
                isinstance(templates.get(label), dict)
                and isinstance(templates[label].get("prototypes"), list)
                and len(templates[label]["prototypes"]) == 2
                for label in VISION_GESTURES
            ):
                continue
            return calibration, metadata.get("session_id")
        except (OSError, ValueError, TypeError):
            continue
    return None, None


class EventHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue] = set()
        self.last_event = {
            "type": "vision_status",
            "phase": "idle",
            "message": "Enable the camera, then start an EMG session.",
        }

    def publish(self, event: dict) -> None:
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

    def subscribe(self) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=32)
        subscriber.put(self.last_event)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)


class PacedSyntheticSource:
    """Real-time source for exercising the complete UI without the receiver."""

    unit = SyntheticSource.unit

    def __init__(self) -> None:
        self._source = SyntheticSource()
        self._next_sample = time.monotonic()

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def reconnect(self) -> None:
        self._next_sample = time.monotonic()

    def read_sample(self):
        self._next_sample += 1.0 / SAMPLE_RATE_HZ
        delay = self._next_sample - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        return self._source.read_sample()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("serial", "synthetic"), default="serial")
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--baud", type=int, default=921_600)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8767)
    parser.add_argument("--recordings-root", type=Path, default=Path("recordings"))
    parser.add_argument("--baseline-seconds", type=float, default=5.0)
    parser.add_argument("--trial-seconds", type=float, default=2.5)
    parser.add_argument("--vision-confidence", type=float, default=0.62)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--view-only", action="store_true",
                        help="camera pose demo only; reject all recording requests")
    parser.add_argument("--reapplication-protocol", type=Path,
                        help="frozen seven-gesture protocol; enables two posture blocks")
    parser.add_argument("--plan", action="store_true", help="print frozen schedule without opening hardware")
    return parser.parse_args()


def make_handler(controller: VisionCaptureController, hub: EventHub, config: dict):
    class Handler(SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

        def handle(self) -> None:
            # Browsers routinely discard idle keep-alive sockets while polling.
            # Treat that as a normal disconnect instead of printing a traceback.
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, format: str, *args: object) -> None:
            if self.path not in (
                "/api/vision/events",
                "/api/vision/observation",
                "/api/vision/status",
            ):
                super().log_message(format, *args)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self.send_response(HTTPStatus.TEMPORARY_REDIRECT)
                self.send_header("Location", "/vision.html")
                self.end_headers()
            elif self.path == "/api/vision/config":
                self._json({**config, **controller.status()})
            elif self.path == "/api/vision/status":
                self._json(controller.status())
            elif self.path == "/api/vision/events":
                self._serve_events()
            else:
                super().do_GET()

        def do_POST(self) -> None:
            if config.get("viewOnly"):
                self._json({"ok": False, "error": "view-only demo: recording is disabled"}, HTTPStatus.FORBIDDEN)
                return
            try:
                payload = self._request_json()
                if self.path == "/api/vision/session/start":
                    result = controller.start_session(payload)
                elif self.path == "/api/vision/observation":
                    result = controller.observe(payload)
                elif self.path == "/api/vision/baseline/start":
                    result = controller.start_baseline()
                elif self.path == "/api/vision/trial/start":
                    result = controller.start_trial(str(payload.get("label", "")))
                elif self.path == "/api/vision/session/stop":
                    result = controller.stop_session()
                elif self.path == "/api/vision/posture/advance":
                    result = controller.advance_posture()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._json({"ok": True, **result})
            except (ValueError, json.JSONDecodeError) as error:
                self._json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self._json(
                    {"ok": False, "error": f"{type(error).__name__}: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def _request_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 256_000:
                raise ValueError("request body is too large")
            if length == 0:
                return {}
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def _json(self, value: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
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
                self.wfile.write(
                    f"data: {json.dumps(hub.last_event)}\n\n".encode("utf-8")
                )
                self.wfile.flush()
                while True:
                    try:
                        event = subscriber.get(timeout=10.0)
                        content = f"data: {json.dumps(event)}\n\n".encode("utf-8")
                    except queue.Empty:
                        content = b": keep-alive\n\n"
                    self.wfile.write(content)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            finally:
                hub.unsubscribe(subscriber)

    return Handler


def main() -> int:
    args = parse_args()
    reapplication = (load_protocol(args.reapplication_protocol, ROOT)
                     if args.reapplication_protocol else None)
    if args.plan:
        if reapplication is None:
            raise ValueError("--plan requires --reapplication-protocol")
        print(json.dumps({"protocol": reapplication["protocol"],
                          "plan_sha256": reapplication["_sha256"],
                          "steps": schedule(reapplication)}, indent=2))
        return 0
    if reapplication and (args.baseline_seconds != 5.0 or args.trial_seconds != 2.5):
        raise ValueError("frozen reapplication capture requires 5 s baselines and 2.5 s trials")
    required = (
        WEB_ROOT / "vendor" / "mediapipe" / "vision_bundle.mjs",
        WEB_ROOT / "vendor" / "mediapipe" / "wasm",
        WEB_ROOT / "models" / "hand_landmarker.task",
        WEB_ROOT / "models" / "pose_landmarker_lite.task",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Missing browser vision assets:\n  " + "\n  ".join(missing), file=sys.stderr)
        print("Run npm.cmd install and restore the model files.", file=sys.stderr)
        return 1

    if args.view_only:
        source_factory = PacedSyntheticSource
        source_name = "camera_only"
        source_unit = PacedSyntheticSource.unit
    elif args.source == "synthetic":
        source_factory = PacedSyntheticSource
        source_name = "paced_synthetic"
        source_unit = PacedSyntheticSource.unit
    else:
        source_factory = lambda: SerialFrameSource(args.port, args.baud)
        source_name = f"serial:{args.port}@{args.baud}"
        source_unit = SerialFrameSource.unit

    hub = EventHub()
    controller = VisionCaptureController(
        args.recordings_root,
        source_factory,
        hub.publish,
        source_name=source_name,
        source_unit=source_unit,
        baseline_seconds=args.baseline_seconds,
        trial_seconds=args.trial_seconds,
        minimum_vision_confidence=args.vision_confidence,
        reapplication_protocol=reapplication,
    )
    saved_calibration, saved_calibration_session = load_latest_vision_calibration(
        args.recordings_root
    )
    config = {
        "gestures": list(VISION_GESTURES),
        "source": "camera_only" if args.view_only else args.source,
        "serialPort": args.port if args.source == "serial" and not args.view_only else None,
        "baselineSeconds": args.baseline_seconds,
        "trialSeconds": args.trial_seconds,
        "minimumVisionConfidence": args.vision_confidence,
        "savedVisionCalibration": saved_calibration,
        "savedVisionCalibrationSession": saved_calibration_session,
        "viewOnly": args.view_only,
        "reapplicationMode": bool(reapplication),
        "reapplicationProtocolSha256": reapplication["_sha256"] if reapplication else None,
    }
    server = ThreadingHTTPServer(
        (args.host, args.http_port), make_handler(controller, hub, config)
    )
    url = f"http://{args.host}:{args.http_port}/vision.html"
    print(f"Vision-assisted EMG recorder available at {url}")
    if args.view_only:
        print("Camera-only demo: recording requests are disabled; COM10 will not be opened.")
    elif args.source == "serial":
        print(f"Receiver: {args.port} at {args.baud} baud. Turn the transmitter on before Start EMG session.")
    else:
        print("Synthetic EMG source active; safe for camera/UI testing.")
    print("Camera processing remains local in this browser session. Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping vision recorder.")
    finally:
        controller.stop_session()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
