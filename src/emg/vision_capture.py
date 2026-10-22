"""Synchronized, auditable vision-labelled EMG capture.

The browser proposes labels from camera landmarks. This module owns the serial
timeline and only promotes a buffered candidate to a labelled gesture segment when
the same sufficiently confident visual label remains fresh throughout the hold.
Rejected candidates are retained as ``prepare`` data, which the existing dataset
loader deliberately excludes from training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
import threading
import time
from typing import Callable

from .recording import SessionRecorder
from .types import SAMPLE_RATE_HZ, Sample
from .vision_reapplication import PROTOCOL_NAME, empty_counts


VISION_GESTURES = (
    "rest",
    "fist",
    "splayed",
    "peace",
    "point",
    "pinch",
    "thumbs_up",
)
UNKNOWN_LABELS = {"unknown", "no_hand"}


def _identifier(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", text):
        raise ValueError(
            f"{field_name} must use only letters, digits, underscores or hyphens"
        )
    return text


def _finite_number(value: object, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be numeric") from error
    if not float("-inf") < result < float("inf"):
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclass(slots=True)
class VisionObservation:
    label: str
    confidence: float
    client_time_ms: float
    media_time_ms: float
    received_monotonic: float
    payload: dict


@dataclass(slots=True)
class Candidate:
    kind: str
    label: str
    identity: str
    target_samples: int
    started_monotonic: float
    starting_confidence: float
    initial_metrics: dict
    samples: list[Sample] = field(default_factory=list)
    observation_count: int = 0
    invalid_observations: int = 0
    confidence_sum: float = 0.0
    confidence_observations: int = 0
    metric_sums: dict[str, float] = field(default_factory=dict)
    metric_counts: dict[str, int] = field(default_factory=dict)
    posture_counts: dict[str, int] = field(default_factory=dict)


class VisionCaptureController:
    """Thread-safe bridge between browser observations and an EMG sample source."""

    def __init__(
        self,
        recordings_root: Path,
        source_factory: Callable[[], object],
        publish: Callable[[dict], None],
        *,
        source_name: str,
        source_unit: str,
        baseline_seconds: float = 5.0,
        trial_seconds: float = 2.5,
        minimum_vision_confidence: float = 0.62,
        observation_timeout_seconds: float = 0.75,
        maximum_invalid_fraction: float = 0.35,
        reapplication_protocol: dict | None = None,
    ) -> None:
        if baseline_seconds <= 0 or trial_seconds <= 0:
            raise ValueError("baseline and trial durations must be positive")
        if not 0 <= minimum_vision_confidence <= 1:
            raise ValueError("minimum vision confidence must be between zero and one")
        if not 0 <= maximum_invalid_fraction < 1:
            raise ValueError("maximum invalid fraction must be between zero and one")
        self.recordings_root = Path(recordings_root)
        self.source_factory = source_factory
        self.publish = publish
        self.source_name = source_name
        self.source_unit = source_unit
        self.baseline_samples = round(baseline_seconds * SAMPLE_RATE_HZ)
        self.trial_samples = round(trial_seconds * SAMPLE_RATE_HZ)
        self.minimum_vision_confidence = minimum_vision_confidence
        self.observation_timeout_seconds = observation_timeout_seconds
        self.maximum_invalid_fraction = maximum_invalid_fraction
        self.reapplication_protocol = reapplication_protocol
        if reapplication_protocol is not None and reapplication_protocol.get("protocol") != PROTOCOL_NAME:
            raise ValueError("invalid reapplication protocol")

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._recorder: SessionRecorder | None = None
        self._observations = None
        self._candidate: Candidate | None = None
        self._latest_observation: VisionObservation | None = None
        self._session_directory: Path | None = None
        self._session_id: str | None = None
        self._counts = {label: 0 for label in VISION_GESTURES}
        self._counts_by_posture = empty_counts(reapplication_protocol) if reapplication_protocol else None
        self._protocol_posture_index = 0
        self._rejected_count = 0
        self._baseline_complete = False
        self._last_source_index = -1
        self._phase = "idle"
        self._last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_session(self, config: dict) -> dict:
        with self._lock:
            if self.running or self._recorder is not None:
                raise ValueError("a vision capture session is already active")
            subject = _identifier(config.get("subject_id", "S01"), "subject_id")
            placement = _identifier(
                config.get("electrode_placement_id", "unspecified"),
                "electrode_placement_id",
            )
            requested = config.get("session_id")
            session_id = _identifier(
                requested
                or datetime.now().astimezone().strftime("vision_%Y%m%d_%H%M%S"),
                "session_id",
            )
            directory = self.recordings_root / subject / session_id
            if directory.exists():
                raise ValueError(f"session directory already exists: {directory}")
            posture = str(config.get("arm_posture", "reference")).strip() or "reference"
            calibration = config.get("vision_calibration", {})
            if not isinstance(calibration, dict):
                raise ValueError("vision_calibration must be an object")
            templates = calibration.get("pose_templates", {})
            if not isinstance(templates, dict):
                raise ValueError("pose_templates must be an object")
            saved_templates = {}
            for label in VISION_GESTURES:
                template = templates.get(label)
                if not isinstance(template, dict):
                    raise ValueError(f"missing visual template for {label}")
                prototypes = template.get("prototypes")
                if not isinstance(prototypes, list) or len(prototypes) != 2:
                    raise ValueError(f"visual template for {label} must contain two posture prototypes")
                saved_prototypes = []
                for prototype in prototypes:
                    if not isinstance(prototype, dict):
                        raise ValueError(f"invalid posture prototype for {label}")
                    posture_name = str(prototype.get("posture", ""))
                    mean = prototype.get("mean")
                    std = prototype.get("std")
                    if posture_name not in {"vertical", "horizontal"}:
                        raise ValueError(f"invalid visual posture for {label}: {posture_name}")
                    if not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 10 or len(std) != 10:
                        raise ValueError(f"visual prototype for {label} must contain ten-element mean and std arrays")
                    saved_prototypes.append(
                        {
                            "posture": posture_name,
                            "mean": [_finite_number(value, f"{label}.{posture_name}.mean") for value in mean],
                            "std": [_finite_number(value, f"{label}.{posture_name}.std") for value in std],
                            "sample_count": int(prototype.get("sampleCount", 0)),
                        }
                    )
                if {item["posture"] for item in saved_prototypes} != {"vertical", "horizontal"}:
                    raise ValueError(f"visual template for {label} needs vertical and horizontal prototypes")
                saved_templates[label] = {"prototypes": saved_prototypes}
            saved_calibration = {
                "pose_templates": saved_templates,
            }
            initial_counts = {label: 0 for label in VISION_GESTURES}
            if self.reapplication_protocol:
                initial_counts_by_posture = empty_counts(self.reapplication_protocol)
            else:
                initial_counts_by_posture = None
            metadata = {
                "protocol": PROTOCOL_NAME if self.reapplication_protocol else "vision_ground_truth_v1",
                "subject_id": subject,
                "session_id": session_id,
                "electrode_placement_id": placement,
                "posture": posture,
                "arm_posture": posture,
                "source": self.source_name,
                "value_unit": self.source_unit,
                "gestures": list(VISION_GESTURES),
                "label_timing_valid": True,
                "created_by": "vision_assisted_local_webapp",
                "recording_started_at": datetime.now().astimezone().isoformat(),
                "vision_model": {
                    "hand": "MediaPipe Hand Landmarker float16 v1",
                    "pose": "MediaPipe Pose Landmarker Lite float16 v1",
                    "classification": "personal landmark template classifier v1",
                },
                "vision_calibration": saved_calibration,
                "capture_policy": {
                    "baseline_samples": self.baseline_samples,
                    "trial_samples": self.trial_samples,
                    "minimum_vision_confidence": self.minimum_vision_confidence,
                    "observation_timeout_seconds": self.observation_timeout_seconds,
                    "maximum_invalid_fraction": self.maximum_invalid_fraction,
                    "validation_unit": "camera_observation",
                },
                "baseline": None,
                **({"baselines": [], "reapplication_plan_sha256": self.reapplication_protocol["_sha256"],
                    "protocol_postures": self.reapplication_protocol["test_design"]["postures"],
                    "accepted_counts_by_posture": initial_counts_by_posture}
                   if self.reapplication_protocol else {}),
                "trials": [],
                "rejected_trials": [],
                "accepted_counts": dict(initial_counts),
                "training_ready": False,
            }
            recorder = SessionRecorder(
                directory,
                metadata,
                condition_fields=("vision_elbow_posture", "vision_forearm_orientation")
                + (("protocol_posture",) if self.reapplication_protocol else ()),
            )
            observations = (directory / "vision_observations.jsonl").open(
                "w", encoding="utf-8", buffering=1
            )
            self._recorder = recorder
            self._observations = observations
            self._session_directory = directory
            self._session_id = session_id
            self._counts = initial_counts_by_posture["vertical"] if initial_counts_by_posture else initial_counts
            self._counts_by_posture = initial_counts_by_posture
            self._protocol_posture_index = 0
            self._rejected_count = 0
            self._baseline_complete = False
            self._candidate = None
            self._latest_observation = None
            self._last_source_index = -1
            self._last_error = None
            self._phase = "connecting"
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._sample_worker,
                name="vision-emg-capture",
                daemon=True,
            )
            self._thread.start()
            result = self.status()
        self.publish(
            {
                "type": "vision_status",
                "phase": "connecting",
                "message": "Opening the EMG stream. Keep the hand relaxed and visible.",
                **result,
            }
        )
        return result

    def observe(self, payload: dict) -> dict:
        label = str(payload.get("label", "unknown"))
        if label not in set(VISION_GESTURES) | UNKNOWN_LABELS:
            raise ValueError(f"unsupported vision label: {label}")
        confidence = _finite_number(payload.get("confidence", 0), "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        client_time = _finite_number(payload.get("clientTimeMs", 0), "clientTimeMs")
        media_time = _finite_number(payload.get("mediaTimeMs", 0), "mediaTimeMs")
        landmarks = payload.get("handLandmarks")
        if label not in UNKNOWN_LABELS and (
            not isinstance(landmarks, list) or len(landmarks) != 21
        ):
            raise ValueError("recognized observations require 21 hand landmarks")
        clean = {
            "label": label,
            "confidence": confidence,
            "clientTimeMs": client_time,
            "mediaTimeMs": media_time,
            "handedness": payload.get("handedness"),
            "handLandmarks": landmarks,
            "poseLandmarks": payload.get("poseLandmarks"),
            "poseWorldLandmarks": payload.get("poseWorldLandmarks"),
            "metrics": payload.get("metrics", {}),
        }
        now = time.monotonic()
        observation = VisionObservation(
            label, confidence, client_time, media_time, now, clean
        )
        with self._lock:
            if self._observations is None:
                raise ValueError("start an acquisition session before sending observations")
            self._latest_observation = observation
            if self._candidate is not None:
                self._record_candidate_observation(self._candidate, observation)
            row = {
                "server_received_at": datetime.now().astimezone().isoformat(),
                "server_monotonic_seconds": now,
                "sample_index_at_receipt": self._last_source_index + 1,
                **clean,
            }
            self._observations.write(json.dumps(row, separators=(",", ":")) + "\n")
            active = self._candidate.identity if self._candidate else None
        return {"ok": True, "activeCandidate": active}

    def start_baseline(self) -> dict:
        with self._lock:
            self._require_ready()
            if self._baseline_complete:
                raise ValueError("baseline already completed for this session")
            self._require_stable_observation("rest")
            posture_suffix = f"_{self._protocol_posture()}" if self.reapplication_protocol else ""
            identity = f"{self._session_id}_vision_baseline{posture_suffix}"
            self._candidate = Candidate(
                "baseline",
                "rest",
                identity,
                self.baseline_samples,
                time.monotonic(),
                self._latest_observation.confidence,
                dict(self._latest_observation.payload.get("metrics", {})),
            )
            self._record_candidate_observation(
                self._candidate, self._latest_observation
            )
            self._phase = "baseline"
            self._write_event("vision_baseline_requested", "rest", identity)
            status = self.status()
        self.publish(
            {
                "type": "vision_status",
                "phase": "baseline",
                "message": "Baseline recording: keep the hand and wrist completely relaxed.",
                **status,
            }
        )
        return status

    def start_trial(self, label: str) -> dict:
        if label not in VISION_GESTURES:
            raise ValueError(f"unsupported trial label: {label}")
        with self._lock:
            self._require_ready()
            if not self._baseline_complete:
                raise ValueError("capture a valid relaxed baseline first")
            if self.reapplication_protocol and self._counts[label] >= 6:
                raise ValueError(f"six {label} trials already accepted in this posture")
            if self.reapplication_protocol and self._counts[label] != min(self._counts.values()):
                raise ValueError("complete the current repetition for every gesture before repeating one")
            self._require_stable_observation(label)
            repetition = self._counts[label] + 1
            posture_suffix = f"_{self._protocol_posture()}" if self.reapplication_protocol else ""
            identity = f"{self._session_id}_vision_{label}{posture_suffix}_r{repetition:02d}"
            self._candidate = Candidate(
                "trial",
                label,
                identity,
                self.trial_samples,
                time.monotonic(),
                self._latest_observation.confidence,
                dict(self._latest_observation.payload.get("metrics", {})),
            )
            self._record_candidate_observation(
                self._candidate, self._latest_observation
            )
            self._phase = "capturing"
            self._write_event("vision_trial_requested", label, identity)
            status = self.status()
        self.publish(
            {
                "type": "vision_status",
                "phase": "capturing",
                "message": f"Recording {label.replace('_', ' ')}. Keep the pose steady.",
                **status,
            }
        )
        return status

    def advance_posture(self) -> dict:
        """Operator confirms the completed vertical block and starts horizontal baseline."""
        if not self.reapplication_protocol:
            raise ValueError("posture blocks are only used in reapplication mode")
        with self._lock:
            self._require_ready()
            if self._candidate is not None:
                raise ValueError("wait for the active capture to finish")
            if self._protocol_posture_index != 0:
                raise ValueError("already in the final posture block")
            if not self._baseline_complete or any(count != 6 for count in self._counts.values()):
                raise ValueError("complete the vertical baseline and all 42 trials first")
            self._protocol_posture_index = 1
            self._counts = self._counts_by_posture[self._protocol_posture()]
            self._baseline_complete = False
            self._phase = "awaiting_baseline"
            self._set_recorder_conditions(self._latest_metrics())
            self._write_event("protocol_posture_start", identity=self._protocol_posture())
            status = self.status()
        self.publish({"type": "vision_status", "phase": "awaiting_baseline",
                      "message": "Horizontal block: hold a relaxed hand for its new baseline.", **status})
        return status

    def _protocol_posture(self) -> str | None:
        if not self.reapplication_protocol:
            return None
        return self.reapplication_protocol["test_design"]["postures"][self._protocol_posture_index]

    def stop_session(self) -> dict:
        with self._lock:
            if self._recorder is None:
                return self.status()
            self._stop.set()
            self._phase = "stopping"
            thread = self._thread
            status = self.status()
        self.publish(
            {
                "type": "vision_status",
                "phase": "stopping",
                "message": "Stop requested. Closing COM10 and saving accepted data...",
                **status,
            }
        )
        if thread and thread is not threading.current_thread():
            # Serial reconnect backs off for at most six seconds. Allow that
            # interruptible cycle to unwind before reporting the final state.
            thread.join(timeout=8.0)
        with self._lock:
            return self.status()

    def status(self) -> dict:
        with self._lock:
            candidate = self._candidate
            progress = (
                len(candidate.samples) / candidate.target_samples if candidate else 0.0
            )
            allowed_invalid = (
                self._allowed_invalid_observations(candidate) if candidate else 0
            )
            return {
                "running": self.running,
                "phase": self._phase,
                "sessionId": self._session_id,
                "sessionDirectory": (
                    str(self._session_directory) if self._session_directory else None
                ),
                "baselineComplete": self._baseline_complete,
                "activeLabel": candidate.label if candidate else None,
                "activeKind": candidate.kind if candidate else None,
                "progress": min(1.0, progress),
                "invalidObservations": candidate.invalid_observations if candidate else 0,
                "allowedInvalidObservations": allowed_invalid,
                "counts": dict(self._counts),
                "countsByPosture": self._counts_by_posture,
                "protocolPosture": self._protocol_posture(),
                "reapplicationMode": bool(self.reapplication_protocol),
                "rejectedCount": self._rejected_count,
                "lastError": self._last_error,
            }

    def _require_ready(self) -> None:
        if not self.running or self._recorder is None:
            raise ValueError("the EMG acquisition stream is not running")
        if self._candidate is not None:
            raise ValueError("another baseline or gesture candidate is already active")

    def _require_stable_observation(self, label: str) -> None:
        observation = self._latest_observation
        if observation is None:
            raise ValueError("no camera observation has been received")
        if time.monotonic() - observation.received_monotonic > self.observation_timeout_seconds:
            raise ValueError("the most recent camera observation is stale")
        if observation.label != label:
            raise ValueError(
                f"camera currently sees {observation.label}, not {label}"
            )
        if observation.confidence < self.minimum_vision_confidence:
            raise ValueError("camera confidence is below the capture threshold")

    def _write_event(self, event: str, label: str = "", identity: str | None = None) -> None:
        assert self._recorder is not None
        self._recorder.write_event(
            self._last_source_index + 1, event, label, identity
        )
        self._recorder.flush()

    def _observation_valid(
        self, candidate: Candidate, observation: VisionObservation, now: float
    ) -> bool:
        metrics = observation.payload.get("metrics", {})
        arm_visibility = metrics.get("armVisibility", 0)
        return (
            now - observation.received_monotonic <= self.observation_timeout_seconds
            and observation.label == candidate.label
            and observation.confidence >= self.minimum_vision_confidence
            and isinstance(arm_visibility, (int, float))
            and float(arm_visibility) >= 0.35
        )

    def _record_candidate_observation(
        self, candidate: Candidate, observation: VisionObservation
    ) -> None:
        """Score each camera result once instead of once per faster EMG sample."""

        candidate.observation_count += 1
        valid = self._observation_valid(candidate, observation, time.monotonic())
        if not valid:
            candidate.invalid_observations += 1
            return
        metrics = observation.payload.get("metrics", {})
        if valid:
            candidate.confidence_sum += observation.confidence
            candidate.confidence_observations += 1
            for key in (
                "elbowAngleDegrees",
                "forearmElevationDegrees",
                "wristDeflectionDegrees",
                "elbowPositionX",
                "elbowPositionY",
            ):
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    candidate.metric_sums[key] = candidate.metric_sums.get(key, 0.0) + float(value)
                    candidate.metric_counts[key] = candidate.metric_counts.get(key, 0) + 1
            for key in ("postureBucket", "forearmOrientation"):
                value = metrics.get(key)
                if isinstance(value, str) and value:
                    token = f"{key}:{value}"
                    candidate.posture_counts[token] = candidate.posture_counts.get(token, 0) + 1

    def _sample_worker(self) -> None:
        try:
            self.publish(
                {
                    "type": "vision_status",
                    "phase": "connecting",
                    "message": "Connecting to the EMG source...",
                    **self.status(),
                }
            )
            with self.source_factory() as source:
                with self._lock:
                    self._phase = "awaiting_baseline"
                self.publish(
                    {
                        "type": "vision_status",
                        "phase": "awaiting_baseline",
                        "message": "EMG connected. Hold a relaxed neutral hand for the automatic baseline.",
                        **self.status(),
                    }
                )
                quiet_timeouts = 0
                reconnect_attempts = 0
                stream_was_interrupted = False
                while not self._stop.is_set():
                    try:
                        sample = source.read_sample()
                        quiet_timeouts = 0
                    except TimeoutError:
                        quiet_timeouts += 1
                        stream_was_interrupted = True
                        with self._lock:
                            self._reject_candidate("stream reconnect during candidate")
                            self._phase = "reconnecting"
                        if quiet_timeouts < 3:
                            self.publish(
                                {
                                    "type": "vision_status",
                                    "phase": "reconnecting",
                                    "message": (
                                        "EMG stream paused; waiting briefly before reopening "
                                        f"COM10 ({quiet_timeouts}/3)."
                                    ),
                                    **self.status(),
                                }
                            )
                            continue
                        reconnect_attempts += 1
                        with self._lock:
                            self._write_event("stream_reconnect")
                        self.publish(
                            {
                                "type": "vision_status",
                                "phase": "reconnecting",
                                "message": (
                                    "No EMG frames received; reopening COM10 "
                                    f"(attempt {reconnect_attempts}). Accepted data is preserved."
                                ),
                                **self.status(),
                            }
                        )
                        while not self._stop.is_set():
                            try:
                                source.reconnect()
                                break
                            except Exception as error:
                                self.publish(
                                    {
                                        "type": "vision_status",
                                        "phase": "reconnecting",
                                        "message": (
                                            f"COM10 reopen failed ({type(error).__name__}); "
                                            "retrying. You can still Stop & save."
                                        ),
                                        **self.status(),
                                    }
                                )
                                if self._stop.wait(2.0):
                                    break
                        quiet_timeouts = 0
                        continue
                    if stream_was_interrupted:
                        with self._lock:
                            self._phase = (
                                "ready" if self._baseline_complete else "awaiting_baseline"
                            )
                            recovery_status = self.status()
                        self.publish(
                            {
                                "type": "vision_status",
                                "phase": self._phase,
                                "message": "EMG stream resumed. Continue with the next pose.",
                                **recovery_status,
                            }
                        )
                        stream_was_interrupted = False
                    with self._lock:
                        self._last_source_index = sample.index
                        candidate = self._candidate
                        if candidate is None:
                            self._set_recorder_conditions(self._latest_metrics())
                            self._recorder.write_sample(
                                sample, phase="prepare", gesture="rest", trial=None
                            )
                        else:
                            candidate.samples.append(sample)
                            if (
                                len(candidate.samples) < candidate.target_samples
                                and (
                                    len(candidate.samples) == 1
                                    or len(candidate.samples) % max(1, round(SAMPLE_RATE_HZ / 5)) == 0
                                )
                            ):
                                elapsed_seconds = len(candidate.samples) / SAMPLE_RATE_HZ
                                total_seconds = candidate.target_samples / SAMPLE_RATE_HZ
                                remaining_seconds = max(0.0, total_seconds - elapsed_seconds)
                                description = (
                                    "Baseline"
                                    if candidate.kind == "baseline"
                                    else candidate.label.replace("_", " ").title()
                                )
                                self.publish(
                                    {
                                        "type": "vision_progress",
                                        "phase": self._phase,
                                        "message": (
                                            f"{description}: {remaining_seconds:.1f} s remaining. "
                                            "Keep the pose relaxed and visible."
                                            if candidate.kind == "baseline"
                                            else f"{description}: {remaining_seconds:.1f} s remaining. Hold steady."
                                        ),
                                        **self.status(),
                                    }
                                )
                            if len(candidate.samples) >= candidate.target_samples:
                                self._finish_candidate()
                        if self._recorder.sample_count % 125 == 0:
                            self._recorder.flush()
            with self._lock:
                self._finish_gracefully()
        except Exception as error:
            with self._lock:
                self._last_error = f"{type(error).__name__}: {error}"
                self._phase = "error"
                self._close_incomplete()
            self.publish(
                {
                    "type": "vision_error",
                    "phase": "error",
                    "message": self._last_error,
                    **self.status(),
                }
            )
        finally:
            with self._lock:
                self._thread = None

    def _finish_candidate(self) -> None:
        candidate = self._candidate
        assert candidate is not None and self._recorder is not None
        allowed_invalid_observations = self._allowed_invalid_observations(candidate)
        duration_seconds = candidate.target_samples / SAMPLE_RATE_HZ
        minimum_observations = max(3, round(duration_seconds * 2.0))
        latest_is_fresh = (
            self._latest_observation is not None
            and time.monotonic() - self._latest_observation.received_monotonic
            <= self.observation_timeout_seconds
        )
        accepted = (
            latest_is_fresh
            and candidate.observation_count >= minimum_observations
            and candidate.invalid_observations <= allowed_invalid_observations
        )
        start = candidate.samples[0].index
        end = candidate.samples[-1].index + 1
        average_confidence = (
            candidate.confidence_sum / candidate.confidence_observations
            if candidate.confidence_observations
            else 0.0
        )
        summary_metrics = {
            key: candidate.metric_sums[key] / candidate.metric_counts[key]
            for key in candidate.metric_sums
            if candidate.metric_counts.get(key, 0)
        }
        for key in ("postureBucket", "forearmOrientation"):
            choices = {
                token.split(":", 1)[1]: count
                for token, count in candidate.posture_counts.items()
                if token.startswith(f"{key}:")
            }
            summary_metrics[key] = max(choices, key=choices.get) if choices else candidate.initial_metrics.get(key, "unknown")
        self._set_recorder_conditions(summary_metrics)
        if accepted:
            phase = "baseline" if candidate.kind == "baseline" else "gesture"
            for sample in candidate.samples:
                self._recorder.write_sample(
                    sample,
                    phase=phase,
                    gesture=candidate.label,
                    trial=candidate.identity,
                )
            self._recorder.write_event(
                start,
                f"vision_{candidate.kind}_start",
                candidate.label,
                candidate.identity,
            )
            self._recorder.write_event(
                end,
                f"vision_{candidate.kind}_end",
                candidate.label,
                candidate.identity,
            )
            record = {
                "trial_id": candidate.identity,
                "gesture": candidate.label,
                "start_sample": start,
                "end_sample": end,
                "sample_count": len(candidate.samples),
                "mean_vision_confidence": average_confidence,
                "vision_observation_count": candidate.observation_count,
                "invalid_vision_observations": candidate.invalid_observations,
                "allowed_invalid_observations": allowed_invalid_observations,
                "vision_posture": summary_metrics,
                **({"protocol_posture": self._protocol_posture()} if self.reapplication_protocol else {}),
            }
            if candidate.kind == "baseline":
                self._baseline_complete = True
                if self.reapplication_protocol:
                    self._recorder._metadata["baselines"].append(record)
                else:
                    self._recorder._metadata["baseline"] = record
            else:
                self._counts[candidate.label] += 1
                record["repetition_index"] = self._counts[candidate.label]
                self._recorder._metadata["trials"].append(record)
            message = (
                ("Baseline accepted. Complete one trial per gesture in each round."
                 if self.reapplication_protocol else
                 "Baseline accepted. You can now perform gestures in any order.")
                if candidate.kind == "baseline"
                else f"Accepted {candidate.label.replace('_', ' ')} repetition {self._counts[candidate.label]}."
            )
            event_type = "vision_accepted"
        else:
            for sample in candidate.samples:
                self._recorder.write_sample(
                    sample,
                    phase="prepare",
                    gesture="rest",
                    trial=candidate.identity,
                )
            record = {
                "trial_id": candidate.identity,
                "kind": candidate.kind,
                "proposed_gesture": candidate.label,
                "start_sample": start,
                "end_sample": end,
                "sample_count": len(candidate.samples),
                "mean_vision_confidence": average_confidence,
                "vision_observation_count": candidate.observation_count,
                "invalid_vision_observations": candidate.invalid_observations,
                "allowed_invalid_observations": allowed_invalid_observations,
                "vision_posture": summary_metrics,
                **({"protocol_posture": self._protocol_posture()} if self.reapplication_protocol else {}),
                "reason": "vision label/confidence was not stable for the required hold",
            }
            self._recorder._metadata["rejected_trials"].append(record)
            self._rejected_count += 1
            message = (
                "Baseline rejected after a camera dropout. Briefly show a different pose, "
                "then return to relaxed rest; capture will retry automatically."
                if candidate.kind == "baseline"
                else f"Rejected {candidate.label.replace('_', ' ')}; return to neutral and repeat."
            )
            event_type = "vision_rejected"
        self._recorder._metadata["accepted_counts"] = dict(self._counts)
        if self.reapplication_protocol:
            self._recorder._metadata["accepted_counts_by_posture"] = self._counts_by_posture
        self._recorder.update_metadata({})
        self._recorder.flush()
        self._candidate = None
        self._set_recorder_conditions(self._latest_metrics())
        self._phase = "ready" if self._baseline_complete else "awaiting_baseline"
        self.publish(
            {
                "type": event_type,
                "phase": self._phase,
                "message": message,
                "accepted": accepted,
                "label": candidate.label,
                "kind": candidate.kind,
                "trialId": candidate.identity,
                **self.status(),
            }
        )

    def _allowed_invalid_observations(self, candidate: Candidate) -> int:
        """Allow a few shaky camera frames while requiring a mostly valid hold."""

        return max(1, round(candidate.observation_count * self.maximum_invalid_fraction))

    def _reject_candidate(self, reason: str) -> None:
        candidate = self._candidate
        if candidate is None or self._recorder is None:
            return
        self._set_recorder_conditions(candidate.initial_metrics)
        for sample in candidate.samples:
            self._recorder.write_sample(
                sample, phase="prepare", gesture="rest", trial=candidate.identity
            )
        if candidate.samples:
            self._recorder._metadata["rejected_trials"].append(
                {
                    "trial_id": candidate.identity,
                    "kind": candidate.kind,
                    "proposed_gesture": candidate.label,
                    "start_sample": candidate.samples[0].index,
                    "end_sample": candidate.samples[-1].index + 1,
                    "sample_count": len(candidate.samples),
                    "reason": reason,
                }
            )
            self._rejected_count += 1
        self._candidate = None
        self._set_recorder_conditions(self._latest_metrics())

    def _latest_metrics(self) -> dict:
        if self._latest_observation is None:
            return {}
        metrics = self._latest_observation.payload.get("metrics", {})
        return metrics if isinstance(metrics, dict) else {}

    def _set_recorder_conditions(self, metrics: dict) -> None:
        if self._recorder is None:
            return
        self._recorder.conditions = {
            "vision_elbow_posture": metrics.get("postureBucket", "unknown"),
            "vision_forearm_orientation": metrics.get("forearmOrientation", "unknown"),
            **({"protocol_posture": self._protocol_posture()} if self.reapplication_protocol else {}),
        }

    def _finish_gracefully(self) -> None:
        if self._recorder is None:
            return
        self._reject_candidate("session stopped before candidate completed")
        self._recorder.write_event(self._last_source_index + 1, "session_end")
        self._recorder.update_metadata(
            {
                "recording_finished_at": datetime.now().astimezone().isoformat(),
                "accepted_counts": dict(self._counts),
                **({"accepted_counts_by_posture": self._counts_by_posture}
                   if self.reapplication_protocol else {}),
                "vision_observation_file": "vision_observations.jsonl",
                "training_ready": (self._protocol_posture_index == 1
                                   and len(self._recorder._metadata["baselines"]) == 2
                                   and all(count == 6 for counts in self._counts_by_posture.values()
                                           for count in counts.values()))
                if self.reapplication_protocol else self._baseline_complete
                and all(self._counts[label] > 0 for label in VISION_GESTURES),
            }
        )
        self._recorder.mark_complete()
        self._recorder.flush()
        self._recorder.close()
        if self._observations is not None:
            self._observations.close()
        directory = str(self._session_directory)
        self._recorder = None
        self._observations = None
        self._candidate = None
        self._phase = "complete"
        self.publish(
            {
                "type": "vision_complete",
                "phase": "complete",
                "message": f"Vision-labelled recording saved to {directory}",
                **self.status(),
            }
        )

    def _close_incomplete(self) -> None:
        self._reject_candidate("capture failed before candidate completed")
        if self._recorder is not None:
            self._recorder.update_metadata(
                {
                    "recording_failed_at": datetime.now().astimezone().isoformat(),
                    "failure": self._last_error,
                    "accepted_counts": dict(self._counts),
                }
            )
            self._recorder.flush()
            self._recorder.close()
        if self._observations is not None:
            self._observations.close()
        self._recorder = None
        self._observations = None
        self._candidate = None
