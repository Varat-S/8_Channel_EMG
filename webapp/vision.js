import { FilesetResolver, HandLandmarker, PoseLandmarker } from "./vendor/mediapipe/vision_bundle.mjs";
import { classifyHandPose, landmarkMotion, poseFeatureVector } from "./vision_geometry.js";

const $ = (selector) => document.querySelector(selector);
const video = $("#camera");
const canvas = $("#overlay");
const context = canvas.getContext("2d");
const gestures = ["rest", "fist", "splayed", "peace", "point", "pinch", "thumbs_up"];
const calibrationPostures = ["vertical", "horizontal"];
const calibrationSteps = calibrationPostures.flatMap((posture) => gestures.map((label) => ({ label, posture })));
const poseDescriptions = {
  rest: "Naturally grouped fingers; neutral wrist",
  fist: "All four fingers closed",
  splayed: "All fingers extended and separated",
  peace: "Index and middle extended",
  point: "Index extended; other fingers closed",
  pinch: "Thumb and index tips touching",
  thumbs_up: "Four fingers closed; thumb extended",
};
const connections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

let config;
let handLandmarker;
let poseLandmarker;
let cameraStream;
let cameraRunning = false;
const cameraChannel = "BroadcastChannel" in window ? new BroadcastChannel("ureca-vision-camera") : null;
let sessionRunning = false;
let serverState = { phase: "idle", counts: Object.fromEntries(gestures.map((g) => [g, 0])), progress: 0 };
let lastInferenceAt = 0;
let previousLandmarks = null;
let stableLabel = "no_hand";
let stableSince = 0;
let currentResult = { label: "no_hand", confidence: 0, metrics: {} };
let postInFlight = false;
let lastObservationPostAt = 0;
let captureRequestInFlight = false;
let releaseRequired = false;
let poseTemplates = null;
let visualCalibrationState = null;
const calibrationSamplesPerPose = 18;
const calibrationPreparationMs = 1800;
const calibrationStorageKey = "ureca-emg-vision-calibration-v1";

function usablePoseTemplates(value) {
  return Boolean(value) && gestures.every((label) =>
    Array.isArray(value[label]?.prototypes) && value[label].prototypes.length === 2
  );
}

async function request(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok || result.ok === false) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}

function updateStatus(state) {
  const previousPhase = serverState.phase;
  serverState = { ...serverState, ...state };
  if (state.type === "vision_rejected") releaseRequired = false;
  $("#phasePill").textContent = String(serverState.phase || "idle").replaceAll("_", " ").toUpperCase();
  $("#statusTitle").textContent = serverState.phase === "baseline"
    ? "Capturing relaxed baseline"
    : serverState.activeLabel
      ? `Recording ${serverState.activeLabel.replaceAll("_", " ")}`
      : titleForPhase(serverState.phase);
  if (state.message) {
    $("#statusMessage").textContent = state.message;
  } else if (state.phase && state.phase !== previousPhase) {
    const fallbackMessages = {
      awaiting_baseline: "EMG stream active. Hold a relaxed neutral hand for the baseline.",
      ready: "EMG stream active again. You can continue with the next pose.",
      reconnecting: "EMG frames paused. The recorder is reopening the receiver.",
      complete: "Session saved and COM10 released.",
    };
    if (fallbackMessages[state.phase]) $("#statusMessage").textContent = fallbackMessages[state.phase];
  }
  $("#captureBar").style.width = `${100 * (serverState.progress || 0)}%`;
  if (["idle", "complete", "error"].includes(serverState.phase)) sessionRunning = false;
  else if (serverState.running || serverState.phase === "connecting") sessionRunning = true;
  $("#sessionButton").disabled = !cameraRunning || !poseTemplates || sessionRunning;
  $("#visualCalibrationButton").disabled = !cameraRunning || sessionRunning || Boolean(visualCalibrationState);
  $("#stopButton").disabled = !sessionRunning;
  if (config?.reapplicationMode) {
    const posture = serverState.protocolPosture || "vertical";
    const counts = serverState.counts || {};
    const finished = gestures.every((gesture) => counts[gesture] === 6);
    $("#protocolBlock").hidden = false;
    $("#protocolBlock").textContent = `Frozen reapplication · ${posture} forearm · ${Object.values(counts).reduce((sum, count) => sum + count, 0)}/42 accepted trials`;
    $("#advancePostureButton").hidden = posture !== "vertical" || !sessionRunning;
    $("#advancePostureButton").disabled = !finished || !serverState.baselineComplete || Boolean(serverState.activeLabel);
  }
  renderPoseGrid();
}

function titleForPhase(phase) {
  return ({
    idle: "Enable the camera",
    connecting: "Connecting to EMG",
    awaiting_baseline: "Relax for baseline",
    baseline: "Capturing baseline",
    ready: "Perform any pose",
    reconnecting: "Reconnecting receiver",
    stopping: "Saving session",
    complete: "Session saved",
    error: "Recorder error",
  })[phase] || "Vision-assisted capture";
}

function renderPoseGrid() {
  const target = Number($("#targetReps").value || 6);
  $("#poseGrid").innerHTML = gestures.map((gesture) => {
    const count = serverState.counts?.[gesture] || 0;
    const ticks = Array.from({ length: target }, (_, i) => `<i class="tick ${i < count ? "done" : ""}"></i>`).join("");
    const calibrating = visualCalibrationState && calibrationSteps[visualCalibrationState.index].label === gesture;
    const classes = ["pose-card", currentResult.label === gesture || calibrating ? "active" : "", count >= target ? "complete" : ""].join(" ");
    if (config?.viewOnly) return `<article class="${classes}"><h3>${gesture.replaceAll("_", " ")}</h3><p>${poseDescriptions[gesture]}</p></article>`;
    return `<article class="${classes}"><header><h3>${gesture.replaceAll("_", " ")}</h3><span class="count">${count}/${target}</span></header><p>${poseDescriptions[gesture]}</p><div class="ticks">${ticks}</div></article>`;
  }).join("");
}

function releaseCamera() {
  cameraRunning = false;
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  video.pause();
  video.srcObject = null;
  $("#cameraEmpty").hidden = false;
  $("#cameraButton").disabled = false;
  $("#cameraButton").textContent = "Enable camera";
  if (config?.viewOnly) {
    $("#statusTitle").textContent = "Camera off";
    $("#statusMessage").textContent = "Camera released. Enable it again to continue the demo.";
  }
}

cameraChannel?.addEventListener("message", (event) => {
  if (event.data === "release-camera" && !sessionRunning) releaseCamera();
});

async function enableCamera() {
  $("#cameraButton").disabled = true;
  $("#statusMessage").textContent = "Loading local hand and pose models…";
  try {
    cameraChannel?.postMessage("release-camera");
    const vision = await FilesetResolver.forVisionTasks("./vendor/mediapipe/wasm");
    const createWithFallback = async (Task, options) => {
      try {
        return await Task.createFromOptions(vision, options);
      } catch (gpuError) {
        console.warn("GPU delegate unavailable; using the local CPU/WASM delegate.", gpuError);
        return Task.createFromOptions(vision, {
          ...options,
          baseOptions: { modelAssetPath: options.baseOptions.modelAssetPath },
        });
      }
    };
    [handLandmarker, poseLandmarker] = await Promise.all([
      createWithFallback(HandLandmarker, {
        baseOptions: { modelAssetPath: "./models/hand_landmarker.task", delegate: "GPU" },
        runningMode: "VIDEO", numHands: 1,
        minHandDetectionConfidence: 0.60, minHandPresenceConfidence: 0.60, minTrackingConfidence: 0.60,
      }),
      createWithFallback(PoseLandmarker, {
        baseOptions: { modelAssetPath: "./models/pose_landmarker_lite.task", delegate: "GPU" },
        runningMode: "VIDEO", numPoses: 1,
        minPoseDetectionConfidence: 0.55, minPosePresenceConfidence: 0.55, minTrackingConfidence: 0.55,
      }),
    ]);
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
    video.srcObject = cameraStream;
    await video.play();
    cameraRunning = true;
    $("#cameraEmpty").hidden = true;
    $("#cameraButton").textContent = "Camera enabled";
    $("#visualCalibrationButton").disabled = false;
    if (poseTemplates) {
      $("#visualCalibrationButton").textContent = "Recalibrate 7 poses Ã— 2 postures";
      $("#sessionButton").disabled = false;
    }
    $("#statusTitle").textContent = "Camera ready";
    $("#statusMessage").textContent = config?.viewOnly
      ? "Camera-only demo is running. Pose labels update live; nothing is saved."
      : poseTemplates
      ? "Previous visual calibration loaded. Keep the same camera framing, or recalibrate if it changed."
      : "Frame your elbow, wrist and hand; then run the short visual calibration.";
    requestAnimationFrame(inferenceLoop);
  } catch (error) {
    if (cameraStream) releaseCamera();
    $("#cameraButton").disabled = false;
    $("#statusTitle").textContent = "Camera unavailable";
    $("#statusMessage").textContent = ["NotReadableError", "AbortError"].includes(error.name)
      ? "Camera is in use. Close any earlier vision tab or camera app, then click Enable camera again."
      : error.message;
  }
}

function drawHand(points, pose, metrics) {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (pose && metrics?.armSide) {
    const elbowIndex = metrics.armSide === "left" ? 13 : 14;
    const wristIndex = metrics.armSide === "left" ? 15 : 16;
    const elbow = pose[elbowIndex];
    const wrist = pose[wristIndex];
    if (elbow && wrist && (elbow.visibility ?? 0) >= .35 && (wrist.visibility ?? 0) >= .35) {
      context.strokeStyle = "#52e0a4";
      context.lineWidth = Math.max(6, canvas.width / 150);
      context.lineCap = "round";
      context.beginPath();
      context.moveTo(elbow.x * canvas.width, elbow.y * canvas.height);
      context.lineTo(wrist.x * canvas.width, wrist.y * canvas.height);
      context.stroke();
      context.fillStyle = "#52e0a4";
      context.beginPath();
      context.arc(elbow.x * canvas.width, elbow.y * canvas.height, Math.max(8, canvas.width / 105), 0, Math.PI * 2);
      context.fill();
      context.fillStyle = "#07130f";
      context.beginPath();
      context.arc(elbow.x * canvas.width, elbow.y * canvas.height, Math.max(3, canvas.width / 280), 0, Math.PI * 2);
      context.fill();
    }
  }
  if (!points) return;
  context.strokeStyle = "#9cb0c2";
  context.fillStyle = "#effff8";
  context.lineWidth = Math.max(2, canvas.width / 400);
  for (const [a, b] of connections) {
    context.beginPath();
    context.moveTo(points[a].x * canvas.width, points[a].y * canvas.height);
    context.lineTo(points[b].x * canvas.width, points[b].y * canvas.height);
    context.stroke();
  }
  for (const point of points) {
    context.beginPath();
    context.arc(point.x * canvas.width, point.y * canvas.height, Math.max(2.5, canvas.width / 300), 0, Math.PI * 2);
    context.fill();
  }
}

async function inferenceLoop(timestamp) {
  if (!cameraRunning) return;
  if (video.readyState >= 2 && timestamp - lastInferenceAt >= 85) {
    lastInferenceAt = timestamp;
    const handResult = handLandmarker.detectForVideo(video, timestamp);
    const poseResult = poseLandmarker.detectForVideo(video, timestamp);
    const hand = handResult.landmarks?.[0] || null;
    const world = handResult.worldLandmarks?.[0] || null;
    const pose = poseResult.landmarks?.[0] || null;
    const poseWorld = poseResult.worldLandmarks?.[0] || null;
    const handedness = handResult.handedness?.[0]?.[0]?.categoryName || null;
    const classified = classifyHandPose(world, hand, pose, { poseTemplates });
    const motion = hand ? landmarkMotion(previousLandmarks, hand) : 1;
    previousLandmarks = hand ? structuredClone(hand) : null;
    const visuallyStable = motion < 0.065 && classified.confidence >= (config?.minimumVisionConfidence || .62);
    if (classified.label !== stableLabel || !visuallyStable) {
      stableLabel = classified.label;
      stableSince = timestamp;
    }
    const stableMs = visuallyStable ? timestamp - stableSince : 0;
    currentResult = { ...classified, confidence: visuallyStable ? classified.confidence : Math.min(classified.confidence, 0.59), motion, stableMs, hand, pose, poseWorld, handedness };
    updateVisualCalibration(timestamp, currentResult.metrics, motion);
    updateVisionReadout();
    drawHand(hand, pose, classified.metrics);
    sendObservation();
    maybeCapture();
  }
  requestAnimationFrame(inferenceLoop);
}

function updateVisionReadout() {
  const calibrationLabel = visualCalibrationState ? calibrationSteps[visualCalibrationState.index].label : null;
  $("#poseLabel").textContent = calibrationLabel ? `LEARNING ${calibrationLabel.replaceAll("_", " ").toUpperCase()}` : currentResult.label.replaceAll("_", " ").toUpperCase();
  $("#poseConfidence").textContent = calibrationLabel ? `${visualCalibrationState.samples.length}/${calibrationSamplesPerPose}` : `${Math.round(100 * currentResult.confidence)}%`;
  $("#stableTime").textContent = `${(currentResult.stableMs / 1000).toFixed(1)} s`;
  $("#motionValue").textContent = Number.isFinite(currentResult.motion) ? currentResult.motion.toFixed(3) : "—";
  const wrist = currentResult.metrics.wristDeflectionDegrees;
  $("#wristValue").textContent = wrist == null ? "elbow not seen" : `${wrist.toFixed(0)}°`;
  const elbow = currentResult.metrics.elbowAngleDegrees;
  $("#elbowValue").textContent = elbow == null ? "not visible" : `${elbow.toFixed(0)}°`;
  $("#postureValue").textContent = (currentResult.metrics.postureBucket || "—").replaceAll("_", " ");
  $("#forearmValue").textContent = currentResult.metrics.forearmOrientation || "—";
  const spread = currentResult.metrics.spread;
  $("#spreadValue").textContent = Number.isFinite(spread) ? spread.toFixed(2) : "—";
  $("#stabilityBar").style.width = `${Math.min(100, currentResult.stableMs / 12)}%`;
  renderPoseGrid();
}

function sendObservation() {
  const now = performance.now();
  if (!sessionRunning || postInFlight || now - lastObservationPostAt < 200) return;
  lastObservationPostAt = now;
  postInFlight = true;
  request("/api/vision/observation", {
    label: currentResult.label,
    confidence: currentResult.confidence,
    clientTimeMs: performance.timeOrigin + performance.now(),
    mediaTimeMs: video.currentTime * 1000,
    handedness: currentResult.handedness,
    handLandmarks: currentResult.hand,
    poseLandmarks: currentResult.pose,
    poseWorldLandmarks: currentResult.poseWorld,
    metrics: { ...currentResult.metrics, landmarkMotion: currentResult.motion, stableMs: currentResult.stableMs },
  }).catch((error) => $("#statusMessage").textContent = error.message).finally(() => postInFlight = false);
}

async function maybeCapture() {
  if (!sessionRunning || captureRequestInFlight || currentResult.stableMs < 1100 || currentResult.confidence < config.minimumVisionConfidence) return;
  if (!gestures.includes(currentResult.label)) return;
  if (serverState.activeLabel || ["connecting", "baseline", "capturing", "reconnecting", "error", "complete"].includes(serverState.phase)) return;
  if (releaseRequired) {
    if (currentResult.label !== releaseRequired) releaseRequired = false;
    else return;
  }
  captureRequestInFlight = true;
  try {
    if (!serverState.baselineComplete && currentResult.label === "rest") {
      await request("/api/vision/baseline/start");
      releaseRequired = "rest";
    } else if (serverState.baselineComplete) {
      const target = Number($("#targetReps").value || 6);
      if ((serverState.counts?.[currentResult.label] || 0) >= target) return;
      if (config.reapplicationMode && (serverState.counts?.[currentResult.label] || 0) !== Math.min(...Object.values(serverState.counts))) return;
      await request("/api/vision/trial/start", { label: currentResult.label });
      releaseRequired = currentResult.label;
    }
  } catch (error) {
    $("#statusMessage").textContent = error.message;
  } finally {
    captureRequestInFlight = false;
  }
}

async function advancePosture() {
  try {
    updateStatus(await request("/api/vision/posture/advance"));
    releaseRequired = false;
  } catch (error) { $("#statusMessage").textContent = error.message; }
}

function startVisualCalibration() {
  poseTemplates = {};
  visualCalibrationState = {
    index: 0,
    phase: "prepare",
    phaseStarted: performance.now(),
    samples: [],
  };
  $("#visualCalibrationButton").disabled = true;
  $("#sessionButton").disabled = true;
  showCalibrationPrompt();
  renderPoseGrid();
}

function showCalibrationPrompt() {
  const { label, posture } = calibrationSteps[visualCalibrationState.index];
  $("#statusTitle").textContent = `Prepare ${label.replaceAll("_", " ")} · ${posture} forearm`;
  $("#statusMessage").textContent = `${poseDescriptions[label]}. Keep the forearm ${posture}; collection begins after the countdown.`;
  $("#captureBar").style.width = "0%";
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function finishCalibrationPose() {
  const state = visualCalibrationState;
  const { label, posture } = calibrationSteps[state.index];
  const width = state.samples[0].length;
  const mean = Array.from({ length: width }, (_, index) => average(state.samples.map((sample) => sample[index])));
  const std = Array.from({ length: width }, (_, index) => {
    const variance = average(state.samples.map((sample) => (sample[index] - mean[index]) ** 2));
    return Math.sqrt(variance);
  });
  if (!poseTemplates[label]) poseTemplates[label] = { prototypes: [] };
  poseTemplates[label].prototypes.push({ posture, mean, std, sampleCount: state.samples.length });
  state.index += 1;
  if (state.index >= calibrationSteps.length) {
    visualCalibrationState = null;
    try { localStorage.setItem(calibrationStorageKey, JSON.stringify(poseTemplates)); } catch (_) {}
    $("#visualCalibrationButton").textContent = "✓ Seven poses · two postures learned";
    $("#visualCalibrationButton").disabled = false;
    $("#sessionButton").disabled = false;
    $("#statusTitle").textContent = "Visual calibration complete";
    $("#statusMessage").textContent = "Test each live label now. Recalibrate if the camera or arm orientation changes.";
    $("#captureBar").style.width = "0%";
    renderPoseGrid();
    return;
  }
  state.phase = "prepare";
  state.phaseStarted = performance.now();
  state.samples = [];
  showCalibrationPrompt();
  renderPoseGrid();
}

function updateVisualCalibration(timestamp, metrics, motion) {
  const state = visualCalibrationState;
  if (!state) return;
  const vector = poseFeatureVector(metrics);
  if (!vector || (metrics.armVisibility || 0) < .35) {
    if (state.phase === "prepare") state.phaseStarted = timestamp;
    $("#statusMessage").textContent = "Keep one hand, wrist and elbow clearly visible to continue.";
    return;
  }
  const { label, posture } = calibrationSteps[state.index];
  if (metrics.forearmOrientation !== posture) {
    if (state.phase === "prepare") state.phaseStarted = timestamp;
    $("#statusTitle").textContent = `Adjust to ${posture} forearm`;
    $("#statusMessage").textContent = `The green forearm line must read ${posture} before ${label.replaceAll("_", " ")} can be learned.`;
    return;
  }
  if (state.phase === "prepare") {
    const elapsed = timestamp - state.phaseStarted;
    const remaining = Math.max(0, Math.ceil((calibrationPreparationMs - elapsed) / 1000));
    $("#statusTitle").textContent = `Prepare ${label.replaceAll("_", " ")} · ${posture} · ${remaining}`;
    $("#captureBar").style.width = `${Math.min(100, elapsed / calibrationPreparationMs * 100)}%`;
    if (elapsed >= calibrationPreparationMs) {
      state.phase = "collect";
      state.samples = [];
      $("#statusTitle").textContent = `Learning ${label.replaceAll("_", " ")} · ${posture}`;
      $("#statusMessage").textContent = "Hold steady while the landmark examples are collected.";
    }
    return;
  }
  if (motion >= .085) {
    $("#statusMessage").textContent = `Hold ${label.replaceAll("_", " ")} more steadily.`;
    return;
  }
  state.samples.push(vector);
  $("#captureBar").style.width = `${Math.min(100, state.samples.length / calibrationSamplesPerPose * 100)}%`;
  if (state.samples.length >= calibrationSamplesPerPose) finishCalibrationPose();
}

async function startSession() {
  $("#sessionButton").disabled = true;
  try {
    const state = await request("/api/vision/session/start", {
      subject_id: $("#subjectId").value.trim(),
      electrode_placement_id: $("#placementId").value.trim(),
      arm_posture: $("#armPosture").value.trim(),
      vision_calibration: {
        pose_templates: poseTemplates,
      },
    });
    sessionRunning = true;
    releaseRequired = false;
    updateStatus(state);
  } catch (error) {
    $("#statusMessage").textContent = error.message;
    $("#sessionButton").disabled = false;
  }
}

async function stopSession() {
  $("#stopButton").disabled = true;
  $("#stopButton").textContent = "Stopping & savingâ€¦";
  try {
    const state = await request("/api/vision/session/stop");
    sessionRunning = Boolean(state.running);
    updateStatus(state);
  } catch (error) {
    $("#statusMessage").textContent = error.message;
  } finally {
    $("#stopButton").textContent = "Stop & save";
    $("#stopButton").disabled = !sessionRunning;
  }
}

async function initialize() {
  config = await fetch("/api/vision/config").then((response) => response.json());
  if (config.viewOnly) {
    document.body.classList.add("demo-view-only");
    $("#privacyBadge").textContent = "Camera only · no recording";
    $(".lede").textContent = "Live camera pose labels and landmarks. No EMG session or recording is started.";
    $("#statusMessage").textContent = "Enable the camera to preview the seven poses. Nothing is saved.";
    $("#protocolTitle").textContent = "Demo mode";
    $("#protocolList").innerHTML = "<li>Enable the camera, then show one gesture at a time.</li><li>Keep your hand, wrist and elbow visible so the pose label can update.</li>";
  }
  try {
    const localTemplates = JSON.parse(localStorage.getItem(calibrationStorageKey) || "null");
    if (usablePoseTemplates(localTemplates)) poseTemplates = localTemplates;
  } catch (_) {}
  const recoveredTemplates = config.savedVisionCalibration?.pose_templates;
  if (!poseTemplates && usablePoseTemplates(recoveredTemplates)) poseTemplates = recoveredTemplates;
  updateStatus(config);
  if (config.reapplicationMode) {
    // P3 belongs to the frozen source session. Give this application a new ID.
    if ($("#placementId").value === "P3") $("#placementId").value = "P4";
    $("#targetReps").value = "6";
    $("#targetReps").disabled = true;
    $("#sessionButton").textContent = "Start frozen reapplication";
    $("#protocolTitle").textContent = "Frozen electrode reapplication protocol";
    $("#protocolList").innerHTML = `
      <li>Calibrate camera templates for seven gestures in both forearm orientations.</li>
      <li>Set the forearm vertical. Capture a fresh relaxed baseline, then six rounds with one accepted trial per gesture in each round.</li>
      <li>After all 42 vertical trials, press Start horizontal block. Set the forearm horizontal and capture a new relaxed baseline.</li>
      <li>Complete the same six rounds horizontally. Rejected trials do not consume a repetition. Stop &amp; save only after all 84 are accepted.</li>
      <li>Trials 1–3 per gesture and posture are reserved for calibration; trials 4–6 are fixed test data. Do not inspect model scores during collection.</li>`;
  }
  $("#sourceLine").textContent = config.viewOnly
    ? "Camera-only demo · no serial connection or recording"
    : config.source === "serial" ? `EMG source: ${config.serialPort} · transmitter must be on before starting`
    : "EMG source: synthetic test stream · no electrodes required";
  renderPoseGrid();
  const events = new EventSource("/api/vision/events");
  events.onmessage = ({ data }) => updateStatus(JSON.parse(data));
  setInterval(async () => {
    try {
      const response = await fetch("/api/vision/status", { cache: "no-store" });
      if (response.ok) updateStatus(await response.json());
    } catch (_) {
      // EventSource shows the connection warning; the next poll retries.
    }
  }, 1000);
  events.onerror = () => $("#statusMessage").textContent = "Recorder connection interrupted; retrying locally…";
}

$("#cameraButton").addEventListener("click", enableCamera);
$("#visualCalibrationButton").addEventListener("click", startVisualCalibration);
$("#sessionButton").addEventListener("click", startSession);
$("#stopButton").addEventListener("click", stopSession);
$("#advancePostureButton").addEventListener("click", advancePosture);
$("#targetReps").addEventListener("change", renderPoseGrid);
document.addEventListener("visibilitychange", () => {
  if (document.hidden && config?.viewOnly) releaseCamera();
});
window.addEventListener("pagehide", releaseCamera);
initialize().catch((error) => $("#statusMessage").textContent = error.message);

