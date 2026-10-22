import { setPose } from "./hand_scene.js";

const $ = (selector) => document.querySelector(selector);
const labels = ["fist", "peace", "pinch", "point", "rest", "splayed", "thumbs_up"];
function show(event) {
  if (event.phase) $("#phase").textContent = event.phase.toUpperCase();
  if (event.message) $("#message").textContent = event.message;
  const running = ["connecting", "baseline", "live", "reconnecting"].includes(event.phase);
  $("#start").disabled = running;
  $("#stop").disabled = !running;
  if (event.type === "signal_quality") {
    $("#channels").textContent = event.baseline_rms_by_channel
      .map((rms, index) => `CH${index + 1}: ${rms}`).join(" · ");
  }
  if (event.type !== "prediction") return;
  if (event.label !== "uncertain") setPose(event.label);
  $("#liveChannels").textContent = event.live_rms_relative_to_baseline
    .map((rms, index) => `CH${index + 1}: ${rms}`).join(" · ");
  $("#label").textContent = event.label.replaceAll("_", " ").toUpperCase();
  $("#confidence").textContent = `${Math.round(event.confidence * 100)}%`;
  $("#probabilities").innerHTML = labels.map((label) => {
    const probability = event.probabilities[label] || 0;
    return `<p>${label.replaceAll("_", " ")}: ${Math.round(probability * 100)}%</p>`;
  }).join("");
}
async function action(name) {
  try {
    const response = await fetch(`/api/live/${name}`, { method: "POST" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    if (!result.ok) throw new Error("Inference is already running");
    show({ phase: name === "start" ? "connecting" : "idle" });
    if (name === "stop") setPose("rest");
  } catch (error) { $("#message").textContent = error.message; }
}
$("#start").onclick = () => action("start");
$("#stop").onclick = () => action("stop");
fetch("/api/live/status").then((response) => response.json()).then(show);
fetch("/api/live/config").then((response) => response.json()).then((config) => {
  $("#source").textContent = config.source === "synthetic"
    ? "Synthetic test stream — predictions are a software check only."
    : "Serial EMG receiver";
  if (config.baselineMode === "fresh") {
    $("#modeDescription").textContent = `${config.modelName} with a new five-second rest baseline. No recording is saved.`;
    $("#baselineTitle").textContent = "Fresh baseline RMS";
  } else {
    $("#modeDescription").textContent = `${config.modelName} with recorded baseline from ${config.baselineSession}. No recording is saved.`;
    $("#baselineTitle").textContent = `Recorded baseline RMS (${config.baselineSession})`;
  }
});
const events = new EventSource("/api/live/events");
events.onmessage = ({data}) => show(JSON.parse(data));
events.onerror = () => { $("#message").textContent = "Connection interrupted; retrying locally."; };
