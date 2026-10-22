const gesture = document.querySelector("#gesture");
const status = document.querySelector("#status");
const start = document.querySelector("#start");
const stop = document.querySelector("#stop");
let setPose = () => {};
import("./hand_scene.js?v=2").then(module => {
  setPose = module.setPose;
}).catch(error => {
  status.textContent = `3D hand could not start: ${error.message}`;
});

function show(event) {
  const phase = event.phase || "idle";
  const running = ["connecting", "baseline", "live", "reconnecting"].includes(phase);
  start.disabled = running;
  stop.disabled = !running;
  if (event.message) status.textContent = event.message;
  if (event.type === "prediction") {
    gesture.textContent = event.label.replaceAll("_", " ").toUpperCase();
    if (event.label !== "uncertain") setPose(event.label);
  }
}

async function action(name) {
  try {
    const response = await fetch(`/api/live/${name}`, { method: "POST" });
    if (!response.ok) throw new Error(`Connection request failed (${response.status})`);
    const result = await response.json();
    if (!result.ok) throw new Error("Receiver is already in use");
    show({ phase: name === "start" ? "connecting" : "idle",
           message: name === "start" ? "Connecting to receiver…" : "Disconnected" });
    if (name === "stop") {
      setPose("rest");
      gesture.textContent = "REST";
    }
  } catch (error) { status.textContent = error.message; }
}

start.addEventListener("click", () => action("start"));
stop.addEventListener("click", () => action("stop"));
fetch("/api/live/status").then(response => response.json()).then(show)
  .catch(error => status.textContent = error.message);
const events = new EventSource("/api/live/events");
events.onmessage = ({data}) => show(JSON.parse(data));
events.onerror = () => { status.textContent = "Connection interrupted; retrying…"; };
