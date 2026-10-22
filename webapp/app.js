import { setPose } from './hand_scene.js';

const ui = {
  title: document.querySelector('#state-title'), status: document.querySelector('#status'),
  output: document.querySelector('#output-pose'), detected: document.querySelector('#detected-pose'),
  bar: document.querySelector('#confidence-bar'), confidence: document.querySelector('#confidence-value'),
  calibrationBlock: document.querySelector('#calibration-block'), calibrationBar: document.querySelector('#calibration-bar'),
  calibrationPhase: document.querySelector('#calibration-phase'), calibrationCount: document.querySelector('#calibration-count'),
  sequence: document.querySelector('#sequence'), help: document.querySelector('#sequence-help'),
  posePreview: document.querySelector('#pose-preview'),
  calibrate: document.querySelector('#calibrate'), connect: document.querySelector('#connect'),
  disarm: document.querySelector('#disarm'), reset: document.querySelector('#reset'),
};
let wakeSequence = [];
let modelCompatible = true;
let freshCalibrationEnabled = true;
const poseGuidance = {
  wrist_up: 'Bend the wrist back so the knuckles move toward the top of the forearm; keep fingers relaxed.',
  wrist_down: 'Bend the wrist forward so the palm moves toward the inner forearm; keep fingers relaxed.',
};
function pretty(value) { return (value || '—').replaceAll('_', ' ').toUpperCase(); }
function renderSequence(progress = 0, armed = false) {
  ui.sequence.replaceChildren(...wakeSequence.map((gesture, index) => {
    const item = document.createElement('span');
    item.className = `step ${armed || index < progress ? 'done' : index === progress ? 'next' : ''}`;
    item.textContent = pretty(gesture);
    return item;
  }));
  ui.help.textContent = armed ? 'Unlocked — classifier now controls the hand.' : 'Hold each highlighted pose steadily.';
}
function renderPosePreview(gestures) {
  ui.posePreview.replaceChildren(...gestures.map(gesture => {
    const button = document.createElement('button');
    button.className = 'pose-chip';
    button.textContent = pretty(gesture);
    if (poseGuidance[gesture]) button.title = poseGuidance[gesture];
    button.addEventListener('click', () => {
      setPose(gesture);
      ui.output.textContent = pretty(gesture);
      ui.posePreview.querySelectorAll('.pose-chip').forEach(item => item.classList.toggle('active', item === button));
    });
    return button;
  }));
}
function handleEvent(event) {
  if (event.type === 'config') {
    freshCalibrationEnabled = event.freshCalibrationEnabled !== false;
    ui.calibrate.classList.toggle('hidden', !freshCalibrationEnabled);
    ui.calibrate.disabled = !freshCalibrationEnabled;
    wakeSequence = event.wakeSequence; renderSequence();
    renderPosePreview(['rest', ...event.calibrationGestures]);
    modelCompatible = event.modelCompatible;
    ui.calibrationCount.textContent = `0 / ${event.calibrationGestures.length * event.calibrationTrials}`;
    ui.connect.disabled = !modelCompatible;
    ui.connect.textContent = modelCompatible ? 'Test saved model' : 'Fresh calibration required';
    ui.status.textContent = modelCompatible
      ? (event.modelDescription || `Mode: ${event.mode}. Wake hold: ${event.holdSeconds}s at ≥ ${Math.round(event.minimumConfidence * 100)}% confidence.`)
      : 'The saved classifier uses the previous gesture set. Run Fresh calibration before live testing.';
  } else if (event.type === 'status') {
    ui.title.textContent = event.phase === 'baseline' ? 'Hold still' : event.phase === 'live' ? 'Locked' : pretty(event.phase);
    ui.status.textContent = event.message;
    ui.connect.disabled = !modelCompatible || !['idle', 'error'].includes(event.phase);
    ui.calibrate.disabled = !freshCalibrationEnabled || !['idle', 'error'].includes(event.phase);
    ui.reset.disabled = event.phase === 'idle';
    if (event.phase === 'live') ui.calibrationBlock.classList.add('hidden');
  } else if (event.type === 'calibration') {
    const names = { baseline:'Relax', prepare:'Get ready', countdown:String(event.secondsLeft || ''), hold:'Hold', rest:'Relax', training:'Training', complete:'Model ready', connecting:'Connecting', reconnecting:'Reconnecting' };
    ui.title.textContent = names[event.phase] || pretty(event.phase);
    ui.status.textContent = poseGuidance[event.promptGesture]
      ? `${event.message} ${poseGuidance[event.promptGesture]}`
      : event.message;
    ui.output.textContent = pretty(event.promptGesture);
    ui.detected.textContent = 'CALIBRATION CUE';
    setPose(event.promptGesture || 'rest');
    ui.calibrationBlock.classList.remove('hidden');
    const total = Math.max(1, event.totalSteps || 1);
    ui.calibrationBar.style.width = `${Math.min(100, 100 * (event.step || 0) / total)}%`;
    ui.calibrationPhase.textContent = pretty(event.phase);
    ui.calibrationCount.textContent = `${event.step || 0} / ${event.totalSteps || total}`;
    if (event.phase === 'complete') {
      modelCompatible = true;
      ui.connect.textContent = 'Test saved model';
    }
    ui.calibrate.disabled = true;
    ui.connect.disabled = true;
  } else if (event.type === 'error') {
    ui.title.textContent = 'Connection error'; ui.status.textContent = event.message; ui.connect.disabled = !modelCompatible; ui.calibrate.disabled = !freshCalibrationEnabled;
  } else if (event.type === 'prediction') {
    ui.title.textContent = event.armed ? 'Mirroring' : 'Locked';
    ui.output.textContent = pretty(event.gesture);
    ui.detected.textContent = `${pretty(event.rawGesture)}${event.rejected ? ' · HELD' : ''}`;
    ui.confidence.textContent = `${Math.round(event.confidence * 100)}%`; ui.bar.style.width = `${event.confidence * 100}%`;
    setPose(event.gesture); renderSequence(event.progress, event.armed);
    if (event.justArmed) ui.status.textContent = 'Wake sequence accepted. Live gesture mirroring is active.';
    else if (event.rejected) ui.status.textContent = `Low-confidence prediction held; output remains ${pretty(event.gesture)}.`;
  }
}

const events = new EventSource('/api/events');
events.onmessage = message => handleEvent(JSON.parse(message.data));
events.onerror = () => { ui.title.textContent = 'Reconnecting'; ui.status.textContent = 'The browser is reconnecting to the local Python server.'; };
ui.connect.addEventListener('click', async () => {
  ui.connect.disabled = true;
  const response = await fetch('/api/start', { method: 'POST' });
  const result = await response.json();
  if (!result.started) {
    ui.connect.disabled = !modelCompatible;
    ui.calibrate.disabled = !freshCalibrationEnabled;
    ui.status.textContent = 'Another serial task is still active. Use Reset connection, wait for Ready, then retry.';
  }
});
ui.calibrate.addEventListener('click', async () => {
  ui.calibrate.disabled = true;
  ui.connect.disabled = true;
  const response = await fetch('/api/calibrate', { method: 'POST' });
  const result = await response.json();
  if (!result.started) { ui.calibrate.disabled = !freshCalibrationEnabled; ui.connect.disabled = false; }
});
ui.disarm.addEventListener('click', () => fetch('/api/disarm', { method: 'POST' }));
ui.reset.addEventListener('click', async () => {
  ui.reset.disabled = true;
  ui.connect.disabled = true;
  ui.calibrate.disabled = true;
  await fetch('/api/reset', { method: 'POST' });
});
