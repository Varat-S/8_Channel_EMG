import * as THREE from './vendor/three.module.js';

const stage = document.querySelector('#stage');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x09100e, 0.035);
const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
camera.position.set(0, 0.1, 11);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
stage.prepend(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xc9ffe7, 0x18211d, 2.2));
const key = new THREE.DirectionalLight(0xffffff, 4.2);
key.position.set(-4, 6, 8);
key.castShadow = true;
scene.add(key);
const rim = new THREE.PointLight(0x55ffbf, 18, 18);
rim.position.set(5, 1, -2);
scene.add(rim);

const hand = new THREE.Group();
hand.rotation.set(-0.12, -0.12, -0.03);
scene.add(hand);

// Keep the forearm fixed while the palm and digits rotate about the wrist.
const wristPivot = new THREE.Group();
wristPivot.position.y = -1.75;
hand.add(wristPivot);
const palmAssembly = new THREE.Group();
palmAssembly.position.y = 1.75;
wristPivot.add(palmAssembly);

const skin = new THREE.MeshPhysicalMaterial({ color: 0xd6a27f, roughness: 0.6, metalness: 0, clearcoat: 0.12 });
const jointMaterial = new THREE.MeshStandardMaterial({ color: 0xb97c61, roughness: 0.7 });

function roundedBox(width, height, depth, radius = 0.18) {
  const shape = new THREE.Shape();
  const x = -width / 2, y = -height / 2;
  shape.moveTo(x + radius, y);
  shape.lineTo(x + width - radius, y); shape.quadraticCurveTo(x + width, y, x + width, y + radius);
  shape.lineTo(x + width, y + height - radius); shape.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  shape.lineTo(x + radius, y + height); shape.quadraticCurveTo(x, y + height, x, y + height - radius);
  shape.lineTo(x, y + radius); shape.quadraticCurveTo(x, y, x + radius, y);
  const geometry = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true, bevelSize: .07, bevelThickness: .07, bevelSegments: 3 });
  geometry.center();
  return geometry;
}

const palm = new THREE.Mesh(roundedBox(3.05, 3.5, .72, .52), skin);
palm.position.y = -0.25;
palm.castShadow = palm.receiveShadow = true;
palmAssembly.add(palm);

const wrist = new THREE.Mesh(new THREE.CapsuleGeometry(1.08, 1.45, 8, 18), skin);
wrist.position.y = -2.65;
wrist.scale.z = .48;
wrist.castShadow = true;
hand.add(wrist);

const digits = {};
function addDigit(name, x, y, lengths, width, baseRotationZ = 0) {
  const root = new THREE.Group();
  root.position.set(x, y, 0);
  root.rotation.z = baseRotationZ;
  root.userData.targetZ = baseRotationZ;
  palmAssembly.add(root);
  const joints = [];
  let parent = root;
  for (let i = 0; i < lengths.length; i++) {
    const pivot = new THREE.Group();
    parent.add(pivot);
    const segment = new THREE.Mesh(new THREE.CapsuleGeometry(width * (1 - i * .08), Math.max(.05, lengths[i] - width * 2), 8, 16), skin);
    segment.position.y = lengths[i] / 2;
    segment.rotation.z = 0;
    segment.castShadow = true;
    pivot.add(segment);
    const joint = new THREE.Mesh(new THREE.SphereGeometry(width * 1.03, 14, 10), jointMaterial);
    joint.scale.z = .78;
    pivot.add(joint);
    joints.push(pivot);
    const next = new THREE.Group();
    next.position.y = lengths[i];
    pivot.add(next);
    parent = next;
  }
  digits[name] = { root, joints, baseRotationZ };
}

addDigit('index', -0.92, 1.38, [1.23, .92, .7], .34, .035);
addDigit('middle', -0.31, 1.52, [1.34, 1.0, .74], .36, .012);
addDigit('ring', .36, 1.48, [1.25, .92, .69], .35, -.018);
addDigit('pinky', 1.0, 1.28, [1.02, .76, .58], .3, -.065);
addDigit('thumb', -1.43, -.72, [1.0, .78, .62], .38, 1.05);

const poses = {
  rest: {
    index:[.12,.10,.08], middle:[.10,.09,.07], ring:[.15,.13,.10], pinky:[.2,.16,.12], thumb:[.15,.1,.08],
    roots:{ index:.035, middle:.012, ring:-.018, pinky:-.065, thumb:1.05 }, wristX:0,
  },
  splayed: {
    index:[0,0,0], middle:[0,0,0], ring:[0,0,0], pinky:[0,0,0], thumb:[.02,.02,.01],
    roots:{ index:.16, middle:.045, ring:-.08, pinky:-.25, thumb:1.28 }, wristX:0,
  },
  fist: {
    index:[1.25,1.3,1.05], middle:[1.3,1.35,1.1], ring:[1.34,1.35,1.12], pinky:[1.4,1.4,1.15], thumb:[.5,.65,.45],
    roots:{ index:.025, middle:.008, ring:-.012, pinky:-.045, thumb:.62 }, wristX:0,
  },
  wrist_up: {
    index:[.12,.10,.08], middle:[.10,.09,.07], ring:[.15,.13,.10], pinky:[.2,.16,.12], thumb:[.15,.1,.08],
    roots:{ index:.035, middle:.012, ring:-.018, pinky:-.065, thumb:1.05 }, wristX:-.62,
  },
  wrist_down: {
    index:[.12,.10,.08], middle:[.10,.09,.07], ring:[.15,.13,.10], pinky:[.2,.16,.12], thumb:[.15,.1,.08],
    roots:{ index:.035, middle:.012, ring:-.018, pinky:-.065, thumb:1.05 }, wristX:.62,
  },
  pinch: {
    index:[.62,.85,.62], middle:[.16,.12,.09], ring:[.2,.16,.12], pinky:[.24,.18,.12], thumb:[.58,.68,.42],
    roots:{ index:.10, middle:.012, ring:-.018, pinky:-.065, thumb:.64 }, wristX:0,
  },
  point: {
    index:[0,0,0], middle:[1.3,1.35,1.1], ring:[1.34,1.35,1.12], pinky:[1.4,1.4,1.15], thumb:[.5,.65,.45],
    roots:{ index:.035, middle:.008, ring:-.012, pinky:-.045, thumb:.62 }, wristX:0,
  },
  peace: {
    index:[0,0,0], middle:[0,0,0], ring:[1.34,1.35,1.12], pinky:[1.4,1.4,1.15], thumb:[.5,.65,.45],
    roots:{ index:.18, middle:-.10, ring:-.012, pinky:-.045, thumb:.62 }, wristX:0,
  },
  thumbs_up: {
    index:[1.25,1.3,1.05], middle:[1.3,1.35,1.1], ring:[1.34,1.35,1.12], pinky:[1.4,1.4,1.15], thumb:[0,0,0],
    roots:{ index:.025, middle:.008, ring:-.012, pinky:-.045, thumb:1.05 }, wristX:0,
  },
};
let targetPose = poses.rest;
export function setPose(name) { targetPose = poses[name] || poses.rest; }

let dragging = false, lastX = 0, lastY = 0;
stage.addEventListener('pointerdown', e => {
  if (e.target.closest('button, a, input, select, textarea')) return;
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  stage.setPointerCapture(e.pointerId);
});
stage.addEventListener('pointerup', () => dragging = false);
stage.addEventListener('pointermove', e => {
  if (!dragging) return;
  hand.rotation.y += (e.clientX - lastX) * .008;
  hand.rotation.x += (e.clientY - lastY) * .008;
  lastX = e.clientX; lastY = e.clientY;
});
stage.addEventListener('wheel', e => { camera.position.z = THREE.MathUtils.clamp(camera.position.z + e.deltaY * .008, 7, 16); }, { passive: true });

const clock = new THREE.Clock();
function animate() {
  const dt = Math.min(clock.getDelta(), .05);
  const blend = 1 - Math.exp(-7.5 * dt);
  for (const [name, digit] of Object.entries(digits)) {
    const target = targetPose[name];
    digit.joints.forEach((joint, index) => joint.rotation.x = THREE.MathUtils.lerp(joint.rotation.x, target[index], blend));
    const targetRoot = targetPose.roots?.[name] ?? digit.baseRotationZ;
    digit.root.rotation.z = THREE.MathUtils.lerp(digit.root.rotation.z, targetRoot, blend);
  }
  wristPivot.rotation.x = THREE.MathUtils.lerp(wristPivot.rotation.x, targetPose.wristX || 0, blend);
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

function resize() {
  const { clientWidth, clientHeight } = stage;
  renderer.setSize(clientWidth, clientHeight, false);
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);
resize(); animate();

