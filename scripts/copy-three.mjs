import { copyFile, cp, mkdir } from 'node:fs/promises';

await mkdir('webapp/vendor', { recursive: true });
await copyFile('node_modules/three/build/three.module.js', 'webapp/vendor/three.module.js');
await copyFile('node_modules/three/build/three.core.js', 'webapp/vendor/three.core.js');
await mkdir('webapp/vendor/mediapipe', { recursive: true });
await copyFile(
  'node_modules/@mediapipe/tasks-vision/vision_bundle.mjs',
  'webapp/vendor/mediapipe/vision_bundle.mjs',
);
await cp(
  'node_modules/@mediapipe/tasks-vision/wasm',
  'webapp/vendor/mediapipe/wasm',
  { recursive: true, force: true },
);
console.log('Copied Three.js and MediaPipe browser assets into webapp/vendor');
