import assert from "node:assert/strict";
import { classifyHandPose, landmarkMotion, poseFeatureVector } from "../webapp/vision_geometry.js";

function openHand() {
  const points = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.8, z: 0 }));
  points[0] = { x: .5, y: .88, z: 0 };
  const knuckles = [[.38,.72], [.46,.70], [.54,.70], [.62,.72]];
  [[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]].forEach((finger, columnIndex) => {
    const [x, y] = knuckles[columnIndex];
    const dx = x - points[0].x;
    const dy = y - points[0].y;
    finger.forEach((index, joint) => points[index] = { x: points[0].x + dx * (joint + 1), y: points[0].y + dy * (joint + 1), z: 0 });
  });
  points[1] = { x: .41, y: .75, z: 0 }; points[2] = { x: .34, y: .67, z: 0 };
  points[3] = { x: .26, y: .59, z: 0 }; points[4] = { x: .17, y: .50, z: 0 };
  return points;
}

const open = openHand();
const pose = Array.from({ length: 33 }, () => ({ x: 10, y: 10, z: 0, visibility: 0 }));
pose[11] = { x: .58, y: 1.24, z: 0, visibility: 1 };
pose[13] = { x: .54, y: 1.06, z: 0, visibility: 1 };
pose[15] = { ...open[0], visibility: 1 };
const splayed = classifyHandPose(open, open, pose);
assert.equal(splayed.label, "splayed");
assert.ok(splayed.confidence >= .62);
assert.ok(Math.abs(splayed.metrics.wristDeflectionDegrees) < 1);

const grouped = structuredClone(open);
const columns = [.44, .48, .52, .56];
[[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]].forEach((finger, i) => {
  finger.forEach((index, joint) => grouped[index] = { x: columns[i], y: .69 - joint * .13, z: 0 });
});
const rest = classifyHandPose(grouped, grouped, pose);
assert.equal(rest.label, "rest");
assert.ok(rest.confidence >= .62);
const personalRest = classifyHandPose(grouped, grouped, pose, {
  restReference: {
    meanCurl: rest.metrics.meanCurl,
    spread: rest.metrics.spread,
    tipSpan: rest.metrics.tipSpan,
  },
});
assert.equal(personalRest.label, "rest");
assert.ok(personalRest.confidence >= .85);
const poseWithoutShoulder = structuredClone(pose);
poseWithoutShoulder[11].visibility = 0;
const croppedRest = classifyHandPose(grouped, grouped, poseWithoutShoulder, {
  restReference: personalRest.metrics,
});
assert.equal(croppedRest.label, "rest");
assert.ok(croppedRest.confidence >= .62);
assert.equal(croppedRest.metrics.postureBucket, "shoulder_out_of_frame");

const bent = grouped.map((point) => {
  const theta = Math.PI / 4;
  const dx = point.x - grouped[0].x;
  const dy = point.y - grouped[0].y;
  return {
    ...point,
    x: grouped[0].x + dx * Math.cos(theta) - dy * Math.sin(theta),
    y: grouped[0].y + dx * Math.sin(theta) + dy * Math.cos(theta),
  };
});
const uncalibrated = classifyHandPose(bent, bent, pose);
assert.equal(uncalibrated.label, "unknown");

const templates = {
  rest: { prototypes: [
    { posture: "vertical", mean: poseFeatureVector(rest.metrics), std: Array(10).fill(.01) },
    { posture: "horizontal", mean: poseFeatureVector(uncalibrated.metrics), std: Array(10).fill(.01) },
  ] },
  splayed: { prototypes: [
    { posture: "vertical", mean: poseFeatureVector(splayed.metrics), std: Array(10).fill(.01) },
    { posture: "horizontal", mean: poseFeatureVector(splayed.metrics), std: Array(10).fill(.01) },
  ] },
};
assert.equal(poseFeatureVector(rest.metrics).length, 10);
poseFeatureVector(rest.metrics).forEach((value, index) => {
  assert.ok(Math.abs(value - poseFeatureVector(uncalibrated.metrics)[index]) < 1e-8);
});
assert.equal(classifyHandPose(grouped, grouped, pose, { poseTemplates: templates }).label, "rest");
assert.equal(classifyHandPose(open, open, pose, { poseTemplates: templates }).label, "splayed");
assert.equal(classifyHandPose(null, null, null).label, "no_hand");
assert.equal(landmarkMotion(open, structuredClone(open)), 0);

const moved = structuredClone(open);
moved[8].x += .1;
assert.ok(landmarkMotion(open, moved) > 0);
console.log("vision geometry tests passed");

