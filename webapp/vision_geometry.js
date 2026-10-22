const FINGERS = [
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
];

const clamp = (value, low = 0, high = 1) => Math.max(low, Math.min(high, value));

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0));
}

function angle(a, b, c) {
  const u = { x: a.x - b.x, y: a.y - b.y, z: (a.z ?? 0) - (b.z ?? 0) };
  const v = { x: c.x - b.x, y: c.y - b.y, z: (c.z ?? 0) - (b.z ?? 0) };
  const denominator = Math.hypot(u.x, u.y, u.z) * Math.hypot(v.x, v.y, v.z);
  if (denominator < 1e-8) return 180;
  return Math.acos(clamp((u.x * v.x + u.y * v.y + u.z * v.z) / denominator, -1, 1)) * 180 / Math.PI;
}

function fingerCurl(points, [mcp, pip, dip, tip]) {
  const proximal = angle(points[0], points[mcp], points[pip]);
  const middle = angle(points[mcp], points[pip], points[dip]);
  const distal = angle(points[pip], points[dip], points[tip]);
  return clamp(1 - (0.45 * proximal + 0.35 * middle + 0.20 * distal) / 180);
}

function selectVisibleArm(hand, pose) {
  if (!pose || pose.length < 17) return null;
  const arms = [
    { side: "left", shoulder: pose[11], elbow: pose[13], wrist: pose[15] },
    { side: "right", shoulder: pose[12], elbow: pose[14], wrist: pose[16] },
  ].filter(({ shoulder, elbow, wrist }) => shoulder && elbow && wrist);
  if (!arms.length) return null;
  arms.sort((a, b) => distance(a.wrist, hand[0]) - distance(b.wrist, hand[0]));
  const arm = arms[0];
  arm.visibility = Math.min(
    arm.elbow.visibility ?? 0,
    arm.wrist.visibility ?? 0,
  );
  arm.shoulderVisibility = arm.shoulder.visibility ?? 0;
  return arm;
}

function armMetrics(hand, pose) {
  const arm = selectVisibleArm(hand, pose);
  if (!arm) {
    return {
      armSide: null, armVisibility: 0, elbowAngleDegrees: null,
      shoulderVisibility: 0, elbowPositionX: null, elbowPositionY: null,
      forearmElevationDegrees: null, signedWristDeflectionDegrees: null,
      wristDeflectionDegrees: null, postureBucket: "unknown",
      forearmOrientation: "unknown",
    };
  }
  const elbowAngle = arm.shoulderVisibility >= 0.35
    ? angle(arm.shoulder, arm.elbow, arm.wrist)
    : null;
  const forearm = { x: arm.wrist.x - arm.elbow.x, y: arm.wrist.y - arm.elbow.y };
  const handAxis = { x: hand[9].x - hand[0].x, y: hand[9].y - hand[0].y };
  const forearmLength = Math.hypot(forearm.x, forearm.y);
  const handLength = Math.hypot(handAxis.x, handAxis.y);
  let signedDeflection = null;
  if (forearmLength > 1e-6 && handLength > 1e-6) {
    const dot = clamp(
      (forearm.x * handAxis.x + forearm.y * handAxis.y) / (forearmLength * handLength),
      -1,
      1,
    );
    const magnitude = Math.acos(dot) * 180 / Math.PI;
    const cross = forearm.x * handAxis.y - forearm.y * handAxis.x;
    signedDeflection = (cross < 0 ? -1 : 1) * magnitude;
  }
  const elevation = Math.atan2(Math.abs(forearm.y), Math.abs(forearm.x)) * 180 / Math.PI;
  return {
    armSide: arm.side,
    armVisibility: Number(arm.visibility.toFixed(4)),
    shoulderVisibility: Number(arm.shoulderVisibility.toFixed(4)),
    elbowPositionX: Number(arm.elbow.x.toFixed(4)),
    elbowPositionY: Number(arm.elbow.y.toFixed(4)),
    elbowAngleDegrees: elbowAngle === null ? null : Number(elbowAngle.toFixed(2)),
    forearmElevationDegrees: Number(elevation.toFixed(2)),
    signedWristDeflectionDegrees: signedDeflection === null ? null : Number(signedDeflection.toFixed(2)),
    wristDeflectionDegrees: signedDeflection === null ? null : Number(Math.abs(signedDeflection).toFixed(2)),
    postureBucket: elbowAngle === null ? "shoulder_out_of_frame" : elbowAngle >= 145 ? "elbow_extended" : elbowAngle <= 115 ? "elbow_bent" : "elbow_mid",
    forearmOrientation: elevation <= 30 ? "horizontal" : elevation >= 60 ? "vertical" : "diagonal",
  };
}

export function landmarkMotion(previous, current) {
  if (!previous || !current || previous.length !== current.length) return 1;
  const palm = Math.max(distance(current[5], current[17]), 1e-4);
  const squared = current.reduce((sum, point, index) => {
    const delta = distance(point, previous[index]) / palm;
    return sum + delta * delta;
  }, 0);
  return Math.sqrt(squared / current.length);
}

const TEMPLATE_SCALES = [0.16, 0.16, 0.16, 0.16, 0.18, 0.30, 0.30, 0.30, 0.34, 0.30];

export function poseFeatureVector(metrics) {
  const curls = metrics?.fingerCurls;
  const gaps = metrics?.fingerGaps;
  if (!Array.isArray(curls) || curls.length !== 4 || !Array.isArray(gaps) || gaps.length !== 3 || !Number.isFinite(metrics?.thumbCurl) || !Number.isFinite(metrics?.thumbIndexDistance) || !Number.isFinite(metrics?.thumbReach)) {
    return null;
  }
  return [...curls, metrics.thumbCurl, ...gaps, metrics.thumbIndexDistance, metrics.thumbReach];
}

function classifyFromTemplates(metrics, templates) {
  const vector = poseFeatureVector(metrics);
  if (!vector) return { label: "unknown", confidence: 0 };
  const candidates = Object.entries(templates).flatMap(([label, template]) => {
    const prototypes = Array.isArray(template?.prototypes) ? template.prototypes : [template];
    return prototypes.map((prototype) => {
      if (!Array.isArray(prototype?.mean) || prototype.mean.length !== vector.length) return null;
      const variance = Array.isArray(prototype.std) ? prototype.std : [];
      const squared = vector.reduce((sum, value, index) => {
        const scale = Math.max(TEMPLATE_SCALES[index], 3 * (variance[index] || 0));
        return sum + ((value - prototype.mean[index]) / scale) ** 2;
      }, 0);
      return { label, posture: prototype.posture || "unspecified", distance: Math.sqrt(squared / vector.length) };
    }).filter(Boolean);
  });
  const bestByLabel = new Map();
  for (const candidate of candidates) {
    const previous = bestByLabel.get(candidate.label);
    if (!previous || candidate.distance < previous.distance) bestByLabel.set(candidate.label, candidate);
  }
  const distances = [...bestByLabel.values()].sort((a, b) => a.distance - b.distance);
  if (!distances.length) return { label: "unknown", confidence: 0 };
  const best = distances[0];
  const margin = distances[1] ? distances[1].distance - best.distance : 1;
  metrics.templateDistance = Number(best.distance.toFixed(4));
  metrics.templateMargin = Number(margin.toFixed(4));
  metrics.templateRunnerUp = distances[1]?.label || null;
  metrics.templatePosture = best.posture;
  if (best.distance > 1.60 || (margin < 0.13 && best.distance > 0.35)) {
    return { label: "unknown", confidence: 0 };
  }
  return {
    label: best.label,
    confidence: clamp(0.97 - best.distance * 0.22 + Math.min(0.06, margin * 0.06)),
  };
}

export function classifyHandPose(handWorld, handImage, poseImage, options = {}) {
  const points = handWorld?.length === 21 ? handWorld : handImage;
  if (!points || points.length !== 21 || !handImage || handImage.length !== 21) {
    return { label: "no_hand", confidence: 0, metrics: {} };
  }

  const curls = FINGERS.map((finger) => fingerCurl(points, finger));
  const [indexCurl, middleCurl, ringCurl, pinkyCurl] = curls;
  const meanCurl = curls.reduce((sum, value) => sum + value, 0) / curls.length;
  // Gesture geometry comes from MediaPipe's metric 3D hand landmarks when
  // available, making it far less sensitive to in-plane forearm rotation and
  // perspective than image-coordinate distances.
  const palmWidth = Math.max(distance(points[5], points[17]), 1e-4);
  const fingerGaps = [
    distance(points[8], points[12]) / palmWidth,
    distance(points[12], points[16]) / palmWidth,
    distance(points[16], points[20]) / palmWidth,
  ];
  const spread = fingerGaps.reduce((sum, value) => sum + value, 0);
  const tipSpan = distance(points[8], points[20]) / palmWidth;
  const thumbIndexDistance = distance(points[4], points[8]) / palmWidth;
  const thumbCurl = fingerCurl(points, [1, 2, 3, 4]);
  const thumbReach = distance(points[0], points[4]) / palmWidth;
  const arm = armMetrics(handImage, poseImage);
  const metrics = {
    meanCurl: Number(meanCurl.toFixed(4)),
    fingerCurls: curls.map((value) => Number(value.toFixed(4))),
    fingerGaps: fingerGaps.map((value) => Number(value.toFixed(4))),
    spread: Number(spread.toFixed(4)),
    tipSpan: Number(tipSpan.toFixed(4)),
    thumbIndexDistance: Number(thumbIndexDistance.toFixed(4)),
    thumbCurl: Number(thumbCurl.toFixed(4)),
    thumbReach: Number(thumbReach.toFixed(4)),
    ...arm,
  };

  if (options.poseTemplates && Object.keys(options.poseTemplates).length) {
    const prediction = classifyFromTemplates(metrics, options.poseTemplates);
    if (arm.armVisibility < 0.45) prediction.confidence = Math.min(prediction.confidence, 0.55);
    return { ...prediction, metrics };
  }

  let label = "unknown";
  let confidence = 0;
  const extended = (curl) => curl <= 0.25;
  const curled = (curl) => curl >= 0.40;
  const reference = options.restReference;
  const restMatch = reference
    ? Math.abs(meanCurl - reference.meanCurl) <= 0.20 &&
      Math.abs(spread - reference.spread) <= 0.42 &&
      Math.abs(tipSpan - reference.tipSpan) <= 0.30
    : meanCurl <= 0.52 && spread <= 1.55 && tipSpan <= 1.28;
  const splayedRelativeToRest = !reference || (
    spread >= reference.spread + 0.32 && tipSpan >= reference.tipSpan + 0.16
  );

  if (
    thumbIndexDistance <= 0.34 && indexCurl <= 0.48 &&
    ![middleCurl, ringCurl, pinkyCurl].every(curled)
  ) {
    label = "pinch";
    confidence = clamp(0.66 + (0.34 - thumbIndexDistance) * 1.15);
  } else if (
    extended(indexCurl) && extended(middleCurl) && curled(ringCurl) && curled(pinkyCurl) &&
    fingerGaps[0] >= 0.34
  ) {
    label = "peace";
    confidence = clamp(0.65 + (fingerGaps[0] - 0.34) * 0.55 + (ringCurl + pinkyCurl - 0.8) * 0.25);
  } else if (extended(indexCurl) && curled(middleCurl) && curled(ringCurl) && curled(pinkyCurl)) {
    label = "point";
    confidence = clamp(0.66 + (0.25 - indexCurl) * 0.45 + (middleCurl + ringCurl + pinkyCurl - 1.2) * 0.24);
  } else if (curls.filter(curled).length >= 4 && thumbCurl <= 0.30 && thumbReach >= 0.90) {
    label = "thumbs_up";
    confidence = clamp(0.65 + (0.30 - thumbCurl) * 0.5 + (thumbReach - 0.90) * 0.18);
  } else if (curls.filter(curled).length >= 4) {
    label = "fist";
    confidence = clamp(0.64 + (meanCurl - 0.40) * 1.25);
  } else if (
    meanCurl <= 0.28 && spread >= 1.48 && tipSpan >= 1.12 &&
    splayedRelativeToRest &&
    fingerGaps.filter((gap) => gap >= 0.30).length >= 2
  ) {
    label = "splayed";
    confidence = clamp(0.64 + (spread - 1.48) * 0.26 + (tipSpan - 1.12) * 0.22);
  } else if (
    restMatch &&
    (arm.wristDeflectionDegrees === null || arm.wristDeflectionDegrees <= 18)
  ) {
    label = "rest";
    if (reference) {
      const error = (
        Math.abs(meanCurl - reference.meanCurl) / 0.20 +
        Math.abs(spread - reference.spread) / 0.42 +
        Math.abs(tipSpan - reference.tipSpan) / 0.30
      ) / 3;
      confidence = clamp(0.92 - error * 0.27);
    } else {
      confidence = clamp(0.70 + (1.55 - spread) * 0.12 + (1.28 - tipSpan) * 0.08);
    }
  }

  // A visible elbow-wrist chain is required; the shoulder is optional because
  // forearm orientation and elbow location do not depend on it.
  if (arm.armVisibility < 0.45) confidence = Math.min(confidence, 0.55);
  return { label, confidence, metrics };
}

