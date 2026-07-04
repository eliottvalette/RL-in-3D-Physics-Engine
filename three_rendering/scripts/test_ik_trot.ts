import assert from "node:assert/strict";
import {
  buildIkTrotLegStates,
  buildIkTrotPoseFrame,
  computeIkTrotElbowCenterBody,
  computeIkTrotFootCenterBody,
} from "../lib/robotPose";
import { DEFAULT_BOX_QUADRUPED_CALIBRATION } from "../lib/boxQuadrupedConfig";

const CYCLE_SECONDS = 1.15;
const EPS = 1e-6;

function assertClose(actual: number, expected: number, tolerance: number) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} within ${tolerance} of ${expected}`,
  );
}

for (const phase of [0, 0.2, 0.5, 0.8]) {
  const timeSeconds = phase * CYCLE_SECONDS;
  const frame = buildIkTrotPoseFrame(timeSeconds);
  assert.deepEqual(frame.root_quaternion, [0, 0, 0, 1]);

  for (const state of buildIkTrotLegStates(timeSeconds)) {
    const shoulder = frame.joints[`${state.legName}_shoulder`];
    const elbow = frame.joints[`${state.legName}_elbow`];
    assert.ok(shoulder >= -Math.PI / 2 && shoulder <= Math.PI / 2);
    assert.ok(elbow >= -Math.PI && elbow <= 0);

    const foot = computeIkTrotFootCenterBody(state.legName, shoulder, elbow);
    assertClose(foot[0], state.footTargetBody[0], EPS);
    assertClose(foot[1], state.footTargetBody[1], 0.02);
    assertClose(foot[2], state.footTargetBody[2], 0.02);

    if (state.legName.startsWith("front_")) {
      const shoulderAnchor = DEFAULT_BOX_QUADRUPED_CALIBRATION.shoulderAnchors[state.legName];
      const elbowCenter = computeIkTrotElbowCenterBody(state.legName, shoulder);
      assert.ok(
        elbowCenter[2] < shoulderAnchor[2],
        `${state.legName} knee should point toward blue/back`,
      );
    }
  }
}

{
  const states = Object.fromEntries(
    buildIkTrotLegStates(0.2 * CYCLE_SECONDS).map((state) => [state.legName, state]),
  );
  assert.equal(states.front_right.inStance, true);
  assert.equal(states.back_left.inStance, true);
  assert.equal(states.front_left.inStance, false);
  assert.equal(states.back_right.inStance, false);
}

console.log("IK trot tests passed");
