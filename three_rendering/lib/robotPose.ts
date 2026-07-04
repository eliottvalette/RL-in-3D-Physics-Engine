import {
  DEFAULT_BOX_QUADRUPED_CALIBRATION,
  type BoxLegName,
  type BoxQuadrupedCalibration,
} from "./boxQuadrupedConfig";
import type { RobotJointName } from "./robotRigMap";

export type RobotPoseFrame = {
  root_position: [number, number, number];
  root_quaternion: [number, number, number, number];
  joints: Record<RobotJointName, number>;
};

export type TrotLegState = {
  legName: BoxLegName;
  phase: number;
  inStance: boolean;
  footTargetBody: [number, number, number];
};

export const DEFAULT_POSE_FRAME: RobotPoseFrame = {
  root_position: [0, 0, 0],
  root_quaternion: [0, 0, 0, 1],
  joints: {
    front_left_shoulder: 0,
    front_left_elbow: 0,
    front_right_shoulder: 0,
    front_right_elbow: 0,
    back_left_shoulder: 0,
    back_left_elbow: 0,
    back_right_shoulder: 0,
    back_right_elbow: 0,
  },
};

const LEG_TO_JOINTS: Record<BoxLegName, { shoulder: RobotJointName; elbow: RobotJointName }> = {
  front_left: { shoulder: "front_left_shoulder", elbow: "front_left_elbow" },
  front_right: { shoulder: "front_right_shoulder", elbow: "front_right_elbow" },
  back_left: { shoulder: "back_left_shoulder", elbow: "back_left_elbow" },
  back_right: { shoulder: "back_right_shoulder", elbow: "back_right_elbow" },
};

const LEG_NAMES: BoxLegName[] = ["front_left", "front_right", "back_left", "back_right"];
const DIAGONAL_PHASE_OFFSETS: Record<BoxLegName, number> = {
  front_right: 0,
  back_left: 0,
  front_left: 0.5,
  back_right: 0.5,
};

const SHOULDER_MIN = -Math.PI / 2;
const SHOULDER_MAX = Math.PI / 2;
const ELBOW_MIN = -Math.PI;
const ELBOW_MAX = 0;

const TROT_CYCLE_SECONDS = 1.15;
const DUTY_FACTOR = 0.62;
const STRIDE_LENGTH = 3.4;
const FOOT_LIFT = 1.35;
const FOOT_GROUND_Y = -7.55;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function fract(value: number) {
  return value - Math.floor(value);
}

function smoothstep(value: number) {
  const t = clamp(value, 0, 1);
  return t * t * (3 - 2 * t);
}

function rotateAroundX(vector: [number, number, number], angle: number): [number, number, number] {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [
    vector[0],
    cos * vector[1] - sin * vector[2],
    sin * vector[1] + cos * vector[2],
  ];
}

function lowerFootCenterOffset(
  legName: BoxLegName,
  calibration: BoxQuadrupedCalibration,
): [number, number, number] {
  const lowerCenter = calibration.lowerLegCenterOffsets[legName];
  return [
    lowerCenter[0],
    lowerCenter[1] - calibration.lowerLegSize[1] / 2,
    lowerCenter[2],
  ];
}

function footCenterX(legName: BoxLegName, calibration: BoxQuadrupedCalibration) {
  return (
    calibration.shoulderAnchors[legName][0]
    + calibration.elbowOffsets[legName][0]
    + lowerFootCenterOffset(legName, calibration)[0]
  );
}

export function computeIkTrotFootCenterBody(
  legName: BoxLegName,
  shoulderAngle: number,
  elbowAngle: number,
  calibration: BoxQuadrupedCalibration = DEFAULT_BOX_QUADRUPED_CALIBRATION,
): [number, number, number] {
  const shoulder = calibration.shoulderAnchors[legName];
  const upperAngle =
    calibration.upperLegRestAxisAngle[legName].angle
    + calibration.shoulderAngleSign * shoulderAngle;
  const lowerAngle =
    upperAngle
    + calibration.lowerLegRestAxisAngle[legName].angle
    + calibration.elbowAngleSign * elbowAngle;
  const elbowVector = rotateAroundX(calibration.elbowOffsets[legName], upperAngle);
  const lowerVector = rotateAroundX(lowerFootCenterOffset(legName, calibration), lowerAngle);

  return [
    shoulder[0] + elbowVector[0] + lowerVector[0],
    shoulder[1] + elbowVector[1] + lowerVector[1],
    shoulder[2] + elbowVector[2] + lowerVector[2],
  ];
}

export function computeIkTrotElbowCenterBody(
  legName: BoxLegName,
  shoulderAngle: number,
  calibration: BoxQuadrupedCalibration = DEFAULT_BOX_QUADRUPED_CALIBRATION,
): [number, number, number] {
  const shoulder = calibration.shoulderAnchors[legName];
  const upperAngle =
    calibration.upperLegRestAxisAngle[legName].angle
    + calibration.shoulderAngleSign * shoulderAngle;
  const elbowVector = rotateAroundX(calibration.elbowOffsets[legName], upperAngle);
  return [
    shoulder[0] + elbowVector[0],
    shoulder[1] + elbowVector[1],
    shoulder[2] + elbowVector[2],
  ];
}

function solveLegIk(
  legName: BoxLegName,
  footTargetBody: [number, number, number],
  calibration: BoxQuadrupedCalibration,
) {
  let shoulder = -0.6;
  let elbow = -0.35;
  const h = 0.0001;
  const damping = 0.0005;

  for (let i = 0; i < 36; i += 1) {
    const current = computeIkTrotFootCenterBody(legName, shoulder, elbow, calibration);
    const errorY = current[1] - footTargetBody[1];
    const errorZ = current[2] - footTargetBody[2];
    if (Math.hypot(errorY, errorZ) < 0.01) {
      break;
    }

    const shoulderSample = computeIkTrotFootCenterBody(legName, shoulder + h, elbow, calibration);
    const elbowSample = computeIkTrotFootCenterBody(legName, shoulder, elbow + h, calibration);
    const j00 = (shoulderSample[1] - current[1]) / h;
    const j10 = (shoulderSample[2] - current[2]) / h;
    const j01 = (elbowSample[1] - current[1]) / h;
    const j11 = (elbowSample[2] - current[2]) / h;

    const a00 = j00 * j00 + j10 * j10 + damping;
    const a01 = j00 * j01 + j10 * j11;
    const a11 = j01 * j01 + j11 * j11 + damping;
    const b0 = j00 * errorY + j10 * errorZ;
    const b1 = j01 * errorY + j11 * errorZ;
    const determinant = a00 * a11 - a01 * a01;
    if (Math.abs(determinant) < 1e-9) {
      break;
    }

    const deltaShoulder = clamp((a11 * b0 - a01 * b1) / determinant, -0.2, 0.2);
    const deltaElbow = clamp((-a01 * b0 + a00 * b1) / determinant, -0.2, 0.2);
    shoulder = clamp(shoulder - deltaShoulder, SHOULDER_MIN, SHOULDER_MAX);
    elbow = clamp(elbow - deltaElbow, ELBOW_MIN, ELBOW_MAX);
  }

  return { shoulder, elbow };
}

export function buildIkTrotLegStates(
  timeSeconds: number,
  calibration: BoxQuadrupedCalibration = DEFAULT_BOX_QUADRUPED_CALIBRATION,
): TrotLegState[] {
  const cycle = timeSeconds / TROT_CYCLE_SECONDS;

  return LEG_NAMES.map((legName) => {
    const phase = fract(cycle + DIAGONAL_PHASE_OFFSETS[legName]);
    const shoulder = calibration.shoulderAnchors[legName];
    const inStance = phase < DUTY_FACTOR;

    let zOffset: number;
    let yOffset = 0;
    if (inStance) {
      const stanceT = smoothstep(phase / DUTY_FACTOR);
      zOffset = STRIDE_LENGTH * (0.5 - stanceT);
    } else {
      const swingT = smoothstep((phase - DUTY_FACTOR) / (1 - DUTY_FACTOR));
      zOffset = STRIDE_LENGTH * (swingT - 0.5);
      yOffset = FOOT_LIFT * Math.sin(Math.PI * swingT);
    }

    return {
      legName,
      phase,
      inStance,
      footTargetBody: [
        footCenterX(legName, calibration),
        FOOT_GROUND_Y + yOffset,
        shoulder[2] + zOffset,
      ],
    };
  });
}

export function buildIkTrotPoseFrame(
  timeSeconds: number,
  calibration: BoxQuadrupedCalibration = DEFAULT_BOX_QUADRUPED_CALIBRATION,
): RobotPoseFrame {
  const joints = { ...DEFAULT_POSE_FRAME.joints };

  for (const state of buildIkTrotLegStates(timeSeconds, calibration)) {
    const solved = solveLegIk(state.legName, state.footTargetBody, calibration);
    const jointNames = LEG_TO_JOINTS[state.legName];
    joints[jointNames.shoulder] = solved.shoulder;
    joints[jointNames.elbow] = solved.elbow;
  }

  return {
    root_position: [0, 0, 0],
    root_quaternion: [0, 0, 0, 1],
    joints,
  };
}

export function buildMockPoseFrame(timeSeconds: number): RobotPoseFrame {
  return buildIkTrotPoseFrame(timeSeconds);
}
