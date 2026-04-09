export type Vec3 = [number, number, number];
export type AxisAngle = { axis: Vec3; angle: number };

export type BoxLegName = "front_left" | "front_right" | "back_left" | "back_right";

export type BoxQuadrupedCalibration = {
  modelScale: number;
  rootOffset: Vec3;
  bodyCenterOffset: Vec3;
  bodySize: Vec3;
  upperLegSize: Vec3;
  lowerLegSize: Vec3;
  shoulderAngleSign: 1 | -1;
  elbowAngleSign: 1 | -1;
  shoulderAnchors: Record<BoxLegName, Vec3>;
  upperLegCenterOffsets: Record<BoxLegName, Vec3>;
  elbowOffsets: Record<BoxLegName, Vec3>;
  lowerLegCenterOffsets: Record<BoxLegName, Vec3>;
  upperLegRestAxisAngle: Record<BoxLegName, AxisAngle>;
  lowerLegRestAxisAngle: Record<BoxLegName, AxisAngle>;
};

type LegSigns = {
  side: -1 | 1;
  depth: -1 | 1;
};

const LEG_SIGNS: Record<BoxLegName, LegSigns> = {
  front_left: { side: -1, depth: 1 },
  front_right: { side: 1, depth: 1 },
  back_left: { side: -1, depth: -1 },
  back_right: { side: 1, depth: -1 },
};

const MODEL_SCALE = 1.0;
const ROOT_OFFSET: Vec3 = [0, 0, 0];
const BODY_CENTER_OFFSET: Vec3 = [0, -0.05, 0.15];

const BODY_SIZE: Vec3 = [5.7461, 1.938, 12.0739];
const UPPER_LEG_SIZE: Vec3 = [1.3, 3.9, 1.6753];
const LOWER_LEG_SIZE: Vec3 = [0.752, 4.6, 1.1237];

const SHOULDER_ANGLE_SIGN = -1;
const ELBOW_ANGLE_SIGN = -1;

const FRONT_SHOULDER_X = 2.77;
const FRONT_SHOULDER_Y = 0.55;
const FRONT_SHOULDER_Z = 4.95;

const BACK_SHOULDER_X = 2.77;
const BACK_SHOULDER_Y = 0.55;
const BACK_SHOULDER_Z = 5.3;

const UPPER_CENTER_X = 0.7572;
const UPPER_CENTER_Y = -2.4212;

const ELBOW_X = 0.7572;
const ELBOW_Y = -4.2;
const ELBOW_Z = -0.15;

const FRONT_LOWER_CENTER_X = 1.0332;
const FRONT_LOWER_CENTER_Y = -2.5;
const FRONT_LOWER_CENTER_Z = -0.2;

const BACK_LOWER_CENTER_X = 1.0332;
const BACK_LOWER_CENTER_Y = -2.53735;
const BACK_LOWER_CENTER_Z = 0;

const FRONT_UPPER_REST_AXIS: Vec3 = [1, 0, 0];
const BACK_UPPER_REST_AXIS: Vec3 = [1, 0, 0];
const FRONT_LOWER_REST_AXIS: Vec3 = [1, 0, 0];
const BACK_LOWER_REST_AXIS: Vec3 = [1, 0, 0];


const FRONT_UPPER_REST_ANGLE = -0.15;
const FRONT_LOWER_REST_ANGLE = -0.63;

const BACK_UPPER_REST_ANGLE = -0.15;
const BACK_LOWER_REST_ANGLE = -0.63;

function buildPerLegVec3(
  mapper: (legName: BoxLegName, signs: LegSigns) => Vec3,
): Record<BoxLegName, Vec3> {
  return {
    front_left: mapper("front_left", LEG_SIGNS.front_left),
    front_right: mapper("front_right", LEG_SIGNS.front_right),
    back_left: mapper("back_left", LEG_SIGNS.back_left),
    back_right: mapper("back_right", LEG_SIGNS.back_right),
  };
}

function buildPerLegAxisAngle(
  mapper: (legName: BoxLegName, signs: LegSigns) => AxisAngle,
): Record<BoxLegName, AxisAngle> {
  return {
    front_left: mapper("front_left", LEG_SIGNS.front_left),
    front_right: mapper("front_right", LEG_SIGNS.front_right),
    back_left: mapper("back_left", LEG_SIGNS.back_left),
    back_right: mapper("back_right", LEG_SIGNS.back_right),
  };
}

function sideDepthVec(xAbs: number, y: number, zAbs: number): Record<BoxLegName, Vec3> {
  return buildPerLegVec3((_legName, signs) => [signs.side * xAbs, y, signs.depth * zAbs]);
}

function sideOnlyVec(xAbs: number, y: number, z: number): Record<BoxLegName, Vec3> {
  return buildPerLegVec3((_legName, signs) => [signs.side * xAbs, y, z]);
}

function frontBackSideVec(
  front: { xAbs: number; y: number; z: number },
  back: { xAbs: number; y: number; z: number },
): Record<BoxLegName, Vec3> {
  return buildPerLegVec3((legName, signs) => {
    const source = legName.startsWith("front_") ? front : back;
    return [signs.side * source.xAbs, source.y, signs.depth * source.z];
  });
}

function frontBackSideOnlyVec(
  front: { xAbs: number; y: number; z: number },
  back: { xAbs: number; y: number; z: number },
): Record<BoxLegName, Vec3> {
  return buildPerLegVec3((legName, signs) => {
    const source = legName.startsWith("front_") ? front : back;
    return [signs.side * source.xAbs, source.y, source.z];
  });
}

export const DEFAULT_BOX_QUADRUPED_CALIBRATION: BoxQuadrupedCalibration = {
  modelScale: MODEL_SCALE,
  rootOffset: ROOT_OFFSET,
  bodyCenterOffset: BODY_CENTER_OFFSET,
  bodySize: BODY_SIZE,
  upperLegSize: UPPER_LEG_SIZE,
  lowerLegSize: LOWER_LEG_SIZE,
  shoulderAngleSign: SHOULDER_ANGLE_SIGN,
  elbowAngleSign: ELBOW_ANGLE_SIGN,
  shoulderAnchors: frontBackSideVec(
    { xAbs: FRONT_SHOULDER_X, y: FRONT_SHOULDER_Y, z: FRONT_SHOULDER_Z },
    { xAbs: BACK_SHOULDER_X, y: BACK_SHOULDER_Y, z: BACK_SHOULDER_Z },
  ),
  upperLegCenterOffsets: sideOnlyVec(UPPER_CENTER_X, UPPER_CENTER_Y, 0),
  elbowOffsets: sideOnlyVec(ELBOW_X, ELBOW_Y, ELBOW_Z),
  lowerLegCenterOffsets: frontBackSideOnlyVec(
    { xAbs: FRONT_LOWER_CENTER_X, y: FRONT_LOWER_CENTER_Y, z: FRONT_LOWER_CENTER_Z },
    { xAbs: BACK_LOWER_CENTER_X, y: BACK_LOWER_CENTER_Y, z: BACK_LOWER_CENTER_Z },
  ),
  upperLegRestAxisAngle: buildPerLegAxisAngle((legName) => ({
    axis: legName.startsWith("front_") ? FRONT_UPPER_REST_AXIS : BACK_UPPER_REST_AXIS,
    angle: legName.startsWith("front_") ? FRONT_UPPER_REST_ANGLE : BACK_UPPER_REST_ANGLE,
  })),
  lowerLegRestAxisAngle: buildPerLegAxisAngle((legName) => ({
    axis: legName.startsWith("front_") ? FRONT_LOWER_REST_AXIS : BACK_LOWER_REST_AXIS,
    angle: legName.startsWith("front_") ? FRONT_LOWER_REST_ANGLE : BACK_LOWER_REST_ANGLE,
  })),
};
