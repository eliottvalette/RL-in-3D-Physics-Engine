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

export const DEFAULT_BOX_QUADRUPED_CALIBRATION: BoxQuadrupedCalibration = {
  modelScale: 1.0,
  rootOffset: [0, 0, 0],
  bodyCenterOffset: [0, -0.05, 0.15],
  bodySize: [5.7461, 1.938, 12.0739],
  upperLegSize: [1.5144, 4.8424, 1.6753],
  lowerLegSize: [0.752, 5.0747, 1.1237],
  shoulderAngleSign: -1,
  elbowAngleSign: -1,
  shoulderAnchors: {
    front_left: [-2.8731, -0.1, 5.15],
    front_right: [2.8731, -0.1, 5.15],
    back_left: [-2.8731, -0.1, -5.15],
    back_right: [2.8731, -0.1, -5.15],
  },
  upperLegCenterOffsets: {
    front_left: [-0.7572, -2.4212, 0],
    front_right: [0.7572, -2.4212, 0],
    back_left: [-0.7572, -2.4212, 0],
    back_right: [0.7572, -2.4212, 0],
  },
  elbowOffsets: {
    front_left: [-0.7572, -4.55, 0.18],
    front_right: [0.7572, -4.55, 0.18],
    back_left: [-0.7572, -4.55, 0.18],
    back_right: [0.7572, -4.55, 0.18],
  },
  lowerLegCenterOffsets: {
    front_left: [-1.1332, -2.53735, 0],
    front_right: [1.1332, -2.53735, 0],
    back_left: [-1.1332, -2.53735, 0],
    back_right: [1.1332, -2.53735, 0],
  },
  upperLegRestAxisAngle: {
    front_left: { axis: [1, 0, 0], angle: -0.16202 },
    front_right: { axis: [1, 0, 0], angle: -0.16202 },
    back_left: { axis: [1, 0, 0], angle: -0.15374 },
    back_right: { axis: [1, 0, 0], angle: -0.15374 },
  },
  lowerLegRestAxisAngle: {
    front_left: { axis: [-0.999482, 0, 0.032184], angle: 0.758704 },
    front_right: { axis: [-0.999433, 0, -0.033673], angle: 0.761789 },
    back_left: { axis: [-0.999595, 0, 0.028451], angle: 0.776615 },
    back_right: { axis: [-0.999008, 0, -0.044522], angle: 0.776909 },
  },
};
