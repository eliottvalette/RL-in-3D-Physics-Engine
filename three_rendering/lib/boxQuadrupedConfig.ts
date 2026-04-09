export type Vec3 = [number, number, number];

export type BoxLegName = "front_left" | "front_right" | "back_left" | "back_right";

export type BoxQuadrupedCalibration = {
  rootOffset: Vec3;
  bodyCenterOffset: Vec3;
  bodySize: Vec3;
  upperLegSize: Vec3;
  lowerLegSize: Vec3;
  shoulderAnchors: Record<BoxLegName, Vec3>;
};

export const DEFAULT_BOX_QUADRUPED_CALIBRATION: BoxQuadrupedCalibration = {
  rootOffset: [0, 0, 0],
  bodyCenterOffset: [0, -0.25, 0.15],
  bodySize: [4.8, 1.5, 7.2],
  upperLegSize: [0.9, 2.4, 0.9],
  lowerLegSize: [0.8, 2.8, 0.8],
  shoulderAnchors: {
    front_left: [-1.55, -0.15, 2.2],
    front_right: [1.55, -0.15, 2.2],
    back_left: [-1.55, -0.15, -2.2],
    back_right: [1.55, -0.15, -2.2],
  },
};
