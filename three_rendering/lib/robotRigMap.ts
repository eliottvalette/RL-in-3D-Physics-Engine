export type RobotJointName =
  | "front_left_shoulder"
  | "front_left_elbow"
  | "front_right_shoulder"
  | "front_right_elbow"
  | "back_left_shoulder"
  | "back_left_elbow"
  | "back_right_shoulder"
  | "back_right_elbow";

export type BoneAxis = "x" | "y" | "z";

export type JointRigMapping = {
  boneName: string;
  axis: BoneAxis;
  sign: 1 | -1;
  restAngle: number;
};

export type RobotRigMap = {
  armatureName: string;
  rootBoneName: string;
  joints: Record<RobotJointName, JointRigMapping>;
};

export const DEFAULT_RIG_MAP: RobotRigMap = {
  armatureName: "Armature",
  rootBoneName: "Bone",
  joints: {
    front_left_shoulder: { boneName: "Bone_L.001", axis: "x", sign: 1, restAngle: 0 },
    front_left_elbow: { boneName: "Bone_L.003", axis: "x", sign: 1, restAngle: 0 },
    front_right_shoulder: { boneName: "Bone_R.001", axis: "x", sign: 1, restAngle: 0 },
    front_right_elbow: { boneName: "Bone_R.003", axis: "x", sign: 1, restAngle: 0 },
    back_left_shoulder: { boneName: "Bone_R.006", axis: "x", sign: 1, restAngle: 0 },
    back_left_elbow: { boneName: "Bone_R.008", axis: "x", sign: 1, restAngle: 0 },
    back_right_shoulder: { boneName: "Bone_L.006", axis: "x", sign: 1, restAngle: 0 },
    back_right_elbow: { boneName: "Bone_L.008", axis: "x", sign: 1, restAngle: 0 },
  },
};
