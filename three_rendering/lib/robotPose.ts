import type { RobotJointName } from "./robotRigMap";

export type RobotPoseFrame = {
  root_position: [number, number, number];
  root_quaternion: [number, number, number, number];
  joints: Record<RobotJointName, number>;
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

export function buildMockPoseFrame(timeSeconds: number): RobotPoseFrame {
  const shoulder = 0.35 * Math.sin(timeSeconds * 2.2);
  const elbow = 0.55 * Math.sin(timeSeconds * 2.2 + Math.PI / 2);

  return {
    root_position: [0, 0, 0],
    root_quaternion: [0, 0, 0, 1],
    joints: {
      front_left_shoulder: shoulder,
      front_left_elbow: elbow,
      front_right_shoulder: -shoulder,
      front_right_elbow: -elbow,
      back_left_shoulder: -shoulder,
      back_left_elbow: -elbow,
      back_right_shoulder: shoulder,
      back_right_elbow: elbow,
    },
  };
}
