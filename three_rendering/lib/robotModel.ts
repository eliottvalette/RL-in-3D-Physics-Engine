import * as THREE from "three";
import type { BoneAxis, RobotJointName, RobotRigMap } from "./robotRigMap";
import type { RobotPoseFrame } from "./robotPose";

type BoundJoint = {
  node: THREE.Object3D;
  restQuaternion: THREE.Quaternion;
};

export type BoundRobotRig = {
  modelRoot: THREE.Object3D;
  joints: Record<RobotJointName, BoundJoint>;
};

function axisVector(axis: BoneAxis) {
  switch (axis) {
    case "x":
      return new THREE.Vector3(1, 0, 0);
    case "y":
      return new THREE.Vector3(0, 1, 0);
    case "z":
      return new THREE.Vector3(0, 0, 1);
  }
}

function normalizeNodeName(name: string) {
  return name.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
}

function findNode(root: THREE.Object3D, name: string) {
  let found: THREE.Object3D | null = null;
  const target = normalizeNodeName(name);
  root.traverse((object) => {
    if (object.name === name || normalizeNodeName(object.name) === target) {
      found = object;
    }
  });
  return found;
}

export function bindRobotRig(root: THREE.Object3D, rigMap: RobotRigMap): BoundRobotRig {
  const joints = {} as Record<RobotJointName, BoundJoint>;

  for (const [jointName, mapping] of Object.entries(rigMap.joints) as Array<
    [RobotJointName, RobotRigMap["joints"][RobotJointName]]
  >) {
    const foundNode = findNode(root, mapping.boneName);
    if (!foundNode) {
      const availableNames: string[] = [];
      root.traverse((object) => {
        if (object.name) {
          availableNames.push(`${object.name} <${object.type}>`);
        }
      });
      throw new Error(
        `Missing node "${mapping.boneName}" for joint "${jointName}". Available nodes: ${availableNames.join(", ")}`,
      );
    }
    const node = foundNode as THREE.Object3D;
    joints[jointName] = {
      node,
      restQuaternion: node.quaternion.clone(),
    };
  }

  return {
    modelRoot: root,
    joints,
  };
}

export function applyPoseFrame(bound: BoundRobotRig, poseFrame: RobotPoseFrame, rigMap: RobotRigMap) {
  bound.modelRoot.position.set(...poseFrame.root_position);
  bound.modelRoot.quaternion.set(...poseFrame.root_quaternion);

  for (const [jointName, mapping] of Object.entries(rigMap.joints) as Array<
    [RobotJointName, RobotRigMap["joints"][RobotJointName]]
  >) {
    const boundJoint = bound.joints[jointName];
    const delta = new THREE.Quaternion().setFromAxisAngle(
      axisVector(mapping.axis),
      mapping.sign * poseFrame.joints[jointName] + mapping.restAngle,
    );
    boundJoint.node.quaternion.copy(boundJoint.restQuaternion).multiply(delta);
  }
}
