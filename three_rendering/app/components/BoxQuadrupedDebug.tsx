"use client";

import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import {
  DEFAULT_BOX_QUADRUPED_CALIBRATION,
  type BoxLegName,
  type BoxQuadrupedCalibration,
} from "@/lib/boxQuadrupedConfig";
import { buildMockPoseFrame, type RobotPoseFrame } from "@/lib/robotPose";

type Props = {
  poseFrame: RobotPoseFrame | null;
  useMockPose: boolean;
  calibration?: BoxQuadrupedCalibration;
};

const LEG_TO_JOINTS: Record<
  BoxLegName,
  { shoulder: keyof RobotPoseFrame["joints"]; elbow: keyof RobotPoseFrame["joints"] }
> = {
  front_left: { shoulder: "front_left_shoulder", elbow: "front_left_elbow" },
  front_right: { shoulder: "front_right_shoulder", elbow: "front_right_elbow" },
  back_left: { shoulder: "back_left_shoulder", elbow: "back_left_elbow" },
  back_right: { shoulder: "back_right_shoulder", elbow: "back_right_elbow" },
};

function DebugLeg({
  legName,
  anchor,
  upperLegSize,
  lowerLegSize,
  upperLegCenterOffset,
  elbowOffset,
  lowerLegCenterOffset,
  upperLegRestAxisAngle,
  lowerLegRestAxisAngle,
  groupsRef,
}: {
  legName: BoxLegName;
  anchor: [number, number, number];
  upperLegSize: [number, number, number];
  lowerLegSize: [number, number, number];
  upperLegCenterOffset: [number, number, number];
  elbowOffset: [number, number, number];
  lowerLegCenterOffset: [number, number, number];
  upperLegRestAxisAngle: { axis: [number, number, number]; angle: number };
  lowerLegRestAxisAngle: { axis: [number, number, number]; angle: number };
  groupsRef: React.MutableRefObject<Record<string, THREE.Group | null>>;
}) {
  const upperLegQuaternion = useMemo(
    () =>
      new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(...upperLegRestAxisAngle.axis),
        upperLegRestAxisAngle.angle,
      ),
    [upperLegRestAxisAngle],
  );
  const lowerLegQuaternion = useMemo(
    () =>
      new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(...lowerLegRestAxisAngle.axis),
        lowerLegRestAxisAngle.angle,
      ),
    [lowerLegRestAxisAngle],
  );

  return (
    <group position={anchor}>
      <group
        ref={(node) => {
          groupsRef.current[LEG_TO_JOINTS[legName].shoulder] = node;
        }}
      >
        <group quaternion={upperLegQuaternion}>
          <mesh position={upperLegCenterOffset}>
            <boxGeometry args={upperLegSize} />
            <meshBasicMaterial color="#ffffff" wireframe transparent opacity={0.65} />
          </mesh>
          <group
            position={elbowOffset}
            ref={(node) => {
              groupsRef.current[LEG_TO_JOINTS[legName].elbow] = node;
            }}
          >
            <group quaternion={lowerLegQuaternion}>
              <mesh position={lowerLegCenterOffset}>
                <boxGeometry args={lowerLegSize} />
                <meshBasicMaterial color="#d9d9d9" wireframe transparent opacity={0.65} />
              </mesh>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}

export function BoxQuadrupedDebug({
  poseFrame,
  useMockPose,
  calibration = DEFAULT_BOX_QUADRUPED_CALIBRATION,
}: Props) {
  const rootRef = useRef<THREE.Group | null>(null);
  const bodyAssemblyRef = useRef<THREE.Group | null>(null);
  const jointGroupsRef = useRef<Record<string, THREE.Group | null>>({});

  const legs = useMemo(
    () =>
      Object.entries(calibration.shoulderAnchors) as Array<
        [BoxLegName, [number, number, number]]
      >,
    [calibration],
  );

  useFrame(({ clock }) => {
    const frame = useMockPose ? buildMockPoseFrame(clock.getElapsedTime()) : poseFrame;
    if (!frame || !rootRef.current || !bodyAssemblyRef.current) {
      return;
    }

    rootRef.current.position.set(
      frame.root_position[0] + calibration.rootOffset[0],
      frame.root_position[1] + calibration.rootOffset[1],
      frame.root_position[2] + calibration.rootOffset[2],
    );
    rootRef.current.quaternion.set(...frame.root_quaternion);
    bodyAssemblyRef.current.position.set(...calibration.bodyCenterOffset);

    for (const joints of Object.values(LEG_TO_JOINTS)) {
      const shoulderGroup = jointGroupsRef.current[joints.shoulder];
      const elbowGroup = jointGroupsRef.current[joints.elbow];
      if (shoulderGroup) {
        shoulderGroup.rotation.x = calibration.shoulderAngleSign * frame.joints[joints.shoulder];
      }
      if (elbowGroup) {
        elbowGroup.rotation.x = calibration.elbowAngleSign * frame.joints[joints.elbow];
      }
    }
  });

  return (
    <group ref={rootRef}>
      <group ref={bodyAssemblyRef} scale={calibration.modelScale}>
        <mesh>
          <boxGeometry args={calibration.bodySize} />
          <meshBasicMaterial color="#f5f5f5" wireframe transparent opacity={0.75} />
        </mesh>
        {legs.map(([legName, anchor]) => (
          <DebugLeg
            key={legName}
            legName={legName}
            anchor={anchor}
            upperLegSize={calibration.upperLegSize}
            lowerLegSize={calibration.lowerLegSize}
            upperLegCenterOffset={calibration.upperLegCenterOffsets[legName]}
            elbowOffset={calibration.elbowOffsets[legName]}
            lowerLegCenterOffset={calibration.lowerLegCenterOffsets[legName]}
            upperLegRestAxisAngle={calibration.upperLegRestAxisAngle[legName]}
            lowerLegRestAxisAngle={calibration.lowerLegRestAxisAngle[legName]}
            groupsRef={jointGroupsRef}
          />
        ))}
      </group>
    </group>
  );
}
