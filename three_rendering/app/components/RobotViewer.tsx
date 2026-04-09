"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Grid, Html, OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { clone } from "three/examples/jsm/utils/SkeletonUtils.js";
import { BoxQuadrupedDebug } from "./BoxQuadrupedDebug";
import { bindRobotRig, applyPoseFrame } from "@/lib/robotModel";
import { buildMockPoseFrame, type RobotPoseFrame } from "@/lib/robotPose";
import { DEFAULT_RIG_MAP } from "@/lib/robotRigMap";
import { createPosePoller } from "@/lib/socketClient";
import { DEFAULT_BOX_QUADRUPED_CALIBRATION } from "@/lib/boxQuadrupedConfig";

const GLB_RENDER_SCALE = 0.5;
const GLB_RENDER_POSITION: [number, number, number] = [0, 0, 0];
const GLB_BODY_CENTER_Y_REST = 13.8931;
const BOX_BODY_CENTER_Y_REST = DEFAULT_BOX_QUADRUPED_CALIBRATION.bodyCenterOffset[1];
const OVERLAP_BOX_Y_OFFSET =
  GLB_RENDER_POSITION[1] + GLB_BODY_CENTER_Y_REST * GLB_RENDER_SCALE - BOX_BODY_CENTER_Y_REST;

type RobotSceneProps = {
  poseFrame: RobotPoseFrame | null;
  useMockPose: boolean;
  renderOffset?: [number, number, number];
};

function RobotScene({ poseFrame, useMockPose, renderOffset = [0, 0, 0] }: RobotSceneProps) {
  const { scene } = useGLTF("/models/rigged_robot_walk.glb");
  const rootRef = useRef<THREE.Group | null>(null);
  const bindingsRef = useRef<ReturnType<typeof bindRobotRig> | null>(null);

  const clonedScene = useMemo(() => clone(scene), [scene]);

  useEffect(() => {
    if (!rootRef.current) {
      return;
    }
    bindingsRef.current = bindRobotRig(rootRef.current, DEFAULT_RIG_MAP);
    rootRef.current.traverse((object) => {
      const mesh = object as THREE.Mesh;
      if (mesh.isMesh) {
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        materials.forEach((material) => {
          if (material instanceof THREE.MeshStandardMaterial || material instanceof THREE.MeshPhongMaterial) {
            material.color.set("#cfcfcf");
          }
        });
        if (!Array.isArray(mesh.material) && mesh.material instanceof THREE.MeshBasicMaterial) {
          mesh.material.color.set("#cfcfcf");
        }
      }
    });
  }, [clonedScene]);

  useFrame(({ clock }) => {
    const bindings = bindingsRef.current;
    if (!bindings) {
      return;
    }
    const frame = useMockPose ? buildMockPoseFrame(clock.getElapsedTime()) : poseFrame;
    if (!frame) {
      return;
    }
    applyPoseFrame(bindings, frame, DEFAULT_RIG_MAP);
  });

  return (
    <primitive
      ref={rootRef}
      object={clonedScene}
      scale={GLB_RENDER_SCALE}
      position={[
        renderOffset[0] + GLB_RENDER_POSITION[0],
        renderOffset[1] + GLB_RENDER_POSITION[1],
        renderOffset[2] + GLB_RENDER_POSITION[2],
      ]}
    />
  );
}

export function RobotViewer() {
  const [poseFrame, setPoseFrame] = useState<RobotPoseFrame | null>(null);
  const [streamConnected, setStreamConnected] = useState(false);
  const [useMockPose, setUseMockPose] = useState(true);
  const [overlapMode, setOverlapMode] = useState(false);

  useEffect(() => {
    const stop = createPosePoller("http://127.0.0.1:8765/pose", {
      intervalMs: 80,
      onPose: (frame) => {
        setPoseFrame(frame);
        setStreamConnected(true);
        setUseMockPose(false);
      },
      onDisconnect: () => {
        setStreamConnected(false);
        setUseMockPose(true);
      },
    });

    return stop;
  }, []);

  const glbOffset: [number, number, number] = overlapMode ? [0, 0, 0] : [0, 0, -4.5];
  const boxOffset: [number, number, number] = overlapMode
    ? [0, OVERLAP_BOX_Y_OFFSET, 0]
    : [10, OVERLAP_BOX_Y_OFFSET, 0];

  return (
    <div className="viewer-shell" style={{ position: "relative", width: "100%", height: "100%" }}>
      <div className="viewer-overlay">
        <p>{streamConnected ? "Bridge connected" : "Mock pose loop"}</p>
      </div>
      <div className="viewer-controls">
        <button
          type="button"
          className="viewer-toggle"
          onClick={() => setOverlapMode((value) => !value)}
        >
          {overlapMode ? "Overlap" : "Separated"}
        </button>
      </div>
      <Canvas shadows camera={{ position: [10, 8, 12], fov: 42 }}>
        <color attach="background" args={["#000000"]} />
        <ambientLight intensity={0.9} />
        <directionalLight position={[8, 12, 6]} intensity={2.4} castShadow />
        <Grid
          args={[30, 30]}
          sectionColor="#2f2f2f"
          cellColor="#1a1a1a"
          fadeDistance={42}
          fadeStrength={1}
          position={[0, -4.6, 0]}
        />
        <axesHelper args={[3]} />
        <Suspense fallback={<Html center>Loading GLB…</Html>}>
          <RobotScene poseFrame={poseFrame} useMockPose={useMockPose} renderOffset={glbOffset} />
          <group position={boxOffset}>
            <BoxQuadrupedDebug poseFrame={poseFrame} useMockPose={useMockPose} />
          </group>
        </Suspense>
        <OrbitControls makeDefault />
      </Canvas>
    </div>
  );
}

useGLTF.preload("/models/rigged_robot_walk.glb");
