"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Grid, Html, OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { clone } from "three/examples/jsm/utils/SkeletonUtils.js";
import { bindRobotRig, applyPoseFrame } from "@/lib/robotModel";
import { buildMockPoseFrame, type RobotPoseFrame } from "@/lib/robotPose";
import { DEFAULT_RIG_MAP } from "@/lib/robotRigMap";
import { createPosePoller } from "@/lib/socketClient";

type RobotSceneProps = {
  poseFrame: RobotPoseFrame | null;
  useMockPose: boolean;
};

function RobotScene({ poseFrame, useMockPose }: RobotSceneProps) {
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

  return <primitive ref={rootRef} object={clonedScene} scale={0.5} position={[0, -2.1, 0]} />;
}

export function RobotViewer() {
  const [poseFrame, setPoseFrame] = useState<RobotPoseFrame | null>(null);
  const [streamConnected, setStreamConnected] = useState(false);
  const [useMockPose, setUseMockPose] = useState(true);

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

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div className="viewer-overlay">
        <p>{streamConnected ? "Bridge connected" : "Mock pose loop"}</p>
      </div>
      <Canvas shadows camera={{ position: [10, 8, 12], fov: 42 }}>
        <color attach="background" args={["#06080f"]} />
        <ambientLight intensity={0.8} />
        <directionalLight position={[8, 12, 6]} intensity={2.2} castShadow />
        <Grid
          args={[30, 30]}
          sectionColor="#1f3555"
          cellColor="#142134"
          fadeDistance={42}
          fadeStrength={1}
          position={[0, -4.6, 0]}
        />
        <axesHelper args={[3]} />
        <Suspense fallback={<Html center>Loading GLB…</Html>}>
          <RobotScene poseFrame={poseFrame} useMockPose={useMockPose} />
        </Suspense>
        <OrbitControls makeDefault />
      </Canvas>
    </div>
  );
}

useGLTF.preload("/models/rigged_robot_walk.glb");
