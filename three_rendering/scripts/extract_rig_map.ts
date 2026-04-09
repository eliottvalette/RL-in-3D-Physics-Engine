import fs from "node:fs";

const filePath = process.argv[2];
if (!filePath) {
  throw new Error("Usage: tsx scripts/extract_rig_map.ts <path-to-glb>");
}

const buffer = fs.readFileSync(filePath);
const jsonChunkLength = buffer.readUInt32LE(12);
const gltf = JSON.parse(buffer.toString("utf8", 20, 20 + jsonChunkLength));

const nodeNames = new Set((gltf.nodes ?? []).map((node: { name?: string }) => node.name).filter(Boolean));

const suggestion = {
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

const missing = Object.values(suggestion.joints)
  .map((joint) => joint.boneName)
  .filter((boneName) => !nodeNames.has(boneName));

console.log(JSON.stringify({ suggestion, missing }, null, 2));
