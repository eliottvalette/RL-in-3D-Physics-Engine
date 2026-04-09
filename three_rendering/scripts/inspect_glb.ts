import fs from "node:fs";
import path from "node:path";

function inspectGlb(filePath: string) {
  const buffer = fs.readFileSync(filePath);
  const magic = buffer.readUInt32LE(0);
  if (magic !== 0x46546c67) {
    throw new Error(`${filePath} is not a valid GLB file`);
  }

  const jsonChunkLength = buffer.readUInt32LE(12);
  const jsonChunkType = buffer.readUInt32LE(16);
  if (jsonChunkType !== 0x4e4f534a) {
    throw new Error(`${filePath} does not start with a JSON chunk`);
  }

  const gltf = JSON.parse(buffer.toString("utf8", 20, 20 + jsonChunkLength));

  console.log(`file: ${path.basename(filePath)}`);
  console.log(`nodes: ${gltf.nodes?.length ?? 0}`);
  console.log(`meshes: ${gltf.meshes?.length ?? 0}`);
  console.log(`skins: ${gltf.skins?.length ?? 0}`);
  console.log(`animations: ${gltf.animations?.length ?? 0}`);

  if (gltf.skins?.[0]?.joints) {
    console.log("skin joints:");
    for (const jointIndex of gltf.skins[0].joints) {
      console.log(`  - ${jointIndex}: ${gltf.nodes[jointIndex]?.name ?? "unnamed"}`);
    }
  }

  if (gltf.animations?.[0]?.channels) {
    console.log("animation channels:");
    for (const channel of gltf.animations[0].channels) {
      const nodeIndex = channel.target.node;
      const nodeName = gltf.nodes[nodeIndex]?.name ?? `Node ${nodeIndex}`;
      console.log(`  - ${nodeName} :: ${channel.target.path}`);
    }
  }
}

const filePath = process.argv[2];
if (!filePath) {
  throw new Error("Usage: tsx scripts/inspect_glb.ts <path-to-glb>");
}

inspectGlb(filePath);
