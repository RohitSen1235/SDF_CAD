import { MeshPayload } from "../types";

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function float32ToBase64(values: Float32Array): string {
  return bytesToBase64(new Uint8Array(values.buffer.slice(0)));
}

function uint32ToBase64(values: Uint32Array): string {
  return bytesToBase64(new Uint8Array(values.buffer.slice(0)));
}

function buildMeshPayload(triangleVertices: number[][]): MeshPayload {
  const triCount = triangleVertices.length / 3;
  const vertices = new Float32Array(triCount * 9);
  const indices = new Uint32Array(triCount * 3);
  const normals = new Float32Array(triCount * 9);

  for (let tri = 0; tri < triCount; tri += 1) {
    const v0 = triangleVertices[tri * 3 + 0];
    const v1 = triangleVertices[tri * 3 + 1];
    const v2 = triangleVertices[tri * 3 + 2];
    const base = tri * 9;
    vertices.set(v0, base);
    vertices.set(v1, base + 3);
    vertices.set(v2, base + 6);

    const i0 = tri * 3;
    indices[i0 + 0] = i0 + 0;
    indices[i0 + 1] = i0 + 1;
    indices[i0 + 2] = i0 + 2;

    const ax = v1[0] - v0[0];
    const ay = v1[1] - v0[1];
    const az = v1[2] - v0[2];
    const bx = v2[0] - v0[0];
    const by = v2[1] - v0[1];
    const bz = v2[2] - v0[2];
    const nx = ay * bz - az * by;
    const ny = az * bx - ax * bz;
    const nz = ax * by - ay * bx;
    const len = Math.hypot(nx, ny, nz) || 1.0;
    const nnx = nx / len;
    const nny = ny / len;
    const nnz = nz / len;
    for (let k = 0; k < 3; k += 1) {
      const normalBase = base + k * 3;
      normals[normalBase + 0] = nnx;
      normals[normalBase + 1] = nny;
      normals[normalBase + 2] = nnz;
    }
  }

  return {
    encoding: "mesh-f32-u32-base64-v1",
    vertex_count: triCount * 3,
    face_count: triCount,
    vertices_b64: float32ToBase64(vertices),
    indices_b64: uint32ToBase64(indices),
    normals_b64: float32ToBase64(normals)
  };
}

function parseObj(text: string): number[][] {
  const lines = text.split(/\r?\n/);
  const positions: number[][] = [];
  const triangles: number[][] = [];

  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    if (line.startsWith("v ")) {
      const parts = line.split(/\s+/);
      if (parts.length < 4) {
        continue;
      }
      positions.push([Number(parts[1]), Number(parts[2]), Number(parts[3])]);
      continue;
    }
    if (line.startsWith("f ")) {
      const parts = line.split(/\s+/).slice(1);
      if (parts.length < 3) {
        continue;
      }
      const face: number[] = [];
      for (const token of parts) {
        const [vertexToken] = token.split("/");
        const index = Number(vertexToken);
        if (!Number.isFinite(index) || index === 0) {
          continue;
        }
        const resolved = index > 0 ? index - 1 : positions.length + index;
        if (resolved >= 0 && resolved < positions.length) {
          face.push(resolved);
        }
      }
      if (face.length < 3) {
        continue;
      }
      for (let i = 1; i < face.length - 1; i += 1) {
        triangles.push(positions[face[0]]);
        triangles.push(positions[face[i]]);
        triangles.push(positions[face[i + 1]]);
      }
    }
  }
  return triangles;
}

function parseAsciiStl(text: string): number[][] {
  const vertexRegex = /^\s*vertex\s+([-\deE.+]+)\s+([-\deE.+]+)\s+([-\deE.+]+)/i;
  const triangles: number[][] = [];
  const lines = text.split(/\r?\n/);
  let current: number[][] = [];

  for (const line of lines) {
    const match = line.match(vertexRegex);
    if (!match) {
      continue;
    }
    current.push([Number(match[1]), Number(match[2]), Number(match[3])]);
    if (current.length === 3) {
      triangles.push(current[0], current[1], current[2]);
      current = [];
    }
  }
  return triangles;
}

function parseBinaryStl(buffer: ArrayBuffer): number[][] {
  const view = new DataView(buffer);
  if (buffer.byteLength < 84) {
    return [];
  }
  const triCount = view.getUint32(80, true);
  const expectedSize = 84 + triCount * 50;
  if (expectedSize > buffer.byteLength) {
    return [];
  }
  const triangles: number[][] = [];
  for (let tri = 0; tri < triCount; tri += 1) {
    const base = 84 + tri * 50;
    const v0: number[] = [
      view.getFloat32(base + 12, true),
      view.getFloat32(base + 16, true),
      view.getFloat32(base + 20, true)
    ];
    const v1: number[] = [
      view.getFloat32(base + 24, true),
      view.getFloat32(base + 28, true),
      view.getFloat32(base + 32, true)
    ];
    const v2: number[] = [
      view.getFloat32(base + 36, true),
      view.getFloat32(base + 40, true),
      view.getFloat32(base + 44, true)
    ];
    triangles.push(v0, v1, v2);
  }
  return triangles;
}

function looksLikeAsciiStl(text: string): boolean {
  const trimmed = text.trimStart();
  return trimmed.startsWith("solid") && /facet/i.test(text) && /vertex/i.test(text);
}

export async function parseLocalMeshPreview(file: File): Promise<MeshPayload> {
  const readText = (): Promise<string> => {
    if (typeof (file as File & { text?: () => Promise<string> }).text === "function") {
      return file.text();
    }
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : "");
      reader.onerror = () => reject(new Error("Failed to read uploaded mesh file"));
      reader.readAsText(file);
    });
  };
  const readArrayBuffer = (): Promise<ArrayBuffer> => {
    if (typeof (file as File & { arrayBuffer?: () => Promise<ArrayBuffer> }).arrayBuffer === "function") {
      return file.arrayBuffer();
    }
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (reader.result instanceof ArrayBuffer) {
          resolve(reader.result);
          return;
        }
        reject(new Error("Failed to read uploaded mesh file"));
      };
      reader.onerror = () => reject(new Error("Failed to read uploaded mesh file"));
      reader.readAsArrayBuffer(file);
    });
  };

  const ext = file.name.toLowerCase().split(".").pop();
  let triangles: number[][] = [];
  if (ext === "obj") {
    triangles = parseObj(await readText());
  } else if (ext === "stl") {
    const text = await readText();
    if (looksLikeAsciiStl(text)) {
      triangles = parseAsciiStl(text);
    } else {
      triangles = parseBinaryStl(await readArrayBuffer());
    }
  } else {
    throw new Error("Only .stl and .obj uploads are supported");
  }

  if (triangles.length < 3) {
    throw new Error("Unable to parse uploaded mesh for local preview");
  }
  return buildMeshPayload(triangles);
}
