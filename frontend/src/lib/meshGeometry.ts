import * as THREE from "three";

import { MeshPayload, SelectionPoint, StructuralOptimizationPreprocessResponse } from "../types";

export interface DecodedMeshData {
  vertices: Float32Array;
  indices: Uint32Array;
  normals: Float32Array;
}

export interface MeshIntersectionInfo {
  point: [number, number, number];
  normal: [number, number, number];
  triangleIndex: number;
}

export interface PlanarFaceSelection {
  triangleIndices: number[];
  normal: [number, number, number];
  centroid: [number, number, number];
  area: number;
  samplePoints: Array<[number, number, number]>;
  triangles: Array<[[number, number, number], [number, number, number], [number, number, number]]>;
}

function decodeBase64Bytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

export function decodeMeshPayload(mesh: MeshPayload | null): DecodedMeshData | null {
  if (!mesh) {
    return null;
  }

  if (mesh.encoding === "mesh-f32-u32-binary-v1") {
    return {
      vertices: mesh.vertices,
      indices: mesh.indices,
      normals: mesh.normals,
    };
  }

  if (mesh.encoding !== "mesh-f32-u32-base64-v1") {
    return null;
  }

  const expectedVerticesBytes = mesh.vertex_count * 3 * Float32Array.BYTES_PER_ELEMENT;
  const expectedIndicesBytes = mesh.face_count * 3 * Uint32Array.BYTES_PER_ELEMENT;
  const expectedNormalsBytes = mesh.vertex_count * 3 * Float32Array.BYTES_PER_ELEMENT;
  const verticesBytes = decodeBase64Bytes(mesh.vertices_b64);
  const indicesBytes = decodeBase64Bytes(mesh.indices_b64);
  const normalsBytes = decodeBase64Bytes(mesh.normals_b64);

  if (
    verticesBytes.byteLength !== expectedVerticesBytes ||
    indicesBytes.byteLength !== expectedIndicesBytes ||
    normalsBytes.byteLength !== expectedNormalsBytes
  ) {
    return null;
  }

  return {
    vertices: new Float32Array(verticesBytes.buffer, verticesBytes.byteOffset, expectedVerticesBytes / 4),
    indices: new Uint32Array(indicesBytes.buffer, indicesBytes.byteOffset, expectedIndicesBytes / 4),
    normals: new Float32Array(normalsBytes.buffer, normalsBytes.byteOffset, expectedNormalsBytes / 4),
  };
}

export function buildMeshGeometry(mesh: MeshPayload | null): THREE.BufferGeometry | null {
  const decoded = decodeMeshPayload(mesh);
  if (!decoded) {
    return null;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(decoded.vertices, 3));
  geometry.setIndex(new THREE.BufferAttribute(decoded.indices, 1));
  if (decoded.normals.length === decoded.vertices.length) {
    geometry.setAttribute("normal", new THREE.BufferAttribute(decoded.normals, 3));
  } else {
    geometry.computeVertexNormals();
  }
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function triangleVertex(vertices: Float32Array, index: number): THREE.Vector3 {
  const base = index * 3;
  return new THREE.Vector3(vertices[base], vertices[base + 1], vertices[base + 2]);
}

function makeEdgeKey(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function computeTriangleNormal(a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3): THREE.Vector3 {
  return new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(c, a)).normalize();
}

function pointInTriangle3D(
  point: THREE.Vector3,
  a: THREE.Vector3,
  b: THREE.Vector3,
  c: THREE.Vector3,
  normal: THREE.Vector3,
  tolerance: number,
): boolean {
  const ab = new THREE.Vector3().subVectors(b, a);
  const bc = new THREE.Vector3().subVectors(c, b);
  const ca = new THREE.Vector3().subVectors(a, c);
  const ap = new THREE.Vector3().subVectors(point, a);
  const bp = new THREE.Vector3().subVectors(point, b);
  const cp = new THREE.Vector3().subVectors(point, c);

  const c1 = ab.clone().cross(ap).dot(normal);
  const c2 = bc.clone().cross(bp).dot(normal);
  const c3 = ca.clone().cross(cp).dot(normal);
  return c1 >= -tolerance && c2 >= -tolerance && c3 >= -tolerance;
}

export function resolvePlanarFaceSelection(
  mesh: MeshPayload | null,
  triangleIndex: number,
  options?: {
    normalToleranceDegrees?: number;
    planeDistanceTolerance?: number;
  },
): PlanarFaceSelection | null {
  const decoded = decodeMeshPayload(mesh);
  if (!decoded) {
    return null;
  }

  const normalToleranceDegrees = options?.normalToleranceDegrees ?? 8;
  const planeDistanceTolerance = options?.planeDistanceTolerance ?? 1e-4;
  const faceCount = decoded.indices.length / 3;
  if (triangleIndex < 0 || triangleIndex >= faceCount) {
    return null;
  }

  const indices = decoded.indices;
  const vertices = decoded.vertices;
  const seedIa = indices[triangleIndex * 3];
  const seedIb = indices[triangleIndex * 3 + 1];
  const seedIc = indices[triangleIndex * 3 + 2];
  const seedA = triangleVertex(vertices, seedIa);
  const seedB = triangleVertex(vertices, seedIb);
  const seedC = triangleVertex(vertices, seedIc);
  const seedNormal = computeTriangleNormal(seedA, seedB, seedC);
  const planeConstant = -seedNormal.dot(seedA);
  const normalDotMin = Math.cos((normalToleranceDegrees * Math.PI) / 180);

  const edgeToTriangles = new Map<string, number[]>();
  for (let tri = 0; tri < faceCount; tri += 1) {
    const ia = indices[tri * 3];
    const ib = indices[tri * 3 + 1];
    const ic = indices[tri * 3 + 2];
    for (const key of [makeEdgeKey(ia, ib), makeEdgeKey(ib, ic), makeEdgeKey(ic, ia)]) {
      const current = edgeToTriangles.get(key) ?? [];
      current.push(tri);
      edgeToTriangles.set(key, current);
    }
  }

  const accepted = new Set<number>();
  const queue = [triangleIndex];
  while (queue.length > 0) {
    const tri = queue.pop();
    if (tri == null || accepted.has(tri)) {
      continue;
    }
    const ia = indices[tri * 3];
    const ib = indices[tri * 3 + 1];
    const ic = indices[tri * 3 + 2];
    const a = triangleVertex(vertices, ia);
    const b = triangleVertex(vertices, ib);
    const c = triangleVertex(vertices, ic);
    const triNormal = computeTriangleNormal(a, b, c);
    const normalDot = Math.abs(triNormal.dot(seedNormal));
    const maxPlaneDistance = Math.max(
      Math.abs(seedNormal.dot(a) + planeConstant),
      Math.abs(seedNormal.dot(b) + planeConstant),
      Math.abs(seedNormal.dot(c) + planeConstant),
    );
    if (normalDot < normalDotMin || maxPlaneDistance > planeDistanceTolerance) {
      continue;
    }
    accepted.add(tri);
    for (const key of [makeEdgeKey(ia, ib), makeEdgeKey(ib, ic), makeEdgeKey(ic, ia)]) {
      for (const adjacent of edgeToTriangles.get(key) ?? []) {
        if (!accepted.has(adjacent)) {
          queue.push(adjacent);
        }
      }
    }
  }

  if (accepted.size === 0) {
    return null;
  }

  const triangles: PlanarFaceSelection["triangles"] = [];
  const uniquePoints = new Map<string, [number, number, number]>();
  let totalArea = 0;
  const weightedCentroid = new THREE.Vector3();
  for (const tri of [...accepted].sort((a, b) => a - b)) {
    const ia = indices[tri * 3];
    const ib = indices[tri * 3 + 1];
    const ic = indices[tri * 3 + 2];
    const a = triangleVertex(vertices, ia);
    const b = triangleVertex(vertices, ib);
    const c = triangleVertex(vertices, ic);
    const triangleArea = new THREE.Triangle(a, b, c).getArea();
    const triangleCentroid = new THREE.Vector3().add(a).add(b).add(c).multiplyScalar(1 / 3);
    totalArea += triangleArea;
    weightedCentroid.addScaledVector(triangleCentroid, triangleArea);
    triangles.push(
      [a.toArray() as [number, number, number], b.toArray() as [number, number, number], c.toArray() as [number, number, number]]
    );
    for (const point of [a, b, c]) {
      uniquePoints.set(point.toArray().map((value) => value.toFixed(6)).join("|"), point.toArray() as [number, number, number]);
    }
  }

  if (totalArea <= 0) {
    return null;
  }

  weightedCentroid.multiplyScalar(1 / totalArea);

  return {
    triangleIndices: [...accepted].sort((a, b) => a - b),
    normal: seedNormal.toArray() as [number, number, number],
    centroid: weightedCentroid.toArray() as [number, number, number],
    area: totalArea,
    samplePoints: [...uniquePoints.values()],
    triangles,
  };
}

export function resolveSurfaceNodePoints(
  preprocess: StructuralOptimizationPreprocessResponse,
  selection: PlanarFaceSelection,
): SelectionPoint[] {
  const [nx, ny, nz] = preprocess.resolution_xyz;
  const nodeCounts = [nx + 1, ny + 1, nz + 1] as const;
  const axisValues = preprocess.bounds.map(([min, max], axis) => {
    const count = nodeCounts[axis];
    const values: number[] = [];
    for (let i = 0; i < count; i += 1) {
      values.push(min + ((max - min) * i) / Math.max(count - 1, 1));
    }
    return values;
  }) as [[number, number], [number, number], [number, number]] | number[][];

  const planeNormal = new THREE.Vector3(...selection.normal).normalize();
  const planePoint = new THREE.Vector3(...selection.centroid);
  const minSpacing = Math.min(
    nodeCounts[0] > 1 ? axisValues[0][1] - axisValues[0][0] : 1.0,
    nodeCounts[1] > 1 ? axisValues[1][1] - axisValues[1][0] : 1.0,
    nodeCounts[2] > 1 ? axisValues[2][1] - axisValues[2][0] : 1.0,
  );
  const planeTolerance = Math.max(minSpacing * 0.45, 1e-5);
  const baryTolerance = Math.max(minSpacing * 0.1, 1e-5);

  const nodes: SelectionPoint[] = [];
  for (const x of axisValues[0]) {
    for (const y of axisValues[1]) {
      for (const z of axisValues[2]) {
        const point = new THREE.Vector3(x, y, z);
        const planeDistance = Math.abs(planeNormal.dot(new THREE.Vector3().subVectors(point, planePoint)));
        if (planeDistance > planeTolerance) {
          continue;
        }

        let inside = false;
        for (const triangle of selection.triangles) {
          const a = new THREE.Vector3(...triangle[0]);
          const b = new THREE.Vector3(...triangle[1]);
          const c = new THREE.Vector3(...triangle[2]);
          if (pointInTriangle3D(point, a, b, c, planeNormal, baryTolerance)) {
            inside = true;
            break;
          }
        }
        if (inside) {
          nodes.push({ point_xyz: [x, y, z] });
        }
      }
    }
  }

  const deduped = new Map<string, SelectionPoint>();
  for (const node of nodes) {
    deduped.set(node.point_xyz.map((value) => value.toFixed(6)).join("|"), node);
  }
  return [...deduped.values()];
}
