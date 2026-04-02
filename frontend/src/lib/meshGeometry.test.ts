import { describe, expect, it } from "vitest";

import { resolvePlanarFaceSelection, resolveSurfaceNodePoints } from "./meshGeometry";
import { MeshPayloadBinary, StructuralOptimizationPreprocessResponse } from "../types";

const cubeMesh: MeshPayloadBinary = {
  encoding: "mesh-f32-u32-binary-v1",
  vertex_count: 8,
  face_count: 12,
  vertices: new Float32Array([
    0, 0, 0,
    1, 0, 0,
    1, 1, 0,
    0, 1, 0,
    0, 0, 1,
    1, 0, 1,
    1, 1, 1,
    0, 1, 1,
  ]),
  indices: new Uint32Array([
    0, 1, 2, 0, 2, 3,
    4, 6, 5, 4, 7, 6,
    0, 4, 5, 0, 5, 1,
    1, 5, 6, 1, 6, 2,
    2, 6, 7, 2, 7, 3,
    3, 7, 4, 3, 4, 0,
  ]),
  normals: new Float32Array(8 * 3),
};

const preprocess: StructuralOptimizationPreprocessResponse = {
  design_mesh: cubeMesh,
  non_design_mesh: cubeMesh,
  combined_mesh: cubeMesh,
  bounds: [[0, 1], [0, 1], [0, 1]],
  resolution_xyz: [4, 4, 4],
  diagnostics: [],
};

describe("meshGeometry", () => {
  it("expands a clicked planar triangle to the full coplanar face", () => {
    const selection = resolvePlanarFaceSelection(cubeMesh, 0);

    expect(selection).not.toBeNull();
    expect(selection?.triangleIndices).toEqual([0, 1]);
    expect(selection?.area).toBeCloseTo(1, 5);
    expect(selection?.centroid[2]).toBeCloseTo(0, 5);
  });

  it("does not include non-coplanar adjacent triangles", () => {
    const selection = resolvePlanarFaceSelection(cubeMesh, 4);

    expect(selection).not.toBeNull();
    expect(selection?.triangleIndices).toEqual([4, 5]);
    expect(selection?.triangleIndices).not.toContain(6);
  });

  it("resolves a non-zero surface node set for the selected planar patch", () => {
    const selection = resolvePlanarFaceSelection(cubeMesh, 0);
    expect(selection).not.toBeNull();

    const nodes = resolveSurfaceNodePoints(preprocess, selection!);

    expect(nodes.length).toBeGreaterThan(0);
    expect(nodes.every((node) => node.point_xyz[2] === 0)).toBe(true);
  });
});
