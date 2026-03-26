import { afterEach, describe, expect, it, vi } from "vitest";

import { commitUploadedMesh, previewUploadedMeshField } from "./api";

describe("previewUploadedMeshField", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("reads the uploaded field preview trace id from response headers", async () => {
    const file = new File(["solid test"], "test.stl", { type: "model/stl" });
    const values = new Float32Array(16);
    const response = new Response(values.buffer.slice(0), {
      status: 200,
      headers: {
        "Content-Type": "application/octet-stream",
        "X-SDF-Stats": JSON.stringify({
          eval_ms: 1,
          mesh_ms: null,
          tri_count: 0,
          voxel_count: 8,
          preview_mode: "field"
        }),
        "X-SDF-Resolution-XYZ": "2,2,2",
        "X-SDF-Bounds": JSON.stringify([
          [-1, 1],
          [-1, 1],
          [-1, 1]
        ]),
        "X-SDF-Trace-Id": "trace-abc"
      }
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));

    const result = await previewUploadedMeshField(
      file,
      {
        shellThickness: 0.08,
        latticeType: "gyroid",
        latticePitch: 0.45,
        latticeThickness: 0.09,
        latticePhase: 0.0
      }
    );

    expect(result.trace?.traceId).toBe("trace-abc");
    expect(result.hostField).not.toBeNull();
    expect(result.trace?.clientResponseWaitMs).toBeGreaterThanOrEqual(0);
    expect(result.trace?.clientDownloadMs).toBeGreaterThanOrEqual(0);
    expect(result.trace?.clientDecodeMs).toBeGreaterThanOrEqual(0);
  });
});

describe("commitUploadedMesh", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("falls back to JSON for uploaded mesh commit responses", async () => {
    const file = new File(["solid test"], "test.stl", { type: "model/stl" });
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(new Response(null, { status: 404 }))
        .mockResolvedValueOnce(
          new Response(
            JSON.stringify({
              mesh: {
                encoding: "mesh-f32-u32-base64-v1",
                vertex_count: 0,
                face_count: 0,
                vertices_b64: "",
                indices_b64: "",
                normals_b64: ""
              },
              stats: {
                eval_ms: 0,
                mesh_ms: 1,
                tri_count: 0,
                preview_mode: "mesh",
                field_cache_hit: true,
                mesh_cache_hit: false
              }
            }),
            {
              status: 200,
              headers: {
                "Content-Type": "application/json"
              }
            }
          )
        )
    );

    const result = await commitUploadedMesh(file, {
      shellThickness: 0.08,
      latticeType: "gyroid",
      latticePitch: 0.45,
      latticeThickness: 0.09,
      latticePhase: 0.0
    });

    expect(result.stats.preview_mode).toBe("mesh");
    expect(result.stats.field_cache_hit).toBe(true);
    expect(result.stats.mesh_cache_hit).toBe(false);
  });
});
