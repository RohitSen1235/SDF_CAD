import { afterEach, describe, expect, it, vi } from "vitest";

import { previewUploadedMeshField } from "./api";

describe("previewUploadedMeshField", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("reads the uploaded field preview trace id from response headers", async () => {
    const file = new File(["solid test"], "test.stl", { type: "model/stl" });
    const values = new Float32Array(8);
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
        "X-SDF-Resolution": "2",
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
    expect(result.trace?.clientResponseWaitMs).toBeGreaterThanOrEqual(0);
    expect(result.trace?.clientDownloadMs).toBeGreaterThanOrEqual(0);
    expect(result.trace?.clientDecodeMs).toBeGreaterThanOrEqual(0);
  });
});
