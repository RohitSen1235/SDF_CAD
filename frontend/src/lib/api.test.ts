import { afterEach, describe, expect, it, vi } from "vitest";

import { commitUploadedMesh, previewUploadedMeshField, runStructuralOptimizationPhased } from "./api";

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

describe("runStructuralOptimizationPhased", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("submits a queued job, polls progress, and replays iteration updates in order", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const request = {
      design_space_file_name: "design.stl",
      design_space_file_data_base64: "AA==",
      non_design_space_file_name: "keep.stl",
      non_design_space_file_data_base64: "AA==",
      compute_backend: "cpu" as const,
      mesh_backend: "cpu" as const,
      execution_mode: "queued" as const,
      constraints: [{ kind: "fixed" as const, points: [{ point_xyz: [0, 0, 0] as [number, number, number] }], radius: 0 }],
      loads: [{ kind: "point" as const, points: [{ point_xyz: [1, 0, 0] as [number, number, number] }], direction_xyz: [1, 0, 0] as [number, number, number], magnitude: 1, radius: 0 }],
      material: { youngs_modulus: 1, poissons_ratio: 0.3, density_floor: 1e-3, stiffness_floor_ratio: 1e-3, simp_penalty: 3 },
      config: {
        resolution: 64,
        target_volume_fraction: 0.35,
        max_iterations: 64,
        cg_max_iterations: 200,
        cg_tolerance: 1e-6,
        optimization_tolerance: 1e-3,
        filter_radius_voxels: 1.5,
        min_density: 1e-3,
        oc_move_limit: 0.2,
        density_iso_threshold: 0.3
      }
    };

    fetchMock
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            job_id: "job-1",
            status: "queued",
            status_url: "/api/v1/jobs/job-1",
            result_url: "/api/v1/jobs/job-1/result",
            progress_url: "/api/v1/optimization/structural/jobs/job-1/progress"
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            job_id: "job-1",
            status: "running",
            current_iteration: 2,
            max_iterations: 64,
            iterations: [
              {
                iteration: 1,
                objective_value: 3.5,
                active_volume_fraction: 0.9,
                removed_voxels: 2,
                mesh: null,
                density_field: null,
                stress_field: null
              },
              {
                iteration: 2,
                objective_value: 3.2,
                active_volume_fraction: 0.82,
                removed_voxels: 3,
                mesh: null,
                density_field: null,
                stress_field: null
              }
            ],
            history: [],
            stop_reason: null,
            detail: null,
            final_result: null
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            job_id: "job-1",
            status: "succeeded",
            current_iteration: 2,
            max_iterations: 64,
            iterations: [],
            history: [],
            stop_reason: "objective_converged",
            detail: null,
            final_result: {
              history: [],
              final_iteration: {
                iteration: 2,
                objective_value: 3.2,
                active_volume_fraction: 0.82,
                removed_voxels: 3
              },
              bounds: [[0, 1], [0, 1], [0, 1]],
              resolution_xyz: [64, 64, 64],
              compute_backend_used: "cpu",
              mesh_backend_used: "cpu",
              stop_reason: "objective_converged"
            }
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      );

    const onIteration = vi.fn();
    const promise = runStructuralOptimizationPhased(request, onIteration);
    await vi.runAllTimersAsync();

    await expect(promise).resolves.toMatchObject({ stop_reason: "objective_converged" });
    expect(onIteration).toHaveBeenCalledTimes(2);
    expect(onIteration.mock.calls.map(([iteration]) => iteration.iteration)).toEqual([1, 2]);
    expect(fetchMock.mock.calls[1]?.[0]).toContain("after_iteration=0");
    expect(fetchMock.mock.calls[2]?.[0]).toContain("after_iteration=2");
  });

  it("fails when the polled structural job reports failed status", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    fetchMock
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            job_id: "job-2",
            status: "queued",
            status_url: "/api/v1/jobs/job-2",
            result_url: "/api/v1/jobs/job-2/result",
            progress_url: "/api/v1/optimization/structural/jobs/job-2/progress"
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            job_id: "job-2",
            status: "failed",
            current_iteration: 0,
            max_iterations: 64,
            iterations: [],
            history: [],
            stop_reason: null,
            detail: "Initial support/load connectivity failed",
            final_result: null
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        )
      );

    await expect(
      runStructuralOptimizationPhased({
        design_space_file_name: "design.stl",
        design_space_file_data_base64: "AA==",
        non_design_space_file_name: "keep.stl",
        non_design_space_file_data_base64: "AA==",
        compute_backend: "cpu",
        mesh_backend: "cpu",
        execution_mode: "queued",
        constraints: [{ kind: "fixed", points: [{ point_xyz: [0, 0, 0] }], radius: 0 }],
        loads: [{ kind: "point", points: [{ point_xyz: [1, 0, 0] }], direction_xyz: [1, 0, 0], magnitude: 1, radius: 0 }],
        material: { youngs_modulus: 1, poissons_ratio: 0.3, density_floor: 1e-3, stiffness_floor_ratio: 1e-3, simp_penalty: 3 },
          config: {
            resolution: 64,
            target_volume_fraction: 0.35,
            max_iterations: 64,
            cg_max_iterations: 200,
            cg_tolerance: 1e-6,
            optimization_tolerance: 1e-3,
            filter_radius_voxels: 1.5,
            min_density: 1e-3,
            oc_move_limit: 0.2,
            density_iso_threshold: 0.3
          }
        })
    ).rejects.toThrow(/Initial support\/load connectivity failed/);
  });
});
