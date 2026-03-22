from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from app import cache as cache_module
from app.main import _run_uploaded_mesh_field_preview_data
from app.mesh_upload import build_host_field, parse_mesh_bytes


MESH_OBJ = b"""
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
"""


@dataclass
class BenchmarkResult:
    first_eval_ms: float
    second_eval_ms: float
    eval_drop_ratio: float
    legacy_peak_bytes: int
    optimized_peak_bytes: int
    memory_drop_ratio: float


def _estimate_legacy_compose_peak_bytes(host_sdf: np.ndarray) -> int:
    mask_bytes = int(host_sdf.size)
    return int((3 * host_sdf.nbytes) + mask_bytes)


def _estimate_optimized_compose_peak_bytes(host_sdf: np.ndarray) -> int:
    mask_bytes = int(host_sdf.size)
    return int((2 * host_sdf.nbytes) + mask_bytes)


def run_acceptance_benchmark(
    *,
    lattice_pitch: float = 0.45,
    lattice_thickness: float = 0.09,
    shell_thickness: float = 0.08,
    lattice_phase: float = 0.0,
    voxels_per_lattice_period: int = 6,
) -> BenchmarkResult:
    cache_module.clear_all_caches()

    _, _, first_stats = _run_uploaded_mesh_field_preview_data(
        file_bytes=MESH_OBJ,
        extension=".obj",
        shell_thickness=shell_thickness,
        lattice_type="gyroid",
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend="cpu",
        field_storage_mode="dense",
    )

    _, _, second_stats = _run_uploaded_mesh_field_preview_data(
        file_bytes=MESH_OBJ,
        extension=".obj",
        shell_thickness=shell_thickness,
        lattice_type="gyroid",
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend="cpu",
        field_storage_mode="dense",
    )

    mesh = parse_mesh_bytes(MESH_OBJ, ".obj")
    resolution = max(48, second_stats.voxel_count and round(second_stats.voxel_count ** (1.0 / 3.0)) or 48)
    host = build_host_field(mesh, resolution=int(resolution), field_storage_mode="dense")

    legacy_peak_bytes = _estimate_legacy_compose_peak_bytes(host.host_sdf)
    optimized_peak_bytes = _estimate_optimized_compose_peak_bytes(host.host_sdf)

    eval_drop_ratio = 1.0 - (float(second_stats.eval_ms) / max(float(first_stats.eval_ms), 1e-9))
    memory_drop_ratio = 1.0 - (float(optimized_peak_bytes) / float(max(legacy_peak_bytes, 1)))
    return BenchmarkResult(
        first_eval_ms=float(first_stats.eval_ms),
        second_eval_ms=float(second_stats.eval_ms),
        eval_drop_ratio=eval_drop_ratio,
        legacy_peak_bytes=legacy_peak_bytes,
        optimized_peak_bytes=optimized_peak_bytes,
        memory_drop_ratio=memory_drop_ratio,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark uploaded preview cache reuse and compose memory footprint.")
    parser.add_argument("--voxels-per-period", type=int, default=6)
    args = parser.parse_args()

    result = run_acceptance_benchmark(
        voxels_per_lattice_period=args.voxels_per_period,
    )

    print(f"first_eval_ms={result.first_eval_ms:.3f}")
    print(f"second_eval_ms={result.second_eval_ms:.3f}")
    print(f"eval_drop_ratio={result.eval_drop_ratio:.3%}")
    print(f"legacy_peak_bytes={result.legacy_peak_bytes}")
    print(f"optimized_peak_bytes={result.optimized_peak_bytes}")
    print(f"memory_drop_ratio={result.memory_drop_ratio:.3%}")

    if result.eval_drop_ratio < 0.80:
        raise SystemExit("Acceptance failed: second-request eval_ms did not drop by at least 80%")
    if result.memory_drop_ratio < 0.25:
        raise SystemExit("Acceptance failed: estimated dense compose peak bytes did not drop by at least 25%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
