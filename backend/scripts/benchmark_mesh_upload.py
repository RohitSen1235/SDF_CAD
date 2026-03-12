from __future__ import annotations

import argparse
import time

import numpy as np
from skimage.measure import marching_cubes

from app.mesh_upload import ParsedMesh, build_host_field, compose_hollow_lattice_field_with_backend
from app.meshing import build_mesh_with_backend


def _sphere_fixture(resolution: int, radius: float = 0.75) -> ParsedMesh:
    axis = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    field = np.sqrt(x * x + y * y + z * z) - radius
    vertices, faces, _, _ = marching_cubes(field, level=0.0, allow_degenerate=False)
    return ParsedMesh(vertices=vertices.astype(np.float64, copy=False), faces=faces.astype(np.int32, copy=False))


def _run_case(label: str, mesh: ParsedMesh, quality_resolution: int) -> None:
    host_start = time.perf_counter()
    host = build_host_field(mesh, resolution=quality_resolution)
    host_ms = (time.perf_counter() - host_start) * 1000.0

    field_start = time.perf_counter()
    field, compute_backend = compose_hollow_lattice_field_with_backend(
        host.host_sdf,
        host.bounds,
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
        compute_backend="cpu",
    )
    field_ms = (time.perf_counter() - field_start) * 1000.0

    mesh_start = time.perf_counter()
    out_mesh, mesh_backend = build_mesh_with_backend(
        field,
        host.bounds,
        backend="cpu",
        meshing_mode="uniform",
    )
    mesh_ms = (time.perf_counter() - mesh_start) * 1000.0
    print(
        f"{label}: host_ms={host_ms:.1f} field_ms={field_ms:.1f} mesh_ms={mesh_ms:.1f} "
        f"compute_backend={compute_backend} mesh_backend={mesh_backend} tri_count={out_mesh.faces.shape[0]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark mesh-upload workflow phases.")
    parser.add_argument("--quality-resolution", type=int, default=96, help="Mesh workflow resolution (default: 96)")
    parser.add_argument("--medium-fixture-res", type=int, default=120, help="Medium synthetic fixture resolution")
    parser.add_argument("--large-fixture-res", type=int, default=160, help="Large synthetic fixture resolution")
    args = parser.parse_args()

    medium_mesh = _sphere_fixture(args.medium_fixture_res)
    large_mesh = _sphere_fixture(args.large_fixture_res)
    print(
        f"fixtures: medium_triangles={medium_mesh.faces.shape[0]} large_triangles={large_mesh.faces.shape[0]} "
        f"quality_resolution={args.quality_resolution}"
    )
    _run_case("medium", medium_mesh, args.quality_resolution)
    _run_case("large", large_mesh, args.quality_resolution)


if __name__ == "__main__":
    main()
