from __future__ import annotations

import math
from typing import Iterable

ResolutionXYZ = tuple[int, int, int]


def normalize_resolution_xyz(value: Iterable[int]) -> ResolutionXYZ:
    parts = [int(v) for v in value]
    if len(parts) != 3:
        raise ValueError("resolution_xyz must contain exactly 3 integers")
    nx, ny, nz = parts
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("resolution_xyz entries must be >= 2")
    return (nx, ny, nz)


def voxel_count(resolution_xyz: ResolutionXYZ) -> int:
    nx, ny, nz = resolution_xyz
    return int(nx) * int(ny) * int(nz)


def spacing_from_bounds(bounds: list[list[float]], resolution_xyz: ResolutionXYZ) -> tuple[float, float, float]:
    nx, ny, nz = resolution_xyz
    return (
        (bounds[0][1] - bounds[0][0]) / float(nx - 1),
        (bounds[1][1] - bounds[1][0]) / float(ny - 1),
        (bounds[2][1] - bounds[2][0]) / float(nz - 1),
    )


def scale_resolution_to_total_voxel_cap(
    resolution_xyz: ResolutionXYZ,
    *,
    max_total_voxels: int,
    min_axis_resolution: int = 2,
) -> tuple[ResolutionXYZ, bool]:
    nx, ny, nz = normalize_resolution_xyz(resolution_xyz)
    if max_total_voxels <= 0:
        raise ValueError("max_total_voxels must be > 0")

    current_voxels = voxel_count((nx, ny, nz))
    if current_voxels <= max_total_voxels:
        return (nx, ny, nz), False

    scale = (float(max_total_voxels) / float(current_voxels)) ** (1.0 / 3.0)
    sx = max(min_axis_resolution, int(math.floor(nx * scale)))
    sy = max(min_axis_resolution, int(math.floor(ny * scale)))
    sz = max(min_axis_resolution, int(math.floor(nz * scale)))

    # Ensure we always reduce from the source shape.
    sx = min(sx, nx)
    sy = min(sy, ny)
    sz = min(sz, nz)

    dims = [sx, sy, sz]
    while dims[0] * dims[1] * dims[2] > max_total_voxels:
        largest_idx = max(range(3), key=lambda idx: dims[idx])
        if dims[largest_idx] <= min_axis_resolution:
            break
        dims[largest_idx] -= 1

    return (int(dims[0]), int(dims[1]), int(dims[2])), True


def compute_uploaded_mesh_resolution_xyz(
    mesh_extents: Iterable[float],
    lattice_pitch: float,
    voxels_per_lattice_period: int,
    *,
    min_axis_resolution: int = 24,
    max_total_voxels: int,
) -> tuple[ResolutionXYZ, bool]:
    extents = [float(v) for v in mesh_extents]
    if len(extents) != 3:
        raise ValueError("mesh_extents must contain exactly 3 values")
    if lattice_pitch <= 0.0:
        raise ValueError("lattice_pitch must be > 0")
    if voxels_per_lattice_period < 2:
        raise ValueError("voxels_per_lattice_period must be >= 2")

    target_spacing = float(lattice_pitch) / float(voxels_per_lattice_period)
    pad_each_side = 2.0 * float(lattice_pitch)
    padded_extents = [extent + (2.0 * pad_each_side) for extent in extents]

    raw_res = [
        max(min_axis_resolution, int(math.ceil(padded / target_spacing)) + 1)
        for padded in padded_extents
    ]

    capped, clamped = scale_resolution_to_total_voxel_cap(
        (raw_res[0], raw_res[1], raw_res[2]),
        max_total_voxels=max_total_voxels,
        min_axis_resolution=min_axis_resolution,
    )
    return capped, clamped


def parse_resolution_xyz_header(value: str) -> ResolutionXYZ:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 3:
        raise ValueError("resolution header must have exactly three comma-separated integers")
    return normalize_resolution_xyz(int(item) for item in parts)
