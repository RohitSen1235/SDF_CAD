from __future__ import annotations

import math
import struct
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .sparse_field import OctreeField, SparseBrickField, detect_zero_crossing_blocks
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:
    njit = None  # type: ignore[assignment]
    prange = range  # type: ignore[assignment]
    NUMBA_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False


class MeshUploadError(ValueError):
    pass


@dataclass
class ParsedMesh:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class HostFieldData:
    mesh: ParsedMesh
    bounds: list[list[float]]
    host_sdf: np.ndarray
    field_storage_mode: Literal["dense", "octree_sparse"] = "dense"
    block_size: int | None = None
    active_blocks: list[tuple[int, int, int]] | None = None
    sparse_background_value: float | None = None
    sparse_bricks: dict[tuple[int, int, int], np.ndarray] | None = None
    octree_node_min: np.ndarray | None = None
    octree_node_max: np.ndarray | None = None
    octree_node_depth: np.ndarray | None = None
    octree_node_kind: np.ndarray | None = None
    host_build_strategy: Literal["dense", "octree_sparse"] = "dense"
    host_decision_reason: str = "dense_default"


_AUTO_SPARSE_MAX_EST_NEAR_RATIO = 0.60
_AUTO_SPARSE_MAX_EST_TRI_BLOCK_HITS = 1_200_000
_AUTO_SPARSE_MAX_EST_TRIANGLES_PER_ACTIVE_BLOCK = 4_000.0


def parse_mesh_bytes(data: bytes, extension: str) -> ParsedMesh:
    ext = extension.lower()
    if ext == ".obj":
        return _parse_obj(data)
    if ext == ".stl":
        return _parse_stl(data)
    raise MeshUploadError("Only .stl and .obj uploads are supported")


def validate_triangle_mesh(mesh: ParsedMesh) -> None:
    vertices = mesh.vertices
    faces = mesh.faces

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise MeshUploadError("Mesh vertices must be Nx3")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise MeshUploadError("Mesh faces must be Mx3 triangles")
    if vertices.shape[0] < 4:
        raise MeshUploadError("Mesh has too few vertices")
    if faces.shape[0] == 0:
        raise MeshUploadError("Mesh contains no faces")
    if not np.all(np.isfinite(vertices)):
        raise MeshUploadError("Mesh contains non-finite vertex coordinates")
    if np.min(faces) < 0 or np.max(faces) >= vertices.shape[0]:
        raise MeshUploadError("Mesh face indices are out of range")

    tri = vertices[faces]
    edge_a = tri[:, 1] - tri[:, 0]
    edge_b = tri[:, 2] - tri[:, 0]
    area2 = np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    extents = np.ptp(vertices, axis=0)
    diag = float(np.linalg.norm(extents))
    # Use scale-aware epsilon so very small models are not falsely rejected.
    area_eps = max((diag * diag) * 1e-16, 1e-28)
    if np.any(area2 <= area_eps):
        raise MeshUploadError("Mesh contains degenerate triangles")

    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    if np.any(edges[:, 0] == edges[:, 1]):
        raise MeshUploadError("Mesh contains invalid zero-length edges")

    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts != 2):
        raise MeshUploadError(
            "Mesh must be watertight and edge-manifold (each edge must be shared by exactly two triangles)"
        )


def compute_resolution_for_lattice_pitch(
    mesh_span: float,
    lattice_pitch: float,
    voxels_per_period: int = 6,
) -> int:
    """Compute the voxel grid resolution needed to faithfully sample a lattice.

    Uses the same 12% padding factor as _build_bounds() so the result is
    consistent with the actual padded bounding box.

    Args:
        mesh_span: Largest dimension of the mesh bounding box in world units.
        lattice_pitch: Desired lattice cell size in world units.
        voxels_per_period: Number of voxels per lattice period (6 = safe default,
            4 = minimum, 8 = high quality).

    Returns:
        Required voxel grid resolution (cubic).
    """
    if lattice_pitch <= 0:
        raise MeshUploadError("lattice_pitch must be > 0")
    if voxels_per_period < 2:
        raise MeshUploadError("voxels_per_period must be >= 2")
    padded_span = mesh_span * 1.24  # matches _build_bounds() 12% padding each side
    return max(24, int(math.ceil((padded_span / lattice_pitch) * voxels_per_period)))


def build_host_field(
    mesh: ParsedMesh,
    resolution: int,
    field_storage_mode: Literal["auto", "dense", "octree_sparse"] = "auto",
) -> HostFieldData:
    if resolution < 24:
        raise MeshUploadError("resolution is too low for mesh workflow")

    bounds = _build_bounds(mesh)
    occupancy = _voxelize_and_fill(mesh, bounds, resolution)

    if field_storage_mode == "dense":
        host_sdf = np.asarray(_build_host_sdf_dense(occupancy, bounds, resolution), dtype=np.float32)
        return HostFieldData(
            mesh=mesh,
            bounds=bounds,
            host_sdf=host_sdf,
            field_storage_mode="dense",
            block_size=None,
            active_blocks=None,
            host_build_strategy="dense",
            host_decision_reason="dense_requested",
        )

    allow_dense_fallback = field_storage_mode == "auto"
    host_sdf, block_size, active_blocks, sparse_background_value, decision_reason = _build_host_sdf_octree_sparse(
        mesh,
        occupancy,
        bounds,
        resolution,
        allow_dense_fallback=allow_dense_fallback,
    )
    host_sdf = np.asarray(host_sdf, dtype=np.float32)

    use_sparse = bool(block_size and active_blocks)

    if use_sparse:
        sparse = SparseBrickField.from_dense(
            host_sdf,
            bounds,
            block_size=block_size or 16,
            active_blocks=active_blocks,
            background_value=sparse_background_value,
        )
        octree = OctreeField.from_sparse_bricks(sparse)
        return HostFieldData(
            mesh=mesh,
            bounds=bounds,
            host_sdf=host_sdf,
            field_storage_mode="octree_sparse",
            block_size=int(sparse.block_size),
            active_blocks=sparse.active_blocks(),
            sparse_background_value=float(sparse.background_value),
            sparse_bricks={
                key: np.array(value, copy=True) for key, value in sparse.bricks.items()
            },
            octree_node_min=np.array(octree.node_min, copy=True),
            octree_node_max=np.array(octree.node_max, copy=True),
            octree_node_depth=np.array(octree.node_depth, copy=True),
            octree_node_kind=np.array(octree.node_kind, copy=True),
            host_build_strategy="octree_sparse",
            host_decision_reason=decision_reason,
        )

    return HostFieldData(
        mesh=mesh,
        bounds=bounds,
        host_sdf=host_sdf,
        field_storage_mode="dense",
        block_size=block_size,
        active_blocks=active_blocks,
        sparse_background_value=sparse_background_value,
        host_build_strategy="dense",
        host_decision_reason=decision_reason,
    )


def _build_host_sdf_dense(
    occupancy: np.ndarray,
    bounds: list[list[float]],
    resolution: int,
) -> np.ndarray:
    spacing = _bounds_spacing(bounds, resolution)
    outside = np.logical_not(occupancy)
    dist_out = ndimage.distance_transform_edt(outside, sampling=spacing)
    dist_in = ndimage.distance_transform_edt(occupancy, sampling=spacing)
    return dist_out - dist_in


def _octree_collect_surface_blocks(
    surface_mask: np.ndarray,
    block_size: int,
) -> set[tuple[int, int, int]]:
    resolution = int(surface_mask.shape[0])
    if resolution <= 0:
        return set()

    leaves: set[tuple[int, int, int]] = set()

    def recurse(i0: int, i1: int, j0: int, j1: int, k0: int, k1: int) -> None:
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            return
        if not np.any(surface_mask[i0:i1, j0:j1, k0:k1]):
            return

        if (i1 - i0) <= block_size and (j1 - j0) <= block_size and (k1 - k0) <= block_size:
            leaves.add((i0 // block_size, j0 // block_size, k0 // block_size))
            return

        im = (i0 + i1) // 2
        jm = (j0 + j1) // 2
        km = (k0 + k1) // 2
        i_ranges = ((i0, im), (im, i1)) if im > i0 else ((i0, i1),)
        j_ranges = ((j0, jm), (jm, j1)) if jm > j0 else ((j0, j1),)
        k_ranges = ((k0, km), (km, k1)) if km > k0 else ((k0, k1),)

        for ia, ib in i_ranges:
            for ja, jb in j_ranges:
                for ka, kb in k_ranges:
                    recurse(ia, ib, ja, jb, ka, kb)

    recurse(0, resolution, 0, resolution, 0, resolution)
    return leaves


def _grid_index_floor(value: float, origin: float, spacing: float, resolution: int) -> int:
    idx = int(math.floor((value - origin) / spacing))
    if idx < 0:
        return 0
    if idx >= resolution:
        return resolution - 1
    return idx


def _grid_index_ceil(value: float, origin: float, spacing: float, resolution: int) -> int:
    idx = int(math.ceil((value - origin) / spacing))
    if idx < 0:
        return 0
    if idx >= resolution:
        return resolution - 1
    return idx


def _point_segment_distance_sq_batch(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    segment = b - a
    seg_len_sq = float(np.dot(segment, segment))
    if seg_len_sq <= 1e-24:
        diff = points - a
        return np.einsum("ij,ij->i", diff, diff)

    rel = points - a
    t = np.einsum("ij,j->i", rel, segment) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * segment[None, :]
    diff = points - closest
    return np.einsum("ij,ij->i", diff, diff)


def _point_triangle_distance_sq_batch(points: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a = triangle[0]
    b = triangle[1]
    c = triangle[2]

    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    normal_len_sq = float(np.dot(normal, normal))
    if normal_len_sq <= 1e-24:
        d_ab = _point_segment_distance_sq_batch(points, a, b)
        d_bc = _point_segment_distance_sq_batch(points, b, c)
        d_ca = _point_segment_distance_sq_batch(points, c, a)
        return np.minimum(d_ab, np.minimum(d_bc, d_ca))

    rel = points - a
    signed_plane = np.einsum("ij,j->i", rel, normal)
    projected = points - (signed_plane / normal_len_sq)[:, None] * normal[None, :]
    projected_rel = projected - a

    d00 = float(np.dot(ab, ab))
    d01 = float(np.dot(ab, ac))
    d11 = float(np.dot(ac, ac))
    denom = max(d00 * d11 - d01 * d01, 1e-24)
    d20 = np.einsum("ij,j->i", projected_rel, ab)
    d21 = np.einsum("ij,j->i", projected_rel, ac)

    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    inside = (u >= -1e-9) & (v >= -1e-9) & ((u + v) <= 1.0 + 1e-9)

    plane_dist_sq = (signed_plane * signed_plane) / normal_len_sq
    d_ab = _point_segment_distance_sq_batch(points, a, b)
    d_bc = _point_segment_distance_sq_batch(points, b, c)
    d_ca = _point_segment_distance_sq_batch(points, c, a)
    edge_dist_sq = np.minimum(d_ab, np.minimum(d_bc, d_ca))
    return np.where(inside, plane_dist_sq, edge_dist_sq)


def _triangle_scan_convert_candidates(
    mesh: ParsedMesh,
    bounds: list[list[float]],
    resolution: int,
    block_size: int,
    band_distance: float,
) -> tuple[np.ndarray, dict[tuple[int, int, int], list[int]], list[tuple[int, int, int]], float]:
    spacing = _bounds_spacing(bounds, resolution)
    origin = np.asarray([bounds[0][0], bounds[1][0], bounds[2][0]], dtype=np.float64)
    spacing_vec = np.asarray(spacing, dtype=np.float64)
    triangles = np.ascontiguousarray(mesh.vertices[mesh.faces], dtype=np.float64)

    block_candidates: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    active_blocks: set[tuple[int, int, int]] = set()
    blocks_per_axis = int(math.ceil(resolution / float(block_size)))

    for tri_idx, triangle in enumerate(triangles):
        tri_min = np.min(triangle, axis=0) - band_distance
        tri_max = np.max(triangle, axis=0) + band_distance

        i0 = _grid_index_floor(float(tri_min[0]), float(origin[0]), float(spacing_vec[0]), resolution)
        j0 = _grid_index_floor(float(tri_min[1]), float(origin[1]), float(spacing_vec[1]), resolution)
        k0 = _grid_index_floor(float(tri_min[2]), float(origin[2]), float(spacing_vec[2]), resolution)
        i1 = _grid_index_ceil(float(tri_max[0]), float(origin[0]), float(spacing_vec[0]), resolution)
        j1 = _grid_index_ceil(float(tri_max[1]), float(origin[1]), float(spacing_vec[1]), resolution)
        k1 = _grid_index_ceil(float(tri_max[2]), float(origin[2]), float(spacing_vec[2]), resolution)

        bx0 = i0 // block_size
        by0 = j0 // block_size
        bz0 = k0 // block_size
        bx1 = min(blocks_per_axis - 1, i1 // block_size)
        by1 = min(blocks_per_axis - 1, j1 // block_size)
        bz1 = min(blocks_per_axis - 1, k1 // block_size)

        for bx in range(bx0, bx1 + 1):
            for by in range(by0, by1 + 1):
                for bz in range(bz0, bz1 + 1):
                    key = (bx, by, bz)
                    block_candidates[key].append(tri_idx)
                    active_blocks.add(key)

    active_blocks_sorted = sorted(active_blocks)
    active_voxels = 0.0
    for bx, by, bz in active_blocks_sorted:
        i0 = bx * block_size
        j0 = by * block_size
        k0 = bz * block_size
        i1 = min(resolution, i0 + block_size)
        j1 = min(resolution, j0 + block_size)
        k1 = min(resolution, k0 + block_size)
        active_voxels += float((i1 - i0) * (j1 - j0) * (k1 - k0))

    near_ratio = active_voxels / float(resolution**3)
    return triangles, block_candidates, active_blocks_sorted, near_ratio


def _scan_convert_block_distances(
    triangles: np.ndarray,
    triangle_indices: list[int],
    bounds: list[list[float]],
    resolution: int,
    i0: int,
    i1: int,
    j0: int,
    j1: int,
    k0: int,
    k1: int,
) -> np.ndarray:
    spacing = _bounds_spacing(bounds, resolution)
    x_axis = bounds[0][0] + np.arange(i0, i1, dtype=np.float64) * spacing[0]
    y_axis = bounds[1][0] + np.arange(j0, j1, dtype=np.float64) * spacing[1]
    z_axis = bounds[2][0] + np.arange(k0, k1, dtype=np.float64) * spacing[2]
    xx, yy, zz = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    min_dist_sq = np.full(points.shape[0], np.inf, dtype=np.float64)
    for tri_idx in triangle_indices:
        dist_sq = _point_triangle_distance_sq_batch(points, triangles[tri_idx])
        np.minimum(min_dist_sq, dist_sq, out=min_dist_sq)

    return min_dist_sq.reshape((i1 - i0, j1 - j0, k1 - k0))


def _estimate_sparse_profitability(
    mesh: ParsedMesh,
    bounds: list[list[float]],
    resolution: int,
    block_size: int,
    band_distance: float,
) -> tuple[bool, str, float, int, float]:
    spacing = _bounds_spacing(bounds, resolution)
    origin = np.asarray([bounds[0][0], bounds[1][0], bounds[2][0]], dtype=np.float64)
    spacing_vec = np.asarray(spacing, dtype=np.float64)
    blocks_per_axis = int(math.ceil(resolution / float(block_size)))

    mesh_min = np.min(mesh.vertices, axis=0) - band_distance
    mesh_max = np.max(mesh.vertices, axis=0) + band_distance
    i0 = _grid_index_floor(float(mesh_min[0]), float(origin[0]), float(spacing_vec[0]), resolution)
    j0 = _grid_index_floor(float(mesh_min[1]), float(origin[1]), float(spacing_vec[1]), resolution)
    k0 = _grid_index_floor(float(mesh_min[2]), float(origin[2]), float(spacing_vec[2]), resolution)
    i1 = _grid_index_ceil(float(mesh_max[0]), float(origin[0]), float(spacing_vec[0]), resolution)
    j1 = _grid_index_ceil(float(mesh_max[1]), float(origin[1]), float(spacing_vec[1]), resolution)
    k1 = _grid_index_ceil(float(mesh_max[2]), float(origin[2]), float(spacing_vec[2]), resolution)

    near_voxels = max(1, (i1 - i0 + 1) * (j1 - j0 + 1) * (k1 - k0 + 1))
    near_ratio_est = float(near_voxels) / float(max(1, resolution**3))
    if near_ratio_est >= _AUTO_SPARSE_MAX_EST_NEAR_RATIO:
        return False, "dense_auto_gate_near_ratio", near_ratio_est, 0, 0.0

    bx0 = i0 // block_size
    by0 = j0 // block_size
    bz0 = k0 // block_size
    bx1 = min(blocks_per_axis - 1, i1 // block_size)
    by1 = min(blocks_per_axis - 1, j1 // block_size)
    bz1 = min(blocks_per_axis - 1, k1 // block_size)
    estimated_active_blocks = max(1, (bx1 - bx0 + 1) * (by1 - by0 + 1) * (bz1 - bz0 + 1))

    tri_block_hits_est = 0
    for f0, f1, f2 in mesh.faces:
        triangle = mesh.vertices[[int(f0), int(f1), int(f2)]]
        tri_min = np.min(triangle, axis=0) - band_distance
        tri_max = np.max(triangle, axis=0) + band_distance
        ti0 = _grid_index_floor(float(tri_min[0]), float(origin[0]), float(spacing_vec[0]), resolution)
        tj0 = _grid_index_floor(float(tri_min[1]), float(origin[1]), float(spacing_vec[1]), resolution)
        tk0 = _grid_index_floor(float(tri_min[2]), float(origin[2]), float(spacing_vec[2]), resolution)
        ti1 = _grid_index_ceil(float(tri_max[0]), float(origin[0]), float(spacing_vec[0]), resolution)
        tj1 = _grid_index_ceil(float(tri_max[1]), float(origin[1]), float(spacing_vec[1]), resolution)
        tk1 = _grid_index_ceil(float(tri_max[2]), float(origin[2]), float(spacing_vec[2]), resolution)

        tbx0 = ti0 // block_size
        tby0 = tj0 // block_size
        tbz0 = tk0 // block_size
        tbx1 = min(blocks_per_axis - 1, ti1 // block_size)
        tby1 = min(blocks_per_axis - 1, tj1 // block_size)
        tbz1 = min(blocks_per_axis - 1, tk1 // block_size)
        tri_block_hits_est += max(1, (tbx1 - tbx0 + 1) * (tby1 - tby0 + 1) * (tbz1 - tbz0 + 1))
        if tri_block_hits_est > _AUTO_SPARSE_MAX_EST_TRI_BLOCK_HITS:
            triangles_per_active_block_est = float(tri_block_hits_est) / float(estimated_active_blocks)
            return (
                False,
                "dense_auto_gate_tri_block_hits",
                near_ratio_est,
                tri_block_hits_est,
                triangles_per_active_block_est,
            )

    triangles_per_active_block_est = float(tri_block_hits_est) / float(estimated_active_blocks)
    if triangles_per_active_block_est > _AUTO_SPARSE_MAX_EST_TRIANGLES_PER_ACTIVE_BLOCK:
        return (
            False,
            "dense_auto_gate_triangles_per_active_block",
            near_ratio_est,
            tri_block_hits_est,
            triangles_per_active_block_est,
        )
    return True, "sparse_selected_auto", near_ratio_est, tri_block_hits_est, triangles_per_active_block_est


def _build_host_sdf_from_surface_samples(
    occupancy: np.ndarray,
    bounds: list[list[float]],
    resolution: int,
) -> tuple[np.ndarray, int | None, list[tuple[int, int, int]] | None, float | None, str]:
    spacing = _bounds_spacing(bounds, resolution)
    max_spacing = float(max(spacing))
    eroded = ndimage.binary_erosion(occupancy, iterations=1)
    surface = np.logical_xor(occupancy, eroded)
    if not np.any(surface):
        return _build_host_sdf_dense(occupancy, bounds, resolution), None, None, None, "dense_sparse_no_surface"

    block_size = max(8, min(32, resolution // 6))
    band_voxels = max(12, resolution // 6)
    band_distance = float(band_voxels) * max_spacing

    surface_blocks = _octree_collect_surface_blocks(surface, block_size=block_size)
    if not surface_blocks:
        return _build_host_sdf_dense(occupancy, bounds, resolution), None, None, None, "dense_sparse_no_surface_blocks"

    blocks_per_axis = int(math.ceil(resolution / float(block_size)))
    halo_blocks = max(1, int(math.ceil(float(band_voxels) / float(block_size))))
    active_blocks: set[tuple[int, int, int]] = set()
    for bx, by, bz in surface_blocks:
        for dx in range(-halo_blocks, halo_blocks + 1):
            for dy in range(-halo_blocks, halo_blocks + 1):
                for dz in range(-halo_blocks, halo_blocks + 1):
                    nbx = bx + dx
                    nby = by + dy
                    nbz = bz + dz
                    if 0 <= nbx < blocks_per_axis and 0 <= nby < blocks_per_axis and 0 <= nbz < blocks_per_axis:
                        active_blocks.add((nbx, nby, nbz))

    near_mask = np.zeros_like(occupancy, dtype=bool)
    for bx, by, bz in active_blocks:
        i0 = bx * block_size
        j0 = by * block_size
        k0 = bz * block_size
        i1 = min(resolution, i0 + block_size)
        j1 = min(resolution, j0 + block_size)
        k1 = min(resolution, k0 + block_size)
        near_mask[i0:i1, j0:j1, k0:k1] = True

    near_ratio = float(np.count_nonzero(near_mask)) / float(near_mask.size)
    if near_ratio >= 0.65:
        return (
            _build_host_sdf_dense(occupancy, bounds, resolution),
            None,
            None,
            None,
            "dense_sparse_post_near_ratio",
        )

    surface_idx = np.argwhere(surface)
    query_idx = np.argwhere(near_mask)
    if surface_idx.size == 0 or query_idx.size == 0:
        return _build_host_sdf_dense(occupancy, bounds, resolution), None, None, None, "dense_sparse_no_query_points"

    origin = np.asarray([bounds[0][0], bounds[1][0], bounds[2][0]], dtype=np.float64)
    spacing_vec = np.asarray(spacing, dtype=np.float64)
    surface_pts = origin + surface_idx.astype(np.float64) * spacing_vec
    query_pts = origin + query_idx.astype(np.float64) * spacing_vec

    try:
        tree = cKDTree(surface_pts)
        query_dist, _ = tree.query(query_pts, workers=-1)
    except Exception:
        return _build_host_sdf_dense(occupancy, bounds, resolution), None, None, None, "dense_sparse_kdtree_failure"

    sign = np.where(occupancy, -1.0, 1.0).astype(np.float32, copy=False)
    far_value = np.float32(max(band_distance + 2.0 * max_spacing, max_spacing))
    host_sdf = sign * far_value

    signed_query = query_dist.astype(np.float32, copy=False) * sign[query_idx[:, 0], query_idx[:, 1], query_idx[:, 2]]
    host_sdf[query_idx[:, 0], query_idx[:, 1], query_idx[:, 2]] = signed_query
    active_blocks_sorted = sorted(active_blocks)
    return host_sdf, block_size, active_blocks_sorted, float(far_value), "sparse_selected_surface_samples"


def _build_host_sdf_octree_sparse(
    mesh: ParsedMesh | None,
    occupancy: np.ndarray,
    bounds: list[list[float]],
    resolution: int,
    *,
    allow_dense_fallback: bool = True,
) -> tuple[np.ndarray, int | None, list[tuple[int, int, int]] | None, float | None, str]:
    # For smaller grids, full EDT is typically faster and simpler.
    if resolution <= 96:
        return _build_host_sdf_dense(occupancy, bounds, resolution), None, None, None, "dense_resolution_cutoff"

    spacing = _bounds_spacing(bounds, resolution)
    max_spacing = float(max(spacing))

    block_size = max(8, min(32, resolution // 6))
    band_voxels = max(12, resolution // 6)
    band_distance = float(band_voxels) * max_spacing
    far_value = np.float32(max(band_distance + 2.0 * max_spacing, max_spacing))

    if mesh is None:
        return _build_host_sdf_from_surface_samples(occupancy, bounds, resolution)

    if allow_dense_fallback:
        profitable, reason, _near_ratio, _tri_hits, _tri_per_block = _estimate_sparse_profitability(
            mesh,
            bounds,
            resolution,
            block_size=block_size,
            band_distance=band_distance,
        )
        if not profitable:
            return _build_host_sdf_dense(occupancy, bounds, resolution), None, None, None, reason

    triangles, block_candidates, active_blocks, near_ratio = _triangle_scan_convert_candidates(
        mesh,
        bounds,
        resolution,
        block_size=block_size,
        band_distance=band_distance,
    )
    if not active_blocks:
        return _build_host_sdf_dense(occupancy, bounds, resolution), None, None, None, "dense_sparse_no_active_blocks"

    if allow_dense_fallback and near_ratio >= 0.65:
        # Sparse path helps only when most of the domain is pruned.
        return _build_host_sdf_dense(occupancy, bounds, resolution), None, None, None, "dense_sparse_post_near_ratio"

    sign = np.where(occupancy, -1.0, 1.0).astype(np.float32, copy=False)
    host_sdf = sign * far_value
    band_distance_sq = band_distance * band_distance
    refined_blocks: list[tuple[int, int, int]] = []

    for bx, by, bz in active_blocks:
        triangle_indices = block_candidates.get((bx, by, bz))
        if not triangle_indices:
            continue

        i0 = bx * block_size
        j0 = by * block_size
        k0 = bz * block_size
        i1 = min(resolution, i0 + block_size)
        j1 = min(resolution, j0 + block_size)
        k1 = min(resolution, k0 + block_size)

        min_dist_sq = _scan_convert_block_distances(
            triangles,
            triangle_indices,
            bounds,
            resolution,
            i0,
            i1,
            j0,
            j1,
            k0,
            k1,
        )
        within_band = np.isfinite(min_dist_sq) & (min_dist_sq <= band_distance_sq)
        if not np.any(within_band):
            continue

        dist_block = np.full_like(min_dist_sq, far_value, dtype=np.float64)
        np.sqrt(min_dist_sq, out=dist_block, where=within_band)
        signed_block = dist_block.astype(np.float32, copy=False) * sign[i0:i1, j0:j1, k0:k1]
        current = host_sdf[i0:i1, j0:j1, k0:k1]
        current[within_band] = signed_block[within_band]
        host_sdf[i0:i1, j0:j1, k0:k1] = current
        refined_blocks.append((bx, by, bz))

    if not refined_blocks:
        return _build_host_sdf_dense(occupancy, bounds, resolution), None, None, None, "dense_sparse_no_refined_blocks"

    strategy_reason = "sparse_selected_auto" if allow_dense_fallback else "sparse_selected_explicit"
    return host_sdf, block_size, refined_blocks, float(far_value), strategy_reason


def compose_hollow_lattice_field_sparse_with_backend(
    host_sdf: np.ndarray,
    bounds: list[list[float]],
    shell_thickness: float,
    lattice_type: str,
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    block_size: int | None,
    active_blocks: list[tuple[int, int, int]] | None,
    compute_backend: Literal["auto", "cpu", "cuda"] = "auto",
) -> tuple[np.ndarray, Literal["cpu", "cuda"], list[tuple[int, int, int]] | None]:
    field, backend = compose_hollow_lattice_field_with_backend(
        host_sdf,
        bounds,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        compute_backend=compute_backend,
    )
    updated_blocks: list[tuple[int, int, int]] | None = None
    if block_size is not None:
        updated_blocks = detect_zero_crossing_blocks(
            field,
            block_size=block_size,
            candidate_blocks=active_blocks,
        )
    return field, backend, updated_blocks


def compose_hollow_lattice_field(
    host_sdf: np.ndarray,
    bounds: list[list[float]],
    shell_thickness: float,
    lattice_type: str,
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
) -> np.ndarray:
    field, _ = compose_hollow_lattice_field_with_backend(
        host_sdf,
        bounds,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        compute_backend="cpu",
    )
    return field


def _resolve_compute_backend(requested: Literal["auto", "cpu", "cuda"]) -> Literal["cpu", "cuda"]:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if CUPY_AVAILABLE else "cpu"
    return "cuda" if CUPY_AVAILABLE else "cpu"


def compose_hollow_lattice_field_with_backend(
    host_sdf: np.ndarray,
    bounds: list[list[float]],
    shell_thickness: float,
    lattice_type: str,
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    compute_backend: Literal["auto", "cpu", "cuda"] = "auto",
) -> tuple[np.ndarray, Literal["cpu", "cuda"]]:
    resolved_backend = _resolve_compute_backend(compute_backend)
    if resolved_backend == "cuda":
        try:
            return (
                _compose_hollow_lattice_field_cuda(
                    host_sdf,
                    bounds,
                    shell_thickness=shell_thickness,
                    lattice_type=lattice_type,
                    lattice_pitch=lattice_pitch,
                    lattice_thickness=lattice_thickness,
                    lattice_phase=lattice_phase,
                ),
                "cuda",
            )
        except Exception:
            pass
    return (
        _compose_hollow_lattice_field_cpu(
            host_sdf,
            bounds,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
        ),
        "cpu",
    )


def _compose_hollow_lattice_field_cpu(
    host_sdf: np.ndarray,
    bounds: list[list[float]],
    shell_thickness: float,
    lattice_type: str,
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
) -> np.ndarray:
    if shell_thickness <= 0.0:
        raise MeshUploadError("shell_thickness must be > 0")
    if lattice_pitch <= 0.0:
        raise MeshUploadError("lattice_pitch must be > 0")
    if lattice_thickness <= 0.0:
        raise MeshUploadError("lattice_thickness must be > 0")
    if host_sdf.ndim != 3:
        raise MeshUploadError("host_sdf must be a 3D grid")

    thickness = np.float32(abs(shell_thickness))
    shell_field = np.empty_like(host_sdf, dtype=np.float32)
    np.negative(host_sdf, out=shell_field)
    shell_field -= thickness
    np.maximum(host_sdf, shell_field, out=shell_field)

    out = np.empty_like(host_sdf, dtype=np.float32)
    np.add(host_sdf, thickness, out=out)

    # Expensive TPMS trig is needed only inside cavity voxels.
    mask = out < 0.0
    if np.any(mask):
        resolution = host_sdf.shape[0]
        x_axis = np.linspace(bounds[0][0], bounds[0][1], resolution, dtype=host_sdf.dtype)
        y_axis = np.linspace(bounds[1][0], bounds[1][1], resolution, dtype=host_sdf.dtype)
        z_axis = np.linspace(bounds[2][0], bounds[2][1], resolution, dtype=host_sdf.dtype)
        ix, iy, iz = np.nonzero(mask)

        qx = x_axis[ix]
        qy = y_axis[iy]
        qz = z_axis[iz]
        lattice_values = np.asarray(
            _tpms_field(
                qx,
                qy,
                qz,
                lattice_type=lattice_type,
                lattice_pitch=lattice_pitch,
                lattice_thickness=lattice_thickness,
                lattice_phase=lattice_phase,
            ),
            dtype=host_sdf.dtype,
        )
        out[mask] = np.maximum(lattice_values, out[mask])

    np.minimum(shell_field, out, out=out)
    return out


def _compose_hollow_lattice_field_cuda(
    host_sdf: np.ndarray,
    bounds: list[list[float]],
    shell_thickness: float,
    lattice_type: str,
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
) -> np.ndarray:
    if not CUPY_AVAILABLE or cp is None:
        raise MeshUploadError("CUDA backend requested but CuPy is not available")
    if shell_thickness <= 0.0:
        raise MeshUploadError("shell_thickness must be > 0")
    if lattice_pitch <= 0.0:
        raise MeshUploadError("lattice_pitch must be > 0")
    if lattice_thickness <= 0.0:
        raise MeshUploadError("lattice_thickness must be > 0")
    if host_sdf.ndim != 3:
        raise MeshUploadError("host_sdf must be a 3D grid")

    thickness = cp.float32(abs(shell_thickness))
    host = cp.asarray(host_sdf, dtype=cp.float32)
    shell_field = cp.empty_like(host)
    cp.negative(host, out=shell_field)
    shell_field -= thickness
    cp.maximum(host, shell_field, out=shell_field)

    out = cp.empty_like(host)
    cp.add(host, thickness, out=out)

    mask = out < 0.0
    if bool(cp.any(mask).item()):
        resolution = host.shape[0]
        x_axis = cp.linspace(bounds[0][0], bounds[0][1], resolution, dtype=cp.float32)
        y_axis = cp.linspace(bounds[1][0], bounds[1][1], resolution, dtype=cp.float32)
        z_axis = cp.linspace(bounds[2][0], bounds[2][1], resolution, dtype=cp.float32)
        ix, iy, iz = cp.nonzero(mask)
        qx = x_axis[ix]
        qy = y_axis[iy]
        qz = z_axis[iz]
        lattice_values = _tpms_field_xp(
            cp,
            qx,
            qy,
            qz,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
        )
        out[mask] = cp.maximum(lattice_values, out[mask])

    cp.minimum(shell_field, out, out=out)
    return cp.asnumpy(out)


def build_hollow_lattice_field(
    mesh: ParsedMesh,
    shell_thickness: float,
    lattice_type: str,
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    resolution: int,
) -> tuple[np.ndarray, list[list[float]], np.ndarray]:
    host = build_host_field(mesh, resolution=resolution)
    field = compose_hollow_lattice_field(
        host.host_sdf,
        host.bounds,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
    )
    return field, host.bounds, host.host_sdf


def _parse_obj(data: bytes) -> ParsedMesh:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        head = tokens[0]

        if head == "v":
            if len(tokens) < 4:
                raise MeshUploadError(f"OBJ vertex on line {line_no} is malformed")
            try:
                vx = float(tokens[1])
                vy = float(tokens[2])
                vz = float(tokens[3])
            except ValueError as exc:
                raise MeshUploadError(f"OBJ vertex on line {line_no} is malformed") from exc
            vertices.append((vx, vy, vz))
            continue

        if head != "f":
            continue
        if len(tokens) < 4:
            raise MeshUploadError(f"OBJ face on line {line_no} has fewer than 3 vertices")

        face_idx: list[int] = []
        for item in tokens[1:]:
            raw_index = item.split("/", maxsplit=1)[0]
            if not raw_index:
                raise MeshUploadError(f"OBJ face on line {line_no} has an invalid index")
            try:
                idx = int(raw_index)
            except ValueError as exc:
                raise MeshUploadError(f"OBJ face on line {line_no} has an invalid index") from exc
            if idx < 0:
                idx = len(vertices) + idx + 1
            if idx <= 0 or idx > len(vertices):
                raise MeshUploadError(f"OBJ face on line {line_no} references out-of-range vertex")
            face_idx.append(idx - 1)

        # Handle exporters that repeat the first vertex at the end and/or emit
        # adjacent duplicate indices in polygon loops.
        compact: list[int] = []
        for idx in face_idx:
            if not compact or compact[-1] != idx:
                compact.append(idx)
        if len(compact) >= 2 and compact[0] == compact[-1]:
            compact.pop()

        if len(compact) < 3:
            continue

        anchor = compact[0]
        for i in range(1, len(compact) - 1):
            i1 = compact[i]
            i2 = compact[i + 1]
            if anchor == i1 or i1 == i2 or i2 == anchor:
                continue
            faces.append((anchor, i1, i2))

    if not vertices or not faces:
        raise MeshUploadError("OBJ upload contains no usable triangle data")

    return ParsedMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int32),
    )


def _parse_stl(data: bytes) -> ParsedMesh:
    mesh = _parse_stl_binary(data)
    if mesh is not None:
        return mesh
    return _parse_stl_ascii(data)


def _parse_stl_binary(data: bytes) -> ParsedMesh | None:
    if len(data) < 84:
        return None
    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + tri_count * 50
    if expected != len(data):
        return None

    if tri_count == 0:
        raise MeshUploadError("Binary STL contains no triangles")
    tri_dtype = np.dtype(
        [
            ("normal", "<f4", 3),
            ("v0", "<f4", 3),
            ("v1", "<f4", 3),
            ("v2", "<f4", 3),
            ("attr", "<u2"),
        ]
    )
    records = np.frombuffer(data, dtype=tri_dtype, count=tri_count, offset=84)
    triangles = np.stack([records["v0"], records["v1"], records["v2"]], axis=1).astype(np.float64, copy=False)
    return _triangles_to_indexed_mesh(triangles)


def _parse_stl_ascii(data: bytes) -> ParsedMesh:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")

    triangles: list[list[tuple[float, float, float]]] = []
    current: list[tuple[float, float, float]] = []
    for raw in text.splitlines():
        line = raw.strip().lower()
        if not line.startswith("vertex"):
            continue
        parts = raw.strip().split()
        if len(parts) < 4:
            continue
        try:
            vx = float(parts[1])
            vy = float(parts[2])
            vz = float(parts[3])
        except ValueError:
            continue
        current.append((vx, vy, vz))
        if len(current) == 3:
            triangles.append(current)
            current = []

    if not triangles:
        raise MeshUploadError("ASCII STL contains no triangles")

    tri_np = np.asarray(triangles, dtype=np.float64)
    return _triangles_to_indexed_mesh(tri_np)


def _triangles_to_indexed_mesh(triangles: np.ndarray) -> ParsedMesh:
    flat_vertices = np.ascontiguousarray(triangles.reshape(-1, 3), dtype=np.float64)
    quant = 1e-9
    quantized = np.rint(flat_vertices / quant).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)

    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    first_positions = order[np.concatenate(([0], np.flatnonzero(np.diff(sorted_inverse)) + 1))]
    vertices = flat_vertices[first_positions]
    faces = inverse.reshape(-1, 3).astype(np.int32, copy=False)

    return ParsedMesh(vertices=vertices, faces=faces)


def _build_bounds(mesh: ParsedMesh) -> list[list[float]]:
    mins = np.min(mesh.vertices, axis=0)
    maxs = np.max(mesh.vertices, axis=0)
    extents = maxs - mins
    if np.any(extents <= 1e-8):
        raise MeshUploadError("Mesh bounding box is degenerate; expected a 3D solid")

    span = float(np.max(extents))
    padding = max(
        1e-3,
        span * 0.12,
    )
    return [
        [float(mins[0] - padding), float(maxs[0] + padding)],
        [float(mins[1] - padding), float(maxs[1] + padding)],
        [float(mins[2] - padding), float(maxs[2] + padding)],
    ]


def _bounds_spacing(bounds: list[list[float]], resolution: int) -> tuple[float, float, float]:
    return (
        (bounds[0][1] - bounds[0][0]) / float(resolution - 1),
        (bounds[1][1] - bounds[1][0]) / float(resolution - 1),
        (bounds[2][1] - bounds[2][0]) / float(resolution - 1),
    )


def _clip_round_index(value: float, resolution: int) -> int:
    idx = int(round(value))
    if idx < 0:
        return 0
    if idx >= resolution:
        return resolution - 1
    return idx


def _rasterize_surface_python(verts_grid: np.ndarray, faces: np.ndarray, resolution: int) -> np.ndarray:
    surface = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    for f0, f1, f2 in faces:
        p0 = verts_grid[int(f0)]
        p1 = verts_grid[int(f1)]
        p2 = verts_grid[int(f2)]

        edge01 = float(np.linalg.norm(p1 - p0))
        edge12 = float(np.linalg.norm(p2 - p1))
        edge20 = float(np.linalg.norm(p0 - p2))
        max_edge = max(edge01, edge12, edge20)
        steps = max(2, int(math.ceil(max_edge * 2.0)))

        for i in range(steps + 1):
            u = float(i) / float(steps)
            max_j = steps - i
            for j in range(max_j + 1):
                v = float(j) / float(steps)
                sample = p0 + u * (p1 - p0) + v * (p2 - p0)
                ix = _clip_round_index(float(sample[0]), resolution)
                iy = _clip_round_index(float(sample[1]), resolution)
                iz = _clip_round_index(float(sample[2]), resolution)
                surface[ix, iy, iz] = 1
    return surface


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _clip_round_index_numba(value: float, resolution: int) -> int:
        idx = int(round(value))
        if idx < 0:
            return 0
        if idx >= resolution:
            return resolution - 1
        return idx


    @njit(parallel=True, cache=True)
    def _rasterize_surface_numba(verts_grid: np.ndarray, faces: np.ndarray, resolution: int) -> np.ndarray:
        surface = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
        for face_idx in prange(faces.shape[0]):
            f0 = int(faces[face_idx, 0])
            f1 = int(faces[face_idx, 1])
            f2 = int(faces[face_idx, 2])

            p0x = verts_grid[f0, 0]
            p0y = verts_grid[f0, 1]
            p0z = verts_grid[f0, 2]
            p1x = verts_grid[f1, 0]
            p1y = verts_grid[f1, 1]
            p1z = verts_grid[f1, 2]
            p2x = verts_grid[f2, 0]
            p2y = verts_grid[f2, 1]
            p2z = verts_grid[f2, 2]

            e01x = p1x - p0x
            e01y = p1y - p0y
            e01z = p1z - p0z
            e12x = p2x - p1x
            e12y = p2y - p1y
            e12z = p2z - p1z
            e20x = p0x - p2x
            e20y = p0y - p2y
            e20z = p0z - p2z

            edge01 = math.sqrt(e01x * e01x + e01y * e01y + e01z * e01z)
            edge12 = math.sqrt(e12x * e12x + e12y * e12y + e12z * e12z)
            edge20 = math.sqrt(e20x * e20x + e20y * e20y + e20z * e20z)
            max_edge = edge01
            if edge12 > max_edge:
                max_edge = edge12
            if edge20 > max_edge:
                max_edge = edge20
            steps = max(2, int(math.ceil(max_edge * 2.0)))

            d10x = p1x - p0x
            d10y = p1y - p0y
            d10z = p1z - p0z
            d20x = p2x - p0x
            d20y = p2y - p0y
            d20z = p2z - p0z
            inv_steps = 1.0 / float(steps)

            for i in range(steps + 1):
                u = float(i) * inv_steps
                max_j = steps - i
                for j in range(max_j + 1):
                    v = float(j) * inv_steps
                    sx = p0x + u * d10x + v * d20x
                    sy = p0y + u * d10y + v * d20y
                    sz = p0z + u * d10z + v * d20z

                    ix = _clip_round_index_numba(sx, resolution)
                    iy = _clip_round_index_numba(sy, resolution)
                    iz = _clip_round_index_numba(sz, resolution)
                    surface[ix, iy, iz] = 1
        return surface

else:

    def _rasterize_surface_numba(verts_grid: np.ndarray, faces: np.ndarray, resolution: int) -> np.ndarray:
        raise RuntimeError("Numba is unavailable")


def _voxelize_and_fill(mesh: ParsedMesh, bounds: list[list[float]], resolution: int) -> np.ndarray:
    mins = np.asarray([bounds[0][0], bounds[1][0], bounds[2][0]], dtype=np.float64)
    scales = np.asarray(
        [
            (resolution - 1) / (bounds[0][1] - bounds[0][0]),
            (resolution - 1) / (bounds[1][1] - bounds[1][0]),
            (resolution - 1) / (bounds[2][1] - bounds[2][0]),
        ],
        dtype=np.float64,
    )
    verts_grid = (mesh.vertices - mins) * scales
    verts_grid = np.ascontiguousarray(verts_grid, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)

    surface_u8: np.ndarray
    try:
        if NUMBA_AVAILABLE:
            surface_u8 = _rasterize_surface_numba(verts_grid, faces, resolution)
        else:
            surface_u8 = _rasterize_surface_python(verts_grid, faces, resolution)
    except Exception:
        surface_u8 = _rasterize_surface_python(verts_grid, faces, resolution)
    surface = surface_u8.astype(bool, copy=False)

    surface = ndimage.binary_dilation(surface, iterations=1)
    surface = ndimage.binary_closing(surface, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)

    filled = ndimage.binary_fill_holes(surface)
    if not np.any(filled):
        raise MeshUploadError("Voxelization failed: no solid volume detected")

    if _touches_boundary(filled):
        raise MeshUploadError(
            "Voxelization leaked to grid boundary. Upload a cleaner watertight mesh or increase geometric scale."
        )

    surface_count = int(np.count_nonzero(surface))
    filled_count = int(np.count_nonzero(filled))

    if filled_count <= surface_count:
        # Some meshes voxelize as an already-solid occupancy where hole-fill does
        # not increase voxel count (filled == surface). Accept if the occupancy
        # has interior thickness; otherwise try a stronger sealing pass.
        solid_core = ndimage.binary_erosion(surface, iterations=1)
        if np.any(solid_core):
            return surface

        sealed = ndimage.binary_closing(surface, structure=np.ones((3, 3, 3), dtype=bool), iterations=2)
        sealed_filled = ndimage.binary_fill_holes(sealed)
        if np.any(sealed_filled) and not _touches_boundary(sealed_filled):
            sealed_core = ndimage.binary_erosion(sealed_filled, iterations=1)
            if np.any(sealed_core):
                return sealed_filled

        raise MeshUploadError(
            "Voxelization could not recover interior volume. Mesh may be self-intersecting or non-solid."
        )

    return filled


def _touches_boundary(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :, :])
        or np.any(mask[-1, :, :])
        or np.any(mask[:, 0, :])
        or np.any(mask[:, -1, :])
        or np.any(mask[:, :, 0])
        or np.any(mask[:, :, -1])
    )


def _tpms_field_xp(
    xp: Any,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    lattice_type: str,
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
) -> np.ndarray:
    op = lattice_type.lower()
    if op not in {"gyroid", "schwarz_p", "diamond"}:
        raise MeshUploadError(f"Unsupported lattice_type '{lattice_type}'")

    scale = 2.0 * xp.pi / max(abs(lattice_pitch), 1e-6)
    u = qx * scale + lattice_phase
    v = qy * scale + lattice_phase
    w = qz * scale + lattice_phase

    if op == "gyroid":
        base = xp.sin(u) * xp.cos(v) + xp.sin(v) * xp.cos(w) + xp.sin(w) * xp.cos(u)
    elif op == "schwarz_p":
        base = xp.cos(u) + xp.cos(v) + xp.cos(w)
    else:
        base = (
            xp.sin(u) * xp.sin(v) * xp.sin(w)
            + xp.sin(u) * xp.cos(v) * xp.cos(w)
            + xp.cos(u) * xp.sin(v) * xp.cos(w)
            + xp.cos(u) * xp.cos(v) * xp.sin(w)
        )

    return xp.abs(base) - abs(lattice_thickness)


def _tpms_field(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    lattice_type: str,
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
) -> np.ndarray:
    return _tpms_field_xp(
        np,
        qx,
        qy,
        qz,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
    )
