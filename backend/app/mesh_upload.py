from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import ndimage
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


def build_host_field(
    mesh: ParsedMesh,
    resolution: int,
) -> HostFieldData:
    if resolution < 24:
        raise MeshUploadError("resolution is too low for mesh workflow")

    bounds = _build_bounds(mesh)
    occupancy = _voxelize_and_fill(mesh, bounds, resolution)

    spacing = _bounds_spacing(bounds, resolution)
    outside = np.logical_not(occupancy)
    dist_out = ndimage.distance_transform_edt(outside, sampling=spacing)
    dist_in = ndimage.distance_transform_edt(occupancy, sampling=spacing)
    host_sdf = dist_out - dist_in
    return HostFieldData(mesh=mesh, bounds=bounds, host_sdf=host_sdf)


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

    shell_field = np.maximum(host_sdf, -host_sdf - abs(shell_thickness))
    cavity = host_sdf + abs(shell_thickness)
    lattice_clipped = cavity.copy()

    # Expensive TPMS trig is needed only inside cavity voxels.
    mask = cavity < 0.0
    if np.any(mask):
        resolution = host_sdf.shape[0]
        x_axis = np.linspace(bounds[0][0], bounds[0][1], resolution, dtype=np.float64)
        y_axis = np.linspace(bounds[1][0], bounds[1][1], resolution, dtype=np.float64)
        z_axis = np.linspace(bounds[2][0], bounds[2][1], resolution, dtype=np.float64)
        ix, iy, iz = np.nonzero(mask)

        qx = x_axis[ix]
        qy = y_axis[iy]
        qz = z_axis[iz]
        lattice_values = _tpms_field(
            qx,
            qy,
            qz,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
        )
        lattice_clipped[mask] = np.maximum(lattice_values, cavity[mask])

    return np.minimum(shell_field, lattice_clipped)


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

    host = cp.asarray(host_sdf, dtype=cp.float32)
    shell_field = cp.maximum(host, -host - abs(shell_thickness))
    cavity = host + abs(shell_thickness)
    lattice_clipped = cavity.copy()

    mask = cavity < 0.0
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
        lattice_clipped[mask] = cp.maximum(lattice_values, cavity[mask])

    out = cp.minimum(shell_field, lattice_clipped)
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
