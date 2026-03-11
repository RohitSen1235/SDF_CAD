from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


class MeshUploadError(ValueError):
    pass


@dataclass
class ParsedMesh:
    vertices: np.ndarray
    faces: np.ndarray


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

    edge_counts: dict[tuple[int, int], int] = {}
    for f0, f1, f2 in faces:
        a = int(f0)
        b = int(f1)
        c = int(f2)
        for u, v in ((a, b), (b, c), (c, a)):
            if u == v:
                raise MeshUploadError("Mesh contains invalid zero-length edges")
            key = (u, v) if u < v else (v, u)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    bad_edges = [count for count in edge_counts.values() if count != 2]
    if bad_edges:
        raise MeshUploadError(
            "Mesh must be watertight and edge-manifold (each edge must be shared by exactly two triangles)"
        )


def build_hollow_lattice_field(
    mesh: ParsedMesh,
    shell_thickness: float,
    lattice_type: str,
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    resolution: int,
) -> tuple[np.ndarray, list[list[float]], np.ndarray]:
    if shell_thickness <= 0.0:
        raise MeshUploadError("shell_thickness must be > 0")
    if lattice_pitch <= 0.0:
        raise MeshUploadError("lattice_pitch must be > 0")
    if lattice_thickness <= 0.0:
        raise MeshUploadError("lattice_thickness must be > 0")
    if resolution < 24:
        raise MeshUploadError("resolution is too low for mesh workflow")

    bounds = _build_bounds(mesh, shell_thickness, lattice_pitch, lattice_thickness)
    occupancy = _voxelize_and_fill(mesh, bounds, resolution)

    spacing = _bounds_spacing(bounds, resolution)
    outside = np.logical_not(occupancy)
    dist_out = ndimage.distance_transform_edt(outside, sampling=spacing)
    dist_in = ndimage.distance_transform_edt(occupancy, sampling=spacing)
    host_sdf = dist_out - dist_in

    px, py, pz = _grid_points(bounds, resolution)
    lattice_sdf = _tpms_field(px, py, pz, lattice_type, lattice_pitch, lattice_thickness, lattice_phase)

    shell_field = np.maximum(host_sdf, -host_sdf - abs(shell_thickness))
    cavity = host_sdf + abs(shell_thickness)
    lattice_clipped = np.maximum(lattice_sdf, cavity)
    final_field = np.minimum(shell_field, lattice_clipped)

    return final_field, bounds, host_sdf


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

    triangles = np.empty((tri_count, 3, 3), dtype=np.float64)
    offset = 84
    for idx in range(tri_count):
        v0 = struct.unpack_from("<3f", data, offset + 12)
        v1 = struct.unpack_from("<3f", data, offset + 24)
        v2 = struct.unpack_from("<3f", data, offset + 36)
        triangles[idx, 0, :] = v0
        triangles[idx, 1, :] = v1
        triangles[idx, 2, :] = v2
        offset += 50

    if tri_count == 0:
        raise MeshUploadError("Binary STL contains no triangles")
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
    vert_map: dict[tuple[int, int, int], int] = {}
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    quant = 1e-9

    for tri in triangles:
        face: list[int] = []
        for vx, vy, vz in tri:
            key = (
                int(round(float(vx) / quant)),
                int(round(float(vy) / quant)),
                int(round(float(vz) / quant)),
            )
            existing = vert_map.get(key)
            if existing is None:
                existing = len(vertices)
                vert_map[key] = existing
                vertices.append((float(vx), float(vy), float(vz)))
            face.append(existing)
        faces.append((face[0], face[1], face[2]))

    return ParsedMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int32),
    )


def _build_bounds(
    mesh: ParsedMesh,
    shell_thickness: float,
    lattice_pitch: float,
    lattice_thickness: float,
) -> list[list[float]]:
    mins = np.min(mesh.vertices, axis=0)
    maxs = np.max(mesh.vertices, axis=0)
    extents = maxs - mins
    if np.any(extents <= 1e-8):
        raise MeshUploadError("Mesh bounding box is degenerate; expected a 3D solid")

    span = float(np.max(extents))
    padding = max(
        1e-3,
        span * 0.08,
        abs(shell_thickness) * 1.5,
        abs(lattice_pitch) * 0.75,
        abs(lattice_thickness) * 2.0,
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


def _grid_points(
    bounds: list[list[float]],
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(bounds[0][0], bounds[0][1], resolution, dtype=np.float64)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution, dtype=np.float64)
    z = np.linspace(bounds[2][0], bounds[2][1], resolution, dtype=np.float64)
    return np.meshgrid(x, y, z, indexing="ij")


def _voxelize_and_fill(mesh: ParsedMesh, bounds: list[list[float]], resolution: int) -> np.ndarray:
    surface = np.zeros((resolution, resolution, resolution), dtype=bool)

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

    for f0, f1, f2 in mesh.faces:
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
                ix = int(np.clip(round(sample[0]), 0, resolution - 1))
                iy = int(np.clip(round(sample[1]), 0, resolution - 1))
                iz = int(np.clip(round(sample[2]), 0, resolution - 1))
                surface[ix, iy, iz] = True

    surface = ndimage.binary_dilation(surface, iterations=1)
    surface = ndimage.binary_closing(surface, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)

    filled = ndimage.binary_fill_holes(surface)
    if not np.any(filled):
        raise MeshUploadError("Voxelization failed: no solid volume detected")

    if _touches_boundary(filled):
        raise MeshUploadError(
            "Voxelization leaked to grid boundary. Upload a cleaner watertight mesh or increase geometric scale."
        )

    if np.count_nonzero(filled) <= np.count_nonzero(surface):
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


def _tpms_field(
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

    scale = 2.0 * np.pi / max(abs(lattice_pitch), 1e-6)
    u = qx * scale + lattice_phase
    v = qy * scale + lattice_phase
    w = qz * scale + lattice_phase

    if op == "gyroid":
        base = np.sin(u) * np.cos(v) + np.sin(v) * np.cos(w) + np.sin(w) * np.cos(u)
    elif op == "schwarz_p":
        base = np.cos(u) + np.cos(v) + np.cos(w)
    else:
        base = (
            np.sin(u) * np.sin(v) * np.sin(w)
            + np.sin(u) * np.cos(v) * np.cos(w)
            + np.cos(u) * np.sin(v) * np.cos(w)
            + np.cos(u) * np.cos(v) * np.sin(w)
        )

    return np.abs(base) - abs(lattice_thickness)
