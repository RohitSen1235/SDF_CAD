from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from skimage.measure import marching_cubes


class MeshingError(ValueError):
    pass


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray



def _mesh_single(field: np.ndarray, bounds: list[list[float]]) -> MeshData:
    resolution = field.shape[0]
    dx = (bounds[0][1] - bounds[0][0]) / float(resolution - 1)
    dy = (bounds[1][1] - bounds[1][0]) / float(resolution - 1)
    dz = (bounds[2][1] - bounds[2][0]) / float(resolution - 1)

    vertices, faces, normals, _ = marching_cubes(
        field, level=0.0, spacing=(dx, dy, dz), allow_degenerate=False
    )

    origin = np.array([bounds[0][0], bounds[1][0], bounds[2][0]], dtype=np.float64)
    vertices = vertices + origin

    return MeshData(vertices=vertices, faces=faces.astype(np.int32), normals=normals)



def _mesh_chunked(field: np.ndarray, bounds: list[list[float]], chunk_size: int = 80) -> MeshData:
    resolution = field.shape[0]
    dx = (bounds[0][1] - bounds[0][0]) / float(resolution - 1)
    dy = (bounds[1][1] - bounds[1][0]) / float(resolution - 1)
    dz = (bounds[2][1] - bounds[2][0]) / float(resolution - 1)

    origin = np.array([bounds[0][0], bounds[1][0], bounds[2][0]], dtype=np.float64)
    slab_step = max(12, min(chunk_size, resolution - 1))

    vertex_map: dict[tuple[int, int, int], int] = {}
    vertices: list[list[float]] = []
    normals_accum: list[np.ndarray] = []
    faces: list[list[int]] = []
    quant = 1e-6

    for start in range(0, resolution - 1, slab_step):
        stop = min(resolution - 1, start + slab_step)
        slab = field[start : stop + 1, :, :]
        if slab.size == 0:
            continue

        smin = float(np.min(slab))
        smax = float(np.max(slab))
        if not (smin <= 0.0 <= smax):
            continue

        local_v, local_f, local_n, _ = marching_cubes(
            slab,
            level=0.0,
            spacing=(dx, dy, dz),
            allow_degenerate=False,
        )

        # Convert slab-local coordinates to world coordinates.
        local_v = local_v + origin + np.array([start * dx, 0.0, 0.0], dtype=np.float64)
        remap: list[int] = []

        for idx, vert in enumerate(local_v):
            key = (
                int(round(float(vert[0]) / quant)),
                int(round(float(vert[1]) / quant)),
                int(round(float(vert[2]) / quant)),
            )
            existing = vertex_map.get(key)
            if existing is None:
                existing = len(vertices)
                vertex_map[key] = existing
                vertices.append([float(vert[0]), float(vert[1]), float(vert[2])])
                normals_accum.append(np.array(local_n[idx], dtype=np.float64))
            else:
                normals_accum[existing] = normals_accum[existing] + np.array(
                    local_n[idx], dtype=np.float64
                )
            remap.append(existing)

        for tri in local_f:
            faces.append([remap[int(tri[0])], remap[int(tri[1])], remap[int(tri[2])]])

    if not faces:
        raise MeshingError(
            "No zero level-set detected in current grid bounds. Expand bounds or change parameters."
        )

    normals: list[list[float]] = []
    for n in normals_accum:
        norm = float(np.linalg.norm(n))
        if norm > 1e-12:
            n = n / norm
        else:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        normals.append([float(n[0]), float(n[1]), float(n[2])])

    return MeshData(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float64),
    )



def build_mesh(field: np.ndarray, bounds: list[list[float]], chunk_size: int | None = None) -> MeshData:
    min_val = float(np.min(field))
    max_val = float(np.max(field))
    if not (min_val <= 0.0 <= max_val):
        raise MeshingError(
            "No zero level-set detected in current grid bounds. Expand bounds or change parameters."
        )

    try:
        return _mesh_single(field, bounds)
    except MemoryError:
        return _mesh_chunked(field, bounds, chunk_size=chunk_size or 80)



def mesh_to_obj(mesh: MeshData) -> bytes:
    buf = io.StringIO()
    for vx, vy, vz in mesh.vertices:
        buf.write(f"v {vx:.8f} {vy:.8f} {vz:.8f}\n")
    for nx, ny, nz in mesh.normals:
        buf.write(f"vn {nx:.8f} {ny:.8f} {nz:.8f}\n")
    for i1, i2, i3 in mesh.faces:
        a, b, c = i1 + 1, i2 + 1, i3 + 1
        buf.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
    return buf.getvalue().encode("utf-8")



def mesh_to_stl(mesh: MeshData) -> bytes:
    # Binary STL output.
    header = b"SDF_CAD" + b" " * (80 - len("SDF_CAD"))
    triangles = np.zeros(
        len(mesh.faces),
        dtype=[
            ("normal", "<f4", 3),
            ("v0", "<f4", 3),
            ("v1", "<f4", 3),
            ("v2", "<f4", 3),
            ("attr", "<u2"),
        ],
    )

    for idx, (i0, i1, i2) in enumerate(mesh.faces):
        v0 = mesh.vertices[i0]
        v1 = mesh.vertices[i1]
        v2 = mesh.vertices[i2]
        normal = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
        else:
            normal = np.array([0.0, 0.0, 1.0])

        triangles[idx]["normal"] = normal.astype(np.float32)
        triangles[idx]["v0"] = v0.astype(np.float32)
        triangles[idx]["v1"] = v1.astype(np.float32)
        triangles[idx]["v2"] = v2.astype(np.float32)
        triangles[idx]["attr"] = 0

    out = io.BytesIO()
    out.write(header)
    out.write(np.uint32(len(mesh.faces)).tobytes())
    out.write(triangles.tobytes())
    return out.getvalue()
