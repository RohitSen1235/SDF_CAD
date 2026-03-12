from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from skimage.measure import marching_cubes

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False


def _decode_classic_tri_table() -> np.ndarray:
    from skimage.measure import _marching_cubes_lewiner_luts as mcluts

    shape, text = mcluts.CASESCLASSIC
    byts = base64.decodebytes(text.encode("utf-8"))
    tri_table = np.frombuffer(byts, dtype=np.int8).reshape(shape).copy()
    return tri_table


def _build_edge_mask_and_tri_count(tri_table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edge_mask = np.zeros((256,), dtype=np.int32)
    tri_count = np.zeros((256,), dtype=np.int32)
    for case_idx in range(256):
        row = tri_table[case_idx]
        used_mask = 0
        used_edges = 0
        for edge_id in row:
            if edge_id < 0:
                break
            used_mask |= 1 << int(edge_id)
            used_edges += 1
        edge_mask[case_idx] = used_mask
        tri_count[case_idx] = used_edges // 3
    return edge_mask, tri_count


_TRI_TABLE_HOST = _decode_classic_tri_table()
_EDGE_MASK_HOST, _TRI_COUNT_HOST = _build_edge_mask_and_tri_count(_TRI_TABLE_HOST)
# Corner order follows classic marching-cubes conventions.
_CORNER_OFFSETS_HOST = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int32,
)
_EDGE_ENDPOINTS_HOST = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ],
    dtype=np.int32,
)


def _cuda_runtime_ready() -> bool:
    if not CUPY_AVAILABLE or cp is None:
        return False
    try:
        return bool(cp.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


def _cuda_meshing_available() -> bool:
    # Evaluate runtime availability at call-time so backend selection does not
    # get pinned to import-time device visibility.
    return _cuda_runtime_ready()


def is_cuda_meshing_available() -> bool:
    return _cuda_meshing_available()


CUDA_MARCHING_CUBES_PROVIDER: Literal["builtin", "none"] = (
    "builtin" if _cuda_meshing_available() else "none"
)
CUDA_MARCHING_CUBES_AVAILABLE = CUDA_MARCHING_CUBES_PROVIDER != "none"


@lru_cache(maxsize=1)
def _cuda_luts() -> tuple["cp.ndarray", "cp.ndarray", "cp.ndarray", "cp.ndarray", "cp.ndarray"]:  # type: ignore[name-defined]
    if cp is None:
        raise MeshingError("CUDA meshing requested but CuPy is unavailable")
    tri_table = cp.asarray(_TRI_TABLE_HOST.reshape(-1), dtype=cp.int8)
    edge_mask = cp.asarray(_EDGE_MASK_HOST, dtype=cp.int32)
    tri_count = cp.asarray(_TRI_COUNT_HOST, dtype=cp.int32)
    edge_endpoints = cp.asarray(_EDGE_ENDPOINTS_HOST.reshape(-1), dtype=cp.int32)
    corner_offsets = cp.asarray(_CORNER_OFFSETS_HOST.reshape(-1), dtype=cp.int32)
    return tri_table, edge_mask, tri_count, edge_endpoints, corner_offsets


_MC_COUNT_KERNEL = r"""
extern "C" __global__
void mc_count(
    const float* field,
    const int nx,
    const int ny,
    const int nz,
    int* tri_counts,
    const int* tri_count_lut
) {
    int cells_x = nx - 1;
    int cells_y = ny - 1;
    int cells_z = nz - 1;
    int n_cells = cells_x * cells_y * cells_z;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n_cells) return;

    int stride_yz = cells_y * cells_z;
    int i = tid / stride_yz;
    int rem = tid - i * stride_yz;
    int j = rem / cells_z;
    int k = rem - j * cells_z;

    int idx000 = (i * ny + j) * nz + k;
    int idx100 = ((i + 1) * ny + j) * nz + k;
    int idx110 = ((i + 1) * ny + (j + 1)) * nz + k;
    int idx010 = (i * ny + (j + 1)) * nz + k;
    int idx001 = idx000 + 1;
    int idx101 = idx100 + 1;
    int idx111 = idx110 + 1;
    int idx011 = idx010 + 1;

    int cube_index = 0;
    if (field[idx000] < 0.0f) cube_index |= 1;
    if (field[idx100] < 0.0f) cube_index |= 2;
    if (field[idx110] < 0.0f) cube_index |= 4;
    if (field[idx010] < 0.0f) cube_index |= 8;
    if (field[idx001] < 0.0f) cube_index |= 16;
    if (field[idx101] < 0.0f) cube_index |= 32;
    if (field[idx111] < 0.0f) cube_index |= 64;
    if (field[idx011] < 0.0f) cube_index |= 128;

    tri_counts[tid] = tri_count_lut[cube_index];
}
"""


_MC_GENERATE_KERNEL = r"""
extern "C" __global__
void mc_generate(
    const float* field,
    const int nx,
    const int ny,
    const int nz,
    const float ox,
    const float oy,
    const float oz,
    const float dx,
    const float dy,
    const float dz,
    const int* tri_offsets,
    const signed char* tri_table,
    const int* edge_mask_lut,
    const int* edge_endpoints,
    const int* corner_offsets,
    float* vertices,
    int* faces
) {
    int cells_x = nx - 1;
    int cells_y = ny - 1;
    int cells_z = nz - 1;
    int n_cells = cells_x * cells_y * cells_z;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n_cells) return;

    int stride_yz = cells_y * cells_z;
    int i = tid / stride_yz;
    int rem = tid - i * stride_yz;
    int j = rem / cells_z;
    int k = rem - j * cells_z;

    int idx000 = (i * ny + j) * nz + k;
    int idx100 = ((i + 1) * ny + j) * nz + k;
    int idx110 = ((i + 1) * ny + (j + 1)) * nz + k;
    int idx010 = (i * ny + (j + 1)) * nz + k;
    int idx001 = idx000 + 1;
    int idx101 = idx100 + 1;
    int idx111 = idx110 + 1;
    int idx011 = idx010 + 1;

    float cv[8];
    cv[0] = field[idx000];
    cv[1] = field[idx100];
    cv[2] = field[idx110];
    cv[3] = field[idx010];
    cv[4] = field[idx001];
    cv[5] = field[idx101];
    cv[6] = field[idx111];
    cv[7] = field[idx011];

    int cube_index = 0;
    if (cv[0] < 0.0f) cube_index |= 1;
    if (cv[1] < 0.0f) cube_index |= 2;
    if (cv[2] < 0.0f) cube_index |= 4;
    if (cv[3] < 0.0f) cube_index |= 8;
    if (cv[4] < 0.0f) cube_index |= 16;
    if (cv[5] < 0.0f) cube_index |= 32;
    if (cv[6] < 0.0f) cube_index |= 64;
    if (cv[7] < 0.0f) cube_index |= 128;

    int tri_count = 0;
    for (int t = 0; t < 16; ++t) {
        signed char edge = tri_table[cube_index * 16 + t];
        if (edge < 0) break;
        ++tri_count;
    }
    tri_count /= 3;
    if (tri_count == 0) return;

    int edge_mask = edge_mask_lut[cube_index];
    float vertlist[12][3];

    for (int edge = 0; edge < 12; ++edge) {
        if ((edge_mask & (1 << edge)) == 0) {
            continue;
        }

        int c0 = edge_endpoints[edge * 2 + 0];
        int c1 = edge_endpoints[edge * 2 + 1];

        int c0x = corner_offsets[c0 * 3 + 0];
        int c0y = corner_offsets[c0 * 3 + 1];
        int c0z = corner_offsets[c0 * 3 + 2];
        int c1x = corner_offsets[c1 * 3 + 0];
        int c1y = corner_offsets[c1 * 3 + 1];
        int c1z = corner_offsets[c1 * 3 + 2];

        float v0 = cv[c0];
        float v1 = cv[c1];
        float den = v1 - v0;
        float interp = ((den < 1e-12f) && (den > -1e-12f)) ? 0.5f : (-v0) / den;
        if (interp < 0.0f) interp = 0.0f;
        if (interp > 1.0f) interp = 1.0f;

        float gx = ((float)i + (float)c0x) + interp * (float)(c1x - c0x);
        float gy = ((float)j + (float)c0y) + interp * (float)(c1y - c0y);
        float gz = ((float)k + (float)c0z) + interp * (float)(c1z - c0z);

        vertlist[edge][0] = ox + gx * dx;
        vertlist[edge][1] = oy + gy * dy;
        vertlist[edge][2] = oz + gz * dz;
    }

    int tri_start = tri_offsets[tid];
    for (int tri_local = 0; tri_local < tri_count; ++tri_local) {
        int e0 = (int)tri_table[cube_index * 16 + tri_local * 3 + 0];
        int e1 = (int)tri_table[cube_index * 16 + tri_local * 3 + 1];
        int e2 = (int)tri_table[cube_index * 16 + tri_local * 3 + 2];
        if (e0 < 0 || e1 < 0 || e2 < 0) break;

        int tri_id = tri_start + tri_local;
        int vi = tri_id * 3;

        float v0x = vertlist[e0][0];
        float v0y = vertlist[e0][1];
        float v0z = vertlist[e0][2];
        float v1x = vertlist[e1][0];
        float v1y = vertlist[e1][1];
        float v1z = vertlist[e1][2];
        float v2x = vertlist[e2][0];
        float v2y = vertlist[e2][1];
        float v2z = vertlist[e2][2];

        int out0 = vi * 3;
        int out1 = (vi + 1) * 3;
        int out2 = (vi + 2) * 3;
        vertices[out0 + 0] = v0x;
        vertices[out0 + 1] = v0y;
        vertices[out0 + 2] = v0z;
        vertices[out1 + 0] = v1x;
        vertices[out1 + 1] = v1y;
        vertices[out1 + 2] = v1z;
        vertices[out2 + 0] = v2x;
        vertices[out2 + 1] = v2y;
        vertices[out2 + 2] = v2z;

        int face_base = tri_id * 3;
        faces[face_base + 0] = vi + 0;
        faces[face_base + 1] = vi + 1;
        faces[face_base + 2] = vi + 2;
    }
}
"""


@lru_cache(maxsize=1)
def _cuda_kernels() -> tuple["cp.RawKernel", "cp.RawKernel"]:  # type: ignore[name-defined]
    if cp is None:
        raise MeshingError("CUDA meshing requested but CuPy is unavailable")
    count_kernel = cp.RawKernel(_MC_COUNT_KERNEL, "mc_count")
    gen_kernel = cp.RawKernel(_MC_GENERATE_KERNEL, "mc_generate")
    return count_kernel, gen_kernel


class MeshingError(ValueError):
    pass


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    if faces.size == 0 or vertices.size == 0:
        return normals

    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for axis in range(3):
        np.add.at(normals[:, axis], faces[:, 0], face_normals[:, axis])
        np.add.at(normals[:, axis], faces[:, 1], face_normals[:, axis])
        np.add.at(normals[:, axis], faces[:, 2], face_normals[:, axis])

    lengths = np.linalg.norm(normals, axis=1)
    non_zero = lengths > 1e-12
    normals[non_zero] = normals[non_zero] / lengths[non_zero, None]
    normals[~non_zero] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normals



def _resolve_mesh_backend(requested: Literal["auto", "cpu", "cuda"]) -> Literal["cpu", "cuda"]:
    cuda_ready = _cuda_meshing_available()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if cuda_ready else "cpu"
    return "cuda" if cuda_ready else "cpu"



def _mesh_single_cpu(field: np.ndarray, bounds: list[list[float]]) -> MeshData:
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


def _mesh_single_cuda(field: np.ndarray, bounds: list[list[float]]) -> MeshData:
    if not _cuda_meshing_available() or cp is None:
        raise MeshingError("CUDA meshing requested but CUDA runtime is unavailable")

    resolution = field.shape[0]
    dx = (bounds[0][1] - bounds[0][0]) / float(resolution - 1)
    dy = (bounds[1][1] - bounds[1][0]) / float(resolution - 1)
    dz = (bounds[2][1] - bounds[2][0]) / float(resolution - 1)
    origin = np.array([bounds[0][0], bounds[1][0], bounds[2][0]], dtype=np.float32)

    volume = cp.asarray(np.ascontiguousarray(field), dtype=cp.float32)
    nx, ny, nz = (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))
    if nx < 2 or ny < 2 or nz < 2:
        raise MeshingError("Field resolution must be at least 2 along all axes")

    cell_count = (nx - 1) * (ny - 1) * (nz - 1)
    if cell_count <= 0:
        raise MeshingError("No cells available for marching cubes")

    tri_table, edge_mask, tri_count_lut, edge_endpoints, corner_offsets = _cuda_luts()
    count_kernel, gen_kernel = _cuda_kernels()

    tri_counts = cp.empty((cell_count,), dtype=cp.int32)
    threads = 256
    blocks = (cell_count + threads - 1) // threads
    count_kernel(
        (blocks,),
        (threads,),
        (
            volume,
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            tri_counts,
            tri_count_lut,
        ),
    )

    total_tris = int(cp.sum(tri_counts, dtype=cp.int64).item())
    if total_tris <= 0:
        raise MeshingError(
            "No zero level-set detected in current grid bounds. Expand bounds or change parameters."
        )

    tri_offsets = cp.zeros((cell_count,), dtype=cp.int32)
    if cell_count > 1:
        tri_offsets[1:] = cp.cumsum(tri_counts[:-1], dtype=cp.int32)

    vertices_flat = cp.empty((total_tris * 9,), dtype=cp.float32)
    faces_flat = cp.empty((total_tris * 3,), dtype=cp.int32)
    gen_kernel(
        (blocks,),
        (threads,),
        (
            volume,
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.float32(origin[0]),
            np.float32(origin[1]),
            np.float32(origin[2]),
            np.float32(dx),
            np.float32(dy),
            np.float32(dz),
            tri_offsets,
            tri_table,
            edge_mask,
            edge_endpoints,
            corner_offsets,
            vertices_flat,
            faces_flat,
        ),
    )

    vertices = cp.asnumpy(vertices_flat).reshape(-1, 3).astype(np.float64, copy=False)
    faces = cp.asnumpy(faces_flat).reshape(-1, 3).astype(np.int32, copy=False)
    normals = _compute_vertex_normals(vertices, faces)
    return MeshData(vertices=vertices, faces=faces, normals=normals)



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
 
def _mesh_adaptive_cpu(field: np.ndarray, bounds: list[list[float]], block_size: int = 28) -> MeshData:
    resolution = field.shape[0]
    dx = (bounds[0][1] - bounds[0][0]) / float(resolution - 1)
    dy = (bounds[1][1] - bounds[1][0]) / float(resolution - 1)
    dz = (bounds[2][1] - bounds[2][0]) / float(resolution - 1)

    origin = np.array([bounds[0][0], bounds[1][0], bounds[2][0]], dtype=np.float64)
    step = max(8, min(block_size, resolution - 1))

    vertex_map: dict[tuple[int, int, int], int] = {}
    vertices: list[list[float]] = []
    normals_accum: list[np.ndarray] = []
    faces: list[list[int]] = []
    quant = 1e-6

    for i0 in range(0, resolution - 1, step):
        i1 = min(resolution - 1, i0 + step)
        for j0 in range(0, resolution - 1, step):
            j1 = min(resolution - 1, j0 + step)
            for k0 in range(0, resolution - 1, step):
                k1 = min(resolution - 1, k0 + step)
                block = field[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
                if block.size == 0:
                    continue
                bmin = float(np.min(block))
                bmax = float(np.max(block))
                if not (bmin <= 0.0 <= bmax):
                    continue

                try:
                    local_v, local_f, local_n, _ = marching_cubes(
                        block,
                        level=0.0,
                        spacing=(dx, dy, dz),
                        allow_degenerate=False,
                    )
                except ValueError:
                    continue

                local_v = local_v + origin + np.array([i0 * dx, j0 * dy, k0 * dz], dtype=np.float64)
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


def build_mesh(
    field: np.ndarray,
    bounds: list[list[float]],
    chunk_size: int | None = None,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    meshing_mode: Literal["uniform", "adaptive"] = "uniform",
) -> MeshData:
    mesh, _ = build_mesh_with_backend(
        field,
        bounds,
        chunk_size=chunk_size,
        backend=backend,
        meshing_mode=meshing_mode,
    )
    return mesh


def build_mesh_with_backend(
    field: np.ndarray,
    bounds: list[list[float]],
    chunk_size: int | None = None,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    meshing_mode: Literal["uniform", "adaptive"] = "uniform",
) -> tuple[MeshData, Literal["cpu", "cuda"]]:
    min_val = float(np.min(field))
    max_val = float(np.max(field))
    if not (min_val <= 0.0 <= max_val):
        raise MeshingError(
            "No zero level-set detected in current grid bounds. Expand bounds or change parameters."
        )

    if meshing_mode == "adaptive":
        try:
            return _mesh_adaptive_cpu(field, bounds), "cpu"
        except MemoryError:
            return _mesh_chunked(field, bounds, chunk_size=chunk_size or 80), "cpu"

    resolved_backend = _resolve_mesh_backend(backend)
    if resolved_backend == "cuda":
        try:
            return _mesh_single_cuda(field, bounds), "cuda"
        except Exception as exc:
            if backend == "cuda":
                raise MeshingError(f"CUDA meshing failed: {exc}") from exc

    try:
        return _mesh_single_cpu(field, bounds), "cpu"
    except MemoryError:
        return _mesh_chunked(field, bounds, chunk_size=chunk_size or 80), "cpu"



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
