from __future__ import annotations

import base64
import copy
import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .cache import (
    CompileCacheEntry,
    MeshCacheEntry,
    UploadedHostFieldCacheEntry,
    field_preview_cache,
    hash_field_preview_request,
    hash_preview_request,
    hash_source,
    hash_uploaded_mesh_host_request,
    hash_uploaded_mesh_request,
    mesh_preview_cache,
    scene_compile_cache,
    uploaded_host_field_cache,
    uploaded_mesh_preview_cache,
    UploadedMeshCacheEntry,
)
from .dsl import DslError, compile_source_with_diagnostics
from .evaluator import (
    EvaluationError,
    ensure_scene_valid,
    evaluate_scene_field_with_backend,
    merge_parameter_values,
)
from .mesh_upload import (
    MeshUploadError,
    ParsedMesh,
    build_host_field,
    compose_hollow_lattice_field_with_backend,
    parse_mesh_bytes,
    validate_triangle_mesh,
)
from .meshing import (
    MeshData,
    MeshingError,
    build_mesh_with_backend,
    is_cuda_meshing_available,
    mesh_to_obj,
    mesh_to_stl,
)
from .models import (
    ComputeBackend,
    ComputePrecision,
    CompileSceneRequest,
    CompileSceneResponse,
    ExportMeshRequest,
    FieldPayload,
    GridConfig,
    MeshingMode,
    PreviewFieldRequest,
    PreviewFieldResponse,
    PreviewMeshRequest,
    PreviewMeshResponse,
    PreviewStats,
    PreviewWsRequest,
    PreviewWsResponse,
    QualityProfile,
    MeshBackend,
)

logger = logging.getLogger(__name__)

EVAL_TIMEOUT_SECONDS = 20.0
QUALITY_TO_RESOLUTION: dict[QualityProfile, int] = {
    "interactive": 64,
    "medium": 128,
    "high": 192,
    "ultra": 256,
}
MESH_QUALITY_TO_RESOLUTION: dict[QualityProfile, int] = {
    "interactive": 48,
    "medium": 72,
    "high": 96,
    "ultra": 128,
}
MESH_UPLOAD_MAX_BYTES = 200 * 1024 * 1024

app = FastAPI(title="SDF CAD", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/scene/compile", response_model=CompileSceneResponse)
def compile_scene(payload: CompileSceneRequest) -> CompileSceneResponse:
    source_hash = hash_source(payload.source)
    cached = scene_compile_cache.get(source_hash)
    if cached is not None:
        return CompileSceneResponse(scene_ir=cached.scene_ir, diagnostics=cached.diagnostics)

    try:
        scene_ir, diagnostics = compile_source_with_diagnostics(payload.source)
    except DslError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    scene_compile_cache.set(source_hash, CompileCacheEntry(scene_ir=scene_ir, diagnostics=diagnostics))
    return CompileSceneResponse(scene_ir=scene_ir, diagnostics=diagnostics)


def _resolve_grid(grid: GridConfig | None, quality_profile: QualityProfile) -> GridConfig:
    if grid is not None:
        return grid
    return GridConfig(resolution=QUALITY_TO_RESOLUTION[quality_profile])


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    if faces.size == 0 or vertices.size == 0:
        return normals

    for i0, i1, i2 in faces:
        v0 = vertices[int(i0)]
        v1 = vertices[int(i1)]
        v2 = vertices[int(i2)]
        n = np.cross(v1 - v0, v2 - v0)
        normals[int(i0)] += n
        normals[int(i1)] += n
        normals[int(i2)] += n

    lengths = np.linalg.norm(normals, axis=1)
    non_zero = lengths > 1e-12
    normals[non_zero] = normals[non_zero] / lengths[non_zero, None]
    normals[~non_zero] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normals


def _meshdata_from_parsed(parsed) -> MeshData:
    vertices = np.array(parsed.vertices, dtype=np.float64, copy=True)
    faces = np.array(parsed.faces, dtype=np.int32, copy=True)
    normals = _compute_vertex_normals(vertices, faces)
    return MeshData(vertices=vertices, faces=faces, normals=normals)


def _sample_field_trilinear(
    field: np.ndarray,
    bounds: list[list[float]],
    points: np.ndarray,
) -> np.ndarray:
    resolution = field.shape[0]
    dx = (bounds[0][1] - bounds[0][0]) / float(resolution - 1)
    dy = (bounds[1][1] - bounds[1][0]) / float(resolution - 1)
    dz = (bounds[2][1] - bounds[2][0]) / float(resolution - 1)

    tx = np.clip((points[:, 0] - bounds[0][0]) / dx, 0.0, resolution - 1.000001)
    ty = np.clip((points[:, 1] - bounds[1][0]) / dy, 0.0, resolution - 1.000001)
    tz = np.clip((points[:, 2] - bounds[2][0]) / dz, 0.0, resolution - 1.000001)

    x0 = np.floor(tx).astype(np.int32)
    y0 = np.floor(ty).astype(np.int32)
    z0 = np.floor(tz).astype(np.int32)
    x1 = np.minimum(x0 + 1, resolution - 1)
    y1 = np.minimum(y0 + 1, resolution - 1)
    z1 = np.minimum(z0 + 1, resolution - 1)

    fx = tx - x0
    fy = ty - y0
    fz = tz - z0

    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


def _strip_outer_surface(
    mesh: MeshData,
    host_sdf: np.ndarray,
    bounds: list[list[float]],
) -> MeshData:
    if mesh.faces.size == 0 or mesh.vertices.size == 0:
        return MeshData(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int32),
            normals=np.empty((0, 3), dtype=np.float64),
        )

    resolution = host_sdf.shape[0]
    spacing = np.array(
        [
            (bounds[0][1] - bounds[0][0]) / float(resolution - 1),
            (bounds[1][1] - bounds[1][0]) / float(resolution - 1),
            (bounds[2][1] - bounds[2][0]) / float(resolution - 1),
        ],
        dtype=np.float64,
    )
    outer_tol = float(np.max(spacing) * 0.5)

    vertex_host = _sample_field_trilinear(host_sdf, bounds, mesh.vertices)
    outer_face_mask = np.all(np.abs(vertex_host[mesh.faces]) <= outer_tol, axis=1)
    kept_faces = mesh.faces[np.logical_not(outer_face_mask)]
    if kept_faces.size == 0:
        return MeshData(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int32),
            normals=np.empty((0, 3), dtype=np.float64),
        )

    used_vertices, remap = np.unique(kept_faces.reshape(-1), return_inverse=True)
    new_vertices = mesh.vertices[used_vertices]
    new_faces = remap.reshape(-1, 3).astype(np.int32)
    new_normals = _compute_vertex_normals(new_vertices, new_faces)
    return MeshData(vertices=new_vertices, faces=new_faces, normals=new_normals)


def _merge_meshes(primary: MeshData, secondary: MeshData) -> MeshData:
    if secondary.faces.size == 0:
        return primary
    if primary.faces.size == 0:
        return secondary

    combined_vertices = np.vstack([primary.vertices, secondary.vertices]).astype(np.float64, copy=False)
    combined_normals = np.vstack([primary.normals, secondary.normals]).astype(np.float64, copy=False)
    shifted_faces = secondary.faces + primary.vertices.shape[0]
    combined_faces = np.vstack([primary.faces, shifted_faces]).astype(np.int32, copy=False)
    return MeshData(vertices=combined_vertices, faces=combined_faces, normals=combined_normals)


def _field_cache_key(
    scene_ir,
    merged_params: dict[str, float],
    grid: GridConfig,
    compute_precision: ComputePrecision,
    compute_backend: ComputeBackend,
) -> str:
    return hash_field_preview_request(
        scene_ir,
        merged_params,
        grid,
        compute_precision=compute_precision,
        compute_backend=compute_backend,
    )


def _encode_field(field: np.ndarray) -> str:
    # WebGL 3D textures expect x as the fastest-varying index, while our
    # NumPy field is shaped (x, y, z) in C-order (z fastest). Serialize in
    # Fortran order so the uploaded volume preserves xyz sampling semantics.
    payload = np.asarray(field, dtype=np.float32).tobytes(order="F")
    return base64.b64encode(payload).decode("ascii")


def _run_preview(
    scene_ir,
    param_values: dict[str, float],
    grid: GridConfig,
    compute_precision: ComputePrecision = "float32",
    compute_backend: ComputeBackend = "auto",
    mesh_backend: MeshBackend = "auto",
    meshing_mode: MeshingMode = "uniform",
) -> PreviewMeshResponse:
    ensure_scene_valid(scene_ir)
    merged_params = merge_parameter_values(scene_ir, param_values)

    cache_key = hash_preview_request(
        scene_ir,
        merged_params,
        grid,
        compute_precision=compute_precision,
        compute_backend=compute_backend,
        mesh_backend=mesh_backend,
        meshing_mode=meshing_mode,
    )
    cached_mesh = mesh_preview_cache.get(cache_key)
    if (
        cached_mesh is not None
        and mesh_backend != "cpu"
        and str(cached_mesh.stats.get("mesh_backend", "cpu")) == "cpu"
        and is_cuda_meshing_available()
    ):
        # Do not pin auto/cuda requests to stale CPU cache entries after CUDA
        # becomes available or backend bugs are fixed.
        cached_mesh = None
    if cached_mesh is not None:
        cached_stats = copy.deepcopy(cached_mesh.stats)
        cached_stats["cache_hit"] = True
        return PreviewMeshResponse(
            mesh={
                "vertices": cached_mesh.vertices,
                "indices": cached_mesh.indices,
                "normals": cached_mesh.normals,
            },
            stats=cached_stats,
        )

    eval_start = time.perf_counter()
    field_key = _field_cache_key(scene_ir, merged_params, grid, compute_precision, compute_backend)
    cached_field = field_preview_cache.get(field_key)
    field_cache_hit = cached_field is not None
    eval_backend: Literal["cpu", "cuda"] = "cpu"
    if cached_field is None:
        field, eval_backend = evaluate_scene_field_with_backend(
            scene_ir,
            merged_params,
            grid,
            compute_precision=compute_precision,
            compute_backend=compute_backend,
        )
        field_preview_cache.set(field_key, (field, eval_backend))
    else:
        field, cached_backend = cached_field
        eval_backend = "cuda" if cached_backend == "cuda" else "cpu"
    eval_ms = (time.perf_counter() - eval_start) * 1000.0
    if eval_ms > EVAL_TIMEOUT_SECONDS * 1000.0:
        raise HTTPException(status_code=408, detail="Field evaluation timeout exceeded")

    mesh_start = time.perf_counter()
    mesh_field = field.astype(np.float32, copy=False) if field.dtype == np.float16 else field
    mesh, mesh_backend_used = build_mesh_with_backend(
        mesh_field,
        grid.bounds,
        backend=mesh_backend,
        meshing_mode=meshing_mode,
    )
    mesh_ms = (time.perf_counter() - mesh_start) * 1000.0

    vertices = mesh.vertices.tolist()
    indices = mesh.faces.tolist()
    normals = mesh.normals.tolist()
    stats = PreviewStats(
        eval_ms=eval_ms,
        mesh_ms=mesh_ms,
        tri_count=int(mesh.faces.shape[0]),
        cache_hit=field_cache_hit,
        compute_precision=compute_precision,
        compute_backend=eval_backend,
        mesh_backend=mesh_backend_used,
        preview_mode="mesh",
    )

    mesh_preview_cache.set(
        cache_key,
        MeshCacheEntry(
            vertices=vertices,
            indices=indices,
            normals=normals,
            stats=stats.model_dump(),
        ),
    )

    return PreviewMeshResponse(
        mesh={"vertices": vertices, "indices": indices, "normals": normals},
        stats=stats,
    )


def _run_field_preview(
    scene_ir,
    param_values: dict[str, float],
    grid: GridConfig,
    compute_precision: ComputePrecision = "float32",
    compute_backend: ComputeBackend = "auto",
) -> PreviewFieldResponse:
    ensure_scene_valid(scene_ir)
    merged_params = merge_parameter_values(scene_ir, param_values)
    field_key = _field_cache_key(scene_ir, merged_params, grid, compute_precision, compute_backend)

    eval_start = time.perf_counter()
    cached_field = field_preview_cache.get(field_key)
    field_cache_hit = cached_field is not None
    eval_backend: Literal["cpu", "cuda"] = "cpu"
    if cached_field is None:
        field, eval_backend = evaluate_scene_field_with_backend(
            scene_ir,
            merged_params,
            grid,
            compute_precision=compute_precision,
            compute_backend=compute_backend,
        )
        field_preview_cache.set(field_key, (field, eval_backend))
    else:
        field, cached_backend = cached_field
        eval_backend = "cuda" if cached_backend == "cuda" else "cpu"
    eval_ms = (time.perf_counter() - eval_start) * 1000.0
    if eval_ms > EVAL_TIMEOUT_SECONDS * 1000.0:
        raise HTTPException(status_code=408, detail="Field evaluation timeout exceeded")

    stats = PreviewStats(
        eval_ms=eval_ms,
        mesh_ms=None,
        tri_count=0,
        voxel_count=int(field.size),
        cache_hit=field_cache_hit,
        compute_precision=compute_precision,
        compute_backend=eval_backend,
        mesh_backend="cpu",
        preview_mode="field",
    )
    payload = FieldPayload(
        encoding="f32-base64",
        resolution=int(field.shape[0]),
        bounds=[[float(axis[0]), float(axis[1])] for axis in grid.bounds],
        data=_encode_field(field),
    )
    return PreviewFieldResponse(field=payload, stats=stats)


async def _read_uploaded_mesh(file: UploadFile) -> tuple[bytes, str]:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in {".stl", ".obj"}:
        raise HTTPException(status_code=400, detail="Only .stl and .obj uploads are supported")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded mesh file is empty")
    if len(payload) > MESH_UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds the {MESH_UPLOAD_MAX_BYTES // (1024 * 1024)} MB limit",
        )
    return payload, extension


def _resolve_uploaded_host_field(
    *,
    file_bytes: bytes,
    extension: str,
    quality_profile: QualityProfile,
) -> tuple[ParsedMesh, list[list[float]], np.ndarray]:
    host_key = hash_uploaded_mesh_host_request(
        file_bytes=file_bytes,
        extension=extension,
        quality_profile=quality_profile,
    )
    cached = uploaded_host_field_cache.get(host_key)
    if cached is not None:
        parsed = ParsedMesh(
            vertices=np.array(cached.vertices, dtype=np.float64, copy=True),
            faces=np.array(cached.faces, dtype=np.int32, copy=True),
        )
        bounds = [[float(axis[0]), float(axis[1])] for axis in cached.bounds]
        return parsed, bounds, cached.host_sdf

    parsed = parse_mesh_bytes(file_bytes, extension)
    validate_triangle_mesh(parsed)
    resolution = MESH_QUALITY_TO_RESOLUTION[quality_profile]
    host = build_host_field(parsed, resolution=resolution)

    uploaded_host_field_cache.set(
        host_key,
        UploadedHostFieldCacheEntry(
            vertices=np.array(parsed.vertices, dtype=np.float64, copy=True),
            faces=np.array(parsed.faces, dtype=np.int32, copy=True),
            bounds=[[float(axis[0]), float(axis[1])] for axis in host.bounds],
            host_sdf=host.host_sdf,
        ),
    )
    return parsed, host.bounds, host.host_sdf


def _run_uploaded_mesh_preview(
    *,
    file_bytes: bytes,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    quality_profile: QualityProfile,
    compute_backend: ComputeBackend = "auto",
    mesh_backend: MeshBackend = "auto",
    meshing_mode: MeshingMode = "uniform",
) -> PreviewMeshResponse:
    cache_key = hash_uploaded_mesh_request(
        file_bytes=file_bytes,
        extension=extension,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        quality_profile=quality_profile,
        compute_backend=compute_backend,
        mesh_backend=mesh_backend,
        meshing_mode=meshing_mode,
    )
    cached_mesh = uploaded_mesh_preview_cache.get(cache_key)
    if (
        cached_mesh is not None
        and mesh_backend != "cpu"
        and str(cached_mesh.stats.get("mesh_backend", "cpu")) == "cpu"
        and is_cuda_meshing_available()
    ):
        # Do not pin auto/cuda requests to stale CPU cache entries after CUDA
        # becomes available or backend bugs are fixed.
        cached_mesh = None
    if cached_mesh is not None:
        cached_stats = copy.deepcopy(cached_mesh.stats)
        cached_stats["cache_hit"] = True
        cached_field = None
        if (
            cached_mesh.field_resolution is not None
            and cached_mesh.field_bounds is not None
            and cached_mesh.field_data is not None
        ):
            cached_field = FieldPayload(
                encoding="f32-base64",
                resolution=int(cached_mesh.field_resolution),
                bounds=[[float(axis[0]), float(axis[1])] for axis in cached_mesh.field_bounds],
                data=cached_mesh.field_data,
            )
        return PreviewMeshResponse(
            mesh={
                "vertices": cached_mesh.vertices,
                "indices": cached_mesh.indices,
                "normals": cached_mesh.normals,
            },
            stats=cached_stats,
            field=cached_field,
        )

    eval_start = time.perf_counter()
    host_start = time.perf_counter()
    parsed, bounds, host_sdf = _resolve_uploaded_host_field(
        file_bytes=file_bytes,
        extension=extension,
        quality_profile=quality_profile,
    )
    host_ms = (time.perf_counter() - host_start) * 1000.0
    compose_start = time.perf_counter()
    field, eval_backend_used = compose_hollow_lattice_field_with_backend(
        host_sdf,
        bounds,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        compute_backend=compute_backend,
    )
    compose_ms = (time.perf_counter() - compose_start) * 1000.0
    eval_ms = (time.perf_counter() - eval_start) * 1000.0

    mesh_start = time.perf_counter()
    mesher_start = time.perf_counter()
    generated, mesh_backend_used = build_mesh_with_backend(
        field,
        bounds,
        backend=mesh_backend,
        meshing_mode=meshing_mode,
    )
    mesher_ms = (time.perf_counter() - mesher_start) * 1000.0
    post_mesh_start = time.perf_counter()
    interior = _strip_outer_surface(generated, host_sdf, bounds)
    outer = _meshdata_from_parsed(parsed)
    mesh = _merge_meshes(outer, interior)
    post_mesh_ms = (time.perf_counter() - post_mesh_start) * 1000.0
    mesh_ms = (time.perf_counter() - mesh_start) * 1000.0
    logger.debug(
        "mesh_upload_timing host_ms=%.2f compose_ms=%.2f eval_ms=%.2f mesher_ms=%.2f post_mesh_ms=%.2f mesh_ms=%.2f",
        host_ms,
        compose_ms,
        eval_ms,
        mesher_ms,
        post_mesh_ms,
        mesh_ms,
    )

    vertices = mesh.vertices.tolist()
    indices = mesh.faces.tolist()
    normals = mesh.normals.tolist()
    field_payload = FieldPayload(
        encoding="f32-base64",
        resolution=int(field.shape[0]),
        bounds=[[float(axis[0]), float(axis[1])] for axis in bounds],
        data=_encode_field(field),
    )
    stats = PreviewStats(
        eval_ms=eval_ms,
        mesh_ms=mesh_ms,
        tri_count=int(mesh.faces.shape[0]),
        cache_hit=False,
        compute_backend=eval_backend_used,
        mesh_backend=mesh_backend_used,
        preview_mode="mesh",
    )

    uploaded_mesh_preview_cache.set(
        cache_key,
        UploadedMeshCacheEntry(
            vertices=vertices,
            indices=indices,
            normals=normals,
            stats=stats.model_dump(),
            field_resolution=int(field_payload.resolution),
            field_bounds=[[float(axis[0]), float(axis[1])] for axis in field_payload.bounds],
            field_data=field_payload.data,
        ),
    )
    return PreviewMeshResponse(
        mesh={"vertices": vertices, "indices": indices, "normals": normals},
        stats=stats,
        field=field_payload,
    )


@app.post("/api/v1/preview/mesh", response_model=PreviewMeshResponse)
def preview_mesh(payload: PreviewMeshRequest) -> PreviewMeshResponse:
    grid = _resolve_grid(payload.grid, payload.quality_profile)
    try:
        return _run_preview(
            payload.scene_ir,
            payload.parameter_values,
            grid,
            compute_precision=payload.compute_precision,
            compute_backend=payload.compute_backend,
            mesh_backend=payload.mesh_backend,
            meshing_mode=payload.meshing_mode,
        )
    except (DslError, EvaluationError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/preview/field", response_model=PreviewFieldResponse)
def preview_field(payload: PreviewFieldRequest) -> PreviewFieldResponse:
    grid = _resolve_grid(payload.grid, payload.quality_profile)
    try:
        return _run_field_preview(
            payload.scene_ir,
            payload.parameter_values,
            grid,
            compute_precision=payload.compute_precision,
            compute_backend=payload.compute_backend,
        )
    except (DslError, EvaluationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/export")
def export_mesh(payload: ExportMeshRequest) -> Response:
    grid = _resolve_grid(payload.grid, payload.quality_profile)

    try:
        preview = _run_preview(
            payload.scene_ir,
            payload.parameter_values,
            grid,
            compute_precision=payload.compute_precision,
            compute_backend=payload.compute_backend,
            mesh_backend=payload.mesh_backend,
            meshing_mode=payload.meshing_mode,
        )
    except (DslError, EvaluationError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mesh = MeshData(
        vertices=np.array(preview.mesh.vertices, dtype=np.float64),
        faces=np.array(preview.mesh.indices, dtype=np.int32),
        normals=np.array(preview.mesh.normals, dtype=np.float64),
    )

    if payload.format == "stl":
        content = mesh_to_stl(mesh)
        media_type = "model/stl"
        filename = "model.stl"
    else:
        content = mesh_to_obj(mesh)
        media_type = "text/plain"
        filename = "model.obj"

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=content, media_type=media_type, headers=headers)


@app.post("/api/v1/mesh/preview", response_model=PreviewMeshResponse)
async def preview_uploaded_mesh(
    file: UploadFile = File(...),
    shell_thickness: float = Form(...),
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"] = Form(...),
    lattice_pitch: float = Form(...),
    lattice_thickness: float = Form(...),
    lattice_phase: float = Form(0.0),
    quality_profile: QualityProfile = Form("medium"),
    compute_backend: ComputeBackend = Form("auto"),
    mesh_backend: MeshBackend = Form("auto"),
    meshing_mode: MeshingMode = Form("uniform"),
) -> PreviewMeshResponse:
    file_bytes, extension = await _read_uploaded_mesh(file)
    try:
        return _run_uploaded_mesh_preview(
            file_bytes=file_bytes,
            extension=extension,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            quality_profile=quality_profile,
            compute_backend=compute_backend,
            mesh_backend=mesh_backend,
            meshing_mode=meshing_mode,
        )
    except (MeshUploadError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/mesh/export")
async def export_uploaded_mesh(
    file: UploadFile = File(...),
    shell_thickness: float = Form(...),
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"] = Form(...),
    lattice_pitch: float = Form(...),
    lattice_thickness: float = Form(...),
    lattice_phase: float = Form(0.0),
    quality_profile: QualityProfile = Form("high"),
    compute_backend: ComputeBackend = Form("auto"),
    mesh_backend: MeshBackend = Form("auto"),
    meshing_mode: MeshingMode = Form("uniform"),
    format: Literal["stl", "obj"] = Form("stl"),
) -> Response:
    file_bytes, extension = await _read_uploaded_mesh(file)
    try:
        preview = _run_uploaded_mesh_preview(
            file_bytes=file_bytes,
            extension=extension,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            quality_profile=quality_profile,
            compute_backend=compute_backend,
            mesh_backend=mesh_backend,
            meshing_mode=meshing_mode,
        )
    except (MeshUploadError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mesh = MeshData(
        vertices=np.array(preview.mesh.vertices, dtype=np.float64),
        faces=np.array(preview.mesh.indices, dtype=np.int32),
        normals=np.array(preview.mesh.normals, dtype=np.float64),
    )

    if format == "stl":
        content = mesh_to_stl(mesh)
        media_type = "model/stl"
        filename = "mesh-lattice.stl"
    else:
        content = mesh_to_obj(mesh)
        media_type = "text/plain"
        filename = "mesh-lattice.obj"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=content, media_type=media_type, headers=headers)


@app.websocket("/api/v1/preview")
async def preview_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            raw_payload = await websocket.receive_json()
            try:
                payload = PreviewWsRequest.model_validate(raw_payload)
                target = QUALITY_TO_RESOLUTION[payload.quality_profile]
                base_grid = payload.base_grid or GridConfig()
                bounds = base_grid.bounds

                coarse_res = min(64, target)
                fine_res = target

                coarse_grid = GridConfig(bounds=bounds, resolution=coarse_res)
                fine_grid = GridConfig(bounds=bounds, resolution=fine_res)

                coarse_preview = _run_preview(
                    payload.scene_ir,
                    payload.parameter_values,
                    coarse_grid,
                    compute_precision=payload.compute_precision,
                    compute_backend=payload.compute_backend,
                    mesh_backend=payload.mesh_backend,
                    meshing_mode=payload.meshing_mode,
                )
                await websocket.send_json(
                    PreviewWsResponse(
                        phase="coarse", mesh=coarse_preview.mesh, stats=coarse_preview.stats
                    ).model_dump()
                )

                if fine_res != coarse_res:
                    fine_preview = _run_preview(
                        payload.scene_ir,
                        payload.parameter_values,
                        fine_grid,
                        compute_precision=payload.compute_precision,
                        compute_backend=payload.compute_backend,
                        mesh_backend=payload.mesh_backend,
                        meshing_mode=payload.meshing_mode,
                    )
                    await websocket.send_json(
                        PreviewWsResponse(
                            phase="fine", mesh=fine_preview.mesh, stats=fine_preview.stats
                        ).model_dump()
                    )
            except Exception as exc:
                await websocket.send_json(PreviewWsResponse(phase="error", error=str(exc)).model_dump())
    except WebSocketDisconnect:
        return
