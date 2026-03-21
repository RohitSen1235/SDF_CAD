from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import json
import logging
import struct
import time
from pathlib import Path
from typing import Literal

import numpy as np
from celery.result import AsyncResult
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

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
    compute_resolution_for_lattice_pitch,
    parse_mesh_bytes,
    validate_triangle_mesh,
)
from .gpu_program import compile_scene_program
from .gpu_memory import cleanup_gpu_memory
from .meshing import (
    MeshData,
    MeshingError,
    _compute_vertex_normals as compute_vertex_normals,
    build_mesh_with_backend,
    iter_obj_chunks,
    iter_stl_chunks,
    is_cuda_meshing_available,
)
from .models import (
    ComputeBackend,
    ComputePrecision,
    CompileSceneRequest,
    CompileSceneResponse,
    ExecutionMode,
    ExportMeshRequest,
    FieldPayload,
    GridConfig,
    JobAcceptedResponse,
    JobStatusResponse,
    MeshingMode,
    PreviewFieldRequest,
    PreviewFieldResponse,
    PreviewMeshRequest,
    PreviewMeshResponse,
    PreviewProgramRequest,
    PreviewProgramResponse,
    PreviewStats,
    PreviewWsRequest,
    PreviewWsResponse,
    ProgramCapabilities,
    QualityProfile,
    MeshBackend,
    MeshPayload,
    UploadedMeshPreviewWsRequest,
    UploadedMeshPreviewWsResponse,
)

logger = logging.getLogger(__name__)

try:
    import redis as _redis  # type: ignore

    REDIS_CLIENT_AVAILABLE = _redis is not None
except Exception:
    REDIS_CLIENT_AVAILABLE = False

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
# Hard cap for mesh-upload voxel grids. 1024^3 at float32 = 4 GB.
# The auto-resolution logic will clamp to this value and warn the caller.
MESH_UPLOAD_MAX_RESOLUTION = 1024
MESH_UPLOAD_MAX_BYTES = 200 * 1024 * 1024
AUTO_QUEUE_RESOLUTION_THRESHOLD = 128
AUTO_QUEUE_UPLOAD_BYTES_THRESHOLD = 8 * 1024 * 1024

app = FastAPI(title="SDF CAD", version="0.2.0")

if not REDIS_CLIENT_AVAILABLE:
    logger.info("Redis client not available; execution_mode='auto' will run inline.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-SDF-Stats",
        "X-SDF-Resolution",
        "X-SDF-Bounds",
        "X-SDF-Vertex-Count",
        "X-SDF-Face-Count",
    ],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/scene/compile", response_model=CompileSceneResponse)
async def compile_scene(payload: CompileSceneRequest) -> CompileSceneResponse:
    source_hash = hash_source(payload.source)
    cached = scene_compile_cache.get(source_hash)
    if cached is not None:
        return CompileSceneResponse(scene_ir=cached.scene_ir, diagnostics=cached.diagnostics)

    try:
        scene_ir, diagnostics = await asyncio.to_thread(compile_source_with_diagnostics, payload.source)
    except DslError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    scene_compile_cache.set(source_hash, CompileCacheEntry(scene_ir=scene_ir, diagnostics=diagnostics))
    return CompileSceneResponse(scene_ir=scene_ir, diagnostics=diagnostics)


def _resolve_grid(grid: GridConfig | None, quality_profile: QualityProfile) -> GridConfig:
    if grid is not None:
        return grid
    return GridConfig(resolution=QUALITY_TO_RESOLUTION[quality_profile])


def _resolve_mesh_resolution(
    quality_profile: QualityProfile,
    lattice_pitch: float,
    mesh_span: float,
    voxels_per_lattice_period: int = 6,
) -> tuple[int, bool]:
    """Compute the voxel resolution for a mesh upload.

    When lattice_pitch and mesh_span are available the resolution is derived
    from the lattice sampling requirement.  Otherwise the quality-profile
    fallback is used.

    Returns:
        (resolution, was_clamped) — was_clamped is True when the computed
        resolution exceeded MESH_UPLOAD_MAX_RESOLUTION and was reduced.
    """
    needed = compute_resolution_for_lattice_pitch(
        mesh_span, lattice_pitch, voxels_per_lattice_period
    )
    if needed > MESH_UPLOAD_MAX_RESOLUTION:
        logger.warning(
            "Computed mesh resolution %d exceeds hard cap %d; clamping. "
            "Consider increasing lattice_pitch or reducing voxels_per_lattice_period.",
            needed,
            MESH_UPLOAD_MAX_RESOLUTION,
        )
        return MESH_UPLOAD_MAX_RESOLUTION, True
    return needed, False


def _meshdata_from_parsed(parsed) -> MeshData:
    vertices = np.array(parsed.vertices, dtype=np.float64, copy=True)
    faces = np.array(parsed.faces, dtype=np.int32, copy=True)
    normals = compute_vertex_normals(vertices, faces)
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
    new_normals = compute_vertex_normals(new_vertices, new_faces)
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


def _pack_field_binary(field: np.ndarray) -> bytes:
    # Preserve the same axis-major semantics as _encode_field() used by
    # existing JSON/base64 payloads.
    return np.asarray(field, dtype=np.float32).tobytes(order="F")


def _encode_mesh_payload(mesh: MeshData) -> MeshPayload:
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    normals = np.ascontiguousarray(mesh.normals, dtype=np.float32)
    return MeshPayload(
        encoding="mesh-f32-u32-base64-v1",
        vertex_count=int(vertices.shape[0]),
        face_count=int(faces.shape[0]),
        vertices_b64=base64.b64encode(vertices.tobytes()).decode("ascii"),
        indices_b64=base64.b64encode(faces.tobytes()).decode("ascii"),
        normals_b64=base64.b64encode(normals.tobytes()).decode("ascii"),
    )


def _pack_mesh_binary(mesh: MeshData) -> bytes:
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    normals = np.ascontiguousarray(mesh.normals, dtype=np.float32)
    vertex_count = int(vertices.shape[0])
    face_count = int(faces.shape[0])
    header = b"SDFMESH1" + struct.pack("<II", vertex_count, face_count)
    return header + vertices.tobytes() + faces.tobytes() + normals.tobytes()


def _stats_header_value(stats: PreviewStats) -> str:
    return json.dumps(stats.model_dump(mode="json"), separators=(",", ":"))


def _bounds_header_value(bounds: list[list[float]]) -> str:
    compact = [[float(axis[0]), float(axis[1])] for axis in bounds]
    return json.dumps(compact, separators=(",", ":"))


def _should_queue_scene_job(grid: GridConfig, execution_mode: ExecutionMode) -> bool:
    if execution_mode == "queued":
        return True
    if execution_mode == "inline":
        return False
    if not REDIS_CLIENT_AVAILABLE:
        return False
    return int(grid.resolution) >= AUTO_QUEUE_RESOLUTION_THRESHOLD


def _should_queue_upload_job(file_size: int, quality_profile: QualityProfile, execution_mode: ExecutionMode) -> bool:
    if execution_mode == "queued":
        return True
    if execution_mode == "inline":
        return False
    if not REDIS_CLIENT_AVAILABLE:
        return False
    return file_size >= AUTO_QUEUE_UPLOAD_BYTES_THRESHOLD or quality_profile in {"high", "ultra"}


def _celery_state_to_status(state: str) -> Literal["queued", "running", "succeeded", "failed"]:
    if state == "SUCCESS":
        return "succeeded"
    if state == "STARTED":
        return "running"
    if state in {"FAILURE", "REVOKED"}:
        return "failed"
    return "queued"


def _job_response(task_id: str) -> JobAcceptedResponse:
    return JobAcceptedResponse(
        job_id=task_id,
        status="queued",
        status_url=f"/api/v1/jobs/{task_id}",
        result_url=f"/api/v1/jobs/{task_id}/result",
    )


def _enqueue_preview_mesh_job(payload: PreviewMeshRequest) -> JobAcceptedResponse:
    from .worker_tasks import preview_mesh_job

    try:
        task = preview_mesh_job.delay(payload.model_dump(mode="json"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}") from exc
    return _job_response(task.id)


def _enqueue_export_mesh_job(payload: ExportMeshRequest) -> JobAcceptedResponse:
    from .worker_tasks import export_mesh_job

    try:
        task = export_mesh_job.delay(payload.model_dump(mode="json"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}") from exc
    return _job_response(task.id)


def _enqueue_uploaded_preview_job(
    *,
    file_bytes: bytes,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    quality_profile: QualityProfile,
    compute_backend: ComputeBackend,
    mesh_backend: MeshBackend,
    meshing_mode: MeshingMode,
) -> JobAcceptedResponse:
    from .worker_tasks import preview_uploaded_mesh_job

    payload = {
        "file_data_b64": base64.b64encode(file_bytes).decode("ascii"),
        "extension": extension,
        "shell_thickness": shell_thickness,
        "lattice_type": lattice_type,
        "lattice_pitch": lattice_pitch,
        "lattice_thickness": lattice_thickness,
        "lattice_phase": lattice_phase,
        "quality_profile": quality_profile,
        "compute_backend": compute_backend,
        "mesh_backend": mesh_backend,
        "meshing_mode": meshing_mode,
    }
    try:
        task = preview_uploaded_mesh_job.delay(payload)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}") from exc
    return _job_response(task.id)


def _enqueue_uploaded_export_job(
    *,
    file_bytes: bytes,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    quality_profile: QualityProfile,
    compute_backend: ComputeBackend,
    mesh_backend: MeshBackend,
    meshing_mode: MeshingMode,
    format: Literal["stl", "obj"],
) -> JobAcceptedResponse:
    from .worker_tasks import export_uploaded_mesh_job

    payload = {
        "file_data_b64": base64.b64encode(file_bytes).decode("ascii"),
        "extension": extension,
        "shell_thickness": shell_thickness,
        "lattice_type": lattice_type,
        "lattice_pitch": lattice_pitch,
        "lattice_thickness": lattice_thickness,
        "lattice_phase": lattice_phase,
        "quality_profile": quality_profile,
        "compute_backend": compute_backend,
        "mesh_backend": mesh_backend,
        "meshing_mode": meshing_mode,
        "format": format,
    }
    try:
        task = export_uploaded_mesh_job.delay(payload)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}") from exc
    return _job_response(task.id)


def _run_preview(
    scene_ir,
    param_values: dict[str, float],
    grid: GridConfig,
    compute_precision: ComputePrecision = "float32",
    compute_backend: ComputeBackend = "auto",
    mesh_backend: MeshBackend = "auto",
    meshing_mode: MeshingMode = "uniform",
) -> PreviewMeshResponse:
    mesh, stats, mesh_payload = _run_preview_meshdata(
        scene_ir,
        param_values,
        grid,
        compute_precision=compute_precision,
        compute_backend=compute_backend,
        mesh_backend=mesh_backend,
        meshing_mode=meshing_mode,
    )
    return PreviewMeshResponse(mesh=mesh_payload, stats=stats)


def _run_preview_meshdata(
    scene_ir,
    param_values: dict[str, float],
    grid: GridConfig,
    compute_precision: ComputePrecision = "float32",
    compute_backend: ComputeBackend = "auto",
    mesh_backend: MeshBackend = "auto",
    meshing_mode: MeshingMode = "uniform",
) -> tuple[MeshData, PreviewStats, MeshPayload]:
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
        return (
            MeshData(
                vertices=np.array(cached_mesh.vertices, dtype=np.float64, copy=True),
                faces=np.array(cached_mesh.faces, dtype=np.int32, copy=True),
                normals=np.array(cached_mesh.normals, dtype=np.float64, copy=True),
            ),
            PreviewStats.model_validate(cached_stats),
            cached_mesh.mesh,
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

    mesh_payload = _encode_mesh_payload(mesh)
    mesh_preview_cache.set(
        cache_key,
        MeshCacheEntry(
            mesh=mesh_payload,
            vertices=np.array(mesh.vertices, dtype=np.float64, copy=True),
            faces=np.array(mesh.faces, dtype=np.int32, copy=True),
            normals=np.array(mesh.normals, dtype=np.float64, copy=True),
            stats=stats.model_dump(),
        ),
    )

    return mesh, stats, mesh_payload


def _run_field_preview(
    scene_ir,
    param_values: dict[str, float],
    grid: GridConfig,
    compute_precision: ComputePrecision = "float32",
    compute_backend: ComputeBackend = "auto",
) -> PreviewFieldResponse:
    field, bounds, stats = _run_field_preview_data(
        scene_ir,
        param_values,
        grid,
        compute_precision,
        compute_backend,
    )
    payload = FieldPayload(
        encoding="f32-base64",
        resolution=int(field.shape[0]),
        bounds=bounds,
        data=_encode_field(field),
    )
    return PreviewFieldResponse(field=payload, stats=stats)


def _run_field_preview_data(
    scene_ir,
    param_values: dict[str, float],
    grid: GridConfig,
    compute_precision: ComputePrecision = "float32",
    compute_backend: ComputeBackend = "auto",
) -> tuple[np.ndarray, list[list[float]], PreviewStats]:
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
    bounds = [[float(axis[0]), float(axis[1])] for axis in grid.bounds]
    return field, bounds, stats


def _stream_file_chunks(path: Path, chunk_size: int = 1024 * 1024, delete_after: bool = False):
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    finally:
        if delete_after:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


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
    resolution: int,
) -> tuple[ParsedMesh, list[list[float]], np.ndarray, int | None, list[tuple[int, int, int]] | None]:
    host_key = hash_uploaded_mesh_host_request(
        file_bytes=file_bytes,
        extension=extension,
        quality_profile=f"res_{resolution}",  # use resolution as cache key discriminator
    )
    cached = uploaded_host_field_cache.get(host_key)
    if cached is not None:
        parsed = ParsedMesh(
            vertices=np.array(cached.vertices, dtype=np.float64, copy=True),
            faces=np.array(cached.faces, dtype=np.int32, copy=True),
        )
        bounds = [[float(axis[0]), float(axis[1])] for axis in cached.bounds]
        active_blocks: list[tuple[int, int, int]] | None = None
        if cached.active_blocks:
            active_blocks = [(int(bx), int(by), int(bz)) for bx, by, bz in cached.active_blocks]
        return parsed, bounds, cached.host_sdf, cached.block_size, active_blocks

    parsed = parse_mesh_bytes(file_bytes, extension)
    validate_triangle_mesh(parsed)
    host = build_host_field(parsed, resolution=resolution)

    uploaded_host_field_cache.set(
        host_key,
        UploadedHostFieldCacheEntry(
            vertices=np.array(parsed.vertices, dtype=np.float64, copy=True),
            faces=np.array(parsed.faces, dtype=np.int32, copy=True),
            bounds=[[float(axis[0]), float(axis[1])] for axis in host.bounds],
            host_sdf=host.host_sdf,
            block_size=host.block_size,
            active_blocks=host.active_blocks,
        ),
    )
    return parsed, host.bounds, host.host_sdf, host.block_size, host.active_blocks


def _compute_mesh_upload_resolution(
    file_bytes: bytes,
    extension: str,
    lattice_pitch: float,
    quality_profile: QualityProfile,
    voxels_per_lattice_period: int,
) -> int:
    """Parse the mesh to get its span, then compute the required voxel resolution."""
    parsed = parse_mesh_bytes(file_bytes, extension)
    validate_triangle_mesh(parsed)
    extents = np.ptp(parsed.vertices, axis=0)
    mesh_span = float(np.max(extents))
    resolution, _ = _resolve_mesh_resolution(
        quality_profile, lattice_pitch, mesh_span, voxels_per_lattice_period
    )
    return resolution


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
    voxels_per_lattice_period: int = 6,
    compute_backend: ComputeBackend = "auto",
    mesh_backend: MeshBackend = "auto",
    meshing_mode: MeshingMode = "uniform",
) -> PreviewMeshResponse:
    _, stats, field_payload, mesh_payload = _run_uploaded_mesh_preview_meshdata(
        file_bytes=file_bytes,
        extension=extension,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        quality_profile=quality_profile,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
        mesh_backend=mesh_backend,
        meshing_mode=meshing_mode,
    )
    return PreviewMeshResponse(mesh=mesh_payload, stats=stats, field=field_payload)


def _run_uploaded_mesh_field_preview(
    *,
    file_bytes: bytes,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    quality_profile: QualityProfile,
    voxels_per_lattice_period: int = 6,
    compute_backend: ComputeBackend = "auto",
) -> PreviewFieldResponse:
    field, bounds, stats = _run_uploaded_mesh_field_preview_data(
        file_bytes=file_bytes,
        extension=extension,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        quality_profile=quality_profile,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
    )
    field_payload = FieldPayload(
        encoding="f32-base64",
        resolution=int(field.shape[0]),
        bounds=bounds,
        data=_encode_field(field),
    )
    return PreviewFieldResponse(field=field_payload, stats=stats)


def _run_uploaded_mesh_field_preview_data(
    *,
    file_bytes: bytes,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    quality_profile: QualityProfile,
    voxels_per_lattice_period: int = 6,
    compute_backend: ComputeBackend = "auto",
) -> tuple[np.ndarray, list[list[float]], PreviewStats]:
    eval_start = time.perf_counter()
    resolution = _compute_mesh_upload_resolution(
        file_bytes, extension, lattice_pitch, quality_profile, voxels_per_lattice_period
    )
    _, bounds, host_sdf, _, _ = _resolve_uploaded_host_field(
        file_bytes=file_bytes,
        extension=extension,
        resolution=resolution,
    )
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
    eval_ms = (time.perf_counter() - eval_start) * 1000.0
    out_bounds = [[float(axis[0]), float(axis[1])] for axis in bounds]
    stats = PreviewStats(
        eval_ms=eval_ms,
        mesh_ms=None,
        tri_count=0,
        voxel_count=int(field.size),
        cache_hit=False,
        compute_backend=eval_backend_used,
        mesh_backend="cpu",
        preview_mode="field",
    )
    return field, out_bounds, stats


def _run_uploaded_mesh_preview_meshdata(
    *,
    file_bytes: bytes,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    quality_profile: QualityProfile,
    voxels_per_lattice_period: int = 6,
    compute_backend: ComputeBackend = "auto",
    mesh_backend: MeshBackend = "auto",
    meshing_mode: MeshingMode = "uniform",
) -> tuple[MeshData, PreviewStats, FieldPayload, MeshPayload]:
    cache_key = hash_uploaded_mesh_request(
        file_bytes=file_bytes,
        extension=extension,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        quality_profile=f"{quality_profile}_vpp{voxels_per_lattice_period}",
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
        if cached_field is None:
            raise HTTPException(status_code=500, detail="Cached mesh entry is missing field payload")
        return (
            MeshData(
                vertices=np.array(cached_mesh.vertices, dtype=np.float64, copy=True),
                faces=np.array(cached_mesh.faces, dtype=np.int32, copy=True),
                normals=np.array(cached_mesh.normals, dtype=np.float64, copy=True),
            ),
            PreviewStats.model_validate(cached_stats),
            cached_field,
            cached_mesh.mesh,
        )

    eval_start = time.perf_counter()
    host_start = time.perf_counter()
    # Compute resolution from lattice pitch and mesh geometry.
    resolution = _compute_mesh_upload_resolution(
        file_bytes, extension, lattice_pitch, quality_profile, voxels_per_lattice_period
    )
    parsed, bounds, host_sdf, sparse_block_size, active_blocks = _resolve_uploaded_host_field(
        file_bytes=file_bytes,
        extension=extension,
        resolution=resolution,
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
        active_blocks=active_blocks,
        block_size=sparse_block_size,
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

    mesh_payload = _encode_mesh_payload(mesh)
    uploaded_mesh_preview_cache.set(
        cache_key,
        UploadedMeshCacheEntry(
            mesh=mesh_payload,
            vertices=np.array(mesh.vertices, dtype=np.float64, copy=True),
            faces=np.array(mesh.faces, dtype=np.int32, copy=True),
            normals=np.array(mesh.normals, dtype=np.float64, copy=True),
            stats=stats.model_dump(),
            field_resolution=int(field_payload.resolution),
            field_bounds=[[float(axis[0]), float(axis[1])] for axis in field_payload.bounds],
            field_data=field_payload.data,
        ),
    )
    return mesh, stats, field_payload, mesh_payload


@app.post("/api/v1/preview/mesh", response_model=PreviewMeshResponse | JobAcceptedResponse)
async def preview_mesh(payload: PreviewMeshRequest) -> PreviewMeshResponse | JobAcceptedResponse:
    grid = _resolve_grid(payload.grid, payload.quality_profile)
    if _should_queue_scene_job(grid, payload.execution_mode):
        try:
            return _enqueue_preview_mesh_job(payload)
        except HTTPException as exc:
            if payload.execution_mode == "queued" or exc.status_code != 503:
                raise
            logger.warning("Queue unavailable for preview_mesh auto mode; falling back inline: %s", exc.detail)
    try:
        result = await asyncio.to_thread(
            _run_preview,
            payload.scene_ir,
            payload.parameter_values,
            grid,
            payload.compute_precision,
            payload.compute_backend,
            payload.mesh_backend,
            payload.meshing_mode,
        )
        return result
    except (DslError, EvaluationError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="preview_mesh_inline")


@app.post("/api/v1/preview/mesh.binary", response_model=None)
async def preview_mesh_binary(payload: PreviewMeshRequest) -> Response:
    grid = _resolve_grid(payload.grid, payload.quality_profile)
    if payload.execution_mode == "queued":
        raise HTTPException(
            status_code=409,
            detail="Binary preview endpoint does not support explicit queued mode",
        )
    try:
        mesh, stats, _ = await asyncio.to_thread(
            _run_preview_meshdata,
            payload.scene_ir,
            payload.parameter_values,
            grid,
            payload.compute_precision,
            payload.compute_backend,
            payload.mesh_backend,
            payload.meshing_mode,
        )
        headers = {
            "X-SDF-Stats": _stats_header_value(stats),
            "X-SDF-Vertex-Count": str(int(mesh.vertices.shape[0])),
            "X-SDF-Face-Count": str(int(mesh.faces.shape[0])),
        }
        return Response(content=_pack_mesh_binary(mesh), media_type="application/octet-stream", headers=headers)
    except (DslError, EvaluationError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="preview_mesh_binary_inline")


@app.post("/api/v1/preview/field", response_model=PreviewFieldResponse)
async def preview_field(payload: PreviewFieldRequest) -> PreviewFieldResponse:
    grid = _resolve_grid(payload.grid, payload.quality_profile)
    try:
        return await asyncio.to_thread(
            _run_field_preview,
            payload.scene_ir,
            payload.parameter_values,
            grid,
            payload.compute_precision,
            payload.compute_backend,
        )
    except (DslError, EvaluationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="preview_field_inline")


@app.post("/api/v1/preview/field.binary", response_model=None)
async def preview_field_binary(payload: PreviewFieldRequest) -> Response:
    grid = _resolve_grid(payload.grid, payload.quality_profile)
    try:
        field, bounds, stats = await asyncio.to_thread(
            _run_field_preview_data,
            payload.scene_ir,
            payload.parameter_values,
            grid,
            payload.compute_precision,
            payload.compute_backend,
        )
        headers = {
            "X-SDF-Stats": _stats_header_value(stats),
            "X-SDF-Resolution": str(int(field.shape[0])),
            "X-SDF-Bounds": _bounds_header_value(bounds),
        }
        return Response(
            content=_pack_field_binary(field),
            media_type="application/octet-stream",
            headers=headers,
        )
    except (DslError, EvaluationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="preview_field_binary_inline")


@app.post("/api/v1/preview/program", response_model=PreviewProgramResponse)
async def preview_program(payload: PreviewProgramRequest) -> PreviewProgramResponse:
    grid = _resolve_grid(payload.grid, payload.quality_profile)
    try:
        ensure_scene_valid(payload.scene_ir)
        merged_params = merge_parameter_values(payload.scene_ir, payload.parameter_values)
        program, fallback_reason, compile_ms = await asyncio.to_thread(
            compile_scene_program,
            payload.scene_ir,
            merged_params,
            grid,
            payload.quality_profile,
        )
        if program is None:
            return PreviewProgramResponse(
                program=None,
                capabilities=ProgramCapabilities(analytic_supported=False, fallback_reason=fallback_reason),
                stats=PreviewStats(
                    eval_ms=0.0,
                    mesh_ms=None,
                    tri_count=0,
                    voxel_count=None,
                    cache_hit=False,
                    preview_mode="analytic_raymarch",
                    compile_ms=compile_ms,
                    program_bytes=0,
                    gpu_eval_mode="webgl2",
                    fallback_reason=fallback_reason,
                ),
            )

        program_bytes = len(program.glsl_sdf.encode("utf-8"))
        return PreviewProgramResponse(
            program=program,
            capabilities=ProgramCapabilities(analytic_supported=True, fallback_reason=None),
            stats=PreviewStats(
                eval_ms=0.0,
                mesh_ms=None,
                tri_count=0,
                voxel_count=None,
                cache_hit=False,
                preview_mode="analytic_raymarch",
                compile_ms=compile_ms,
                program_bytes=program_bytes,
                gpu_eval_mode="webgl2",
                fallback_reason=None,
            ),
        )
    except (DslError, EvaluationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/export", response_model=None)
async def export_mesh(payload: ExportMeshRequest) -> Response | JobAcceptedResponse:
    grid = _resolve_grid(payload.grid, payload.quality_profile)
    if _should_queue_scene_job(grid, payload.execution_mode):
        try:
            return _enqueue_export_mesh_job(payload)
        except HTTPException as exc:
            if payload.execution_mode == "queued" or exc.status_code != 503:
                raise
            logger.warning("Queue unavailable for export_mesh auto mode; falling back inline: %s", exc.detail)

    try:
        mesh, _, _ = await asyncio.to_thread(
            _run_preview_meshdata,
            payload.scene_ir,
            payload.parameter_values,
            grid,
            payload.compute_precision,
            payload.compute_backend,
            payload.mesh_backend,
            payload.meshing_mode,
        )
    except (DslError, EvaluationError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="export_mesh_inline")

    if payload.format == "stl":
        media_type = "model/stl"
        filename = "model.stl"
        body = iter_stl_chunks(mesh)
    else:
        media_type = "text/plain"
        filename = "model.obj"
        body = iter_obj_chunks(mesh)

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(body, media_type=media_type, headers=headers)


@app.post("/api/v1/jobs/preview-mesh", response_model=JobAcceptedResponse)
async def submit_preview_mesh_job(payload: PreviewMeshRequest) -> JobAcceptedResponse:
    return _enqueue_preview_mesh_job(payload)


@app.post("/api/v1/jobs/export", response_model=JobAcceptedResponse)
async def submit_export_job(payload: ExportMeshRequest) -> JobAcceptedResponse:
    return _enqueue_export_mesh_job(payload)


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    from .worker import celery_app

    result = AsyncResult(job_id, app=celery_app)
    status = _celery_state_to_status(result.state)
    task_name = str(getattr(result, "name", "") or "unknown")
    detail = None
    if status == "failed" and result.result is not None:
        detail = str(result.result)
    return JobStatusResponse(job_id=job_id, status=status, task_name=task_name, detail=detail)


@app.get("/api/v1/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> Response:
    from .worker import celery_app

    result = AsyncResult(job_id, app=celery_app)
    status = _celery_state_to_status(result.state)
    if status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Job is not completed yet")
    if status == "failed":
        raise HTTPException(status_code=400, detail=str(result.result))
    payload = result.result
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Unexpected job result payload")

    kind = str(payload.get("kind", ""))
    if kind in {"preview_mesh", "preview_uploaded_mesh"}:
        return Response(
            content=PreviewMeshResponse.model_validate(payload["payload"]).model_dump_json(),
            media_type="application/json",
        )
    if kind in {"export_mesh", "export_uploaded_mesh"}:
        file_path = Path(str(payload.get("file_path", "")))
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Export artifact no longer available")
        media_type = str(payload.get("media_type", "application/octet-stream"))
        filename = str(payload.get("filename", "artifact.bin"))
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return StreamingResponse(
            _stream_file_chunks(file_path, delete_after=True),
            media_type=media_type,
            headers=headers,
        )
    raise HTTPException(status_code=500, detail="Unsupported job result kind")


@app.post("/api/v1/mesh/preview", response_model=PreviewMeshResponse | JobAcceptedResponse)
async def preview_uploaded_mesh(
    file: UploadFile = File(...),
    shell_thickness: float = Form(...),
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"] = Form(...),
    lattice_pitch: float = Form(...),
    lattice_thickness: float = Form(...),
    lattice_phase: float = Form(0.0),
    quality_profile: QualityProfile = Form("medium"),
    voxels_per_lattice_period: int = Form(6),
    compute_backend: ComputeBackend = Form("auto"),
    mesh_backend: MeshBackend = Form("auto"),
    meshing_mode: MeshingMode = Form("uniform"),
    execution_mode: ExecutionMode = Form("auto"),
) -> PreviewMeshResponse | JobAcceptedResponse:
    file_bytes, extension = await _read_uploaded_mesh(file)
    if _should_queue_upload_job(len(file_bytes), quality_profile, execution_mode):
        try:
            return _enqueue_uploaded_preview_job(
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
        except HTTPException as exc:
            if execution_mode == "queued" or exc.status_code != 503:
                raise
            logger.warning(
                "Queue unavailable for preview_uploaded_mesh auto mode; falling back inline: %s",
                exc.detail,
            )
    try:
        result = await asyncio.to_thread(
            _run_uploaded_mesh_preview,
            file_bytes=file_bytes,
            extension=extension,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            quality_profile=quality_profile,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
            mesh_backend=mesh_backend,
            meshing_mode=meshing_mode,
        )
        return result
    except (MeshUploadError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="preview_uploaded_mesh_inline")


@app.post("/api/v1/mesh/preview.binary", response_model=None)
async def preview_uploaded_mesh_binary(
    file: UploadFile = File(...),
    shell_thickness: float = Form(...),
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"] = Form(...),
    lattice_pitch: float = Form(...),
    lattice_thickness: float = Form(...),
    lattice_phase: float = Form(0.0),
    quality_profile: QualityProfile = Form("medium"),
    voxels_per_lattice_period: int = Form(6),
    compute_backend: ComputeBackend = Form("auto"),
    mesh_backend: MeshBackend = Form("auto"),
    meshing_mode: MeshingMode = Form("uniform"),
    execution_mode: ExecutionMode = Form("auto"),
) -> Response:
    file_bytes, extension = await _read_uploaded_mesh(file)
    if execution_mode == "queued":
        raise HTTPException(
            status_code=409,
            detail="Binary preview endpoint does not support explicit queued mode",
        )
    try:
        mesh, stats, _, _ = await asyncio.to_thread(
            _run_uploaded_mesh_preview_meshdata,
            file_bytes=file_bytes,
            extension=extension,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            quality_profile=quality_profile,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
            mesh_backend=mesh_backend,
            meshing_mode=meshing_mode,
        )
        headers = {
            "X-SDF-Stats": _stats_header_value(stats),
            "X-SDF-Vertex-Count": str(int(mesh.vertices.shape[0])),
            "X-SDF-Face-Count": str(int(mesh.faces.shape[0])),
        }
        return Response(content=_pack_mesh_binary(mesh), media_type="application/octet-stream", headers=headers)
    except (MeshUploadError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="preview_uploaded_mesh_binary_inline")


@app.post("/api/v1/mesh/field", response_model=PreviewFieldResponse)
async def preview_uploaded_mesh_field(
    file: UploadFile = File(...),
    shell_thickness: float = Form(...),
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"] = Form(...),
    lattice_pitch: float = Form(...),
    lattice_thickness: float = Form(...),
    lattice_phase: float = Form(0.0),
    quality_profile: QualityProfile = Form("medium"),
    voxels_per_lattice_period: int = Form(6),
    compute_backend: ComputeBackend = Form("auto"),
) -> PreviewFieldResponse:
    file_bytes, extension = await _read_uploaded_mesh(file)
    try:
        result = await asyncio.to_thread(
            _run_uploaded_mesh_field_preview,
            file_bytes=file_bytes,
            extension=extension,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            quality_profile=quality_profile,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
        )
        return result
    except (MeshUploadError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="preview_uploaded_mesh_field_inline")


@app.post("/api/v1/mesh/field.binary", response_model=None)
async def preview_uploaded_mesh_field_binary(
    file: UploadFile = File(...),
    shell_thickness: float = Form(...),
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"] = Form(...),
    lattice_pitch: float = Form(...),
    lattice_thickness: float = Form(...),
    lattice_phase: float = Form(0.0),
    quality_profile: QualityProfile = Form("medium"),
    voxels_per_lattice_period: int = Form(6),
    compute_backend: ComputeBackend = Form("auto"),
) -> Response:
    file_bytes, extension = await _read_uploaded_mesh(file)
    try:
        field, bounds, stats = await asyncio.to_thread(
            _run_uploaded_mesh_field_preview_data,
            file_bytes=file_bytes,
            extension=extension,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            quality_profile=quality_profile,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
        )
        headers = {
            "X-SDF-Stats": _stats_header_value(stats),
            "X-SDF-Resolution": str(int(field.shape[0])),
            "X-SDF-Bounds": _bounds_header_value(bounds),
        }
        return Response(
            content=_pack_field_binary(field),
            media_type="application/octet-stream",
            headers=headers,
        )
    except (MeshUploadError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="preview_uploaded_mesh_field_binary_inline")


@app.post("/api/v1/mesh/export", response_model=None)
async def export_uploaded_mesh(
    file: UploadFile = File(...),
    shell_thickness: float = Form(...),
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"] = Form(...),
    lattice_pitch: float = Form(...),
    lattice_thickness: float = Form(...),
    lattice_phase: float = Form(0.0),
    quality_profile: QualityProfile = Form("high"),
    voxels_per_lattice_period: int = Form(6),
    compute_backend: ComputeBackend = Form("auto"),
    mesh_backend: MeshBackend = Form("auto"),
    meshing_mode: MeshingMode = Form("uniform"),
    format: Literal["stl", "obj"] = Form("stl"),
    execution_mode: ExecutionMode = Form("auto"),
) -> Response | JobAcceptedResponse:
    file_bytes, extension = await _read_uploaded_mesh(file)
    if _should_queue_upload_job(len(file_bytes), quality_profile, execution_mode):
        try:
            return _enqueue_uploaded_export_job(
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
                format=format,
            )
        except HTTPException as exc:
            if execution_mode == "queued" or exc.status_code != 503:
                raise
            logger.warning(
                "Queue unavailable for export_uploaded_mesh auto mode; falling back inline: %s",
                exc.detail,
            )
    try:
        mesh, _, _, _ = await asyncio.to_thread(
            _run_uploaded_mesh_preview_meshdata,
            file_bytes=file_bytes,
            extension=extension,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            quality_profile=quality_profile,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
            mesh_backend=mesh_backend,
            meshing_mode=meshing_mode,
        )
    except (MeshUploadError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="export_uploaded_mesh_inline")

    if format == "stl":
        media_type = "model/stl"
        filename = "mesh-lattice.stl"
        body = iter_stl_chunks(mesh)
    else:
        media_type = "text/plain"
        filename = "mesh-lattice.obj"
        body = iter_obj_chunks(mesh)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(body, media_type=media_type, headers=headers)


@app.websocket("/api/v1/mesh/preview/ws")
async def preview_uploaded_mesh_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        raw_payload = await websocket.receive_json()
        try:
            payload = UploadedMeshPreviewWsRequest.model_validate(raw_payload)
            extension = Path(payload.file_name or "").suffix.lower()
            if extension not in {".stl", ".obj"}:
                raise MeshUploadError("Only .stl and .obj uploads are supported")
            try:
                file_bytes = base64.b64decode(payload.file_data_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise MeshUploadError("Uploaded mesh payload is not valid base64") from exc
            if not file_bytes:
                raise MeshUploadError("Uploaded mesh file is empty")
            if len(file_bytes) > MESH_UPLOAD_MAX_BYTES:
                raise MeshUploadError(
                    f"Uploaded file exceeds the {MESH_UPLOAD_MAX_BYTES // (1024 * 1024)} MB limit"
                )

            voxels_per_lattice_period: int = getattr(payload, "voxels_per_lattice_period", 6)
            cache_key = hash_uploaded_mesh_request(
                file_bytes=file_bytes,
                extension=extension,
                shell_thickness=payload.shell_thickness,
                lattice_type=payload.lattice_type,
                lattice_pitch=payload.lattice_pitch,
                lattice_thickness=payload.lattice_thickness,
                lattice_phase=payload.lattice_phase,
                quality_profile=f"{payload.quality_profile}_vpp{voxels_per_lattice_period}",
                compute_backend=payload.compute_backend,
                mesh_backend=payload.mesh_backend,
                meshing_mode=payload.meshing_mode,
            )
            cached_mesh = uploaded_mesh_preview_cache.get(cache_key)
            if (
                cached_mesh is not None
                and payload.mesh_backend != "cpu"
                and str(cached_mesh.stats.get("mesh_backend", "cpu")) == "cpu"
                and is_cuda_meshing_available()
            ):
                cached_mesh = None

            if cached_mesh is not None:
                cached_stats = copy.deepcopy(cached_mesh.stats)
                cached_stats["cache_hit"] = True
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
                    await websocket.send_json(
                        UploadedMeshPreviewWsResponse(
                            phase="field",
                            field=cached_field,
                            stats=PreviewStats(
                                eval_ms=float(cached_stats.get("eval_ms", 0.0)),
                                mesh_ms=None,
                                tri_count=0,
                                voxel_count=int(cached_field.resolution**3),
                                cache_hit=True,
                                compute_backend=str(cached_stats.get("compute_backend", "cpu")),
                                mesh_backend="cpu",
                                preview_mode="field",
                            ),
                        ).model_dump()
                    )
                await websocket.send_json(
                    UploadedMeshPreviewWsResponse(
                        phase="mesh",
                        mesh=cached_mesh.mesh,
                        stats=PreviewStats.model_validate(cached_stats),
                    ).model_dump()
                )
                return

            # Phase 1: field evaluation — offloaded to thread pool to avoid
            # blocking the event loop (which would starve the websocket
            # keepalive ping and cause an AssertionError).
            def _compute_field_phase():
                try:
                    eval_start = time.perf_counter()
                    host_start = time.perf_counter()
                    ws_resolution = _compute_mesh_upload_resolution(
                        file_bytes,
                        extension,
                        payload.lattice_pitch,
                        payload.quality_profile,
                        voxels_per_lattice_period,
                    )
                    parsed, bounds, host_sdf, sparse_block_size, active_blocks = _resolve_uploaded_host_field(
                        file_bytes=file_bytes,
                        extension=extension,
                        resolution=ws_resolution,
                    )
                    host_ms = (time.perf_counter() - host_start) * 1000.0
                    compose_start = time.perf_counter()
                    field, eval_backend_used = compose_hollow_lattice_field_with_backend(
                        host_sdf,
                        bounds,
                        shell_thickness=payload.shell_thickness,
                        lattice_type=payload.lattice_type,
                        lattice_pitch=payload.lattice_pitch,
                        lattice_thickness=payload.lattice_thickness,
                        lattice_phase=payload.lattice_phase,
                        compute_backend=payload.compute_backend,
                    )
                    compose_ms = (time.perf_counter() - compose_start) * 1000.0
                    eval_ms = (time.perf_counter() - eval_start) * 1000.0
                    field_payload = FieldPayload(
                        encoding="f32-base64",
                        resolution=int(field.shape[0]),
                        bounds=[[float(axis[0]), float(axis[1])] for axis in bounds],
                        data=_encode_field(field),
                    )
                    logger.debug(
                        "mesh_upload field_phase host_ms=%.2f compose_ms=%.2f eval_ms=%.2f",
                        host_ms, compose_ms, eval_ms,
                    )
                    return (
                        parsed,
                        bounds,
                        host_sdf,
                        sparse_block_size,
                        active_blocks,
                        field,
                        eval_backend_used,
                        eval_ms,
                        field_payload,
                    )
                finally:
                    cleanup_gpu_memory(reason="uploaded_mesh_ws_field_phase")

            (
                parsed,
                bounds,
                host_sdf,
                sparse_block_size,
                active_blocks,
                field,
                eval_backend_used,
                eval_ms,
                field_payload,
            ) = (
                await asyncio.to_thread(_compute_field_phase)
            )

            field_stats = PreviewStats(
                eval_ms=eval_ms,
                mesh_ms=None,
                tri_count=0,
                voxel_count=int(field.size),
                cache_hit=False,
                compute_backend=eval_backend_used,
                mesh_backend="cpu",
                preview_mode="field",
            )
            await websocket.send_json(
                UploadedMeshPreviewWsResponse(
                    phase="field",
                    field=field_payload,
                    stats=field_stats,
                ).model_dump()
            )

            # Phase 2: meshing — also offloaded to thread pool.
            def _compute_mesh_phase():
                try:
                    mesh_start = time.perf_counter()
                    generated, mesh_backend_used = build_mesh_with_backend(
                        field,
                        bounds,
                        backend=payload.mesh_backend,
                        meshing_mode=payload.meshing_mode,
                        active_blocks=active_blocks,
                        block_size=sparse_block_size,
                    )
                    interior = _strip_outer_surface(generated, host_sdf, bounds)
                    outer = _meshdata_from_parsed(parsed)
                    mesh = _merge_meshes(outer, interior)
                    mesh_ms = (time.perf_counter() - mesh_start) * 1000.0
                    mesh_payload = _encode_mesh_payload(mesh)
                    tri_count = int(mesh.faces.shape[0])
                    logger.debug("mesh_upload mesh_phase mesh_ms=%.2f tri_count=%d", mesh_ms, tri_count)
                    return mesh_payload, mesh, mesh_ms, mesh_backend_used, tri_count
                finally:
                    cleanup_gpu_memory(reason="uploaded_mesh_ws_mesh_phase")

            mesh_payload, mesh_np, mesh_ms, mesh_backend_used, tri_count = (
                await asyncio.to_thread(_compute_mesh_phase)
            )

            stats = PreviewStats(
                eval_ms=eval_ms,
                mesh_ms=mesh_ms,
                tri_count=tri_count,
                cache_hit=False,
                compute_backend=eval_backend_used,
                mesh_backend=mesh_backend_used,
                preview_mode="mesh",
            )
            uploaded_mesh_preview_cache.set(
                cache_key,
                UploadedMeshCacheEntry(
                    mesh=mesh_payload,
                    vertices=np.array(mesh_np.vertices, dtype=np.float64, copy=True),
                    faces=np.array(mesh_np.faces, dtype=np.int32, copy=True),
                    normals=np.array(mesh_np.normals, dtype=np.float64, copy=True),
                    stats=stats.model_dump(),
                    field_resolution=int(field_payload.resolution),
                    field_bounds=[[float(axis[0]), float(axis[1])] for axis in field_payload.bounds],
                    field_data=field_payload.data,
                ),
            )
            await websocket.send_json(
                UploadedMeshPreviewWsResponse(
                    phase="mesh",
                    mesh=mesh_payload,
                    stats=stats,
                ).model_dump()
            )
            return
        except Exception as exc:
            await websocket.send_json(
                UploadedMeshPreviewWsResponse(phase="error", error=str(exc)).model_dump()
            )
            return
    except WebSocketDisconnect:
        return


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

                coarse_preview = await asyncio.to_thread(
                    _run_preview,
                    payload.scene_ir,
                    payload.parameter_values,
                    coarse_grid,
                    compute_precision=payload.compute_precision,
                    compute_backend=payload.compute_backend,
                    mesh_backend=payload.mesh_backend,
                    meshing_mode=payload.meshing_mode,
                )
                cleanup_gpu_memory(reason="preview_ws_coarse")
                await websocket.send_json(
                    PreviewWsResponse(
                        phase="coarse", mesh=coarse_preview.mesh, stats=coarse_preview.stats
                    ).model_dump()
                )

                if fine_res != coarse_res:
                    fine_preview = await asyncio.to_thread(
                        _run_preview,
                        payload.scene_ir,
                        payload.parameter_values,
                        fine_grid,
                        compute_precision=payload.compute_precision,
                        compute_backend=payload.compute_backend,
                        mesh_backend=payload.mesh_backend,
                        meshing_mode=payload.meshing_mode,
                    )
                    cleanup_gpu_memory(reason="preview_ws_fine")
                    await websocket.send_json(
                        PreviewWsResponse(
                            phase="fine", mesh=fine_preview.mesh, stats=fine_preview.stats
                        ).model_dump()
                    )
            except Exception as exc:
                await websocket.send_json(PreviewWsResponse(phase="error", error=str(exc)).model_dump())
    except WebSocketDisconnect:
        return
