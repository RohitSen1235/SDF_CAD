from __future__ import annotations

import gc
import asyncio
import base64
import binascii
import copy
import hashlib
import json
import logging
import os
import struct
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from celery.result import AsyncResult
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .cache import (
    CompileCacheEntry,
    MeshCacheEntry,
    UploadedFieldPreviewTraceEntry,
    UploadedComposedFieldCacheEntry,
    UploadedHostFieldCacheEntry,
    UploadedMeshMetadataCacheEntry,
    UploadedMeshPreprocessTimingEntry,
    field_preview_cache,
    hash_field_preview_request,
    hash_preview_request,
    hash_uploaded_mesh_metadata_request,
    hash_source,
    hash_uploaded_mesh_field_request,
    hash_uploaded_mesh_host_request,
    hash_uploaded_mesh_request,
    mesh_preview_cache,
    scene_compile_cache,
    uploaded_composed_field_cache,
    uploaded_field_preview_trace_store,
    uploaded_host_field_cache,
    uploaded_mesh_metadata_cache,
    uploaded_mesh_preprocess_timing_store,
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
    compose_hollow_lattice_field_sparse_with_backend,
    compose_hollow_lattice_field_with_backend,
    compute_resolution_for_lattice_pitch,
    parse_mesh_bytes,
    validate_triangle_mesh,
    _log_host_field_backend,
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
    UploadedPreviewFieldResponse,
    UploadedFieldStorageMode,
    UploadedFieldPreviewClientTelemetry,
    UploadedMeshPreviewWsRequest,
    UploadedMeshPreviewWsResponse,
)

logger = logging.getLogger(__name__)
_uvicorn_error_logger = logging.getLogger("uvicorn.error")
if _uvicorn_error_logger.handlers:
    logger.handlers = _uvicorn_error_logger.handlers
    logger.setLevel(_uvicorn_error_logger.level or logging.INFO)
    logger.propagate = False
else:
    logger.setLevel(logging.INFO)


class _SuppressTelemetryAccessLog(logging.Filter):
    _silenced_path = "/api/v1/internal/mesh/field-preview-telemetry"

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access":
            return True
        args = getattr(record, "args", ())
        if isinstance(args, tuple) and len(args) >= 3 and args[2] == self._silenced_path:
            return False
        return True


logging.getLogger("uvicorn.access").addFilter(_SuppressTelemetryAccessLog())

try:
    import redis as _redis  # type: ignore

    REDIS_CLIENT_AVAILABLE = _redis is not None
except Exception:
    REDIS_CLIENT_AVAILABLE = False

try:
    import cupy as _cupy  # type: ignore

    CUPY_RUNTIME_AVAILABLE = _cupy is not None
except Exception:
    _cupy = None  # type: ignore[assignment]
    CUPY_RUNTIME_AVAILABLE = False

EVAL_TIMEOUT_SECONDS = 20.0
QUALITY_TO_RESOLUTION: dict[QualityProfile, int] = {
    "interactive": 64,
    "medium": 128,
    "high": 192,
    "ultra": 256,
}
# Hard cap for mesh-upload voxel grids. 1024^3 at float32 = 4 GB.
# The tiered upload policy will never exceed this value.
MESH_UPLOAD_MAX_RESOLUTION = 1024
MESH_UPLOAD_MAX_BYTES = 200 * 1024 * 1024
AUTO_QUEUE_RESOLUTION_THRESHOLD = 128
AUTO_QUEUE_UPLOAD_BYTES_THRESHOLD = 8 * 1024 * 1024
MEMORY_MODEL_CPU_BYTES_PER_VOXEL = 48.0
MEMORY_MODEL_GPU_BYTES_PER_VOXEL = 56.0
MEMORY_MODEL_SAFETY_FACTOR = 1.0

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
        "X-SDF-Trace-Id",
        "X-SDF-Mesh-Span",
        "X-SDF-Available-CPU-Bytes",
        "X-SDF-Available-GPU-Free-Bytes",
        "X-SDF-Available-GPU-Total-Bytes",
        "X-SDF-CPU-Bytes-Per-Voxel",
        "X-SDF-GPU-Bytes-Per-Voxel",
        "X-SDF-Memory-Safety-Factor",
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
    _lattice_pitch: float,
    mesh_span: float,
    _voxels_per_lattice_period: int = 6,
) -> tuple[int, bool]:
    """Resolve the uploaded mesh grid from lattice requirements."""
    needed = compute_resolution_for_lattice_pitch(
        mesh_span,
        _lattice_pitch,
        _voxels_per_lattice_period,
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


def _freeze_cached_array(array: np.ndarray, dtype: np.dtype | type | None = None) -> np.ndarray:
    frozen = np.array(array, dtype=dtype, copy=True)
    frozen.setflags(write=False)
    return frozen


def _freeze_meshdata(mesh: MeshData) -> MeshData:
    return MeshData(
        vertices=_freeze_cached_array(mesh.vertices, np.float64),
        faces=_freeze_cached_array(mesh.faces, np.int32),
        normals=_freeze_cached_array(mesh.normals, np.float64),
    )


@dataclass(frozen=True)
class UploadedMeshMetadata:
    parsed: ParsedMesh
    outer_mesh: MeshData
    mesh_span: float
    cache_hit: bool


@dataclass(frozen=True)
class UploadedMeshMemoryContext:
    mesh_span: float
    available_cpu_bytes: int | None
    available_gpu_free_bytes: int | None
    available_gpu_total_bytes: int | None
    cpu_bytes_per_voxel: float
    gpu_bytes_per_voxel: float
    safety_factor: float


@dataclass(frozen=True)
class UploadedMeshMemoryEstimate:
    context: UploadedMeshMemoryContext
    resolution: int
    required_cpu_bytes: int
    required_gpu_bytes: int
    cpu_fatal: bool
    gpu_fatal: bool
    fatal: bool
    gpu_check_enabled: bool


@dataclass(frozen=True)
class UploadedHostFieldResult:
    parsed: ParsedMesh
    bounds: list[list[float]]
    host_sdf: np.ndarray
    host_compute_backend: Literal["cpu", "cuda"]
    fallback_reason: str | None
    block_size: int | None
    active_blocks: list[tuple[int, int, int]] | None
    field_storage_mode: UploadedFieldStorageMode
    host_build_strategy: str
    host_decision_reason: str
    cache_hit: bool


@dataclass(frozen=True)
class UploadedComposedFieldResult:
    parsed: ParsedMesh
    resolution: int
    bounds: list[list[float]]
    host_sdf: np.ndarray
    sparse_block_size: int | None
    active_blocks: list[tuple[int, int, int]] | None
    field: np.ndarray
    eval_backend_used: Literal["cpu", "cuda"]
    eval_ms: float
    field_cache_hit: bool


@dataclass(frozen=True)
class UploadedFieldPreviewServerAudit:
    metadata_cache_hit: bool
    host_cache_hit: bool
    field_cache_hit: bool
    resolution: int
    voxel_count: int
    payload_bytes: int
    compute_backend: Literal["cpu", "cuda"]
    host_build_strategy: str
    host_decision_reason: str
    server_upload_read_ms: float
    server_preprocessing_ms: float | None
    server_metadata_resolve_ms: float
    server_host_field_ms: float
    server_compose_field_ms: float
    server_pack_binary_ms: float
    server_handler_total_ms: float


def _format_optional_ms(value: float | None) -> str:
    return "None" if value is None else f"{value:.3f}"


def _lookup_uploaded_mesh_preprocessing_ms(file_hash: str) -> float | None:
    entry = uploaded_mesh_preprocess_timing_store.get(file_hash)
    if entry is None:
        return None
    return entry.preprocessing_ms


def _log_uploaded_field_preview_server_trace(
    *,
    trace_id: str,
    route: str,
    extension: str,
    audit: UploadedFieldPreviewServerAudit,
) -> None:
    print(
        "\n".join(
            [
                "uploaded_field_preview_server",
                f"trace_id={trace_id}",
                f"route={route}",
                f"extension={extension}",
                f"resolution={audit.resolution}",
                f"voxel_count={audit.voxel_count}",
                f"payload_bytes={audit.payload_bytes}",
                f"compute_backend={audit.compute_backend}",
                f"host_build_strategy={audit.host_build_strategy}",
                f"host_decision_reason={audit.host_decision_reason}",
                f"metadata_cache_hit={audit.metadata_cache_hit}",
                f"host_cache_hit={audit.host_cache_hit}",
                f"field_cache_hit={audit.field_cache_hit}",
                f"server_upload_read_ms={audit.server_upload_read_ms:.3f}",
                f"server_preprocessing_ms={_format_optional_ms(audit.server_preprocessing_ms)}",
                f"server_metadata_resolve_ms={audit.server_metadata_resolve_ms:.3f}",
                f"server_host_field_ms={audit.server_host_field_ms:.3f}",
                f"server_compose_field_ms={audit.server_compose_field_ms:.3f}",
                f"server_pack_binary_ms={audit.server_pack_binary_ms:.3f}",
                f"server_handler_total_ms={audit.server_handler_total_ms:.3f}",
            ]
        ),
        flush=True,
    )


def _log_uploaded_field_preview_consolidated_trace(entry: UploadedFieldPreviewTraceEntry) -> None:
    print(
        "\n".join(
            [
                "uploaded_field_preview_trace",
                f"trace_id={entry.trace_id}",
                f"route={entry.route}",
                f"extension={entry.extension}",
                f"resolution={entry.resolution}",
                f"voxel_count={entry.voxel_count}",
                f"payload_bytes={entry.payload_bytes}",
                f"compute_backend={entry.compute_backend}",
                f"host_build_strategy={entry.host_build_strategy}",
                f"host_decision_reason={entry.host_decision_reason}",
                f"metadata_cache_hit={entry.metadata_cache_hit}",
                f"host_cache_hit={entry.host_cache_hit}",
                f"field_cache_hit={entry.field_cache_hit}",
                f"mesh_cache_hit={entry.mesh_cache_hit}",
                f"server_upload_read_ms={entry.server_upload_read_ms}",
                f"server_preprocessing_ms={_format_optional_ms(entry.server_preprocessing_ms)}",
                f"server_metadata_resolve_ms={entry.server_metadata_resolve_ms}",
                f"server_host_field_ms={entry.server_host_field_ms}",
                f"server_compose_field_ms={entry.server_compose_field_ms}",
                f"server_pack_binary_ms={entry.server_pack_binary_ms}",
                f"server_handler_total_ms={entry.server_handler_total_ms}",
                f"client_response_wait_ms={entry.client_response_wait_ms}",
                f"client_download_ms={entry.client_download_ms}",
                f"client_decode_ms={entry.client_decode_ms}",
                f"client_texture_upload_and_first_frame_ms={entry.client_texture_upload_and_first_frame_ms}",
                f"client_total_visible_ms={entry.client_total_visible_ms}",
            ]
        ),
        flush=True,
    )


def _record_uploaded_field_preview_server_trace(
    *,
    trace_id: str,
    route: str,
    extension: str,
    audit: UploadedFieldPreviewServerAudit,
) -> None:
    uploaded_field_preview_trace_store.set(
        trace_id,
        UploadedFieldPreviewTraceEntry(
            trace_id=trace_id,
            created_at=time.time(),
            route=route,
            extension=extension,
            resolution=audit.resolution,
            voxel_count=audit.voxel_count,
            payload_bytes=audit.payload_bytes,
            compute_backend=audit.compute_backend,
            host_build_strategy=audit.host_build_strategy,
            host_decision_reason=audit.host_decision_reason,
            field_cache_hit=audit.field_cache_hit,
            mesh_cache_hit=False,
            host_cache_hit=audit.host_cache_hit,
            metadata_cache_hit=audit.metadata_cache_hit,
            server_upload_read_ms=audit.server_upload_read_ms,
            server_preprocessing_ms=audit.server_preprocessing_ms,
            server_metadata_resolve_ms=audit.server_metadata_resolve_ms,
            server_host_field_ms=audit.server_host_field_ms,
            server_compose_field_ms=audit.server_compose_field_ms,
            server_pack_binary_ms=audit.server_pack_binary_ms,
            server_handler_total_ms=audit.server_handler_total_ms,
        ),
    )
    _log_uploaded_field_preview_server_trace(trace_id=trace_id, route=route, extension=extension, audit=audit)


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


def _should_queue_upload_job(file_size: int, resolution: int, execution_mode: ExecutionMode) -> bool:
    if execution_mode == "queued":
        return True
    if execution_mode == "inline":
        return False
    if not REDIS_CLIENT_AVAILABLE:
        return False
    return file_size >= AUTO_QUEUE_UPLOAD_BYTES_THRESHOLD or resolution >= AUTO_QUEUE_RESOLUTION_THRESHOLD


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
    file_hash: str,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    voxels_per_lattice_period: int,
    compute_backend: ComputeBackend,
    mesh_backend: MeshBackend,
    meshing_mode: MeshingMode,
    field_storage_mode: UploadedFieldStorageMode = "auto",
) -> JobAcceptedResponse:
    from .worker_tasks import preview_uploaded_mesh_job

    payload = {
        "file_data_b64": base64.b64encode(file_bytes).decode("ascii"),
        "file_hash": file_hash,
        "extension": extension,
        "shell_thickness": shell_thickness,
        "lattice_type": lattice_type,
        "lattice_pitch": lattice_pitch,
        "lattice_thickness": lattice_thickness,
        "lattice_phase": lattice_phase,
        "voxels_per_lattice_period": voxels_per_lattice_period,
        "compute_backend": compute_backend,
        "mesh_backend": mesh_backend,
        "meshing_mode": meshing_mode,
        "field_storage_mode": field_storage_mode,
    }
    try:
        task = preview_uploaded_mesh_job.delay(payload)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}") from exc
    return _job_response(task.id)


def _enqueue_uploaded_export_job(
    *,
    file_bytes: bytes,
    file_hash: str,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    voxels_per_lattice_period: int,
    compute_backend: ComputeBackend,
    mesh_backend: MeshBackend,
    meshing_mode: MeshingMode,
    field_storage_mode: UploadedFieldStorageMode = "auto",
    format: Literal["stl", "obj"],
) -> JobAcceptedResponse:
    from .worker_tasks import export_uploaded_mesh_job

    payload = {
        "file_data_b64": base64.b64encode(file_bytes).decode("ascii"),
        "file_hash": file_hash,
        "extension": extension,
        "shell_thickness": shell_thickness,
        "lattice_type": lattice_type,
        "lattice_pitch": lattice_pitch,
        "lattice_thickness": lattice_thickness,
        "lattice_phase": lattice_phase,
        "voxels_per_lattice_period": voxels_per_lattice_period,
        "compute_backend": compute_backend,
        "mesh_backend": mesh_backend,
        "meshing_mode": meshing_mode,
        "field_storage_mode": field_storage_mode,
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
    if mesh_payload is None:
        raise RuntimeError("preview mesh payload was not generated")
    return PreviewMeshResponse(mesh=mesh_payload, stats=stats)


def _run_preview_meshdata(
    scene_ir,
    param_values: dict[str, float],
    grid: GridConfig,
    compute_precision: ComputePrecision = "float32",
    compute_backend: ComputeBackend = "auto",
    mesh_backend: MeshBackend = "auto",
    meshing_mode: MeshingMode = "uniform",
    encode_mesh_payload: bool = True,
    cache_result: bool = True,
) -> tuple[MeshData, PreviewStats, MeshPayload | None]:
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
        cached_stats["field_cache_hit"] = True
        cached_stats["mesh_cache_hit"] = True
        cached_stats["eval_ms"] = 0.0
        cached_stats["mesh_ms"] = 0.0
        return (
            MeshData(
                vertices=np.array(cached_mesh.vertices, dtype=np.float64, copy=True),
                faces=np.array(cached_mesh.faces, dtype=np.int32, copy=True),
                normals=np.array(cached_mesh.normals, dtype=np.float64, copy=True),
            ),
            PreviewStats.model_validate(cached_stats),
            cached_mesh.mesh if encode_mesh_payload else None,
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
        eval_ms = (time.perf_counter() - eval_start) * 1000.0
    else:
        field, cached_backend = cached_field
        eval_backend = "cuda" if cached_backend == "cuda" else "cpu"
        eval_ms = 0.0
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
        field_cache_hit=field_cache_hit,
        mesh_cache_hit=False,
        compute_precision=compute_precision,
        compute_backend=eval_backend,
        mesh_backend=mesh_backend_used,
        preview_mode="mesh",
    )

    mesh_payload = _encode_mesh_payload(mesh) if encode_mesh_payload else None
    if cache_result and mesh_payload is not None:
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
        eval_ms = (time.perf_counter() - eval_start) * 1000.0
    else:
        field, cached_backend = cached_field
        eval_backend = "cuda" if cached_backend == "cuda" else "cpu"
        eval_ms = 0.0
    if eval_ms > EVAL_TIMEOUT_SECONDS * 1000.0:
        raise HTTPException(status_code=408, detail="Field evaluation timeout exceeded")

    stats = PreviewStats(
        eval_ms=eval_ms,
        mesh_ms=None,
        tri_count=0,
        voxel_count=int(field.size),
        cache_hit=field_cache_hit,
        field_cache_hit=field_cache_hit,
        mesh_cache_hit=False,
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


def _hash_uploaded_mesh_bytes(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


async def _reject_legacy_uploaded_mesh_quality_profile(request: Request) -> None:
    form = await request.form()
    if "quality_profile" in form:
        raise HTTPException(
            status_code=422,
            detail="Uploaded mesh endpoints no longer accept quality_profile; use voxels_per_lattice_period instead.",
        )


def _resolve_uploaded_mesh_metadata(
    *,
    file_bytes: bytes,
    file_hash: str,
    extension: str,
) -> UploadedMeshMetadata:
    metadata_key = hash_uploaded_mesh_metadata_request(file_hash=file_hash, extension=extension)
    cached = uploaded_mesh_metadata_cache.get(metadata_key)
    if cached is not None:
        parsed = ParsedMesh(vertices=cached.vertices, faces=cached.faces)
        outer_mesh = MeshData(vertices=cached.vertices, faces=cached.faces, normals=cached.normals)
        return UploadedMeshMetadata(parsed=parsed, outer_mesh=outer_mesh, mesh_span=float(cached.mesh_span), cache_hit=True)

    parsed = parse_mesh_bytes(file_bytes, extension)
    validate_triangle_mesh(parsed)
    extents = np.ptp(parsed.vertices, axis=0)
    mesh_span = float(np.max(extents))
    outer_mesh = _freeze_meshdata(_meshdata_from_parsed(parsed))
    frozen_vertices = _freeze_cached_array(parsed.vertices, np.float64)
    frozen_faces = _freeze_cached_array(parsed.faces, np.int32)
    uploaded_mesh_metadata_cache.set(
        metadata_key,
        UploadedMeshMetadataCacheEntry(
            vertices=frozen_vertices,
            faces=frozen_faces,
            normals=outer_mesh.normals,
            mesh_span=mesh_span,
        ),
    )
    return UploadedMeshMetadata(
        parsed=ParsedMesh(vertices=frozen_vertices, faces=frozen_faces),
        outer_mesh=MeshData(vertices=frozen_vertices, faces=frozen_faces, normals=outer_mesh.normals),
        mesh_span=mesh_span,
        cache_hit=False,
    )


def _read_linux_mem_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kib = int(parts[1])
                        if kib > 0:
                            return kib * 1024
                    break
    except Exception:
        pass

    return None


def _read_sysconf_available_bytes() -> int | None:
    try:
        if "SC_AVPHYS_PAGES" in os.sysconf_names and "SC_PAGE_SIZE" in os.sysconf_names:
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            available = pages * page_size
            if available > 0:
                return available
    except Exception:
        pass
    return None


def _read_positive_int_file(path: Path) -> int | None:
    try:
        raw_value = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None

    if not raw_value or raw_value == "max":
        return None

    try:
        value = int(raw_value)
    except ValueError:
        return None

    # cgroup v1 uses a very large sentinel for "unlimited" memory.
    if value <= 0 or value >= 1 << 60:
        return None
    return value


def _read_cgroup_relative_path(controller: str | None = None) -> str | None:
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(":", 2)
                if len(parts) != 3:
                    continue
                _, controllers, path = parts
                if controller is None:
                    if controllers == "":
                        return path
                elif controller in controllers.split(","):
                    return path
    except Exception:
        pass
    return None


def _read_cgroup_memory_headroom_bytes() -> int | None:
    candidates: list[tuple[Path, str, str]] = []

    cgroup_v2_path = _read_cgroup_relative_path(None)
    if cgroup_v2_path is not None:
        base = Path("/sys/fs/cgroup") / cgroup_v2_path.lstrip("/")
        candidates.append((base, "memory.max", "memory.current"))

    cgroup_v1_path = _read_cgroup_relative_path("memory")
    if cgroup_v1_path is not None:
        base = Path("/sys/fs/cgroup/memory") / cgroup_v1_path.lstrip("/")
        candidates.append((base, "memory.limit_in_bytes", "memory.usage_in_bytes"))

    for base, limit_name, usage_name in candidates:
        limit_bytes = _read_positive_int_file(base / limit_name)
        usage_bytes = _read_positive_int_file(base / usage_name)
        if limit_bytes is None or usage_bytes is None:
            continue
        headroom = limit_bytes - usage_bytes
        if headroom <= 0:
            return 0
        return headroom

    return None


def _available_cpu_memory_bytes() -> int | None:
    host_available = _read_linux_mem_available_bytes()
    if host_available is None:
        host_available = _read_sysconf_available_bytes()

    cgroup_headroom = _read_cgroup_memory_headroom_bytes()
    if host_available is None:
        return cgroup_headroom
    if cgroup_headroom is None:
        return host_available

    available = min(host_available, cgroup_headroom)
    delta = abs(host_available - cgroup_headroom)
    bigger = max(host_available, cgroup_headroom)
    if delta >= 512 * 1024 * 1024 or delta / float(bigger) >= 0.20:
        logger.debug(
            "cpu_memory_probe host_memavailable=%d cgroup_headroom=%d selected=%d",
            host_available,
            cgroup_headroom,
            available,
        )
    return available


def _available_gpu_memory_bytes() -> tuple[int | None, int | None]:
    if not CUPY_RUNTIME_AVAILABLE or _cupy is None:
        return None, None
    try:
        if int(_cupy.cuda.runtime.getDeviceCount()) <= 0:
            return None, None
        free_raw, total_raw = _cupy.cuda.runtime.memGetInfo()
        free_bytes = int(free_raw)
        total_bytes = int(total_raw)
        if free_bytes <= 0 or total_bytes <= 0:
            return None, None
        return free_bytes, total_bytes
    except Exception:
        return None, None


def _maybe_evict_uploaded_composed_field_cache_before_meshing(field: np.ndarray) -> bool:
    available_cpu = _available_cpu_memory_bytes()
    if available_cpu is None:
        return False
    field_bytes = int(field.nbytes)
    # Keep the cache if there is enough room for the live field plus a rough
    # meshing working set. Otherwise drop the cached composed field copy.
    if available_cpu >= field_bytes * 4:
        return False

    uploaded_composed_field_cache.clear()
    gc.collect()
    logger.debug(
        "evicted_uploaded_composed_field_cache_before_meshing field_bytes=%d available_cpu=%d",
        field_bytes,
        available_cpu,
    )
    return True


def _snapshot_uploaded_mesh_memory_context(mesh_span: float) -> UploadedMeshMemoryContext:
    available_cpu = _available_cpu_memory_bytes()
    available_gpu_free, available_gpu_total = _available_gpu_memory_bytes()
    return UploadedMeshMemoryContext(
        mesh_span=float(mesh_span),
        available_cpu_bytes=available_cpu,
        available_gpu_free_bytes=available_gpu_free,
        available_gpu_total_bytes=available_gpu_total,
        cpu_bytes_per_voxel=MEMORY_MODEL_CPU_BYTES_PER_VOXEL,
        gpu_bytes_per_voxel=MEMORY_MODEL_GPU_BYTES_PER_VOXEL,
        safety_factor=MEMORY_MODEL_SAFETY_FACTOR,
    )


def _resolve_uploaded_mesh_memory_estimate(
    *,
    file_bytes: bytes,
    file_hash: str,
    extension: str,
    lattice_pitch: float,
    voxels_per_lattice_period: int,
    compute_backend: ComputeBackend,
    metadata: UploadedMeshMetadata | None = None,
) -> tuple[UploadedMeshMemoryEstimate, UploadedMeshMetadata]:
    resolved_metadata = metadata or _resolve_uploaded_mesh_metadata(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
    )
    context = _snapshot_uploaded_mesh_memory_context(resolved_metadata.mesh_span)
    resolution, _ = _resolve_mesh_resolution(
        lattice_pitch,
        resolved_metadata.mesh_span,
        voxels_per_lattice_period,
    )
    voxel_count = int(resolution**3)
    required_cpu_bytes = int(np.ceil(voxel_count * context.cpu_bytes_per_voxel * context.safety_factor))
    required_gpu_bytes = int(np.ceil(voxel_count * context.gpu_bytes_per_voxel * context.safety_factor))
    gpu_check_enabled = compute_backend == "cuda" or (
        compute_backend == "auto" and context.available_gpu_free_bytes is not None
    )
    cpu_fatal = context.available_cpu_bytes is not None and required_cpu_bytes > context.available_cpu_bytes
    gpu_fatal = (
        gpu_check_enabled
        and context.available_gpu_free_bytes is not None
        and required_gpu_bytes > context.available_gpu_free_bytes
    )
    return (
        UploadedMeshMemoryEstimate(
            context=context,
            resolution=resolution,
            required_cpu_bytes=required_cpu_bytes,
            required_gpu_bytes=required_gpu_bytes,
            cpu_fatal=cpu_fatal,
            gpu_fatal=gpu_fatal,
            fatal=cpu_fatal or gpu_fatal,
            gpu_check_enabled=gpu_check_enabled,
        ),
        resolved_metadata,
    )


def _format_bytes_gib(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / float(1024**3):.2f} GiB"


def _uploaded_mesh_memory_guard_message(
    *,
    estimate: UploadedMeshMemoryEstimate,
    lattice_pitch: float,
    voxels_per_lattice_period: int,
    compute_backend: ComputeBackend,
) -> str:
    parts = [
        "Estimated memory requirement exceeds available system memory for this mesh preview.",
        f"resolution={estimate.resolution}, compute_backend={compute_backend}",
        f"required_cpu={_format_bytes_gib(estimate.required_cpu_bytes)}, available_cpu={_format_bytes_gib(estimate.context.available_cpu_bytes)}",
    ]
    if estimate.gpu_check_enabled:
        parts.append(
            f"required_gpu={_format_bytes_gib(estimate.required_gpu_bytes)}, available_gpu_free={_format_bytes_gib(estimate.context.available_gpu_free_bytes)}"
        )
    parts.append(
        "Reduce memory usage by increasing lattice_pitch, lowering voxels_per_lattice_period, or selecting CPU backend."
    )
    parts.append(
        f"current_lattice_pitch={lattice_pitch}, current_voxels_per_lattice_period={voxels_per_lattice_period}"
    )
    return " ".join(parts)


def _enforce_uploaded_mesh_memory_guard(
    *,
    file_bytes: bytes,
    file_hash: str,
    extension: str,
    lattice_pitch: float,
    voxels_per_lattice_period: int,
    compute_backend: ComputeBackend,
    metadata: UploadedMeshMetadata | None = None,
) -> UploadedMeshMemoryEstimate:
    estimate, _ = _resolve_uploaded_mesh_memory_estimate(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
        lattice_pitch=lattice_pitch,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
        metadata=metadata,
    )
    if estimate.fatal:
        raise MeshUploadError(
            _uploaded_mesh_memory_guard_message(
                estimate=estimate,
                lattice_pitch=lattice_pitch,
                voxels_per_lattice_period=voxels_per_lattice_period,
                compute_backend=compute_backend,
            )
        )
    return estimate


def _resolve_uploaded_host_field(
    *,
    file_bytes: bytes,
    file_hash: str,
    extension: str,
    resolution: int,
    compute_backend: ComputeBackend = "auto",
    parsed: ParsedMesh | None = None,
    field_storage_mode: UploadedFieldStorageMode = "auto",
) -> UploadedHostFieldResult:
    if parsed is None:
        parsed = _resolve_uploaded_mesh_metadata(
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
        ).parsed
    host_key = hash_uploaded_mesh_host_request(
        file_hash=file_hash,
        extension=extension,
        resolution=resolution,
        compute_backend=compute_backend,
        field_storage_mode=field_storage_mode,
    )
    cached = uploaded_host_field_cache.get(host_key)
    if cached is not None:
        bounds = [[float(axis[0]), float(axis[1])] for axis in cached.bounds]
        active_blocks: list[tuple[int, int, int]] | None = None
        if cached.active_blocks:
            active_blocks = [(int(bx), int(by), int(bz)) for bx, by, bz in cached.active_blocks]
        _log_host_field_backend(cached.host_compute_backend)
        return UploadedHostFieldResult(
            parsed=parsed,
            bounds=bounds,
            host_sdf=cached.host_sdf,
            host_compute_backend="cuda" if cached.host_compute_backend == "cuda" else "cpu",
            fallback_reason=getattr(cached, "fallback_reason", None),
            block_size=cached.block_size,
            active_blocks=active_blocks,
            field_storage_mode=cached.field_storage_mode,
            host_build_strategy=cached.host_build_strategy,
            host_decision_reason=cached.host_decision_reason,
            cache_hit=True,
        )

    host = build_host_field(
        parsed,
        resolution=resolution,
        compute_backend=compute_backend,
        field_storage_mode=field_storage_mode,
    )
    frozen_host_sdf = _freeze_cached_array(host.host_sdf, np.float32)

    uploaded_host_field_cache.set(
        host_key,
        UploadedHostFieldCacheEntry(
            bounds=[[float(axis[0]), float(axis[1])] for axis in host.bounds],
            host_sdf=frozen_host_sdf,
            host_compute_backend=host.host_compute_backend,
            fallback_reason=getattr(host, "fallback_reason", None),
            field_storage_mode=host.field_storage_mode,
            block_size=host.block_size,
            active_blocks=host.active_blocks,
            sparse_background_value=host.sparse_background_value,
            sparse_bricks=(
                {k: _freeze_cached_array(v, np.float32) for k, v in host.sparse_bricks.items()}
                if host.sparse_bricks is not None
                else None
            ),
            host_build_strategy=host.host_build_strategy,
            host_decision_reason=host.host_decision_reason,
        ),
    )
    return UploadedHostFieldResult(
        parsed=parsed,
        bounds=host.bounds,
        host_sdf=frozen_host_sdf,
        host_compute_backend=host.host_compute_backend,
        fallback_reason=getattr(host, "fallback_reason", None),
        block_size=host.block_size,
        active_blocks=host.active_blocks,
        field_storage_mode=host.field_storage_mode,
        host_build_strategy=host.host_build_strategy,
        host_decision_reason=host.host_decision_reason,
        cache_hit=False,
    )


def _compute_mesh_upload_resolution(
    file_bytes: bytes,
    file_hash: str,
    extension: str,
    _lattice_pitch: float,
    _voxels_per_lattice_period: int,
) -> int:
    """Resolve the uploaded mesh bbox span, then compute the lattice-driven grid."""
    metadata = _resolve_uploaded_mesh_metadata(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
    )
    mesh_span = metadata.mesh_span
    resolution, _ = _resolve_mesh_resolution(_lattice_pitch, mesh_span, _voxels_per_lattice_period)
    return resolution


def _should_queue_uploaded_request(
    *,
    file_bytes: bytes,
    file_hash: str,
    extension: str,
    lattice_pitch: float,
    voxels_per_lattice_period: int,
    execution_mode: ExecutionMode,
) -> bool:
    if execution_mode == "queued":
        return True
    if execution_mode == "inline":
        return False
    if not REDIS_CLIENT_AVAILABLE:
        return False
    if len(file_bytes) >= AUTO_QUEUE_UPLOAD_BYTES_THRESHOLD:
        return True
    resolution = _compute_mesh_upload_resolution(
        file_bytes,
        file_hash,
        extension,
        lattice_pitch,
        voxels_per_lattice_period,
    )
    return _should_queue_upload_job(len(file_bytes), resolution, execution_mode)


def _uploaded_mesh_field_payload_key(
    *,
    file_hash: str,
    extension: str,
    resolution: int,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    voxels_per_lattice_period: int,
    compute_backend: ComputeBackend,
    field_storage_mode: UploadedFieldStorageMode,
) -> str:
    return hash_uploaded_mesh_field_request(
        file_hash=file_hash,
        extension=extension,
        resolution=resolution,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
        field_storage_mode=field_storage_mode,
    )


def _build_uploaded_field_payload(
    field: np.ndarray,
    bounds: list[list[float]],
) -> FieldPayload:
    return FieldPayload(
        encoding="f32-base64",
        resolution=int(field.shape[0]),
        bounds=[[float(axis[0]), float(axis[1])] for axis in bounds],
        data=_encode_field(field),
    )


def _build_uploaded_host_field_payload(
    host_sdf: np.ndarray,
    bounds: list[list[float]],
) -> FieldPayload:
    return FieldPayload(
        encoding="f32-base64",
        resolution=int(host_sdf.shape[0]),
        bounds=[[float(axis[0]), float(axis[1])] for axis in bounds],
        data=_encode_field(host_sdf),
    )


def _build_uploaded_field_payload_from_cache_entry(
    *,
    cache_entry: UploadedMeshCacheEntry,
    file_hash: str,
    extension: str,
    resolution: int,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    voxels_per_lattice_period: int,
    compute_backend: ComputeBackend,
    field_storage_mode: UploadedFieldStorageMode,
) -> FieldPayload:
    if (
        cache_entry.field_resolution is not None
        and cache_entry.field_bounds is not None
        and cache_entry.field_data is not None
    ):
        return FieldPayload(
            encoding="f32-base64",
            resolution=int(cache_entry.field_resolution),
            bounds=[[float(axis[0]), float(axis[1])] for axis in cache_entry.field_bounds],
            data=cache_entry.field_data,
        )

    field_key = _uploaded_mesh_field_payload_key(
        file_hash=file_hash,
        extension=extension,
        resolution=resolution,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
        field_storage_mode=field_storage_mode,
    )
    cached_field = uploaded_composed_field_cache.get(field_key)
    if cached_field is None:
        raise HTTPException(status_code=500, detail="Cached mesh entry is missing composed field payload")
    return _build_uploaded_field_payload(cached_field.field, cached_field.bounds)


def _resolve_uploaded_composed_field(
    *,
    file_bytes: bytes,
    file_hash: str,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    voxels_per_lattice_period: int = 6,
    compute_backend: ComputeBackend = "auto",
    parsed: ParsedMesh | None = None,
    resolution: int | None = None,
    bounds: list[list[float]] | None = None,
    host_sdf: np.ndarray | None = None,
    sparse_block_size: int | None = None,
    host_active_blocks: list[tuple[int, int, int]] | None = None,
    field_storage_mode: UploadedFieldStorageMode = "auto",
) -> UploadedComposedFieldResult:
    if parsed is None or resolution is None:
        metadata = _resolve_uploaded_mesh_metadata(
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
        )
        if parsed is None:
            parsed = metadata.parsed
        if resolution is None:
            resolution, _ = _resolve_mesh_resolution(
                lattice_pitch,
                metadata.mesh_span,
                voxels_per_lattice_period,
            )

    assert parsed is not None
    assert resolution is not None
    if bounds is None or host_sdf is None:
        host_result = _resolve_uploaded_host_field(
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
            resolution=resolution,
            compute_backend=compute_backend,
            parsed=parsed,
            field_storage_mode=field_storage_mode,
        )
        parsed = host_result.parsed
        bounds = host_result.bounds
        host_sdf = host_result.host_sdf
        sparse_block_size = host_result.block_size
        host_active_blocks = host_result.active_blocks

    assert bounds is not None
    assert host_sdf is not None

    field_key = _uploaded_mesh_field_payload_key(
        file_hash=file_hash,
        extension=extension,
        resolution=resolution,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
        field_storage_mode=field_storage_mode,
    )
    cached_field = uploaded_composed_field_cache.get(field_key)
    if cached_field is not None:
        active_blocks: list[tuple[int, int, int]] | None = None
        if cached_field.active_blocks:
            active_blocks = [(int(bx), int(by), int(bz)) for bx, by, bz in cached_field.active_blocks]
        return UploadedComposedFieldResult(
            parsed=parsed,
            resolution=resolution,
            bounds=[[float(axis[0]), float(axis[1])] for axis in cached_field.bounds],
            host_sdf=host_sdf,
            sparse_block_size=cached_field.block_size,
            active_blocks=active_blocks,
            field=cached_field.field,
            eval_backend_used="cuda" if cached_field.eval_backend == "cuda" else "cpu",
            eval_ms=0.0,
            field_cache_hit=True,
        )

    eval_start = time.perf_counter()
    active_blocks = host_active_blocks
    if sparse_block_size is not None and active_blocks:
        field, eval_backend_used, updated_blocks = compose_hollow_lattice_field_sparse_with_backend(
            host_sdf,
            bounds,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            block_size=sparse_block_size,
            active_blocks=active_blocks,
            compute_backend=compute_backend,
        )
        active_blocks = updated_blocks or active_blocks
    else:
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

    uploaded_composed_field_cache.set(
        field_key,
        UploadedComposedFieldCacheEntry(
            field=np.array(field, dtype=np.float32, copy=True),
            bounds=[[float(axis[0]), float(axis[1])] for axis in bounds],
            resolution=resolution,
            eval_backend=eval_backend_used,
            block_size=sparse_block_size,
            active_blocks=(
                [(int(bx), int(by), int(bz)) for bx, by, bz in active_blocks]
                if active_blocks is not None
                else None
            ),
        ),
    )
    eval_ms = (time.perf_counter() - eval_start) * 1000.0
    return UploadedComposedFieldResult(
        parsed=parsed,
        resolution=resolution,
        bounds=bounds,
        host_sdf=host_sdf,
        sparse_block_size=sparse_block_size,
        active_blocks=active_blocks,
        field=field,
        eval_backend_used=eval_backend_used,
        eval_ms=eval_ms,
        field_cache_hit=False,
    )


def _resolve_cached_uploaded_mesh_commit_inputs(
    *,
    file_bytes: bytes,
    file_hash: str,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    voxels_per_lattice_period: int = 6,
    compute_backend: ComputeBackend = "auto",
    field_storage_mode: UploadedFieldStorageMode = "auto",
) -> tuple[
    UploadedMeshMetadata,
    int,
    UploadedHostFieldCacheEntry,
    UploadedComposedFieldCacheEntry,
]:
    metadata = _resolve_uploaded_mesh_metadata(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
    )
    resolution, _ = _resolve_mesh_resolution(
        lattice_pitch,
        metadata.mesh_span,
        voxels_per_lattice_period,
    )
    host_key = hash_uploaded_mesh_host_request(
        file_hash=file_hash,
        extension=extension,
        resolution=resolution,
        compute_backend=compute_backend,
        field_storage_mode=field_storage_mode,
    )
    field_key = _uploaded_mesh_field_payload_key(
        file_hash=file_hash,
        extension=extension,
        resolution=resolution,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
        field_storage_mode=field_storage_mode,
    )
    host_cached = uploaded_host_field_cache.get(host_key)
    field_cached = uploaded_composed_field_cache.get(field_key)
    if host_cached is None or field_cached is None:
        raise MeshUploadError("Preview Field first before committing the mesh.")
    return metadata, resolution, host_cached, field_cached


def _run_uploaded_mesh_commit_meshdata(
    *,
    file_bytes: bytes,
    file_hash: str,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    voxels_per_lattice_period: int = 6,
    compute_backend: ComputeBackend = "auto",
    mesh_backend: MeshBackend = "auto",
    meshing_mode: MeshingMode = "uniform",
    field_storage_mode: UploadedFieldStorageMode = "auto",
    encode_response_payloads: bool = True,
) -> tuple[MeshData, PreviewStats, MeshPayload | None]:
    metadata, resolution, host_cached, field_cached = _resolve_cached_uploaded_mesh_commit_inputs(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
        field_storage_mode=field_storage_mode,
    )
    cache_key = hash_uploaded_mesh_request(
        file_hash=file_hash,
        extension=extension,
        resolution=resolution,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
        mesh_backend=mesh_backend,
        meshing_mode=meshing_mode,
        field_storage_mode=field_storage_mode,
    )
    cached_mesh = uploaded_mesh_preview_cache.get(cache_key)
    if (
        cached_mesh is not None
        and mesh_backend != "cpu"
        and str(cached_mesh.stats.get("mesh_backend", "cpu")) == "cpu"
        and is_cuda_meshing_available()
    ):
        cached_mesh = None

    if cached_mesh is not None:
        cached_stats = copy.deepcopy(cached_mesh.stats)
        cached_stats["cache_hit"] = True
        cached_stats["field_cache_hit"] = True
        cached_stats["mesh_cache_hit"] = True
        cached_stats["eval_ms"] = 0.0
        cached_stats["mesh_ms"] = 0.0
        cached_field = _build_uploaded_field_payload_from_cache_entry(
            cache_entry=cached_mesh,
            file_hash=file_hash,
            extension=extension,
            resolution=resolution,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
            field_storage_mode=field_storage_mode,
        )
        cached_mesh_payload = cached_mesh.mesh
        if encode_response_payloads and cached_mesh_payload is None:
            cached_mesh_payload = _encode_mesh_payload(
                MeshData(
                    vertices=cached_mesh.vertices,
                    faces=cached_mesh.faces,
                    normals=cached_mesh.normals,
                )
            )
        uploaded_mesh_preview_cache.set(
            cache_key,
            UploadedMeshCacheEntry(
                mesh=cached_mesh_payload if encode_response_payloads else cached_mesh.mesh,
                vertices=cached_mesh.vertices,
                faces=cached_mesh.faces,
                normals=cached_mesh.normals,
                stats=cached_stats,
                field_resolution=int(cached_field.resolution),
                field_bounds=[[float(axis[0]), float(axis[1])] for axis in cached_field.bounds],
                field_data=cached_field.data,
            )
        )
        return (
            MeshData(
                vertices=cached_mesh.vertices,
                faces=cached_mesh.faces,
                normals=cached_mesh.normals,
            ),
            PreviewStats.model_validate(cached_stats),
            cached_mesh_payload if encode_response_payloads else None,
        )

    field = field_cached.field
    bounds = [[float(axis[0]), float(axis[1])] for axis in field_cached.bounds]
    active_blocks: list[tuple[int, int, int]] | None = None
    if field_cached.active_blocks:
        active_blocks = [(int(bx), int(by), int(bz)) for bx, by, bz in field_cached.active_blocks]

    mesh_start = time.perf_counter()
    generated, mesh_backend_used = build_mesh_with_backend(
        field,
        bounds,
        backend=mesh_backend,
        meshing_mode=meshing_mode,
        active_blocks=active_blocks,
        block_size=host_cached.block_size or field_cached.block_size,
    )
    interior = _strip_outer_surface(generated, host_cached.host_sdf, bounds)
    mesh = _merge_meshes(metadata.outer_mesh, interior)
    mesh_ms = (time.perf_counter() - mesh_start) * 1000.0

    stats = PreviewStats(
        eval_ms=0.0,
        mesh_ms=mesh_ms,
        tri_count=int(mesh.faces.shape[0]),
        cache_hit=False,
        field_cache_hit=True,
        mesh_cache_hit=False,
        compute_backend="cuda" if field_cached.eval_backend == "cuda" else "cpu",
        mesh_backend=mesh_backend_used,
        preview_mode="mesh",
    )

    mesh_payload = _encode_mesh_payload(mesh) if encode_response_payloads else None
    if encode_response_payloads:
        uploaded_mesh_preview_cache.set(
            cache_key,
            UploadedMeshCacheEntry(
                mesh=mesh_payload,
                vertices=_freeze_cached_array(mesh.vertices, np.float64),
                faces=_freeze_cached_array(mesh.faces, np.int32),
                normals=_freeze_cached_array(mesh.normals, np.float64),
                stats=stats.model_dump(),
                field_resolution=int(field.shape[0]),
                field_bounds=[[float(axis[0]), float(axis[1])] for axis in bounds],
                field_data=_build_uploaded_field_payload(field, bounds).data,
            ),
        )
    return mesh, stats, mesh_payload


def _run_uploaded_mesh_preview(
    *,
    file_bytes: bytes,
    file_hash: str,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    voxels_per_lattice_period: int = 6,
    compute_backend: ComputeBackend = "auto",
    mesh_backend: MeshBackend = "auto",
    meshing_mode: MeshingMode = "uniform",
    field_storage_mode: UploadedFieldStorageMode = "auto",
) -> PreviewMeshResponse:
    _, stats, field_payload, mesh_payload = _run_uploaded_mesh_preview_meshdata(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
        mesh_backend=mesh_backend,
        meshing_mode=meshing_mode,
        field_storage_mode=field_storage_mode,
    )
    if mesh_payload is None or field_payload is None:
        raise RuntimeError("uploaded preview payload was not generated")
    return PreviewMeshResponse(mesh=mesh_payload, stats=stats, field=field_payload)


def _run_uploaded_mesh_field_preview(
    *,
    file_bytes: bytes,
    file_hash: str,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    voxels_per_lattice_period: int = 6,
    compute_backend: ComputeBackend = "auto",
    field_storage_mode: UploadedFieldStorageMode = "auto",
) -> PreviewFieldResponse:
    field, bounds, stats = _run_uploaded_mesh_field_preview_data(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
        field_storage_mode=field_storage_mode,
    )
    field_payload = _build_uploaded_field_payload(field, bounds)
    return PreviewFieldResponse(field=field_payload, stats=stats)


def _run_uploaded_mesh_field_preview_data_with_audit(
    *,
    file_bytes: bytes,
    file_hash: str,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    voxels_per_lattice_period: int = 6,
    compute_backend: ComputeBackend = "auto",
    field_storage_mode: UploadedFieldStorageMode = "auto",
) -> tuple[np.ndarray, np.ndarray, list[list[float]], PreviewStats, UploadedFieldPreviewServerAudit]:
    metadata_start = time.perf_counter()
    metadata = _resolve_uploaded_mesh_metadata(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
    )
    server_metadata_resolve_ms = (time.perf_counter() - metadata_start) * 1000.0
    resolution, _ = _resolve_mesh_resolution(
        lattice_pitch,
        metadata.mesh_span,
        voxels_per_lattice_period,
    )

    host_start = time.perf_counter()
    host_result = _resolve_uploaded_host_field(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
        resolution=resolution,
        compute_backend=compute_backend,
        parsed=metadata.parsed,
        field_storage_mode=field_storage_mode,
    )
    server_host_field_ms = (time.perf_counter() - host_start) * 1000.0

    composed = _resolve_uploaded_composed_field(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
        parsed=host_result.parsed,
        resolution=resolution,
        bounds=host_result.bounds,
        host_sdf=host_result.host_sdf,
        sparse_block_size=host_result.block_size,
        host_active_blocks=host_result.active_blocks,
        field_storage_mode=field_storage_mode,
    )

    out_bounds = [[float(axis[0]), float(axis[1])] for axis in composed.bounds]
    stats = PreviewStats(
        eval_ms=composed.eval_ms,
        mesh_ms=None,
        tri_count=0,
        voxel_count=int(composed.field.size),
        cache_hit=composed.field_cache_hit,
        field_cache_hit=composed.field_cache_hit,
        mesh_cache_hit=False,
        compute_backend=composed.eval_backend_used,
        mesh_backend="cpu",
        preview_mode="field",
        fallback_reason=host_result.fallback_reason,
    )
    preprocessing_ms = _lookup_uploaded_mesh_preprocessing_ms(file_hash)
    audit = UploadedFieldPreviewServerAudit(
        metadata_cache_hit=metadata.cache_hit,
        host_cache_hit=host_result.cache_hit,
        field_cache_hit=composed.field_cache_hit,
        resolution=int(composed.field.shape[0]),
        voxel_count=int(composed.field.size),
        payload_bytes=int(composed.field.size * np.dtype(np.float32).itemsize * 2),
        compute_backend=composed.eval_backend_used,
        host_build_strategy=host_result.host_build_strategy,
        host_decision_reason=host_result.host_decision_reason,
        server_upload_read_ms=0.0,
        server_preprocessing_ms=preprocessing_ms,
        server_metadata_resolve_ms=server_metadata_resolve_ms,
        server_host_field_ms=server_host_field_ms,
        server_compose_field_ms=composed.eval_ms,
        server_pack_binary_ms=0.0,
        server_handler_total_ms=0.0,
    )
    return host_result.host_sdf, composed.field, out_bounds, stats, audit


def _run_uploaded_mesh_field_preview_data(
    *,
    file_bytes: bytes,
    file_hash: str,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    voxels_per_lattice_period: int = 6,
    compute_backend: ComputeBackend = "auto",
    field_storage_mode: UploadedFieldStorageMode = "auto",
) -> tuple[np.ndarray, list[list[float]], PreviewStats]:
    _host_field, field, bounds, stats, _audit = _run_uploaded_mesh_field_preview_data_with_audit(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
        field_storage_mode=field_storage_mode,
    )
    return field, bounds, stats


def _run_uploaded_mesh_preview_meshdata(
    *,
    file_bytes: bytes,
    file_hash: str,
    extension: str,
    shell_thickness: float,
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"],
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    voxels_per_lattice_period: int = 6,
    compute_backend: ComputeBackend = "auto",
    mesh_backend: MeshBackend = "auto",
    meshing_mode: MeshingMode = "uniform",
    field_storage_mode: UploadedFieldStorageMode = "auto",
    encode_response_payloads: bool = True,
    cache_result: bool = True,
) -> tuple[MeshData, PreviewStats, FieldPayload | None, MeshPayload | None]:
    metadata = _resolve_uploaded_mesh_metadata(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
    )
    resolution, _ = _resolve_mesh_resolution(
        lattice_pitch,
        metadata.mesh_span,
        voxels_per_lattice_period,
    )
    cache_key = hash_uploaded_mesh_request(
        file_hash=file_hash,
        extension=extension,
        resolution=resolution,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
        mesh_backend=mesh_backend,
        meshing_mode=meshing_mode,
        field_storage_mode=field_storage_mode,
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
        cached_stats["field_cache_hit"] = True
        cached_stats["mesh_cache_hit"] = True
        cached_stats["eval_ms"] = 0.0
        cached_stats["mesh_ms"] = 0.0
        cached_field: FieldPayload | None = None
        if encode_response_payloads:
            cached_field = _build_uploaded_field_payload_from_cache_entry(
                cache_entry=cached_mesh,
                file_hash=file_hash,
                extension=extension,
                resolution=resolution,
                shell_thickness=shell_thickness,
                lattice_type=lattice_type,
                lattice_pitch=lattice_pitch,
                lattice_thickness=lattice_thickness,
                lattice_phase=lattice_phase,
                voxels_per_lattice_period=voxels_per_lattice_period,
                compute_backend=compute_backend,
                field_storage_mode=field_storage_mode,
            )
        cached_mesh_payload = cached_mesh.mesh
        if encode_response_payloads and cached_mesh_payload is None:
            cached_mesh_payload = _encode_mesh_payload(
                MeshData(
                    vertices=cached_mesh.vertices,
                    faces=cached_mesh.faces,
                    normals=cached_mesh.normals,
                )
            )
        return (
            MeshData(
                vertices=cached_mesh.vertices,
                faces=cached_mesh.faces,
                normals=cached_mesh.normals,
            ),
            PreviewStats.model_validate(cached_stats),
            cached_field,
            cached_mesh_payload if encode_response_payloads else None,
        )
    composed = _resolve_uploaded_composed_field(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
        voxels_per_lattice_period=voxels_per_lattice_period,
        compute_backend=compute_backend,
        parsed=metadata.parsed,
        resolution=resolution,
        field_storage_mode=field_storage_mode,
    )
    parsed = composed.parsed
    resolution = composed.resolution
    bounds = composed.bounds
    host_sdf = composed.host_sdf
    sparse_block_size = composed.sparse_block_size
    active_blocks = composed.active_blocks
    field = composed.field
    eval_backend_used = composed.eval_backend_used
    eval_ms = composed.eval_ms
    field_cache_hit = composed.field_cache_hit

    _maybe_evict_uploaded_composed_field_cache_before_meshing(field)
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
    mesh = _merge_meshes(metadata.outer_mesh, interior)
    post_mesh_ms = (time.perf_counter() - post_mesh_start) * 1000.0
    mesh_ms = (time.perf_counter() - mesh_start) * 1000.0
    logger.debug(
        "mesh_upload_timing resolution=%d eval_ms=%.2f cache_hit=%s mesher_ms=%.2f post_mesh_ms=%.2f mesh_ms=%.2f",
        resolution,
        eval_ms,
        field_cache_hit,
        mesher_ms,
        post_mesh_ms,
        mesh_ms,
    )

    field_payload: FieldPayload | None = None
    if encode_response_payloads:
        field_payload = _build_uploaded_field_payload(field, bounds)
    stats = PreviewStats(
        eval_ms=eval_ms,
        mesh_ms=mesh_ms,
        tri_count=int(mesh.faces.shape[0]),
        cache_hit=field_cache_hit,
        field_cache_hit=field_cache_hit,
        mesh_cache_hit=False,
        compute_backend=eval_backend_used,
        mesh_backend=mesh_backend_used,
        preview_mode="mesh",
    )

    mesh_payload = _encode_mesh_payload(mesh) if encode_response_payloads else None
    if cache_result:
        uploaded_mesh_preview_cache.set(
            cache_key,
            UploadedMeshCacheEntry(
                mesh=mesh_payload,
                vertices=_freeze_cached_array(mesh.vertices, np.float64),
                faces=_freeze_cached_array(mesh.faces, np.int32),
                normals=_freeze_cached_array(mesh.normals, np.float64),
                stats=stats.model_dump(),
                field_resolution=int(field_payload.resolution) if field_payload is not None else int(field.shape[0]),
                field_bounds=(
                    [[float(axis[0]), float(axis[1])] for axis in field_payload.bounds]
                    if field_payload is not None
                    else [[float(axis[0]), float(axis[1])] for axis in bounds]
                ),
                field_data=field_payload.data if field_payload is not None else None,
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
            False,
            False,
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
            False,
            False,
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
    request: Request,
    file: UploadFile = File(...),
    shell_thickness: float = Form(...),
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"] = Form(...),
    lattice_pitch: float = Form(...),
    lattice_thickness: float = Form(...),
    lattice_phase: float = Form(0.0),
    voxels_per_lattice_period: int = Form(6),
    compute_backend: ComputeBackend = Form("auto"),
    mesh_backend: MeshBackend = Form("auto"),
    meshing_mode: MeshingMode = Form("uniform"),
    field_storage_mode: UploadedFieldStorageMode = Form("auto"),
    execution_mode: ExecutionMode = Form("auto"),
) -> PreviewMeshResponse | JobAcceptedResponse:
    await _reject_legacy_uploaded_mesh_quality_profile(request)
    file_bytes, extension = await _read_uploaded_mesh(file)
    file_hash = _hash_uploaded_mesh_bytes(file_bytes)
    try:
        _enforce_uploaded_mesh_memory_guard(
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
            lattice_pitch=lattice_pitch,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
        )
    except MeshUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if _should_queue_uploaded_request(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
        lattice_pitch=lattice_pitch,
        voxels_per_lattice_period=voxels_per_lattice_period,
        execution_mode=execution_mode,
    ):
        try:
            return _enqueue_uploaded_preview_job(
                file_bytes=file_bytes,
                file_hash=file_hash,
                extension=extension,
                shell_thickness=shell_thickness,
                lattice_type=lattice_type,
                lattice_pitch=lattice_pitch,
                lattice_thickness=lattice_thickness,
                lattice_phase=lattice_phase,
                voxels_per_lattice_period=voxels_per_lattice_period,
                compute_backend=compute_backend,
                mesh_backend=mesh_backend,
                meshing_mode=meshing_mode,
                field_storage_mode=field_storage_mode,
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
            file_hash=file_hash,
            extension=extension,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
            mesh_backend=mesh_backend,
            meshing_mode=meshing_mode,
            field_storage_mode=field_storage_mode,
        )
        return result
    except (MeshUploadError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="preview_uploaded_mesh_inline")


@app.post("/api/v1/mesh/preview.binary", response_model=None)
async def preview_uploaded_mesh_binary(
    request: Request,
    file: UploadFile = File(...),
    shell_thickness: float = Form(...),
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"] = Form(...),
    lattice_pitch: float = Form(...),
    lattice_thickness: float = Form(...),
    lattice_phase: float = Form(0.0),
    voxels_per_lattice_period: int = Form(6),
    compute_backend: ComputeBackend = Form("auto"),
    mesh_backend: MeshBackend = Form("auto"),
    meshing_mode: MeshingMode = Form("uniform"),
    field_storage_mode: UploadedFieldStorageMode = Form("auto"),
    execution_mode: ExecutionMode = Form("auto"),
) -> Response:
    await _reject_legacy_uploaded_mesh_quality_profile(request)
    file_bytes, extension = await _read_uploaded_mesh(file)
    file_hash = _hash_uploaded_mesh_bytes(file_bytes)
    if execution_mode == "queued":
        raise HTTPException(
            status_code=409,
            detail="Binary preview endpoint does not support explicit queued mode",
        )
    try:
        _enforce_uploaded_mesh_memory_guard(
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
            lattice_pitch=lattice_pitch,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
        )
    except MeshUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        mesh, stats, _, _ = await asyncio.to_thread(
            _run_uploaded_mesh_preview_meshdata,
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
            mesh_backend=mesh_backend,
            meshing_mode=meshing_mode,
            field_storage_mode=field_storage_mode,
            encode_response_payloads=False,
            cache_result=True,
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


@app.post("/api/v1/mesh/commit", response_model=PreviewMeshResponse)
async def commit_uploaded_mesh(
    request: Request,
    file: UploadFile = File(...),
    shell_thickness: float = Form(...),
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"] = Form(...),
    lattice_pitch: float = Form(...),
    lattice_thickness: float = Form(...),
    lattice_phase: float = Form(0.0),
    voxels_per_lattice_period: int = Form(6),
    compute_backend: ComputeBackend = Form("auto"),
    mesh_backend: MeshBackend = Form("auto"),
    meshing_mode: MeshingMode = Form("uniform"),
    field_storage_mode: UploadedFieldStorageMode = Form("auto"),
) -> PreviewMeshResponse:
    await _reject_legacy_uploaded_mesh_quality_profile(request)
    file_bytes, extension = await _read_uploaded_mesh(file)
    file_hash = _hash_uploaded_mesh_bytes(file_bytes)
    try:
        mesh, stats, _ = await asyncio.to_thread(
            _run_uploaded_mesh_commit_meshdata,
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
            mesh_backend=mesh_backend,
            meshing_mode=meshing_mode,
            field_storage_mode=field_storage_mode,
            encode_response_payloads=True,
        )
        return PreviewMeshResponse(mesh=_encode_mesh_payload(mesh), stats=stats)
    except MeshUploadError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MeshingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="commit_uploaded_mesh_inline")


@app.post("/api/v1/mesh/commit.binary", response_model=None)
async def commit_uploaded_mesh_binary(
    request: Request,
    file: UploadFile = File(...),
    shell_thickness: float = Form(...),
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"] = Form(...),
    lattice_pitch: float = Form(...),
    lattice_thickness: float = Form(...),
    lattice_phase: float = Form(0.0),
    voxels_per_lattice_period: int = Form(6),
    compute_backend: ComputeBackend = Form("auto"),
    mesh_backend: MeshBackend = Form("auto"),
    meshing_mode: MeshingMode = Form("uniform"),
    field_storage_mode: UploadedFieldStorageMode = Form("auto"),
) -> Response:
    await _reject_legacy_uploaded_mesh_quality_profile(request)
    file_bytes, extension = await _read_uploaded_mesh(file)
    file_hash = _hash_uploaded_mesh_bytes(file_bytes)
    try:
        mesh, stats, _ = await asyncio.to_thread(
            _run_uploaded_mesh_commit_meshdata,
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
            mesh_backend=mesh_backend,
            meshing_mode=meshing_mode,
            field_storage_mode=field_storage_mode,
            encode_response_payloads=False,
        )
        headers = {
            "X-SDF-Stats": _stats_header_value(stats),
            "X-SDF-Vertex-Count": str(int(mesh.vertices.shape[0])),
            "X-SDF-Face-Count": str(int(mesh.faces.shape[0])),
        }
        return Response(content=_pack_mesh_binary(mesh), media_type="application/octet-stream", headers=headers)
    except MeshUploadError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MeshingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="commit_uploaded_mesh_binary_inline")


@app.post("/api/v1/mesh/field", response_model=UploadedPreviewFieldResponse)
async def preview_uploaded_mesh_field(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    shell_thickness: float = Form(...),
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"] = Form(...),
    lattice_pitch: float = Form(...),
    lattice_thickness: float = Form(...),
    lattice_phase: float = Form(0.0),
    voxels_per_lattice_period: int = Form(6),
    compute_backend: ComputeBackend = Form("auto"),
    field_storage_mode: UploadedFieldStorageMode = Form("auto"),
) -> UploadedPreviewFieldResponse:
    trace_id = uuid4().hex
    route_started = time.perf_counter()
    await _reject_legacy_uploaded_mesh_quality_profile(request)
    upload_read_start = time.perf_counter()
    file_bytes, extension = await _read_uploaded_mesh(file)
    file_hash = _hash_uploaded_mesh_bytes(file_bytes)
    server_upload_read_ms = (time.perf_counter() - upload_read_start) * 1000.0
    try:
        _enforce_uploaded_mesh_memory_guard(
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
            lattice_pitch=lattice_pitch,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
        )
    except MeshUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        host_sdf, field, bounds, stats, audit = await asyncio.to_thread(
            _run_uploaded_mesh_field_preview_data_with_audit,
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
            field_storage_mode=field_storage_mode,
        )
        final_audit = replace(
            audit,
            server_upload_read_ms=server_upload_read_ms,
            server_handler_total_ms=(time.perf_counter() - route_started) * 1000.0,
        )
        _record_uploaded_field_preview_server_trace(
            trace_id=trace_id,
            route="/api/v1/mesh/field",
            extension=extension,
            audit=final_audit,
        )
        response.headers["X-SDF-Trace-Id"] = trace_id
        return UploadedPreviewFieldResponse(
            field=_build_uploaded_field_payload(field, bounds),
            host_field=_build_uploaded_host_field_payload(host_sdf, bounds),
            stats=stats,
        )
    except (MeshUploadError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="preview_uploaded_mesh_field_inline")


@app.post("/api/v1/mesh/field.binary", response_model=None)
async def preview_uploaded_mesh_field_binary(
    request: Request,
    file: UploadFile = File(...),
    shell_thickness: float = Form(...),
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"] = Form(...),
    lattice_pitch: float = Form(...),
    lattice_thickness: float = Form(...),
    lattice_phase: float = Form(0.0),
    voxels_per_lattice_period: int = Form(6),
    compute_backend: ComputeBackend = Form("auto"),
    field_storage_mode: UploadedFieldStorageMode = Form("auto"),
) -> Response:
    trace_id = uuid4().hex
    route_started = time.perf_counter()
    await _reject_legacy_uploaded_mesh_quality_profile(request)
    upload_read_start = time.perf_counter()
    file_bytes, extension = await _read_uploaded_mesh(file)
    file_hash = _hash_uploaded_mesh_bytes(file_bytes)
    server_upload_read_ms = (time.perf_counter() - upload_read_start) * 1000.0
    try:
        _enforce_uploaded_mesh_memory_guard(
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
            lattice_pitch=lattice_pitch,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
        )
    except MeshUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        host_sdf, field, bounds, stats, audit = await asyncio.to_thread(
            _run_uploaded_mesh_field_preview_data_with_audit,
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
            field_storage_mode=field_storage_mode,
        )
        pack_started = time.perf_counter()
        packet = _pack_field_binary(host_sdf) + _pack_field_binary(field)
        server_pack_binary_ms = (time.perf_counter() - pack_started) * 1000.0
        final_audit = replace(
            audit,
            payload_bytes=len(packet),
            server_upload_read_ms=server_upload_read_ms,
            server_pack_binary_ms=server_pack_binary_ms,
            server_handler_total_ms=(time.perf_counter() - route_started) * 1000.0,
        )
        _record_uploaded_field_preview_server_trace(
            trace_id=trace_id,
            route="/api/v1/mesh/field.binary",
            extension=extension,
            audit=final_audit,
        )
        headers = {
            "X-SDF-Stats": _stats_header_value(stats),
            "X-SDF-Resolution": str(int(field.shape[0])),
            "X-SDF-Bounds": _bounds_header_value(bounds),
            "X-SDF-Trace-Id": trace_id,
        }
        return Response(
            content=packet,
            media_type="application/octet-stream",
            headers=headers,
        )
    except (MeshUploadError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="preview_uploaded_mesh_field_binary_inline")


@app.post("/api/v1/mesh/preprocess", response_model=None)
async def preprocess_uploaded_mesh(
    request: Request,
    file: UploadFile = File(...),
    lattice_pitch: float = Form(...),
    voxels_per_lattice_period: int = Form(6),
    compute_backend: ComputeBackend = Form("auto"),
    field_storage_mode: UploadedFieldStorageMode = Form("auto"),
) -> Response:
    """
    Upload a mesh file and warm the metadata cache.

    Returns the raw outer mesh (no lattice infill) as a binary payload so the
    frontend can display the geometry immediately. The expensive host SDF
    voxelization and EDT steps now happen when the user clicks "Preview Field"
    so the lattice grid can be computed from the selected lattice settings.

    Call this endpoint immediately on file selection, before the user clicks
    "Preview Field".
    """
    preprocess_started = time.perf_counter()
    await _reject_legacy_uploaded_mesh_quality_profile(request)
    file_bytes, extension = await _read_uploaded_mesh(file)
    file_hash = _hash_uploaded_mesh_bytes(file_bytes)

    try:
        # Step 1: Resolve metadata (parse + validate + normals) — warms metadata cache
        metadata = await asyncio.to_thread(
            _resolve_uploaded_mesh_metadata,
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
        )
        memory_estimate, _ = await asyncio.to_thread(
            _resolve_uploaded_mesh_memory_estimate,
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
            lattice_pitch=lattice_pitch,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
            metadata=metadata,
        )
        uploaded_mesh_preprocess_timing_store.set(
            file_hash,
            UploadedMeshPreprocessTimingEntry(
                created_at=time.time(),
                preprocessing_ms=(time.perf_counter() - preprocess_started) * 1000.0,
            ),
        )
    except (MeshUploadError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="preprocess_uploaded_mesh_inline")

    # Return the raw outer mesh so the frontend can display it immediately.
    outer = metadata.outer_mesh
    return Response(
        content=_pack_mesh_binary(outer),
        media_type="application/octet-stream",
        headers={
            "X-SDF-Vertex-Count": str(int(outer.vertices.shape[0])),
            "X-SDF-Face-Count": str(int(outer.faces.shape[0])),
            "X-SDF-Mesh-Span": str(float(memory_estimate.context.mesh_span)),
            "X-SDF-Available-CPU-Bytes": (
                str(int(memory_estimate.context.available_cpu_bytes))
                if memory_estimate.context.available_cpu_bytes is not None
                else ""
            ),
            "X-SDF-Available-GPU-Free-Bytes": (
                str(int(memory_estimate.context.available_gpu_free_bytes))
                if memory_estimate.context.available_gpu_free_bytes is not None
                else ""
            ),
            "X-SDF-Available-GPU-Total-Bytes": (
                str(int(memory_estimate.context.available_gpu_total_bytes))
                if memory_estimate.context.available_gpu_total_bytes is not None
                else ""
            ),
            "X-SDF-CPU-Bytes-Per-Voxel": str(float(memory_estimate.context.cpu_bytes_per_voxel)),
            "X-SDF-GPU-Bytes-Per-Voxel": str(float(memory_estimate.context.gpu_bytes_per_voxel)),
            "X-SDF-Memory-Safety-Factor": str(float(memory_estimate.context.safety_factor)),
        },
    )


@app.post("/api/v1/internal/mesh/field-preview-telemetry", response_model=None, status_code=204)
async def record_uploaded_field_preview_telemetry(payload: UploadedFieldPreviewClientTelemetry) -> Response:
    trace_entry = uploaded_field_preview_trace_store.pop(payload.trace_id)
    if trace_entry is None:
        raise HTTPException(status_code=404, detail="Unknown or expired uploaded field preview trace_id")

    merged = replace(
        trace_entry,
        client_response_wait_ms=payload.client_response_wait_ms,
        client_download_ms=payload.client_download_ms,
        client_decode_ms=payload.client_decode_ms,
        client_texture_upload_and_first_frame_ms=payload.client_texture_upload_and_first_frame_ms,
        client_total_visible_ms=payload.client_total_visible_ms,
    )
    _log_uploaded_field_preview_consolidated_trace(merged)
    return Response(status_code=204)


@app.post("/api/v1/mesh/export", response_model=None)
async def export_uploaded_mesh(
    request: Request,
    file: UploadFile = File(...),
    shell_thickness: float = Form(...),
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"] = Form(...),
    lattice_pitch: float = Form(...),
    lattice_thickness: float = Form(...),
    lattice_phase: float = Form(0.0),
    voxels_per_lattice_period: int = Form(6),
    compute_backend: ComputeBackend = Form("auto"),
    mesh_backend: MeshBackend = Form("auto"),
    meshing_mode: MeshingMode = Form("uniform"),
    field_storage_mode: UploadedFieldStorageMode = Form("auto"),
    format: Literal["stl", "obj"] = Form("stl"),
    execution_mode: ExecutionMode = Form("auto"),
) -> Response | JobAcceptedResponse:
    await _reject_legacy_uploaded_mesh_quality_profile(request)
    file_bytes, extension = await _read_uploaded_mesh(file)
    file_hash = _hash_uploaded_mesh_bytes(file_bytes)
    if _should_queue_uploaded_request(
        file_bytes=file_bytes,
        file_hash=file_hash,
        extension=extension,
        lattice_pitch=lattice_pitch,
        voxels_per_lattice_period=voxels_per_lattice_period,
        execution_mode=execution_mode,
    ):
        try:
            return _enqueue_uploaded_export_job(
                file_bytes=file_bytes,
                file_hash=file_hash,
                extension=extension,
                shell_thickness=shell_thickness,
                lattice_type=lattice_type,
                lattice_pitch=lattice_pitch,
                lattice_thickness=lattice_thickness,
                lattice_phase=lattice_phase,
                voxels_per_lattice_period=voxels_per_lattice_period,
                compute_backend=compute_backend,
                mesh_backend=mesh_backend,
                meshing_mode=meshing_mode,
                field_storage_mode=field_storage_mode,
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
            file_hash=file_hash,
            extension=extension,
            shell_thickness=shell_thickness,
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
            voxels_per_lattice_period=voxels_per_lattice_period,
            compute_backend=compute_backend,
            mesh_backend=mesh_backend,
            meshing_mode=meshing_mode,
            field_storage_mode=field_storage_mode,
            encode_response_payloads=False,
            cache_result=False,
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
            request_start = time.perf_counter()
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
            file_hash = _hash_uploaded_mesh_bytes(file_bytes)
            metadata = _resolve_uploaded_mesh_metadata(
                file_bytes=file_bytes,
                file_hash=file_hash,
                extension=extension,
            )
            resolution, _ = _resolve_mesh_resolution(
                payload.lattice_pitch,
                metadata.mesh_span,
                voxels_per_lattice_period,
            )
            cache_key = hash_uploaded_mesh_request(
                file_hash=file_hash,
                extension=extension,
                resolution=resolution,
                shell_thickness=payload.shell_thickness,
                lattice_type=payload.lattice_type,
                lattice_pitch=payload.lattice_pitch,
                lattice_thickness=payload.lattice_thickness,
                lattice_phase=payload.lattice_phase,
                voxels_per_lattice_period=voxels_per_lattice_period,
                compute_backend=payload.compute_backend,
                mesh_backend=payload.mesh_backend,
                meshing_mode=payload.meshing_mode,
                field_storage_mode=payload.field_storage_mode,
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
                cached_stats["field_cache_hit"] = True
                cached_stats["mesh_cache_hit"] = True
                cached_stats["eval_ms"] = 0.0
                cached_stats["mesh_ms"] = 0.0
                cached_field = _build_uploaded_field_payload_from_cache_entry(
                    cache_entry=cached_mesh,
                    file_hash=file_hash,
                    extension=extension,
                    resolution=resolution,
                    shell_thickness=payload.shell_thickness,
                    lattice_type=payload.lattice_type,
                    lattice_pitch=payload.lattice_pitch,
                    lattice_thickness=payload.lattice_thickness,
                    lattice_phase=payload.lattice_phase,
                    voxels_per_lattice_period=voxels_per_lattice_period,
                    compute_backend=payload.compute_backend,
                    field_storage_mode=payload.field_storage_mode,
                )
                if cached_field is not None:
                    await websocket.send_json(
                        UploadedMeshPreviewWsResponse(
                            phase="field",
                            field=cached_field,
                            stats=PreviewStats(
                                eval_ms=0.0,
                                mesh_ms=None,
                                tri_count=0,
                                voxel_count=int(cached_field.resolution**3),
                                cache_hit=True,
                                field_cache_hit=True,
                                mesh_cache_hit=False,
                                compute_backend=str(cached_stats.get("compute_backend", "cpu")),
                                mesh_backend="cpu",
                                preview_mode="field",
                            ),
                        ).model_dump()
                    )
                cached_mesh_payload = cached_mesh.mesh
                if cached_mesh_payload is None:
                    cached_mesh_payload = _encode_mesh_payload(
                        MeshData(
                            vertices=cached_mesh.vertices,
                            faces=cached_mesh.faces,
                            normals=cached_mesh.normals,
                        )
                    )
                await websocket.send_json(
                    UploadedMeshPreviewWsResponse(
                        phase="mesh",
                        mesh=cached_mesh_payload,
                        stats=PreviewStats.model_validate(cached_stats),
                    ).model_dump()
                )
                return

            # Phase 1: field evaluation — offloaded to thread pool to avoid
            # blocking the event loop (which would starve the websocket
            # keepalive ping and cause an AssertionError).
            def _compute_field_phase():
                try:
                    composed = _resolve_uploaded_composed_field(
                        file_bytes=file_bytes,
                        file_hash=file_hash,
                        extension=extension,
                        shell_thickness=payload.shell_thickness,
                        lattice_type=payload.lattice_type,
                        lattice_pitch=payload.lattice_pitch,
                        lattice_thickness=payload.lattice_thickness,
                        lattice_phase=payload.lattice_phase,
                        voxels_per_lattice_period=voxels_per_lattice_period,
                        compute_backend=payload.compute_backend,
                        parsed=metadata.parsed,
                        resolution=resolution,
                        field_storage_mode=payload.field_storage_mode,
                    )
                    parsed = composed.parsed
                    bounds = composed.bounds
                    host_sdf = composed.host_sdf
                    sparse_block_size = composed.sparse_block_size
                    active_blocks = composed.active_blocks
                    field = composed.field
                    eval_backend_used = composed.eval_backend_used
                    eval_ms = composed.eval_ms
                    field_cache_hit = composed.field_cache_hit
                    field_payload = _build_uploaded_field_payload(field, bounds)
                    logger.debug(
                        "mesh_upload field_phase eval_ms=%.2f cache_hit=%s",
                        eval_ms,
                        field_cache_hit,
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
                        field_cache_hit,
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
                field_cache_hit,
                field_payload,
            ) = (
                await asyncio.to_thread(_compute_field_phase)
            )

            field_stats = PreviewStats(
                eval_ms=eval_ms,
                mesh_ms=None,
                tri_count=0,
                voxel_count=int(field.size),
                cache_hit=field_cache_hit,
                field_cache_hit=field_cache_hit,
                mesh_cache_hit=False,
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
                    _maybe_evict_uploaded_composed_field_cache_before_meshing(field)
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
                    mesh = _merge_meshes(metadata.outer_mesh, interior)
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
                cache_hit=field_cache_hit,
                field_cache_hit=field_cache_hit,
                mesh_cache_hit=False,
                compute_backend=eval_backend_used,
                mesh_backend=mesh_backend_used,
                preview_mode="mesh",
            )
            uploaded_mesh_preview_cache.set(
                cache_key,
                UploadedMeshCacheEntry(
                    mesh=mesh_payload,
                    vertices=_freeze_cached_array(mesh_np.vertices, np.float64),
                    faces=_freeze_cached_array(mesh_np.faces, np.int32),
                    normals=_freeze_cached_array(mesh_np.normals, np.float64),
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
