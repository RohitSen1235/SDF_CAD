from __future__ import annotations

import copy
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
from .evaluator import EvaluationError, ensure_scene_valid, evaluate_scene_field, merge_parameter_values
from .mesh_upload import (
    MeshUploadError,
    ParsedMesh,
    build_host_field,
    compose_hollow_lattice_field,
    parse_mesh_bytes,
    validate_triangle_mesh,
)
from .meshing import MeshData, MeshingError, build_mesh, mesh_to_obj, mesh_to_stl
from .models import (
    CompileSceneRequest,
    CompileSceneResponse,
    ExportMeshRequest,
    GridConfig,
    PreviewMeshRequest,
    PreviewMeshResponse,
    PreviewStats,
    PreviewWsRequest,
    PreviewWsResponse,
    QualityProfile,
)

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
MESH_UPLOAD_MAX_BYTES = 50 * 1024 * 1024

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


def _run_preview(
    scene_ir,
    param_values: dict[str, float],
    grid: GridConfig,
) -> PreviewMeshResponse:
    ensure_scene_valid(scene_ir)
    merged_params = merge_parameter_values(scene_ir, param_values)

    cache_key = hash_preview_request(scene_ir, merged_params, grid)
    cached_mesh = mesh_preview_cache.get(cache_key)
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
    field = field_preview_cache.get(cache_key)
    field_cache_hit = field is not None
    if field is None:
        field = evaluate_scene_field(scene_ir, merged_params, grid)
        field_preview_cache.set(cache_key, field)
    eval_ms = (time.perf_counter() - eval_start) * 1000.0
    if eval_ms > EVAL_TIMEOUT_SECONDS * 1000.0:
        raise HTTPException(status_code=408, detail="Field evaluation timeout exceeded")

    mesh_start = time.perf_counter()
    mesh = build_mesh(field, grid.bounds)
    mesh_ms = (time.perf_counter() - mesh_start) * 1000.0

    vertices = mesh.vertices.tolist()
    indices = mesh.faces.tolist()
    normals = mesh.normals.tolist()
    stats = PreviewStats(
        eval_ms=eval_ms,
        mesh_ms=mesh_ms,
        tri_count=int(mesh.faces.shape[0]),
        cache_hit=field_cache_hit,
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
    )
    cached_mesh = uploaded_mesh_preview_cache.get(cache_key)
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
    parsed, bounds, host_sdf = _resolve_uploaded_host_field(
        file_bytes=file_bytes,
        extension=extension,
        quality_profile=quality_profile,
    )
    field = compose_hollow_lattice_field(
        host_sdf,
        bounds,
        shell_thickness=shell_thickness,
        lattice_type=lattice_type,
        lattice_pitch=lattice_pitch,
        lattice_thickness=lattice_thickness,
        lattice_phase=lattice_phase,
    )
    eval_ms = (time.perf_counter() - eval_start) * 1000.0

    mesh_start = time.perf_counter()
    generated = build_mesh(field, bounds)
    interior = _strip_outer_surface(generated, host_sdf, bounds)
    outer = _meshdata_from_parsed(parsed)
    mesh = _merge_meshes(outer, interior)
    mesh_ms = (time.perf_counter() - mesh_start) * 1000.0

    vertices = mesh.vertices.tolist()
    indices = mesh.faces.tolist()
    normals = mesh.normals.tolist()
    stats = PreviewStats(
        eval_ms=eval_ms,
        mesh_ms=mesh_ms,
        tri_count=int(mesh.faces.shape[0]),
        cache_hit=False,
    )

    uploaded_mesh_preview_cache.set(
        cache_key,
        UploadedMeshCacheEntry(
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


@app.post("/api/v1/preview/mesh", response_model=PreviewMeshResponse)
def preview_mesh(payload: PreviewMeshRequest) -> PreviewMeshResponse:
    grid = _resolve_grid(payload.grid, payload.quality_profile)
    try:
        return _run_preview(payload.scene_ir, payload.parameter_values, grid)
    except (DslError, EvaluationError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/export")
def export_mesh(payload: ExportMeshRequest) -> Response:
    grid = _resolve_grid(payload.grid, payload.quality_profile)

    try:
        preview = _run_preview(payload.scene_ir, payload.parameter_values, grid)
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

                coarse_preview = _run_preview(payload.scene_ir, payload.parameter_values, coarse_grid)
                await websocket.send_json(
                    PreviewWsResponse(
                        phase="coarse", mesh=coarse_preview.mesh, stats=coarse_preview.stats
                    ).model_dump()
                )

                if fine_res != coarse_res:
                    fine_preview = _run_preview(payload.scene_ir, payload.parameter_values, fine_grid)
                    await websocket.send_json(
                        PreviewWsResponse(
                            phase="fine", mesh=fine_preview.mesh, stats=fine_preview.stats
                        ).model_dump()
                    )
            except Exception as exc:
                await websocket.send_json(PreviewWsResponse(phase="error", error=str(exc)).model_dump())
    except WebSocketDisconnect:
        return
