from __future__ import annotations

import copy
import time

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .cache import (
    CompileCacheEntry,
    MeshCacheEntry,
    field_preview_cache,
    hash_preview_request,
    hash_source,
    mesh_preview_cache,
    scene_compile_cache,
)
from .dsl import DslError, compile_source_with_diagnostics
from .evaluator import EvaluationError, ensure_scene_valid, evaluate_scene_field, merge_parameter_values
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
