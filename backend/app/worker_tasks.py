from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import base64

from .gpu_memory import cleanup_runtime_memory
from .meshing import iter_obj_chunks, iter_stl_chunks
from .models import ExportMeshRequest, PreviewMeshRequest
from .worker import celery_app

JOB_EXPORT_DIR = Path(os.getenv("SDF_CAD_JOB_EXPORT_DIR", "/tmp/sdf_cad_jobs"))
JOB_EXPORT_DIR.mkdir(parents=True, exist_ok=True)


@celery_app.task(name="jobs.preview_mesh")
def preview_mesh_job(payload: dict[str, Any]) -> dict[str, Any]:
    from .main import _resolve_grid, _run_preview

    req = PreviewMeshRequest.model_validate(payload)
    grid = _resolve_grid(req.grid, req.quality_profile)
    try:
        preview = _run_preview(
            req.scene_ir,
            req.parameter_values,
            grid,
            compute_precision=req.compute_precision,
            compute_backend=req.compute_backend,
            mesh_backend=req.mesh_backend,
            meshing_mode=req.meshing_mode,
        )
        return {"kind": "preview_mesh", "payload": preview.model_dump(mode="json")}
    finally:
        cleanup_runtime_memory(reason="preview_mesh_job")


@celery_app.task(name="jobs.export_mesh")
def export_mesh_job(payload: dict[str, Any]) -> dict[str, Any]:
    from .main import _resolve_grid, _run_preview_meshdata

    req = ExportMeshRequest.model_validate(payload)
    grid = _resolve_grid(req.grid, req.quality_profile)
    try:
        mesh, _, _ = _run_preview_meshdata(
            req.scene_ir,
            req.parameter_values,
            grid,
            compute_precision=req.compute_precision,
            compute_backend=req.compute_backend,
            mesh_backend=req.mesh_backend,
            meshing_mode=req.meshing_mode,
            encode_mesh_payload=False,
            cache_result=False,
        )
        suffix = ".stl" if req.format == "stl" else ".obj"
        fd, tmp_path = tempfile.mkstemp(prefix="sdfcad-job-", suffix=suffix, dir=str(JOB_EXPORT_DIR))
        with os.fdopen(fd, "wb") as out:
            if req.format == "stl":
                for chunk in iter_stl_chunks(mesh):
                    out.write(chunk)
                media_type = "model/stl"
                filename = "model.stl"
            else:
                for chunk in iter_obj_chunks(mesh):
                    out.write(chunk)
                media_type = "text/plain"
                filename = "model.obj"
        return {
            "kind": "export_mesh",
            "file_path": tmp_path,
            "media_type": media_type,
            "filename": filename,
        }
    finally:
        cleanup_runtime_memory(reason="export_mesh_job")


@celery_app.task(name="jobs.preview_uploaded_mesh")
def preview_uploaded_mesh_job(payload: dict[str, Any]) -> dict[str, Any]:
    from .main import _run_uploaded_mesh_preview

    try:
        preview = _run_uploaded_mesh_preview(
            file_bytes=base64.b64decode(str(payload["file_data_b64"])),
            extension=str(payload["extension"]),
            shell_thickness=float(payload["shell_thickness"]),
            lattice_type=str(payload["lattice_type"]),
            lattice_pitch=float(payload["lattice_pitch"]),
            lattice_thickness=float(payload["lattice_thickness"]),
            lattice_phase=float(payload.get("lattice_phase", 0.0)),
            quality_profile=str(payload["quality_profile"]),
            compute_backend=str(payload.get("compute_backend", "auto")),
            mesh_backend=str(payload.get("mesh_backend", "auto")),
            meshing_mode=str(payload.get("meshing_mode", "uniform")),
            field_storage_mode=str(payload.get("field_storage_mode", "auto")),
        )
        return {"kind": "preview_uploaded_mesh", "payload": preview.model_dump(mode="json")}
    finally:
        cleanup_runtime_memory(reason="preview_uploaded_mesh_job")


@celery_app.task(name="jobs.export_uploaded_mesh")
def export_uploaded_mesh_job(payload: dict[str, Any]) -> dict[str, Any]:
    from .main import _run_uploaded_mesh_preview_meshdata

    try:
        mesh, _, _, _ = _run_uploaded_mesh_preview_meshdata(
            file_bytes=base64.b64decode(str(payload["file_data_b64"])),
            extension=str(payload["extension"]),
            shell_thickness=float(payload["shell_thickness"]),
            lattice_type=str(payload["lattice_type"]),
            lattice_pitch=float(payload["lattice_pitch"]),
            lattice_thickness=float(payload["lattice_thickness"]),
            lattice_phase=float(payload.get("lattice_phase", 0.0)),
            quality_profile=str(payload["quality_profile"]),
            compute_backend=str(payload.get("compute_backend", "auto")),
            mesh_backend=str(payload.get("mesh_backend", "auto")),
            meshing_mode=str(payload.get("meshing_mode", "uniform")),
            field_storage_mode=str(payload.get("field_storage_mode", "auto")),
            encode_response_payloads=False,
            cache_result=False,
        )
        export_format = str(payload.get("format", "stl")).lower()
        suffix = ".stl" if export_format == "stl" else ".obj"
        fd, tmp_path = tempfile.mkstemp(prefix="sdfcad-job-", suffix=suffix, dir=str(JOB_EXPORT_DIR))
        with os.fdopen(fd, "wb") as out:
            if export_format == "stl":
                for chunk in iter_stl_chunks(mesh):
                    out.write(chunk)
                media_type = "model/stl"
                filename = "mesh-lattice.stl"
            else:
                for chunk in iter_obj_chunks(mesh):
                    out.write(chunk)
                media_type = "text/plain"
                filename = "mesh-lattice.obj"
        return {
            "kind": "export_uploaded_mesh",
            "file_path": tmp_path,
            "media_type": media_type,
            "filename": filename,
        }
    finally:
        cleanup_runtime_memory(reason="export_uploaded_mesh_job")


def _purge_old_job_files(max_age_seconds: int = 6 * 3600) -> None:
    now = int(time.time())
    for entry in JOB_EXPORT_DIR.glob("sdfcad-job-*"):
        try:
            age = now - int(entry.stat().st_mtime)
            if age > max_age_seconds:
                entry.unlink(missing_ok=True)
        except Exception:
            continue
