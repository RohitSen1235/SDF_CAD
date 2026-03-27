from __future__ import annotations

import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import base64

from .gpu_memory import cleanup_runtime_memory
from .meshing import iter_obj_chunks, iter_stl_chunks
from .models import (
    ExportMeshRequest,
    PreviewMeshRequest,
    StructuralOptimizationIterationWebhookRequest,
    StructuralOptimizationRequest,
)
from .optimization import run_structural_optimization
from .structural_progress import get_structural_progress_store
from .worker import celery_app

JOB_EXPORT_DIR = Path(os.getenv("SDF_CAD_JOB_EXPORT_DIR", "/tmp/sdf_cad_jobs"))
JOB_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
STRUCTURAL_CALLBACK_BASE_URL = os.getenv("SDF_CAD_INTERNAL_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
STRUCTURAL_CALLBACK_RETRIES = int(os.getenv("SDF_CAD_STRUCTURAL_CALLBACK_RETRIES", "5"))
STRUCTURAL_CALLBACK_TIMEOUT_SECONDS = float(os.getenv("SDF_CAD_STRUCTURAL_CALLBACK_TIMEOUT_SECONDS", "30.0"))
STRUCTURAL_CALLBACK_BACKOFF_SECONDS = float(os.getenv("SDF_CAD_STRUCTURAL_CALLBACK_BACKOFF_SECONDS", "1.0"))


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
    from .main import _hash_uploaded_mesh_bytes, _run_uploaded_mesh_preview

    try:
        file_bytes = base64.b64decode(str(payload["file_data_b64"]))
        file_hash = str(payload.get("file_hash") or _hash_uploaded_mesh_bytes(file_bytes))
        preview = _run_uploaded_mesh_preview(
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=str(payload["extension"]),
            shell_thickness=float(payload["shell_thickness"]),
            lattice_type=str(payload["lattice_type"]),
            lattice_pitch=float(payload["lattice_pitch"]),
            lattice_thickness=float(payload["lattice_thickness"]),
            lattice_phase=float(payload.get("lattice_phase", 0.0)),
            voxels_per_lattice_period=int(payload.get("voxels_per_lattice_period", 6)),
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
    from .main import _hash_uploaded_mesh_bytes, _run_uploaded_mesh_preview_meshdata

    try:
        file_bytes = base64.b64decode(str(payload["file_data_b64"]))
        file_hash = str(payload.get("file_hash") or _hash_uploaded_mesh_bytes(file_bytes))
        mesh, _, _, _ = _run_uploaded_mesh_preview_meshdata(
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=str(payload["extension"]),
            shell_thickness=float(payload["shell_thickness"]),
            lattice_type=str(payload["lattice_type"]),
            lattice_pitch=float(payload["lattice_pitch"]),
            lattice_thickness=float(payload["lattice_thickness"]),
            lattice_phase=float(payload.get("lattice_phase", 0.0)),
            voxels_per_lattice_period=int(payload.get("voxels_per_lattice_period", 6)),
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


def _post_structural_iteration_callback(
    *,
    job_id: str,
    callback_token: str,
    payload: StructuralOptimizationIterationWebhookRequest,
) -> None:
    url = f"{STRUCTURAL_CALLBACK_BASE_URL}/api/v1/internal/optimization/structural/jobs/{job_id}/iterations"
    body = payload.model_dump_json().encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(1, STRUCTURAL_CALLBACK_RETRIES + 1):
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {callback_token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=STRUCTURAL_CALLBACK_TIMEOUT_SECONDS) as response:
                if 200 <= int(getattr(response, "status", 0)) < 300:
                    return
                raise RuntimeError(f"Unexpected structural callback status: {getattr(response, 'status', 'unknown')}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"Structural callback rejected with HTTP {exc.code}: {detail or exc.reason}")
        except Exception as exc:  # pragma: no cover - exercised via retry tests
            last_error = exc
        if attempt < STRUCTURAL_CALLBACK_RETRIES:
            time.sleep(STRUCTURAL_CALLBACK_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"Structural optimization callback failed after {STRUCTURAL_CALLBACK_RETRIES} attempts") from last_error


def _run_structural_optimization_job(payload: dict[str, Any], *, job_id: str) -> dict[str, Any]:
    callback_token = str(payload.get("_callback_token", "") or "")
    try:
        if not callback_token:
            raise RuntimeError("Missing structural optimization callback token")
        if not job_id:
            raise RuntimeError("Structural optimization job is missing a task id")
        payload = dict(payload)
        payload.pop("_callback_token", None)
        store = get_structural_progress_store()
        store.mark_running(job_id)

        req = StructuralOptimizationRequest.model_validate(payload)
        result = run_structural_optimization(
            req,
            iteration_callback=lambda iteration_payload: _post_structural_iteration_callback(
                job_id=job_id,
                callback_token=callback_token,
                payload=iteration_payload,
            ),
        )
        return {"kind": "structural_optimization", "payload": result.model_dump(mode="json")}
    except Exception as exc:
        if job_id:
            try:
                failure_payload = StructuralOptimizationIterationWebhookRequest(
                    failure_detail=str(exc),
                    is_final=True,
                )
                if callback_token:
                    _post_structural_iteration_callback(
                        job_id=job_id,
                        callback_token=callback_token,
                        payload=failure_payload,
                    )
                else:
                    get_structural_progress_store().mark_failed(job_id, str(exc))
            except Exception:
                try:
                    get_structural_progress_store().mark_failed(job_id, str(exc))
                except Exception:
                    pass
        raise


@celery_app.task(bind=True, name="jobs.structural_optimization")
def structural_optimization_job(self, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        job_id = str(getattr(self.request, "id", "") or "")
        return _run_structural_optimization_job(payload, job_id=job_id)
    finally:
        cleanup_runtime_memory(reason="structural_optimization_job")


def _purge_old_job_files(max_age_seconds: int = 6 * 3600) -> None:
    now = int(time.time())
    for entry in JOB_EXPORT_DIR.glob("sdfcad-job-*"):
        try:
            age = now - int(entry.stat().st_mtime)
            if age > max_age_seconds:
                entry.unlink(missing_ok=True)
        except Exception:
            continue
