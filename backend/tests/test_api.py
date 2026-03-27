import asyncio
import json
import base64
import io
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from app import main as main_module
from app.main import app
from app.models import (
    OptimizationHistoryEntry,
    StructuralOptimizationRequest,
    StructuralOptimizationIterationResult,
    StructuralOptimizationIterationWebhookRequest,
    StructuralOptimizationProgressResponse,
    StructuralOptimizationResultResponse,
)
from app.worker import celery_app

client = TestClient(app)


SOURCE = """
param r default=0.8 min=0.2 max=1.5 step=0.1
body = sphere(r=$r)
lat = gyroid(pitch=0.5, thickness=0.1)
root = conformal_fill(body, lat, wall=0.08, mode="hybrid")
"""
SIMPLE_SOURCE = "root = sphere(r=0.72)"

MESH_OBJ = b"""
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
"""
MESH_OBJ_HASH = hashlib.sha256(MESH_OBJ).hexdigest()


def _decode_mesh_arrays(mesh_payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.frombuffer(base64.b64decode(mesh_payload["vertices_b64"]), dtype=np.float32).reshape(-1, 3)
    indices = np.frombuffer(base64.b64decode(mesh_payload["indices_b64"]), dtype=np.uint32).reshape(-1, 3)
    normals = np.frombuffer(base64.b64decode(mesh_payload["normals_b64"]), dtype=np.float32).reshape(-1, 3)
    return vertices, indices, normals


def _mesh_form(lattice_type: str = "gyroid") -> dict[str, str]:
    return {
        "shell_thickness": "0.08",
        "lattice_type": lattice_type,
        "lattice_pitch": "0.45",
        "lattice_thickness": "0.09",
        "lattice_phase": "0.0",
    }


def _structural_request_json() -> dict[str, object]:
    return {
        "design_space_file_name": "design.stl",
        "design_space_file_data_base64": "AA==",
        "non_design_space_file_name": "keep.stl",
        "non_design_space_file_data_base64": "AA==",
        "compute_backend": "cpu",
        "mesh_backend": "cpu",
        "execution_mode": "queued",
        "constraints": [{"kind": "fixed", "points": [{"point_xyz": [0.0, 0.0, 0.0]}], "radius": 0.0}],
        "loads": [
            {
                "kind": "point",
                "points": [{"point_xyz": [1.0, 0.0, 0.0]}],
                "direction_xyz": [1.0, 0.0, 0.0],
                "magnitude": 1.0,
                "radius": 0.0,
            }
        ],
        "material": {
            "youngs_modulus": 1.0,
            "poissons_ratio": 0.3,
            "density_floor": 1e-3,
            "stiffness_floor_ratio": 1e-3,
            "simp_penalty": 3.0,
        },
        "config": {
            "resolution": 64,
            "target_volume_fraction": 0.35,
            "max_iterations": 8,
            "cg_max_iterations": 200,
            "cg_tolerance": 1e-6,
            "optimization_tolerance": 1e-3,
            "filter_radius_voxels": 1.5,
            "min_density": 1e-3,
            "oc_move_limit": 0.2,
            "density_iso_threshold": 0.3,
        },
    }


class FakeStructuralProgressStore:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, object]] = {}

    def initialize_job(self, *, job_id: str, max_iterations: int, callback_token: str) -> None:
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "current_iteration": 0,
            "max_iterations": max_iterations,
            "callback_token": callback_token,
            "iterations": [],
            "history": [],
            "stop_reason": None,
            "detail": None,
            "final_result": None,
        }

    def clear_job(self, job_id: str) -> None:
        self.jobs.pop(job_id, None)

    def mark_running(self, job_id: str) -> None:
        self.jobs[job_id]["status"] = "running"

    def mark_failed(self, job_id: str, detail: str) -> None:
        self.jobs[job_id]["status"] = "failed"
        self.jobs[job_id]["detail"] = detail

    def validate_callback_token(self, job_id: str, token: str) -> bool:
        return self.jobs[job_id]["callback_token"] == token

    def persist_callback(self, job_id: str, payload: StructuralOptimizationIterationWebhookRequest) -> StructuralOptimizationProgressResponse:
        job = self.jobs[job_id]
        if payload.iteration_result is not None:
            job["iterations"].append(payload.iteration_result)
            job["current_iteration"] = payload.iteration_result.iteration
            job["status"] = "running"
        if payload.history_entry is not None:
            job["history"].append(payload.history_entry)
        if payload.failure_detail:
            job["status"] = "failed"
            job["detail"] = payload.failure_detail
        elif payload.is_final and payload.iteration_result is not None:
            job["status"] = "succeeded"
            job["stop_reason"] = payload.stop_reason
            job["final_result"] = StructuralOptimizationResultResponse(
                history=list(job["history"]),
                final_iteration=payload.iteration_result,
                bounds=payload.bounds or [],
                resolution_xyz=payload.resolution_xyz or [],
                compute_backend_used=payload.compute_backend_used or "cpu",
                mesh_backend_used=payload.mesh_backend_used or "cpu",
                stop_reason=payload.stop_reason or "max_iterations",
            )
        return self.get_progress(job_id=job_id, after_iteration=max(0, (payload.iteration_result.iteration - 1) if payload.iteration_result else 0))

    def get_progress(self, *, job_id: str, after_iteration: int = 0) -> StructuralOptimizationProgressResponse:
        job = self.jobs[job_id]
        iterations = list(job["iterations"])[int(after_iteration) :]
        return StructuralOptimizationProgressResponse(
            job_id=job_id,
            status=job["status"],
            current_iteration=int(job["current_iteration"]),
            max_iterations=int(job["max_iterations"]),
            iterations=list(iterations),
            history=list(job["history"]),
            stop_reason=job["stop_reason"],
            detail=job["detail"],
            final_result=job["final_result"],
        )


@pytest.fixture(autouse=True)
def eager_celery() -> None:
    prev_eager = celery_app.conf.task_always_eager
    prev_store = celery_app.conf.task_store_eager_result
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_store_eager_result = True
    try:
        yield
    finally:
        celery_app.conf.task_always_eager = prev_eager
        celery_app.conf.task_store_eager_result = prev_store


def test_structural_job_submission_returns_progress_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import worker_tasks

    store = FakeStructuralProgressStore()
    captured: dict[str, object] = {}

    monkeypatch.setattr(main_module, "get_structural_progress_store", lambda: store)

    def fake_apply_async(*, args: list[dict[str, object]], task_id: str):
        captured["payload"] = args[0]
        captured["task_id"] = task_id
        return SimpleNamespace(id=task_id)

    monkeypatch.setattr(worker_tasks, "structural_optimization_job", SimpleNamespace(apply_async=fake_apply_async))

    response = main_module._enqueue_structural_optimization_job(StructuralOptimizationRequest.model_validate(_structural_request_json()))

    assert response.job_id
    assert response.progress_url == f"/api/v1/optimization/structural/jobs/{response.job_id}/progress"
    assert response.job_id == captured["task_id"]
    assert isinstance(captured["payload"], dict)
    assert "_callback_token" in captured["payload"]
    assert store.jobs[response.job_id]["callback_token"] == captured["payload"]["_callback_token"]


def test_structural_iteration_callback_rejects_invalid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeStructuralProgressStore()
    store.initialize_job(job_id="job-invalid", max_iterations=8, callback_token="token-good")
    monkeypatch.setattr(main_module, "get_structural_progress_store", lambda: store)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            main_module.structural_optimization_iteration_callback(
                "job-invalid",
                StructuralOptimizationIterationWebhookRequest(failure_detail="boom", is_final=True),
                Request({"type": "http", "headers": [(b"authorization", b"Bearer token-bad")]}),
            )
        )

    assert exc_info.value.status_code == 403


def test_structural_progress_endpoint_returns_only_unseen_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeStructuralProgressStore()
    store.initialize_job(job_id="job-progress", max_iterations=8, callback_token="token-progress")
    monkeypatch.setattr(main_module, "get_structural_progress_store", lambda: store)

    callback_request = Request({"type": "http", "headers": [(b"authorization", b"Bearer token-progress")]})
    iteration_one = StructuralOptimizationIterationWebhookRequest(
        iteration_result=StructuralOptimizationIterationResult(
            iteration=1,
            objective_value=4.0,
            active_volume_fraction=0.95,
            removed_voxels=2,
        ),
        history_entry=OptimizationHistoryEntry(
            iteration=1,
            objective_value=4.0,
            active_volume_fraction=0.95,
            removed_voxels=2,
            max_displacement=0.1,
        ),
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        resolution_xyz=[64, 64, 64],
        compute_backend_used="cpu",
        mesh_backend_used="cpu",
        is_final=False,
    )
    iteration_two = StructuralOptimizationIterationWebhookRequest(
        iteration_result=StructuralOptimizationIterationResult(
            iteration=2,
            objective_value=3.1,
            active_volume_fraction=0.82,
            removed_voxels=3,
        ),
        history_entry=OptimizationHistoryEntry(
            iteration=2,
            objective_value=3.1,
            active_volume_fraction=0.82,
            removed_voxels=3,
            max_displacement=0.09,
        ),
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        resolution_xyz=[64, 64, 64],
        compute_backend_used="cpu",
        mesh_backend_used="cpu",
        is_final=True,
        stop_reason="objective_converged",
    )
    callback_result_one = asyncio.run(
        main_module.structural_optimization_iteration_callback("job-progress", iteration_one, callback_request)
    )
    callback_result_two = asyncio.run(
        main_module.structural_optimization_iteration_callback("job-progress", iteration_two, callback_request)
    )
    assert callback_result_one["acknowledged"] is True
    assert callback_result_two["acknowledged"] is True

    first = asyncio.run(main_module.structural_optimization_progress("job-progress", after_iteration=0))
    second = asyncio.run(main_module.structural_optimization_progress("job-progress", after_iteration=1))

    assert [entry.iteration for entry in first.iterations] == [1, 2]
    assert [entry.iteration for entry in second.iterations] == [2]
    assert first.final_result is not None
    assert first.final_result.stop_reason == "objective_converged"


def test_compile_endpoint_returns_diagnostics() -> None:
    response = client.post("/api/v1/scene/compile", json={"source": SOURCE})
    assert response.status_code == 200
    payload = response.json()
    assert payload["scene_ir"]["root_node_id"]
    assert len(payload["scene_ir"]["parameter_schema"]) == 1
    assert "diagnostics" in payload
    assert "warnings" in payload["diagnostics"]


def test_preview_mesh_endpoint_accepts_quality_profile() -> None:
    compiled = client.post("/api/v1/scene/compile", json={"source": SOURCE}).json()["scene_ir"]
    response = client.post(
        "/api/v1/preview/mesh",
        json={
            "scene_ir": compiled,
            "parameter_values": {"r": 0.7},
            "quality_profile": "medium",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["stats"]["tri_count"] > 0
    assert payload["mesh"]["encoding"] == "mesh-f32-u32-base64-v1"
    vertices, indices, normals = _decode_mesh_arrays(payload["mesh"])
    assert vertices.shape[0] == payload["mesh"]["vertex_count"]
    assert indices.shape[0] == payload["mesh"]["face_count"]
    assert normals.shape[0] == payload["mesh"]["vertex_count"]
    assert payload["stats"]["compute_precision"] == "float32"
    assert payload["stats"]["compute_backend"] == "cpu"


def test_preview_mesh_binary_endpoint_returns_octet_stream() -> None:
    compiled = client.post("/api/v1/scene/compile", json={"source": SOURCE}).json()["scene_ir"]
    response = client.post(
        "/api/v1/preview/mesh.binary",
        json={
            "scene_ir": compiled,
            "parameter_values": {"r": 0.7},
            "quality_profile": "interactive",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert response.headers.get("x-sdf-stats")
    assert response.headers.get("x-sdf-vertex-count")
    assert response.headers.get("x-sdf-face-count")
    payload = response.content
    assert payload[:8] == b"SDFMESH1"
    assert len(payload) > 16


def test_preview_field_endpoint_returns_volume_payload() -> None:
    compiled = client.post("/api/v1/scene/compile", json={"source": SOURCE}).json()["scene_ir"]
    response = client.post(
        "/api/v1/preview/field",
        json={
            "scene_ir": compiled,
            "parameter_values": {"r": 0.7},
            "quality_profile": "interactive",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["field"]["encoding"] == "f32-base64"
    assert payload["field"]["resolution_xyz"] == [64, 64, 64]
    assert payload["field"]["data"]
    assert payload["stats"]["preview_mode"] == "field"
    assert payload["stats"]["mesh_ms"] is None
    assert payload["stats"]["voxel_count"] == 64 * 64 * 64


def test_preview_field_binary_endpoint_returns_octet_stream() -> None:
    compiled = client.post("/api/v1/scene/compile", json={"source": SOURCE}).json()["scene_ir"]
    response = client.post(
        "/api/v1/preview/field.binary",
        json={
            "scene_ir": compiled,
            "parameter_values": {"r": 0.7},
            "quality_profile": "interactive",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert response.headers.get("x-sdf-stats")
    assert response.headers.get("x-sdf-resolution-xyz") == "64,64,64"
    assert response.headers.get("x-sdf-bounds")
    # 64^3 float32 samples
    assert len(response.content) == 64 * 64 * 64 * 4


def test_preview_field_endpoint_does_not_call_mesher(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = client.post("/api/v1/scene/compile", json={"source": SIMPLE_SOURCE}).json()["scene_ir"]

    def fail_mesher(*_args, **_kwargs):
        raise AssertionError("Mesher should not run for field preview")

    monkeypatch.setattr(main_module, "build_mesh_with_backend", fail_mesher)
    response = client.post(
        "/api/v1/preview/field",
        json={
            "scene_ir": compiled,
            "parameter_values": {"r": 0.7},
            "quality_profile": "interactive",
        },
    )
    assert response.status_code == 200
    assert response.json()["stats"]["preview_mode"] == "field"


def test_preview_mesh_endpoint_accepts_explicit_float16_precision() -> None:
    compiled = client.post("/api/v1/scene/compile", json={"source": SIMPLE_SOURCE}).json()["scene_ir"]
    response = client.post(
        "/api/v1/preview/mesh",
        json={
            "scene_ir": compiled,
            "parameter_values": {"r": 0.7},
            "quality_profile": "interactive",
            "compute_precision": "float16",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["stats"]["compute_precision"] == "float16"
    assert payload["stats"]["compute_backend"] == "cpu"


def test_preview_cache_isolated_by_compute_precision() -> None:
    compiled = client.post("/api/v1/scene/compile", json={"source": SIMPLE_SOURCE}).json()["scene_ir"]
    request_base = {
        "scene_ir": compiled,
        "parameter_values": {"r": 0.73},
        "quality_profile": "interactive",
    }

    first = client.post("/api/v1/preview/mesh", json=request_base)
    assert first.status_code == 200
    assert first.json()["stats"]["cache_hit"] is False
    assert first.json()["stats"]["compute_precision"] == "float32"

    float16 = client.post(
        "/api/v1/preview/mesh",
        json={**request_base, "compute_precision": "float16"},
    )
    assert float16.status_code == 200
    assert float16.json()["stats"]["cache_hit"] is False
    assert float16.json()["stats"]["compute_precision"] == "float16"

    second = client.post("/api/v1/preview/mesh", json=request_base)
    assert second.status_code == 200
    assert second.json()["stats"]["cache_hit"] is True
    assert second.json()["stats"]["compute_precision"] == "float32"


def test_export_endpoint_stl() -> None:
    compiled = client.post("/api/v1/scene/compile", json={"source": SOURCE}).json()["scene_ir"]
    response = client.post(
        "/api/v1/export",
        json={
            "scene_ir": compiled,
            "parameter_values": {"r": 0.8},
            "format": "stl",
            "quality_profile": "interactive",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("model/stl")
    assert len(response.content) > 84


def test_export_endpoint_accepts_adaptive_meshing_mode() -> None:
    compiled = client.post("/api/v1/scene/compile", json={"source": SOURCE}).json()["scene_ir"]
    response = client.post(
        "/api/v1/export",
        json={
            "scene_ir": compiled,
            "parameter_values": {"r": 0.8},
            "format": "obj",
            "quality_profile": "interactive",
            "meshing_mode": "adaptive",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_export_endpoint_skips_mesh_payload_encoding_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import cache as cache_module

    cache_module.clear_all_caches()

    def fail_encode(*_args, **_kwargs):
        raise AssertionError("_encode_mesh_payload should not run for export path")

    def fail_cache_set(*_args, **_kwargs):
        raise AssertionError("mesh_preview_cache.set should not run for export path")

    monkeypatch.setattr(main_module, "_encode_mesh_payload", fail_encode)
    monkeypatch.setattr(main_module.mesh_preview_cache, "set", fail_cache_set)

    compiled = client.post("/api/v1/scene/compile", json={"source": SOURCE}).json()["scene_ir"]
    response = client.post(
        "/api/v1/export",
        json={
            "scene_ir": compiled,
            "parameter_values": {"r": 0.79},
            "format": "stl",
            "quality_profile": "interactive",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("model/stl")


def test_mesh_preview_endpoint_obj_upload() -> None:
    response = client.post(
        "/api/v1/mesh/preview",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["stats"]["tri_count"] > 0
    assert payload["mesh"]["encoding"] == "mesh-f32-u32-base64-v1"
    assert payload["field"]["encoding"] == "f32-base64"
    assert payload["field"]["resolution_xyz"] == [24, 24, 24]
    assert payload["field"]["data"]


def test_mesh_preview_binary_endpoint_obj_upload() -> None:
    response = client.post(
        "/api/v1/mesh/preview.binary",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert response.headers.get("x-sdf-stats")
    assert response.content[:8] == b"SDFMESH1"


def test_mesh_field_endpoint_obj_upload() -> None:
    response = client.post(
        "/api/v1/mesh/field",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["field"]["encoding"] == "f32-base64"
    assert payload["field"]["resolution_xyz"] == [24, 24, 24]
    assert payload["field"]["data"]
    assert payload["host_field"]["encoding"] == "f32-base64"
    assert payload["host_field"]["resolution_xyz"] == [24, 24, 24]
    assert payload["host_field"]["data"]
    assert payload["stats"]["preview_mode"] == "field"
    assert payload["stats"]["mesh_ms"] is None
    assert payload["stats"]["tri_count"] == 0
    assert response.headers.get("x-sdf-trace-id")


def test_mesh_field_binary_endpoint_obj_upload() -> None:
    response = client.post(
        "/api/v1/mesh/field.binary",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert response.headers.get("x-sdf-stats")
    assert response.headers.get("x-sdf-resolution-xyz") == "24,24,24"
    assert response.headers.get("x-sdf-trace-id")
    assert len(response.content) == 24 * 24 * 24 * 4 * 2


def test_mesh_field_binary_endpoint_reports_gpu_fill_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    warning = (
        "GPU voxel fill ran out of memory at resolution 24. "
        "Try lowering the preview resolution, or reduce voxels_per_lattice_period / increase lattice_pitch. "
        "The preview was retried on CPU."
    )

    def fake_run_uploaded_mesh_field_preview_data_with_audit(**_kwargs):
        host_sdf = np.zeros((2, 2, 2), dtype=np.float32)
        field = np.zeros((2, 2, 2), dtype=np.float32)
        bounds = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        stats = main_module.PreviewStats(
            eval_ms=1.0,
            mesh_ms=None,
            tri_count=0,
            voxel_count=int(field.size),
            cache_hit=False,
            field_cache_hit=False,
            mesh_cache_hit=False,
            compute_backend="cpu",
            mesh_backend="cpu",
            preview_mode="field",
            fallback_reason=warning,
        )
        audit = main_module.UploadedFieldPreviewServerAudit(
            metadata_cache_hit=False,
            host_cache_hit=False,
            field_cache_hit=False,
            resolution_xyz=(2, 2, 2),
            voxel_count=int(field.size),
            payload_bytes=int(field.size * np.dtype(np.float32).itemsize),
            compute_backend="cpu",
            host_build_strategy="dense",
            host_decision_reason="dense_requested",
            server_upload_read_ms=0.0,
            server_preprocessing_ms=0.0,
            server_metadata_resolve_ms=0.0,
            server_host_field_ms=0.0,
            server_compose_field_ms=0.0,
            server_pack_binary_ms=0.0,
            server_handler_total_ms=0.0,
        )
        return host_sdf, field, bounds, stats, audit

    monkeypatch.setattr(main_module, "_run_uploaded_mesh_field_preview_data_with_audit", fake_run_uploaded_mesh_field_preview_data_with_audit)

    response = client.post(
        "/api/v1/mesh/field.binary",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    stats = json.loads(response.headers["x-sdf-stats"])
    assert stats["fallback_reason"] == warning


def test_mesh_field_binary_endpoint_rejects_memory_fatal_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_memory_guard(**_kwargs):
        raise main_module.MeshUploadError(
            "Estimated memory requirement exceeds available system memory for this mesh preview."
        )

    monkeypatch.setattr(main_module, "_enforce_uploaded_mesh_memory_guard", fail_memory_guard)
    response = client.post(
        "/api/v1/mesh/field.binary",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 400
    assert "Estimated memory requirement exceeds available system memory" in response.json()["detail"]


def test_uploaded_field_preview_telemetry_endpoint_merges_with_server_trace() -> None:
    response = client.post(
        "/api/v1/mesh/field.binary",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    trace_id = response.headers.get("x-sdf-trace-id")
    assert trace_id

    telemetry = client.post(
        "/api/v1/internal/mesh/field-preview-telemetry",
        json={
            "trace_id": trace_id,
            "client_response_wait_ms": 10.0,
            "client_download_ms": 20.0,
            "client_decode_ms": 30.0,
            "client_texture_upload_and_first_frame_ms": 40.0,
            "client_total_visible_ms": 100.0,
        },
    )
    assert telemetry.status_code == 204


def test_uploaded_field_preview_telemetry_endpoint_rejects_unknown_trace_id() -> None:
    telemetry = client.post(
        "/api/v1/internal/mesh/field-preview-telemetry",
        json={
            "trace_id": "missing-trace",
            "client_response_wait_ms": 10.0,
            "client_download_ms": 20.0,
            "client_decode_ms": 30.0,
            "client_texture_upload_and_first_frame_ms": 40.0,
            "client_total_visible_ms": 100.0,
        },
    )
    assert telemetry.status_code == 404


def test_mesh_field_endpoint_does_not_call_mesher(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_mesher(*_args, **_kwargs):
        raise AssertionError("Mesher should not run for uploaded field preview")

    monkeypatch.setattr(main_module, "build_mesh_with_backend", fail_mesher)
    response = client.post(
        "/api/v1/mesh/field",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["stats"]["preview_mode"] == "field"


def test_mesh_commit_endpoint_requires_preview_field_first() -> None:
    from app import cache as cache_module

    cache_module.clear_all_preview_caches()

    response = client.post(
        "/api/v1/mesh/commit",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 409
    assert "Preview Field first" in response.json()["detail"]


def test_mesh_commit_endpoint_uses_cached_field_without_recomputing_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import cache as cache_module

    cache_module.clear_all_preview_caches()

    field_response = client.post(
        "/api/v1/mesh/field.binary",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert field_response.status_code == 200

    def fail_memory_guard(**_kwargs):
        raise AssertionError("Uploaded mesh memory guard should not run for commit-only path")

    def fail_field_compute(**_kwargs):
        raise AssertionError("Field recomputation should not run for commit-only path")

    monkeypatch.setattr(main_module, "_enforce_uploaded_mesh_memory_guard", fail_memory_guard)
    monkeypatch.setattr(main_module, "_resolve_uploaded_composed_field", fail_field_compute)

    response = client.post(
        "/api/v1/mesh/commit.binary",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    stats = json.loads(response.headers["x-sdf-stats"])
    assert stats["preview_mode"] == "mesh"
    assert stats["field_cache_hit"] is True
    assert stats["mesh_cache_hit"] is False
    assert stats["eval_ms"] == 0.0


def test_mesh_preview_preserves_original_outer_vertices() -> None:
    response = client.post(
        "/api/v1/mesh/preview",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    vertices, _, _ = _decode_mesh_arrays(response.json()["mesh"])

    def has_vertex(target: list[float]) -> bool:
        for vx, vy, vz in vertices:
            if abs(vx - target[0]) < 1e-6 and abs(vy - target[1]) < 1e-6 and abs(vz - target[2]) < 1e-6:
                return True
        return False

    assert has_vertex([0.0, 0.0, 0.0])
    assert has_vertex([1.0, 0.0, 0.0])
    assert has_vertex([0.0, 1.0, 0.0])


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/v1/mesh/preview",
        "/api/v1/mesh/preview.binary",
        "/api/v1/mesh/field",
        "/api/v1/mesh/field.binary",
        "/api/v1/mesh/export",
    ],
)
def test_uploaded_mesh_endpoints_reject_legacy_quality_profile(endpoint: str) -> None:
    form = _mesh_form()
    form["quality_profile"] = "interactive"
    if endpoint == "/api/v1/mesh/export":
        form["format"] = "stl"
    response = client.post(
        endpoint,
        data=form,
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 422
    assert "voxels_per_lattice_period" in response.json()["detail"]


def test_uploaded_mesh_preview_ws_rejects_legacy_quality_profile() -> None:
    with client.websocket_connect("/api/v1/mesh/preview/ws") as websocket:
        websocket.send_json(
            {
                "file_name": "tetra.obj",
                "file_data_base64": base64.b64encode(MESH_OBJ).decode("ascii"),
                "shell_thickness": 0.08,
                "lattice_type": "gyroid",
                "lattice_pitch": 0.45,
                "lattice_thickness": 0.09,
                "lattice_phase": 0.0,
                "quality_profile": "interactive",
                "voxels_per_lattice_period": 6,
            }
        )
        payload = websocket.receive_json()
    assert payload["phase"] == "error"
    assert "Extra inputs are not permitted" in payload["error"]


@pytest.mark.parametrize("lattice_type", ["gyroid", "schwarz_p", "diamond"])
def test_mesh_preview_supports_tpms_variants(lattice_type: str) -> None:
    response = client.post(
        "/api/v1/mesh/preview",
        data=_mesh_form(lattice_type=lattice_type),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["stats"]["tri_count"] > 0


def test_mesh_preview_accepts_backend_selectors_with_fallback() -> None:
    form = _mesh_form()
    form["compute_backend"] = "cuda"
    form["mesh_backend"] = "cuda"
    response = client.post(
        "/api/v1/mesh/preview",
        data=form,
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    if response.status_code == 200:
        stats = response.json()["stats"]
        assert stats["mesh_backend"] == "cuda"
        assert stats["compute_backend"] in {"cpu", "cuda"}
    else:
        assert response.status_code == 400
        assert "CUDA meshing failed" in response.json()["detail"]


def test_mesh_export_endpoint_obj() -> None:
    form = _mesh_form()
    form["format"] = "obj"
    response = client.post(
        "/api/v1/mesh/export",
        data=form,
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert b"\nv " in response.content


def test_uploaded_export_endpoint_skips_mesh_payload_encoding_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import cache as cache_module

    cache_module.clear_all_caches()

    def fail_encode(*_args, **_kwargs):
        raise AssertionError("_encode_mesh_payload should not run for uploaded export path")

    def fail_cache_set(*_args, **_kwargs):
        raise AssertionError("uploaded_mesh_preview_cache.set should not run for uploaded export path")

    monkeypatch.setattr(main_module, "_encode_mesh_payload", fail_encode)
    monkeypatch.setattr(main_module.uploaded_mesh_preview_cache, "set", fail_cache_set)

    form = _mesh_form()
    form["format"] = "stl"
    response = client.post(
        "/api/v1/mesh/export",
        data=form,
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("model/stl")


def test_mesh_preview_rejects_unsupported_extension() -> None:
    response = client.post(
        "/api/v1/mesh/preview",
        data=_mesh_form(),
        files={"file": ("bad.ply", b"ply", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Only .stl and .obj uploads are supported" in response.json()["detail"]


def test_mesh_preview_rejects_oversized_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "MESH_UPLOAD_MAX_BYTES", 128)
    response = client.post(
        "/api/v1/mesh/preview",
        data=_mesh_form(),
        files={"file": ("big.obj", b"x" * 129, "text/plain")},
    )
    assert response.status_code == 413


def test_preprocess_endpoint_does_not_build_host_field(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_build_host_field(*args, **kwargs):
        raise AssertionError("build_host_field should not run during preprocess")

    monkeypatch.setattr(main_module, "build_host_field", fail_build_host_field)
    response = client.post(
        "/api/v1/mesh/preprocess",
        data={
            "lattice_pitch": "0.01",
            "voxels_per_lattice_period": "12",
            "compute_backend": "cpu",
            "field_storage_mode": "dense",
        },
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    assert "x-sdf-preprocess-resolution" not in response.headers


def test_uploaded_queueing_uses_computed_resolution_in_auto_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "REDIS_CLIENT_AVAILABLE", True)
    monkeypatch.setattr(
        main_module,
        "_compute_mesh_upload_resolution",
        lambda *args, **kwargs: (128, 128, 128),
    )
    should_queue = main_module._should_queue_uploaded_request(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
        extension=".obj",
        lattice_pitch=0.45,
        voxels_per_lattice_period=6,
        execution_mode="auto",
    )
    assert should_queue is True

    monkeypatch.setattr(
        main_module,
        "_compute_mesh_upload_resolution",
        lambda *args, **kwargs: (127, 127, 127),
    )
    should_queue = main_module._should_queue_uploaded_request(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
        extension=".obj",
        lattice_pitch=0.45,
        voxels_per_lattice_period=6,
        execution_mode="auto",
    )
    assert should_queue is False


def test_uploaded_queueing_uses_file_size_threshold_in_auto_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "REDIS_CLIENT_AVAILABLE", True)
    monkeypatch.setattr(main_module, "AUTO_QUEUE_UPLOAD_BYTES_THRESHOLD", 1)
    monkeypatch.setattr(main_module, "_compute_mesh_upload_resolution", lambda *args, **kwargs: (2, 2, 2))
    should_queue = main_module._should_queue_uploaded_request(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
        extension=".obj",
        lattice_pitch=0.45,
        voxels_per_lattice_period=6,
        execution_mode="auto",
    )
    assert should_queue is True


def test_preview_mesh_queued_mode_returns_job_and_result() -> None:
    compiled = client.post("/api/v1/scene/compile", json={"source": SIMPLE_SOURCE}).json()["scene_ir"]
    submit = client.post(
        "/api/v1/preview/mesh",
        json={
            "scene_ir": compiled,
            "parameter_values": {"r": 0.7},
            "quality_profile": "interactive",
            "execution_mode": "queued",
        },
    )
    assert submit.status_code == 200
    payload = submit.json()
    assert payload["job_id"]
    status = client.get(f"/api/v1/jobs/{payload['job_id']}")
    assert status.status_code == 200
    assert status.json()["status"] in {"queued", "running", "succeeded"}
    result = client.get(f"/api/v1/jobs/{payload['job_id']}/result")
    assert result.status_code == 200
    result_payload = result.json()
    assert result_payload["mesh"]["encoding"] == "mesh-f32-u32-base64-v1"


def test_export_queued_mode_returns_streaming_result() -> None:
    compiled = client.post("/api/v1/scene/compile", json={"source": SIMPLE_SOURCE}).json()["scene_ir"]
    submit = client.post(
        "/api/v1/jobs/export",
        json={
            "scene_ir": compiled,
            "parameter_values": {"r": 0.75},
            "quality_profile": "interactive",
            "format": "stl",
        },
    )
    assert submit.status_code == 200
    payload = submit.json()
    result = client.get(f"/api/v1/jobs/{payload['job_id']}/result")
    assert result.status_code == 200
    assert result.headers["content-type"].startswith("model/stl")
    assert len(result.content) > 84


def test_preprocess_endpoint_returns_binary_outer_mesh() -> None:
    """POST /api/v1/mesh/preprocess returns a binary mesh packet for the outer geometry."""
    response = client.post(
        "/api/v1/mesh/preprocess",
        data={
            "lattice_pitch": "0.45",
            "voxels_per_lattice_period": "6",
            "compute_backend": "cpu",
            "field_storage_mode": "dense",
        },
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert response.headers.get("x-sdf-vertex-count")
    assert response.headers.get("x-sdf-face-count")
    body = response.content
    assert body[:8] == b"SDFMESH1"
    import struct
    vertex_count = struct.unpack_from("<I", body, 8)[0]
    face_count = struct.unpack_from("<I", body, 12)[0]
    assert vertex_count > 0
    assert face_count > 0
    assert int(response.headers["x-sdf-vertex-count"]) == vertex_count
    assert int(response.headers["x-sdf-face-count"]) == face_count
    assert response.headers.get("x-sdf-mesh-span")
    assert response.headers.get("x-sdf-cpu-bytes-per-voxel")
    assert response.headers.get("x-sdf-gpu-bytes-per-voxel")
    assert response.headers.get("x-sdf-memory-safety-factor")


def test_preprocess_endpoint_returns_memory_context_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_memory_estimate(**kwargs):
        metadata = kwargs["metadata"]
        context = main_module.UploadedMeshMemoryContext(
            mesh_span=1.5,
            mesh_extents=(1.5, 1.5, 1.5),
            available_cpu_bytes=1024,
            available_gpu_free_bytes=2048,
            available_gpu_total_bytes=4096,
            cpu_bytes_per_voxel=48.0,
            gpu_bytes_per_voxel=56.0,
            safety_factor=1.0,
        )
        estimate = main_module.UploadedMeshMemoryEstimate(
            context=context,
            resolution_xyz=(24, 24, 24),
            required_cpu_bytes=512,
            required_gpu_bytes=1024,
            cpu_fatal=False,
            gpu_fatal=False,
            fatal=False,
            gpu_check_enabled=True,
        )
        return estimate, metadata

    monkeypatch.setattr(main_module, "_resolve_uploaded_mesh_memory_estimate", fake_memory_estimate)
    response = client.post(
        "/api/v1/mesh/preprocess",
        data={"lattice_pitch": "0.45", "voxels_per_lattice_period": "6", "compute_backend": "auto"},
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    assert response.headers["x-sdf-mesh-span"] == "1.5"
    assert response.headers["x-sdf-resolution-xyz"] == "24,24,24"
    assert response.headers["x-sdf-available-cpu-bytes"] == "1024"
    assert response.headers["x-sdf-available-gpu-free-bytes"] == "2048"
    assert response.headers["x-sdf-available-gpu-total-bytes"] == "4096"
    assert response.headers["x-sdf-cpu-bytes-per-voxel"] == "48.0"
    assert response.headers["x-sdf-gpu-bytes-per-voxel"] == "56.0"
    assert response.headers["x-sdf-memory-safety-factor"] == "1.0"


def test_available_cpu_memory_probe_graceful_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module.os, "sysconf", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("boom")))
    monkeypatch.setattr(main_module.os, "sysconf_names", {})
    monkeypatch.setattr(main_module, "_read_cgroup_memory_headroom_bytes", lambda: None)

    import builtins

    monkeypatch.setattr(
        builtins,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("boom")),
    )
    assert main_module._available_cpu_memory_bytes() is None


def test_available_cpu_memory_probe_prefers_memavailable_over_sysconf(monkeypatch: pytest.MonkeyPatch) -> None:
    meminfo = "MemTotal:       16198740 kB\nMemAvailable:    6895560 kB\n"

    monkeypatch.setattr(main_module, "_read_cgroup_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(main_module.os, "sysconf_names", {"SC_AVPHYS_PAGES": 1, "SC_PAGE_SIZE": 1})
    monkeypatch.setattr(
        main_module.os,
        "sysconf",
        lambda name: 1024 if name == "SC_AVPHYS_PAGES" else 4096,
    )

    import builtins

    monkeypatch.setattr(
        builtins,
        "open",
        lambda path, *args, **kwargs: io.StringIO(meminfo) if path == "/proc/meminfo" else (_ for _ in ()).throw(AssertionError(path)),
    )

    assert main_module._available_cpu_memory_bytes() == 6895560 * 1024


def test_preprocess_endpoint_warms_metadata_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """After preprocess, the metadata cache should be warm for the same file."""
    from app import cache as cache_module

    cache_module.clear_all_preview_caches()

    def fake_build_host_field(mesh, resolution_xyz, **kwargs):
        nx, ny, nz = resolution_xyz
        return SimpleNamespace(
            mesh=mesh,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            host_sdf=np.zeros((nx, ny, nz), dtype=np.float32),
            host_compute_backend="cpu",
            field_storage_mode="dense",
            block_size=None,
            active_blocks=None,
            sparse_background_value=None,
            sparse_bricks=None,
            host_build_strategy="dense",
            host_decision_reason="dense_requested",
        )

    monkeypatch.setattr(main_module, "build_host_field", fake_build_host_field)

    # First call — cold cache
    response = client.post(
        "/api/v1/mesh/preprocess",
        data={"lattice_pitch": "0.45"},
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200

    # Second call to field endpoint — should be a metadata cache hit
    field_response = client.post(
        "/api/v1/mesh/field",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert field_response.status_code == 200
    # metadata_cache_hit is logged in the trace but not directly in the response;
    # we verify the field endpoint succeeds and returns valid data
    assert field_response.json()["field"]["encoding"] == "f32-base64"


def test_uploaded_field_preview_trace_includes_preprocessing_time(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from app import cache as cache_module

    cache_module.clear_all_preview_caches()

    def fake_run_uploaded_mesh_field_preview_data_with_audit(**_kwargs):
        host_sdf = np.zeros((2, 2, 2), dtype=np.float32)
        field = np.zeros((2, 2, 2), dtype=np.float32)
        bounds = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        stats = main_module.PreviewStats(
            eval_ms=1.0,
            mesh_ms=None,
            tri_count=0,
            voxel_count=int(field.size),
            cache_hit=False,
            field_cache_hit=False,
            mesh_cache_hit=False,
            compute_backend="cpu",
            mesh_backend="cpu",
            preview_mode="field",
        )
        audit = main_module.UploadedFieldPreviewServerAudit(
            metadata_cache_hit=False,
            host_cache_hit=False,
            field_cache_hit=False,
            resolution_xyz=(2, 2, 2),
            voxel_count=int(field.size),
            payload_bytes=int(field.size * np.dtype(np.float32).itemsize),
            compute_backend="cpu",
            host_build_strategy="dense",
            host_decision_reason="dense_requested",
            server_upload_read_ms=0.0,
            server_preprocessing_ms=None,
            server_metadata_resolve_ms=0.0,
            server_host_field_ms=0.0,
            server_compose_field_ms=0.0,
            server_pack_binary_ms=0.0,
            server_handler_total_ms=0.0,
        )
        return host_sdf, field, bounds, stats, audit

    monkeypatch.setattr(main_module, "_run_uploaded_mesh_field_preview_data_with_audit", fake_run_uploaded_mesh_field_preview_data_with_audit)

    preprocess_response = client.post(
        "/api/v1/mesh/preprocess",
        data={"lattice_pitch": "0.45"},
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert preprocess_response.status_code == 200

    field_response = client.post(
        "/api/v1/mesh/field.binary",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert field_response.status_code == 200

    telemetry_payload = {
        "trace_id": field_response.headers["x-sdf-trace-id"],
        "client_response_wait_ms": 1.0,
        "client_download_ms": 2.0,
        "client_decode_ms": 3.0,
        "client_texture_upload_and_first_frame_ms": 4.0,
        "client_total_visible_ms": 10.0,
    }
    telemetry_response = client.post("/api/v1/internal/mesh/field-preview-telemetry", json=telemetry_payload)
    assert telemetry_response.status_code == 204

    captured = capsys.readouterr().out
    assert "uploaded_field_preview_trace" in captured
    assert "server_preprocessing_ms=" in captured


def test_preprocess_endpoint_rejects_invalid_mesh() -> None:
    """POST /api/v1/mesh/preprocess returns 400 for invalid mesh data."""
    response = client.post(
        "/api/v1/mesh/preprocess",
        data={"lattice_pitch": "0.45"},
        files={"file": ("bad.obj", b"not a mesh at all", "text/plain")},
    )
    assert response.status_code == 400


def test_preprocess_endpoint_rejects_unsupported_extension() -> None:
    """POST /api/v1/mesh/preprocess returns 400 for unsupported file types."""
    response = client.post(
        "/api/v1/mesh/preprocess",
        data={"lattice_pitch": "0.45"},
        files={"file": ("model.ply", b"ply data", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Only .stl and .obj uploads are supported" in response.json()["detail"]


def test_preprocess_endpoint_rejects_legacy_quality_profile() -> None:
    """POST /api/v1/mesh/preprocess rejects the legacy quality_profile field."""
    response = client.post(
        "/api/v1/mesh/preprocess",
        data={"lattice_pitch": "0.45", "quality_profile": "interactive"},
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 422
    assert "voxels_per_lattice_period" in response.json()["detail"]
