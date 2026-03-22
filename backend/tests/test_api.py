import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app
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
    assert payload["field"]["resolution"] == 64
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
    assert response.headers.get("x-sdf-resolution") == "64"
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
    assert payload["field"]["resolution"] == 24
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
    assert payload["field"]["resolution"] == 24
    assert payload["field"]["data"]
    assert payload["stats"]["preview_mode"] == "field"
    assert payload["stats"]["mesh_ms"] is None
    assert payload["stats"]["tri_count"] == 0


def test_mesh_field_binary_endpoint_obj_upload() -> None:
    response = client.post(
        "/api/v1/mesh/field.binary",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert response.headers.get("x-sdf-stats")
    assert response.headers.get("x-sdf-resolution") == "24"


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


def test_uploaded_queueing_uses_computed_resolution_in_auto_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "REDIS_CLIENT_AVAILABLE", True)
    monkeypatch.setattr(
        main_module,
        "_compute_mesh_upload_resolution",
        lambda *args, **kwargs: main_module.AUTO_QUEUE_RESOLUTION_THRESHOLD,
    )
    should_queue = main_module._should_queue_uploaded_request(
        file_bytes=MESH_OBJ,
        extension=".obj",
        lattice_pitch=0.45,
        voxels_per_lattice_period=6,
        execution_mode="auto",
    )
    assert should_queue is True

    monkeypatch.setattr(
        main_module,
        "_compute_mesh_upload_resolution",
        lambda *args, **kwargs: main_module.AUTO_QUEUE_RESOLUTION_THRESHOLD - 1,
    )
    should_queue = main_module._should_queue_uploaded_request(
        file_bytes=MESH_OBJ,
        extension=".obj",
        lattice_pitch=0.45,
        voxels_per_lattice_period=6,
        execution_mode="auto",
    )
    assert should_queue is False


def test_uploaded_queueing_uses_file_size_threshold_in_auto_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "REDIS_CLIENT_AVAILABLE", True)
    monkeypatch.setattr(main_module, "AUTO_QUEUE_UPLOAD_BYTES_THRESHOLD", 1)
    monkeypatch.setattr(main_module, "_compute_mesh_upload_resolution", lambda *args, **kwargs: 1)
    should_queue = main_module._should_queue_uploaded_request(
        file_bytes=MESH_OBJ,
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
