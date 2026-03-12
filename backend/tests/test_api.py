import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app

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


def _mesh_form(lattice_type: str = "gyroid") -> dict[str, str]:
    return {
        "shell_thickness": "0.08",
        "lattice_type": lattice_type,
        "lattice_pitch": "0.45",
        "lattice_thickness": "0.09",
        "lattice_phase": "0.0",
        "quality_profile": "interactive",
    }


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
    assert len(payload["mesh"]["vertices"]) > 0
    assert payload["stats"]["compute_precision"] == "float32"
    assert payload["stats"]["compute_backend"] == "cpu"


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


def test_mesh_preview_endpoint_obj_upload() -> None:
    response = client.post(
        "/api/v1/mesh/preview",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["stats"]["tri_count"] > 0
    assert len(payload["mesh"]["vertices"]) > 0
    assert payload["field"]["encoding"] == "f32-base64"
    assert payload["field"]["resolution"] == 48
    assert payload["field"]["data"]


def test_mesh_preview_preserves_original_outer_vertices() -> None:
    response = client.post(
        "/api/v1/mesh/preview",
        data=_mesh_form(),
        files={"file": ("tetra.obj", MESH_OBJ, "text/plain")},
    )
    assert response.status_code == 200
    vertices = response.json()["mesh"]["vertices"]

    def has_vertex(target: list[float]) -> bool:
        for vx, vy, vz in vertices:
            if abs(vx - target[0]) < 1e-6 and abs(vy - target[1]) < 1e-6 and abs(vz - target[2]) < 1e-6:
                return True
        return False

    assert has_vertex([0.0, 0.0, 0.0])
    assert has_vertex([1.0, 0.0, 0.0])
    assert has_vertex([0.0, 1.0, 0.0])


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
