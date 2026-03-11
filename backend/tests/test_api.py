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
