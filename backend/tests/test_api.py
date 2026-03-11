from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


SOURCE = """
param r default=0.8 min=0.2 max=1.5 step=0.1
body = sphere(r=$r)
lat = gyroid(pitch=0.5, thickness=0.1)
root = conformal_fill(body, lat, wall=0.08, mode="hybrid")
"""


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
