import pytest

from app import cache as cache_module
from app import main as main_module

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


@pytest.fixture(autouse=True)
def clear_uploaded_caches() -> None:
    cache_module.clear_all_caches()
    try:
        yield
    finally:
        cache_module.clear_all_caches()


def test_uploaded_mesh_preview_reuses_composed_field_after_field_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"compose": 0}
    original = main_module.compose_hollow_lattice_field_with_backend

    def track_compose(*args, **kwargs):
        calls["compose"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(main_module, "compose_hollow_lattice_field_with_backend", track_compose)

    _, _, field_stats = main_module._run_uploaded_mesh_field_preview_data(
        file_bytes=MESH_OBJ,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
    )
    assert field_stats.cache_hit is False

    _, mesh_stats, field_payload, mesh_payload = main_module._run_uploaded_mesh_preview_meshdata(
        file_bytes=MESH_OBJ,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
    )
    assert mesh_stats.cache_hit is True
    assert field_payload is not None
    assert mesh_payload is not None
    assert calls["compose"] == 1


def test_uploaded_binary_preview_reuses_composed_field_after_field_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"compose": 0}
    original = main_module.compose_hollow_lattice_field_with_backend

    def track_compose(*args, **kwargs):
        calls["compose"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(main_module, "compose_hollow_lattice_field_with_backend", track_compose)

    _, _, field_stats = main_module._run_uploaded_mesh_field_preview_data(
        file_bytes=MESH_OBJ,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
    )
    assert field_stats.cache_hit is False

    _, mesh_stats, field_payload, mesh_payload = main_module._run_uploaded_mesh_preview_meshdata(
        file_bytes=MESH_OBJ,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
        encode_response_payloads=False,
        cache_result=False,
    )
    assert mesh_stats.cache_hit is True
    assert field_payload is None
    assert mesh_payload is None
    assert calls["compose"] == 1


def test_uploaded_composed_field_cache_ignores_mesh_settings_but_invalidates_on_field_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"compose": 0}
    original = main_module.compose_hollow_lattice_field_with_backend

    def track_compose(*args, **kwargs):
        calls["compose"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(main_module, "compose_hollow_lattice_field_with_backend", track_compose)

    _, first_stats, _, _ = main_module._run_uploaded_mesh_preview_meshdata(
        file_bytes=MESH_OBJ,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
        mesh_backend="cpu",
        meshing_mode="uniform",
        encode_response_payloads=False,
        cache_result=False,
    )
    assert first_stats.cache_hit is False
    assert calls["compose"] == 1

    _, second_stats, _, _ = main_module._run_uploaded_mesh_preview_meshdata(
        file_bytes=MESH_OBJ,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
        mesh_backend="auto",
        meshing_mode="adaptive",
        encode_response_payloads=False,
        cache_result=False,
    )
    assert second_stats.cache_hit is True
    assert calls["compose"] == 1

    _, third_stats, _, _ = main_module._run_uploaded_mesh_preview_meshdata(
        file_bytes=MESH_OBJ,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.10,
        lattice_phase=0.0,
        mesh_backend="auto",
        meshing_mode="adaptive",
        encode_response_payloads=False,
        cache_result=False,
    )
    assert third_stats.cache_hit is False
    assert calls["compose"] == 2

    _, fourth_stats, _, _ = main_module._run_uploaded_mesh_preview_meshdata(
        file_bytes=MESH_OBJ,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
        compute_backend="cpu",
        mesh_backend="auto",
        meshing_mode="adaptive",
        encode_response_payloads=False,
        cache_result=False,
    )
    assert fourth_stats.cache_hit is False
    assert calls["compose"] == 3
