import hashlib
import time

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
MESH_OBJ_HASH = hashlib.sha256(MESH_OBJ).hexdigest()


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
        file_hash=MESH_OBJ_HASH,
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
        file_hash=MESH_OBJ_HASH,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
    )
    assert mesh_stats.cache_hit is True
    assert mesh_stats.field_cache_hit is True
    assert mesh_stats.mesh_cache_hit is False
    assert mesh_stats.eval_ms == 0.0
    assert mesh_stats.mesh_ms is not None and mesh_stats.mesh_ms > 0.0
    assert field_payload is not None
    assert mesh_payload is not None
    assert calls["compose"] == 1


def test_uploaded_mesh_preview_reuses_full_mesh_cache_on_second_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"compose": 0}
    original = main_module.compose_hollow_lattice_field_with_backend

    def track_compose(*args, **kwargs):
        calls["compose"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(main_module, "compose_hollow_lattice_field_with_backend", track_compose)

    _, _, field_stats = main_module._run_uploaded_mesh_field_preview_data(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
    )
    assert field_stats.cache_hit is False

    _, first_mesh_stats, _, _ = main_module._run_uploaded_mesh_preview_meshdata(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
    )
    assert first_mesh_stats.field_cache_hit is True
    assert first_mesh_stats.mesh_cache_hit is False
    assert first_mesh_stats.eval_ms == 0.0

    _, second_mesh_stats, field_payload, mesh_payload = main_module._run_uploaded_mesh_preview_meshdata(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
    )
    assert second_mesh_stats.cache_hit is True
    assert second_mesh_stats.field_cache_hit is True
    assert second_mesh_stats.mesh_cache_hit is True
    assert second_mesh_stats.eval_ms == 0.0
    assert second_mesh_stats.mesh_ms == 0.0
    assert field_payload is not None
    assert mesh_payload is not None
    assert calls["compose"] == 1


def test_uploaded_binary_preview_seeds_and_reuses_full_mesh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"compose": 0}
    original = main_module.compose_hollow_lattice_field_with_backend

    def track_compose(*args, **kwargs):
        calls["compose"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(main_module, "compose_hollow_lattice_field_with_backend", track_compose)

    _, first_stats, field_payload, mesh_payload = main_module._run_uploaded_mesh_preview_meshdata(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
        encode_response_payloads=False,
        cache_result=True,
    )
    assert first_stats.mesh_cache_hit is False
    assert first_stats.mesh_ms is not None and first_stats.mesh_ms > 0.0
    assert field_payload is None
    assert mesh_payload is None

    mesh, second_stats, field_payload, mesh_payload = main_module._run_uploaded_mesh_preview_meshdata(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
        encode_response_payloads=False,
        cache_result=True,
    )
    assert second_stats.cache_hit is True
    assert second_stats.field_cache_hit is True
    assert second_stats.mesh_cache_hit is True
    assert second_stats.eval_ms == 0.0
    assert second_stats.mesh_ms == 0.0
    assert field_payload is None
    assert mesh_payload is None
    assert mesh.faces.shape[0] > 0
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
        file_hash=MESH_OBJ_HASH,
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
    assert first_stats.field_cache_hit is False
    assert first_stats.mesh_cache_hit is False
    assert calls["compose"] == 1

    _, second_stats, _, _ = main_module._run_uploaded_mesh_preview_meshdata(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
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
    assert second_stats.field_cache_hit is True
    assert second_stats.mesh_cache_hit is False
    assert second_stats.eval_ms == 0.0
    assert calls["compose"] == 1

    _, third_stats, _, _ = main_module._run_uploaded_mesh_preview_meshdata(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
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
    assert third_stats.field_cache_hit is False
    assert calls["compose"] == 2

    _, fourth_stats, _, _ = main_module._run_uploaded_mesh_preview_meshdata(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
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
    assert fourth_stats.field_cache_hit is False
    assert calls["compose"] == 3


def test_uploaded_metadata_cache_avoids_reparsing_for_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"parse": 0}
    original = main_module.parse_mesh_bytes

    def track_parse(*args, **kwargs):
        calls["parse"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(main_module, "parse_mesh_bytes", track_parse)

    first = main_module._compute_mesh_upload_resolution(MESH_OBJ, MESH_OBJ_HASH, ".obj", 0.45, 6)
    second = main_module._compute_mesh_upload_resolution(MESH_OBJ, MESH_OBJ_HASH, ".obj", 0.45, 6)

    expected, _ = main_module._resolve_mesh_resolution(0.45, 1.0, 6)
    assert first == expected
    assert first == second
    assert calls["parse"] == 1


def test_uploaded_host_cache_returns_shared_immutable_arrays() -> None:
    metadata = main_module._resolve_uploaded_mesh_metadata(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
        extension=".obj",
    )
    first = main_module._resolve_uploaded_host_field(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
        extension=".obj",
        resolution=24,
        parsed=metadata.parsed,
    )
    second = main_module._resolve_uploaded_host_field(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
        extension=".obj",
        resolution=24,
        parsed=metadata.parsed,
    )

    assert first.parsed.vertices is second.parsed.vertices
    assert first.parsed.faces is second.parsed.faces
    assert first.host_sdf is second.host_sdf
    assert first.host_sdf.flags.writeable is False
    assert isinstance(first.host_build_strategy, str) and first.host_build_strategy
    assert isinstance(first.host_decision_reason, str) and first.host_decision_reason
    assert first.host_build_strategy == second.host_build_strategy
    assert first.host_decision_reason == second.host_decision_reason
    assert first.cache_hit is False
    assert second.cache_hit is True


def test_uploaded_field_preview_audit_reports_zero_compose_time_on_cache_hit() -> None:
    _, _, _, first_audit = main_module._run_uploaded_mesh_field_preview_data_with_audit(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
    )
    _, _, _, second_audit = main_module._run_uploaded_mesh_field_preview_data_with_audit(
        file_bytes=MESH_OBJ,
        file_hash=MESH_OBJ_HASH,
        extension=".obj",
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
    )

    assert first_audit.field_cache_hit is False
    assert first_audit.server_compose_field_ms > 0.0
    assert first_audit.host_build_strategy in {"dense", "octree_sparse"}
    assert isinstance(first_audit.host_decision_reason, str) and first_audit.host_decision_reason
    assert second_audit.field_cache_hit is True
    assert second_audit.server_compose_field_ms == 0.0
    assert second_audit.host_build_strategy in {"dense", "octree_sparse"}


def test_uploaded_field_preview_trace_store_expires_entries() -> None:
    cache_module.uploaded_field_preview_trace_store.ttl_seconds = 0.001
    try:
        cache_module.uploaded_field_preview_trace_store.set(
            "trace-expire",
            cache_module.UploadedFieldPreviewTraceEntry(
                trace_id="trace-expire",
                created_at=time.time() - 1.0,
                route="/api/v1/mesh/field.binary",
            ),
        )
        assert cache_module.uploaded_field_preview_trace_store.get("trace-expire") is None
    finally:
        cache_module.uploaded_field_preview_trace_store.ttl_seconds = 300.0
