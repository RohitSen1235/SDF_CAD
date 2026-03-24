import logging
import gc

import numpy as np

from app import cache, evaluator, gpu_memory, main, meshing


def test_clear_all_caches_empties_process_lru_caches() -> None:
    cache.scene_compile_cache.set("compile", object())
    cache.mesh_preview_cache.set("mesh", object())
    cache.field_preview_cache.set("field", object())
    cache.uploaded_mesh_preview_cache.set("uploaded_mesh", object())
    cache.uploaded_mesh_metadata_cache.set("uploaded_metadata", object())
    cache.uploaded_composed_field_cache.set("uploaded_composed", object())
    cache.uploaded_host_field_cache.set("uploaded_field", object())
    cache.uploaded_field_preview_trace_store.set(
        "uploaded_trace",
        cache.UploadedFieldPreviewTraceEntry(trace_id="uploaded_trace", created_at=0.0, route="/api/v1/mesh/field.binary"),
    )
    cache.uploaded_mesh_preprocess_timing_store.set(
        "uploaded_file",
        cache.UploadedMeshPreprocessTimingEntry(created_at=0.0, preprocessing_ms=12.3),
    )

    cache.clear_all_caches()

    assert cache.scene_compile_cache.get("compile") is None
    assert cache.mesh_preview_cache.get("mesh") is None
    assert cache.field_preview_cache.get("field") is None
    assert cache.uploaded_mesh_preview_cache.get("uploaded_mesh") is None
    assert cache.uploaded_mesh_metadata_cache.get("uploaded_metadata") is None
    assert cache.uploaded_composed_field_cache.get("uploaded_composed") is None
    assert cache.uploaded_host_field_cache.get("uploaded_field") is None
    assert cache.uploaded_field_preview_trace_store.get("uploaded_trace") is None
    assert cache.uploaded_mesh_preprocess_timing_store.get("uploaded_file") is None


def test_clear_evaluator_caches_clears_lru_entries() -> None:
    key = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    evaluator._cached_grid(key, 8, "float32")
    evaluator._cached_axes(key, 8, "float32")
    assert evaluator._cached_grid.cache_info().currsize > 0
    assert evaluator._cached_axes.cache_info().currsize > 0

    evaluator.clear_evaluator_caches()

    assert evaluator._cached_grid.cache_info().currsize == 0
    assert evaluator._cached_axes.cache_info().currsize == 0


def test_cleanup_runtime_memory_orchestrates_all_cleanup(monkeypatch) -> None:
    calls: dict[str, object] = {
        "gpu_reason": None,
        "cache": False,
        "evaluator": False,
        "meshing": False,
        "gc": False,
    }

    monkeypatch.setattr(gpu_memory, "cleanup_gpu_memory", lambda reason="unknown": calls.__setitem__("gpu_reason", reason))
    monkeypatch.setattr(cache, "clear_all_caches", lambda: calls.__setitem__("cache", True))
    monkeypatch.setattr(evaluator, "clear_evaluator_caches", lambda: calls.__setitem__("evaluator", True))
    monkeypatch.setattr(meshing, "clear_meshing_caches", lambda: calls.__setitem__("meshing", True))
    monkeypatch.setattr(gc, "collect", lambda: calls.__setitem__("gc", True))

    gpu_memory.cleanup_runtime_memory(reason="unit-test")

    assert calls["gpu_reason"] == "unit-test"
    assert calls["cache"] is True
    assert calls["evaluator"] is True
    assert calls["meshing"] is True
    assert calls["gc"] is True


def test_uploaded_composed_field_cache_evicts_before_meshing_when_cpu_memory_is_tight(monkeypatch) -> None:
    field = np.zeros((8, 8, 8), dtype=np.float32)
    cache.uploaded_composed_field_cache.set("composed", object())
    gc_calls: list[bool] = []

    monkeypatch.setattr(main, "_available_cpu_memory_bytes", lambda: int(field.nbytes * 2))
    monkeypatch.setattr(main.gc, "collect", lambda: gc_calls.append(True))

    evicted = main._maybe_evict_uploaded_composed_field_cache_before_meshing(field)

    assert evicted is True
    assert cache.uploaded_composed_field_cache.get("composed") is None
    assert gc_calls == [True]


def test_uploaded_composed_field_cache_is_retained_when_cpu_memory_is_plentiful(monkeypatch) -> None:
    field = np.zeros((8, 8, 8), dtype=np.float32)
    cache.uploaded_composed_field_cache.set("composed", object())
    gc_calls: list[bool] = []

    monkeypatch.setattr(main, "_available_cpu_memory_bytes", lambda: int(field.nbytes * 8))
    monkeypatch.setattr(main.gc, "collect", lambda: gc_calls.append(True))

    evicted = main._maybe_evict_uploaded_composed_field_cache_before_meshing(field)

    assert evicted is False
    assert cache.uploaded_composed_field_cache.get("composed") is not None
    assert gc_calls == []


def test_available_cpu_memory_bytes_clamps_to_cgroup_headroom_and_logs(caplog, monkeypatch) -> None:
    host_available = 8 * 1024**3
    cgroup_headroom = 3 * 1024**3

    monkeypatch.setattr(main, "_read_linux_mem_available_bytes", lambda: host_available)
    monkeypatch.setattr(main, "_read_sysconf_available_bytes", lambda: None)
    monkeypatch.setattr(main, "_read_cgroup_memory_headroom_bytes", lambda: cgroup_headroom)

    with caplog.at_level(logging.DEBUG, logger=main.logger.name):
        available = main._available_cpu_memory_bytes()

    assert available == cgroup_headroom
    assert "cpu_memory_probe" in caplog.text


def test_uploaded_mesh_memory_guard_accepts_resolution_360_with_realistic_available_cpu(monkeypatch) -> None:
    metadata = main.UploadedMeshMetadata(
        parsed=main.ParsedMesh(
            vertices=np.zeros((1, 3), dtype=np.float64),
            faces=np.zeros((1, 3), dtype=np.int32),
        ),
        outer_mesh=main.MeshData(
            vertices=np.zeros((1, 3), dtype=np.float64),
            faces=np.zeros((1, 3), dtype=np.int32),
            normals=np.zeros((1, 3), dtype=np.float64),
        ),
        mesh_span=1.0,
        cache_hit=False,
    )
    context = main.UploadedMeshMemoryContext(
        mesh_span=1.0,
        available_cpu_bytes=int(6.9 * 1024**3),
        available_gpu_free_bytes=int(8.0 * 1024**3),
        available_gpu_total_bytes=int(8.0 * 1024**3),
        cpu_bytes_per_voxel=main.MEMORY_MODEL_CPU_BYTES_PER_VOXEL,
        gpu_bytes_per_voxel=main.MEMORY_MODEL_GPU_BYTES_PER_VOXEL,
        safety_factor=main.MEMORY_MODEL_SAFETY_FACTOR,
    )

    monkeypatch.setattr(main, "_resolve_uploaded_mesh_metadata", lambda **_kwargs: metadata)
    monkeypatch.setattr(main, "_resolve_mesh_resolution", lambda *_args, **_kwargs: (360, None))
    monkeypatch.setattr(main, "_snapshot_uploaded_mesh_memory_context", lambda _mesh_span: context)

    estimate, resolved_metadata = main._resolve_uploaded_mesh_memory_estimate(
        file_bytes=b"dummy",
        file_hash="dummy-hash",
        extension=".obj",
        lattice_pitch=2.0,
        voxels_per_lattice_period=6,
        compute_backend="auto",
    )

    assert resolved_metadata is metadata
    assert estimate.resolution == 360
    assert estimate.cpu_fatal is False
    assert estimate.gpu_fatal is False
    assert estimate.fatal is False
