import gc

from app import cache, evaluator, gpu_memory, meshing


def test_clear_all_caches_empties_process_lru_caches() -> None:
    cache.scene_compile_cache.set("compile", object())
    cache.mesh_preview_cache.set("mesh", object())
    cache.field_preview_cache.set("field", object())
    cache.uploaded_mesh_preview_cache.set("uploaded_mesh", object())
    cache.uploaded_mesh_metadata_cache.set("uploaded_metadata", object())
    cache.uploaded_composed_field_cache.set("uploaded_composed", object())
    cache.uploaded_host_field_cache.set("uploaded_field", object())

    cache.clear_all_caches()

    assert cache.scene_compile_cache.get("compile") is None
    assert cache.mesh_preview_cache.get("mesh") is None
    assert cache.field_preview_cache.get("field") is None
    assert cache.uploaded_mesh_preview_cache.get("uploaded_mesh") is None
    assert cache.uploaded_mesh_metadata_cache.get("uploaded_metadata") is None
    assert cache.uploaded_composed_field_cache.get("uploaded_composed") is None
    assert cache.uploaded_host_field_cache.get("uploaded_field") is None


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
