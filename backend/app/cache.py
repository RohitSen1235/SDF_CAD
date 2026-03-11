from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

from .models import CompileDiagnostics, GridConfig, SceneIR

T = TypeVar("T")


class LruCache(Generic[T]):
    def __init__(self, maxsize: int = 64) -> None:
        self.maxsize = maxsize
        self._data: dict[str, T] = {}
        self._order: list[str] = []

    def get(self, key: str) -> T | None:
        if key not in self._data:
            return None
        self._order.remove(key)
        self._order.append(key)
        return self._data[key]

    def set(self, key: str, value: T) -> None:
        if key in self._data:
            self._order.remove(key)
        self._data[key] = value
        self._order.append(key)
        while len(self._order) > self.maxsize:
            old = self._order.pop(0)
            self._data.pop(old, None)


@dataclass
class CompileCacheEntry:
    scene_ir: SceneIR
    diagnostics: CompileDiagnostics


scene_compile_cache: LruCache[CompileCacheEntry] = LruCache(maxsize=64)


@dataclass
class MeshCacheEntry:
    vertices: list[list[float]]
    indices: list[list[int]]
    normals: list[list[float]]
    stats: dict[str, float | int | bool]


mesh_preview_cache: LruCache[MeshCacheEntry] = LruCache(maxsize=24)
field_preview_cache: LruCache[np.ndarray] = LruCache(maxsize=8)


def hash_source(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def hash_preview_request(scene_ir: SceneIR, params: dict[str, float], grid: GridConfig) -> str:
    payload: dict[str, Any] = {
        "scene": scene_ir.model_dump(mode="json"),
        "params": {k: float(v) for k, v in sorted(params.items())},
        "grid": grid.model_dump(mode="json"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
