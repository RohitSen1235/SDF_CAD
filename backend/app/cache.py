from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

from .models import (
    CompileDiagnostics,
    ComputeBackend,
    ComputePrecision,
    GridConfig,
    MeshBackend,
    MeshingMode,
    SceneIR,
)

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
    stats: dict[str, float | int | bool | str]


mesh_preview_cache: LruCache[MeshCacheEntry] = LruCache(maxsize=24)
field_preview_cache: LruCache[tuple[np.ndarray, str]] = LruCache(maxsize=8)


@dataclass
class UploadedMeshCacheEntry:
    vertices: list[list[float]]
    indices: list[list[int]]
    normals: list[list[float]]
    stats: dict[str, float | int | bool | str]


uploaded_mesh_preview_cache: LruCache[UploadedMeshCacheEntry] = LruCache(maxsize=12)


@dataclass
class UploadedHostFieldCacheEntry:
    vertices: np.ndarray
    faces: np.ndarray
    bounds: list[list[float]]
    host_sdf: np.ndarray


uploaded_host_field_cache: LruCache[UploadedHostFieldCacheEntry] = LruCache(maxsize=8)


def hash_source(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def hash_preview_request(
    scene_ir: SceneIR,
    params: dict[str, float],
    grid: GridConfig,
    compute_precision: ComputePrecision = "float32",
    compute_backend: ComputeBackend = "auto",
    mesh_backend: MeshBackend = "auto",
    meshing_mode: MeshingMode = "uniform",
) -> str:
    payload: dict[str, Any] = {
        "scene": scene_ir.model_dump(mode="json"),
        "params": {k: float(v) for k, v in sorted(params.items())},
        "grid": grid.model_dump(mode="json"),
        "compute_precision": compute_precision,
        "compute_backend": compute_backend,
        "mesh_backend": mesh_backend,
        "meshing_mode": meshing_mode,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def hash_field_preview_request(
    scene_ir: SceneIR,
    params: dict[str, float],
    grid: GridConfig,
    compute_precision: ComputePrecision = "float32",
    compute_backend: ComputeBackend = "auto",
) -> str:
    payload: dict[str, Any] = {
        "scene": scene_ir.model_dump(mode="json"),
        "params": {k: float(v) for k, v in sorted(params.items())},
        "grid": grid.model_dump(mode="json"),
        "compute_precision": compute_precision,
        "compute_backend": compute_backend,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def hash_uploaded_mesh_request(
    *,
    file_bytes: bytes,
    extension: str,
    shell_thickness: float,
    lattice_type: str,
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    quality_profile: str,
    compute_backend: str = "auto",
    mesh_backend: str = "auto",
    meshing_mode: str = "uniform",
) -> str:
    payload: dict[str, Any] = {
        "file_hash": hashlib.sha256(file_bytes).hexdigest(),
        "extension": extension.lower(),
        "shell_thickness": float(shell_thickness),
        "lattice_type": lattice_type,
        "lattice_pitch": float(lattice_pitch),
        "lattice_thickness": float(lattice_thickness),
        "lattice_phase": float(lattice_phase),
        "quality_profile": quality_profile,
        "compute_backend": compute_backend,
        "mesh_backend": mesh_backend,
        "meshing_mode": meshing_mode,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def hash_uploaded_mesh_host_request(
    *,
    file_bytes: bytes,
    extension: str,
    quality_profile: str,
) -> str:
    payload: dict[str, Any] = {
        "file_hash": hashlib.sha256(file_bytes).hexdigest(),
        "extension": extension.lower(),
        "quality_profile": quality_profile,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
