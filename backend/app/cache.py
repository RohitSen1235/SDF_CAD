from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

from .models import (
    CompileDiagnostics,
    ComputeBackend,
    ComputePrecision,
    GridConfig,
    MeshBackend,
    MeshPayload,
    MeshingMode,
    SceneIR,
    UploadedFieldStorageMode,
)

T = TypeVar("T")


class LruCache(Generic[T]):
    def __init__(self, maxsize: int = 64) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, T] = OrderedDict()

    def get(self, key: str) -> T | None:
        value = self._data.get(key)
        if value is None:
            return None
        self._data.move_to_end(key, last=True)
        return value

    def set(self, key: str, value: T) -> None:
        if key in self._data:
            self._data.move_to_end(key, last=True)
        self._data[key] = value
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


@dataclass
class CompileCacheEntry:
    scene_ir: SceneIR
    diagnostics: CompileDiagnostics


scene_compile_cache: LruCache[CompileCacheEntry] = LruCache(maxsize=64)


@dataclass
class MeshCacheEntry:
    mesh: MeshPayload
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    stats: dict[str, float | int | bool | str]


mesh_preview_cache: LruCache[MeshCacheEntry] = LruCache(maxsize=24)
field_preview_cache: LruCache[tuple[np.ndarray, str]] = LruCache(maxsize=8)


@dataclass
class UploadedMeshCacheEntry:
    mesh: MeshPayload | None
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    stats: dict[str, float | int | bool | str]
    field_resolution: int | None = None
    field_bounds: list[list[float]] | None = None
    field_data: str | None = None


uploaded_mesh_preview_cache: LruCache[UploadedMeshCacheEntry] = LruCache(maxsize=12)


@dataclass
class UploadedMeshMetadataCacheEntry:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    mesh_span: float


uploaded_mesh_metadata_cache: LruCache[UploadedMeshMetadataCacheEntry] = LruCache(maxsize=12)


@dataclass
class UploadedComposedFieldCacheEntry:
    field: np.ndarray
    bounds: list[list[float]]
    resolution: int
    eval_backend: str
    block_size: int | None = None
    active_blocks: list[tuple[int, int, int]] | None = None


uploaded_composed_field_cache: LruCache[UploadedComposedFieldCacheEntry] = LruCache(maxsize=12)


@dataclass
class UploadedHostFieldCacheEntry:
    bounds: list[list[float]]
    host_sdf: np.ndarray
    field_storage_mode: UploadedFieldStorageMode = "dense"
    block_size: int | None = None
    active_blocks: list[tuple[int, int, int]] | None = None
    sparse_background_value: float | None = None
    sparse_bricks: dict[tuple[int, int, int], np.ndarray] | None = None


uploaded_host_field_cache: LruCache[UploadedHostFieldCacheEntry] = LruCache(maxsize=8)


def clear_all_preview_caches() -> None:
    mesh_preview_cache.clear()
    field_preview_cache.clear()
    uploaded_mesh_preview_cache.clear()
    uploaded_mesh_metadata_cache.clear()
    uploaded_composed_field_cache.clear()
    uploaded_host_field_cache.clear()


def clear_all_caches() -> None:
    scene_compile_cache.clear()
    clear_all_preview_caches()


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
    scene_key = scene_ir.source_hash
    if not scene_key:
        scene_key = json.dumps(scene_ir.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    payload: dict[str, Any] = {
        "scene_key": scene_key,
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
    scene_key = scene_ir.source_hash
    if not scene_key:
        scene_key = json.dumps(scene_ir.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    payload: dict[str, Any] = {
        "scene_key": scene_key,
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
    resolution: int,
    shell_thickness: float,
    lattice_type: str,
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    voxels_per_lattice_period: int,
    compute_backend: str = "auto",
    mesh_backend: str = "auto",
    meshing_mode: str = "uniform",
    field_storage_mode: str = "auto",
) -> str:
    payload: dict[str, Any] = {
        "file_hash": hashlib.sha256(file_bytes).hexdigest(),
        "extension": extension.lower(),
        "resolution": int(resolution),
        "shell_thickness": float(shell_thickness),
        "lattice_type": lattice_type,
        "lattice_pitch": float(lattice_pitch),
        "lattice_thickness": float(lattice_thickness),
        "lattice_phase": float(lattice_phase),
        "voxels_per_lattice_period": int(voxels_per_lattice_period),
        "compute_backend": compute_backend,
        "mesh_backend": mesh_backend,
        "meshing_mode": meshing_mode,
        "field_storage_mode": field_storage_mode,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def hash_uploaded_mesh_metadata_request(
    *,
    file_bytes: bytes,
    extension: str,
) -> str:
    payload: dict[str, Any] = {
        "file_hash": hashlib.sha256(file_bytes).hexdigest(),
        "extension": extension.lower(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def hash_uploaded_mesh_host_request(
    *,
    file_bytes: bytes,
    extension: str,
    resolution: int,
    field_storage_mode: str = "auto",
) -> str:
    payload: dict[str, Any] = {
        "file_hash": hashlib.sha256(file_bytes).hexdigest(),
        "extension": extension.lower(),
        "resolution": int(resolution),
        "field_storage_mode": field_storage_mode,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def hash_uploaded_mesh_field_request(
    *,
    file_bytes: bytes,
    extension: str,
    resolution: int,
    shell_thickness: float,
    lattice_type: str,
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
    voxels_per_lattice_period: int,
    compute_backend: str = "auto",
    field_storage_mode: str = "auto",
) -> str:
    payload: dict[str, Any] = {
        "file_hash": hashlib.sha256(file_bytes).hexdigest(),
        "extension": extension.lower(),
        "resolution": int(resolution),
        "shell_thickness": float(shell_thickness),
        "lattice_type": lattice_type,
        "lattice_pitch": float(lattice_pitch),
        "lattice_thickness": float(lattice_thickness),
        "lattice_phase": float(lattice_phase),
        "voxels_per_lattice_period": int(voxels_per_lattice_period),
        "compute_backend": compute_backend,
        "field_storage_mode": field_storage_mode,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
