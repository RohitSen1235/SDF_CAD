from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


FieldStorageMode = Literal["dense", "sparse_bricks", "octree"]


def sample_dense_trilinear(
    field: np.ndarray,
    bounds: list[list[float]],
    points: np.ndarray,
) -> np.ndarray:
    if field.ndim != 3:
        raise ValueError("field must be a 3D array")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")

    resolution = int(field.shape[0])
    if resolution < 2:
        raise ValueError("field resolution must be >= 2")

    dx = (bounds[0][1] - bounds[0][0]) / float(resolution - 1)
    dy = (bounds[1][1] - bounds[1][0]) / float(resolution - 1)
    dz = (bounds[2][1] - bounds[2][0]) / float(resolution - 1)

    tx = np.clip((points[:, 0] - bounds[0][0]) / dx, 0.0, resolution - 1.000001)
    ty = np.clip((points[:, 1] - bounds[1][0]) / dy, 0.0, resolution - 1.000001)
    tz = np.clip((points[:, 2] - bounds[2][0]) / dz, 0.0, resolution - 1.000001)

    x0 = np.floor(tx).astype(np.int32)
    y0 = np.floor(ty).astype(np.int32)
    z0 = np.floor(tz).astype(np.int32)
    x1 = np.minimum(x0 + 1, resolution - 1)
    y1 = np.minimum(y0 + 1, resolution - 1)
    z1 = np.minimum(z0 + 1, resolution - 1)

    fx = tx - x0
    fy = ty - y0
    fz = tz - z0

    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


def detect_zero_crossing_blocks(
    field: np.ndarray,
    block_size: int,
    candidate_blocks: list[tuple[int, int, int]] | None = None,
) -> list[tuple[int, int, int]]:
    if field.ndim != 3:
        raise ValueError("field must be a 3D array")
    resolution = int(field.shape[0])
    step = max(2, min(int(block_size), resolution - 1))
    if step <= 0:
        return []

    blocks: list[tuple[int, int, int]] = []
    if candidate_blocks is not None:
        candidate_iter = candidate_blocks
    else:
        blocks_per_axis = int(math.ceil(resolution / float(step)))
        candidate_iter = [
            (bx, by, bz)
            for bx in range(blocks_per_axis)
            for by in range(blocks_per_axis)
            for bz in range(blocks_per_axis)
        ]

    for bx, by, bz in candidate_iter:
        i0 = max(0, min(int(bx) * step, resolution - 1))
        j0 = max(0, min(int(by) * step, resolution - 1))
        k0 = max(0, min(int(bz) * step, resolution - 1))
        i1 = min(resolution, i0 + step + 1)
        j1 = min(resolution, j0 + step + 1)
        k1 = min(resolution, k0 + step + 1)
        block = field[i0:i1, j0:j1, k0:k1]
        if block.size == 0:
            continue
        bmin = float(np.min(block))
        bmax = float(np.max(block))
        if bmin <= 0.0 <= bmax:
            blocks.append((int(bx), int(by), int(bz)))
    return blocks


@dataclass
class SparseBrickField:
    bounds: list[list[float]]
    resolution: int
    block_size: int
    background_value: float
    bricks: dict[tuple[int, int, int], np.ndarray]
    _dense_cache: np.ndarray | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_dense(
        cls,
        dense_field: np.ndarray,
        bounds: list[list[float]],
        block_size: int,
        active_blocks: list[tuple[int, int, int]] | None = None,
        background_value: float | None = None,
    ) -> "SparseBrickField":
        if dense_field.ndim != 3:
            raise ValueError("dense_field must be a 3D array")
        resolution = int(dense_field.shape[0])
        step = max(2, min(int(block_size), resolution))

        if background_value is None:
            max_abs = float(np.max(np.abs(dense_field))) if dense_field.size > 0 else 1.0
            background_value = float(max(1.0, max_abs))

        if active_blocks is None:
            active_blocks = detect_zero_crossing_blocks(dense_field, step)

        bricks: dict[tuple[int, int, int], np.ndarray] = {}
        for bx, by, bz in active_blocks:
            i0 = int(bx) * step
            j0 = int(by) * step
            k0 = int(bz) * step
            i1 = min(resolution, i0 + step)
            j1 = min(resolution, j0 + step)
            k1 = min(resolution, k0 + step)
            if i0 >= i1 or j0 >= j1 or k0 >= k1:
                continue
            bricks[(int(bx), int(by), int(bz))] = np.array(
                dense_field[i0:i1, j0:j1, k0:k1],
                copy=True,
            )

        return cls(
            bounds=[[float(axis[0]), float(axis[1])] for axis in bounds],
            resolution=resolution,
            block_size=step,
            background_value=float(background_value),
            bricks=bricks,
        )

    def active_blocks(self) -> list[tuple[int, int, int]]:
        return sorted(self.bricks.keys())

    def materialize_dense(self) -> np.ndarray:
        if self._dense_cache is not None:
            return np.array(self._dense_cache, copy=True)

        dense = np.full(
            (self.resolution, self.resolution, self.resolution),
            np.float32(self.background_value),
            dtype=np.float32,
        )
        step = int(self.block_size)
        for (bx, by, bz), brick in self.bricks.items():
            i0 = int(bx) * step
            j0 = int(by) * step
            k0 = int(bz) * step
            i1 = min(self.resolution, i0 + brick.shape[0])
            j1 = min(self.resolution, j0 + brick.shape[1])
            k1 = min(self.resolution, k0 + brick.shape[2])
            dense[i0:i1, j0:j1, k0:k1] = brick[: i1 - i0, : j1 - j0, : k1 - k0]

        self._dense_cache = dense
        return np.array(dense, copy=True)

    def sample_points(self, points: np.ndarray) -> np.ndarray:
        dense = self.materialize_dense()
        return sample_dense_trilinear(dense, self.bounds, points)


@dataclass
class OctreeField:
    bounds: list[list[float]]
    resolution: int
    block_size: int
    node_min: np.ndarray
    node_max: np.ndarray
    node_depth: np.ndarray
    node_kind: np.ndarray
    sparse_bricks: SparseBrickField

    @classmethod
    def from_sparse_bricks(cls, sparse_bricks: SparseBrickField) -> "OctreeField":
        step = int(sparse_bricks.block_size)
        dx = (sparse_bricks.bounds[0][1] - sparse_bricks.bounds[0][0]) / float(sparse_bricks.resolution - 1)
        dy = (sparse_bricks.bounds[1][1] - sparse_bricks.bounds[1][0]) / float(sparse_bricks.resolution - 1)
        dz = (sparse_bricks.bounds[2][1] - sparse_bricks.bounds[2][0]) / float(sparse_bricks.resolution - 1)

        mins: list[list[float]] = []
        maxs: list[list[float]] = []
        depths: list[int] = []
        kinds: list[int] = []
        for bx, by, bz in sparse_bricks.active_blocks():
            i0 = int(bx) * step
            j0 = int(by) * step
            k0 = int(bz) * step
            i1 = min(sparse_bricks.resolution - 1, i0 + step)
            j1 = min(sparse_bricks.resolution - 1, j0 + step)
            k1 = min(sparse_bricks.resolution - 1, k0 + step)
            mins.append(
                [
                    sparse_bricks.bounds[0][0] + i0 * dx,
                    sparse_bricks.bounds[1][0] + j0 * dy,
                    sparse_bricks.bounds[2][0] + k0 * dz,
                ]
            )
            maxs.append(
                [
                    sparse_bricks.bounds[0][0] + i1 * dx,
                    sparse_bricks.bounds[1][0] + j1 * dy,
                    sparse_bricks.bounds[2][0] + k1 * dz,
                ]
            )
            depths.append(1)
            kinds.append(1)  # near-surface leaf

        if not mins:
            mins = [[sparse_bricks.bounds[0][0], sparse_bricks.bounds[1][0], sparse_bricks.bounds[2][0]]]
            maxs = [[sparse_bricks.bounds[0][1], sparse_bricks.bounds[1][1], sparse_bricks.bounds[2][1]]]
            depths = [0]
            kinds = [0]

        return cls(
            bounds=[[float(axis[0]), float(axis[1])] for axis in sparse_bricks.bounds],
            resolution=int(sparse_bricks.resolution),
            block_size=int(sparse_bricks.block_size),
            node_min=np.asarray(mins, dtype=np.float32),
            node_max=np.asarray(maxs, dtype=np.float32),
            node_depth=np.asarray(depths, dtype=np.int16),
            node_kind=np.asarray(kinds, dtype=np.uint8),
            sparse_bricks=sparse_bricks,
        )

    def active_blocks(self) -> list[tuple[int, int, int]]:
        return self.sparse_bricks.active_blocks()

    def sample_points(self, points: np.ndarray) -> np.ndarray:
        return self.sparse_bricks.sample_points(points)

    def materialize_dense(self) -> np.ndarray:
        return self.sparse_bricks.materialize_dense()
