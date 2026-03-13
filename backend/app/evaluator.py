from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Literal

import numpy as np

from .dsl import DslError
from .models import ComputeBackend, ComputePrecision, GridConfig, SceneIR, SceneNode

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False

try:
    from numba import njit, prange, vectorize

    @vectorize(["float32(float32,float32,float32)", "float64(float64,float64,float64)"], nopython=True, cache=True)
    def _smin_numba(a: float, b: float, k: float) -> float:
        h = 0.5 + 0.5 * (b - a) / k
        if h < 0.0:
            h = 0.0
        elif h > 1.0:
            h = 1.0
        return (b * (1.0 - h) + a * h) - k * h * (1.0 - h)

    @njit(parallel=True, cache=True)
    def _spline_min_dist_numba(
        qx_flat: np.ndarray,
        qy_flat: np.ndarray,
        qz_flat: np.ndarray,
        segments: np.ndarray,
    ) -> np.ndarray:
        out = np.empty(qx_flat.shape[0], dtype=qx_flat.dtype)
        for i in prange(qx_flat.shape[0]):
            px = qx_flat[i]
            py = qy_flat[i]
            pz = qz_flat[i]
            best = np.float32(1e30)
            for s in range(segments.shape[0]):
                ax = segments[s, 0]
                ay = segments[s, 1]
                az = segments[s, 2]
                bx = segments[s, 3]
                by = segments[s, 4]
                bz = segments[s, 5]

                abx = bx - ax
                aby = by - ay
                abz = bz - az
                denom = abx * abx + aby * aby + abz * abz
                if denom < 1e-12:
                    dx = px - ax
                    dy = py - ay
                    dz = pz - az
                else:
                    apx = px - ax
                    apy = py - ay
                    apz = pz - az
                    t = (apx * abx + apy * aby + apz * abz) / denom
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                    cx = ax + t * abx
                    cy = ay + t * aby
                    cz = az + t * abz
                    dx = px - cx
                    dy = py - cy
                    dz = pz - cz

                d = np.sqrt(dx * dx + dy * dy + dz * dz)
                if d < best:
                    best = d
            out[i] = best
        return out

    @njit(parallel=True, cache=True)
    def _blade_field_numba(
        r_flat: np.ndarray,
        theta_flat: np.ndarray,
        qy_flat: np.ndarray,
        r_in: float,
        radial_span: float,
        blade_count: int,
        blade_thickness: float,
        hub_h: float,
        blade_twist_dir: float,
    ) -> np.ndarray:
        out = np.empty(r_flat.shape[0], dtype=r_flat.dtype)
        for i in prange(r_flat.shape[0]):
            rv = r_flat[i]
            tv = (rv - r_in) / radial_span
            if tv < 0.0:
                tv = 0.0
            elif tv > 1.0:
                tv = 1.0
            th = theta_flat[i]
            qy = qy_flat[i]
            best = np.float32(1e30)
            rr = rv if rv > 1e-6 else np.float32(1e-6)
            for idx in range(blade_count):
                base_angle = 2.0 * np.pi * idx / float(blade_count)
                target = base_angle + blade_twist_dir * tv
                ang = (th - target + np.pi) % (2.0 * np.pi) - np.pi
                arc_dist = abs(ang) * rr
                blade_field = max(arc_dist - abs(blade_thickness) * 0.5, abs(qy) - hub_h * 0.5)
                if blade_field < best:
                    best = blade_field
            out[i] = best
        return out

    NUMBA_AVAILABLE = True
except Exception:
    _smin_numba = None
    _spline_min_dist_numba = None
    _blade_field_numba = None
    NUMBA_AVAILABLE = False


class EvaluationError(ValueError):
    pass


LEAF_TYPES_WITH_AABB_SKIP = {"primitive", "field_expr", "turbomachine"}

EVAL_DTYPE_MAP: dict[ComputePrecision, np.dtype[Any]] = {
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float16),
}


def _resolve_eval_dtype(compute_precision: ComputePrecision) -> np.dtype[Any]:
    return EVAL_DTYPE_MAP[compute_precision]


def _resolve_compute_backend(requested: ComputeBackend) -> Literal["cpu", "cuda"]:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if CUPY_AVAILABLE else "cpu"
    return "cuda" if CUPY_AVAILABLE else "cpu"


@lru_cache(maxsize=48)
def _cached_grid(
    bounds_key: tuple[float, ...],
    resolution: int,
    dtype_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds_key
    dtype = np.dtype(dtype_name)
    x = np.linspace(xmin, xmax, resolution, dtype=dtype)
    y = np.linspace(ymin, ymax, resolution, dtype=dtype)
    z = np.linspace(zmin, zmax, resolution, dtype=dtype)
    return np.meshgrid(x, y, z, indexing="ij")


@lru_cache(maxsize=48)
def _cached_axes(
    bounds_key: tuple[float, ...],
    resolution: int,
    dtype_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds_key
    dtype = np.dtype(dtype_name)
    x = np.linspace(xmin, xmax, resolution, dtype=dtype)
    y = np.linspace(ymin, ymax, resolution, dtype=dtype)
    z = np.linspace(zmin, zmax, resolution, dtype=dtype)
    return x, y, z


def _rotation_matrix_xyz(deg: tuple[float, float, float], dtype: np.dtype[Any]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.array(deg, dtype=np.float64))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    mx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=dtype)
    my = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=dtype)
    mz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    return mz @ my @ mx


def _angle_wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _resolve_scalar(value: Any, parameter_values: dict[str, float]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and "$param" in value:
        name = value["$param"]
        if name not in parameter_values:
            raise EvaluationError(f"Missing value for parameter '{name}'")
        return float(parameter_values[name])
    raise EvaluationError(f"Unsupported scalar payload: {value!r}")


def _resolve_string(value: Any, default: str) -> str:
    if isinstance(value, str):
        return value
    return default


def _resolve_vec3(
    values: list[Any],
    parameter_values: dict[str, float],
    fallback: tuple[float, float, float],
) -> tuple[float, float, float]:
    if len(values) != 3:
        return fallback
    return (
        _resolve_scalar(values[0], parameter_values),
        _resolve_scalar(values[1], parameter_values),
        _resolve_scalar(values[2], parameter_values),
    )


def _smooth_union(a: np.ndarray, b: np.ndarray, k: float, xp: Any = np) -> np.ndarray:
    if k <= 0:
        return xp.minimum(a, b)
    if xp is np and NUMBA_AVAILABLE and _smin_numba is not None:
        return _smin_numba(a, b, a.dtype.type(k))
    h = xp.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return (b * (1.0 - h) + a * h) - k * h * (1.0 - h)


def _distance_to_aabb(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
    xp: Any = np,
) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    dx = xp.maximum(xp.maximum(xmin - px, 0.0), px - xmax)
    dy = xp.maximum(xp.maximum(ymin - py, 0.0), py - ymax)
    dz = xp.maximum(xp.maximum(zmin - pz, 0.0), pz - zmax)
    outside = xp.sqrt(dx * dx + dy * dy + dz * dz)

    inside = px - xmin
    inside = xp.minimum(inside, xmax - px)
    inside = xp.minimum(inside, py - ymin)
    inside = xp.minimum(inside, ymax - py)
    inside = xp.minimum(inside, pz - zmin)
    inside = xp.minimum(inside, zmax - pz)
    return outside - xp.maximum(inside, 0.0)


def _dist_to_segment(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    xp: Any = np,
) -> np.ndarray:
    ax, ay, az = a
    bx, by, bz = b
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    denom = abx * abx + aby * aby + abz * abz
    if denom < 1e-12:
        return xp.sqrt((px - ax) ** 2 + (py - ay) ** 2 + (pz - az) ** 2)

    apx = px - ax
    apy = py - ay
    apz = pz - az
    t = (apx * abx + apy * aby + apz * abz) / denom
    t = xp.clip(t, 0.0, 1.0)

    cx = ax + t * abx
    cy = ay + t * aby
    cz = az + t * abz
    return xp.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)


def _repeat_local(coord: np.ndarray, pitch: float) -> np.ndarray:
    if pitch <= 1e-6:
        return coord
    return ((coord + pitch * 0.5) % pitch) - pitch * 0.5


def _parse_float_tokens(raw: str) -> list[float] | None:
    text = raw.replace(";", " ").replace(",", " ")
    tokens = [token for token in text.split() if token]
    if not tokens:
        return None
    out: list[float] = []
    for token in tokens:
        try:
            out.append(float(token))
        except ValueError:
            return None
    return out


def _parse_points_string(raw: str) -> list[tuple[float, float, float]] | None:
    values = _parse_float_tokens(raw)
    if values is None or len(values) < 6 or len(values) % 3 != 0:
        return None
    return [
        (values[idx], values[idx + 1], values[idx + 2])
        for idx in range(0, len(values), 3)
    ]


def _parse_heights_grid(raw: str) -> np.ndarray | None:
    values = _parse_float_tokens(raw)
    if values is None or len(values) != 16:
        return None
    return np.asarray(values, dtype=np.float32).reshape(4, 4)


def _catmull_rom_sample(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        2.0 * p1[0]
        + (-p0[0] + p2[0]) * t
        + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        2.0 * p1[1]
        + (-p0[1] + p2[1]) * t
        + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
    )
    z = 0.5 * (
        2.0 * p1[2]
        + (-p0[2] + p2[2]) * t
        + (2.0 * p0[2] - 5.0 * p1[2] + 4.0 * p2[2] - p3[2]) * t2
        + (-p0[2] + 3.0 * p1[2] - 3.0 * p2[2] + p3[2]) * t3
    )
    return (x, y, z)


def _sample_catmull_rom_polyline(
    points: list[tuple[float, float, float]],
    samples_per_span: int,
    closed: bool,
) -> list[tuple[float, float, float]]:
    if len(points) < 2:
        return points

    samples = max(4, samples_per_span)
    polyline: list[tuple[float, float, float]] = []

    if closed:
        count = len(points)
        for idx in range(count):
            p0 = points[(idx - 1) % count]
            p1 = points[idx]
            p2 = points[(idx + 1) % count]
            p3 = points[(idx + 2) % count]
            for step in range(samples):
                t = float(step) / float(samples)
                polyline.append(_catmull_rom_sample(p0, p1, p2, p3, t))
        polyline.append(polyline[0])
        return polyline

    span_count = len(points) - 1
    for idx in range(span_count):
        p0 = points[idx - 1] if idx > 0 else points[idx]
        p1 = points[idx]
        p2 = points[idx + 1]
        p3 = points[idx + 2] if idx + 2 < len(points) else points[idx + 1]
        for step in range(samples):
            t = float(step) / float(samples)
            polyline.append(_catmull_rom_sample(p0, p1, p2, p3, t))
    polyline.append(points[-1])
    return polyline


def _cubic_bezier_basis(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    one_minus = 1.0 - t
    b0 = one_minus * one_minus * one_minus
    b1 = 3.0 * t * one_minus * one_minus
    b2 = 3.0 * t * t * one_minus
    b3 = t * t * t
    return b0, b1, b2, b3


def _strut_segments(kind: str) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    c = (0.0, 0.0, 0.0)
    corners = [
        (-0.5, -0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, 0.5, 0.5),
        (0.5, -0.5, -0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, -0.5),
        (0.5, 0.5, 0.5),
    ]

    edges = [
        (corners[0], corners[1]), (corners[0], corners[2]), (corners[0], corners[4]),
        (corners[7], corners[6]), (corners[7], corners[5]), (corners[7], corners[3]),
        (corners[1], corners[3]), (corners[1], corners[5]),
        (corners[2], corners[3]), (corners[2], corners[6]),
        (corners[4], corners[5]), (corners[4], corners[6]),
    ]

    if kind == "bcc":
        return [(c, corner) for corner in corners]

    if kind == "fcc":
        face_centers = [
            (0.0, 0.0, 0.5),
            (0.0, 0.0, -0.5),
            (0.0, 0.5, 0.0),
            (0.0, -0.5, 0.0),
            (0.5, 0.0, 0.0),
            (-0.5, 0.0, 0.0),
        ]
        out: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        for fc in face_centers:
            for corner in corners:
                if abs(abs(fc[0]) + abs(fc[1]) + abs(fc[2]) - 0.5) < 1e-9:
                    shared = sum(
                        1
                        for idx in range(3)
                        if abs(fc[idx]) == 0.5 and math.isclose(fc[idx], corner[idx], rel_tol=0.0, abs_tol=1e-9)
                    )
                    if shared >= 1:
                        out.append((fc, corner))
        return out

    if kind == "octet":
        return [(c, corner) for corner in corners] + edges

    return [(c, corner) for corner in corners]


class _FieldRuntime:
    def __init__(
        self,
        scene_ir: SceneIR,
        parameter_values: dict[str, float],
        eval_dtype: np.dtype[Any],
        xp: Any = np,
    ) -> None:
        self.scene_ir = scene_ir
        self.parameter_values = parameter_values
        self.eval_dtype = np.dtype(eval_dtype)
        self.xp = xp
        self._small_eps = 1e-4 if self.eval_dtype == np.dtype(np.float16) else 1e-12
        self.node_lookup: dict[str, SceneNode] = {node.id: node for node in scene_ir.nodes}
        self._bounds_hint_cache: dict[str, tuple[float, float, float, float, float, float] | None] = {
            node.id: self._resolve_bounds_hint(node) for node in scene_ir.nodes
        }
        self._scalar_param_cache: dict[tuple[str, str], float] = {}
        self._string_param_cache: dict[tuple[str, str], str] = {}
        self._transform_cache: dict[str, tuple[float, float, float, float, float, float, np.ndarray | None]] = {}
        self._spline_polyline_cache: dict[
            tuple[str, str, bool, int],
            list[tuple[float, float, float]],
        ] = {}
        self._freeform_grid_cache: dict[tuple[str, str, str, str], np.ndarray] = {}

    def _node_scalar_param(self, node: SceneNode, key: str, default: Any) -> float:
        cache_key = (node.id, key)
        if cache_key in self._scalar_param_cache:
            return self._scalar_param_cache[cache_key]
        value = _resolve_scalar(node.params.get(key, default), self.parameter_values)
        self._scalar_param_cache[cache_key] = value
        return value

    def _node_string_param(self, node: SceneNode, key: str, default: str) -> str:
        cache_key = (node.id, key)
        if cache_key in self._string_param_cache:
            return self._string_param_cache[cache_key]
        value = _resolve_string(node.params.get(key, default), default)
        self._string_param_cache[cache_key] = value
        return value

    def _node_transform(
        self, node: SceneNode
    ) -> tuple[float, float, float, float, float, float, np.ndarray | None]:
        cached = self._transform_cache.get(node.id)
        if cached is not None:
            return cached

        transform_payload = node.transform or {}
        translate = _resolve_vec3(
            transform_payload.get("translate", [0.0, 0.0, 0.0]),
            self.parameter_values,
            (0.0, 0.0, 0.0),
        )
        rotate = _resolve_vec3(
            transform_payload.get("rotate", [0.0, 0.0, 0.0]),
            self.parameter_values,
            (0.0, 0.0, 0.0),
        )
        scale = _resolve_vec3(
            transform_payload.get("scale", [1.0, 1.0, 1.0]),
            self.parameter_values,
            (1.0, 1.0, 1.0),
        )
        sx, sy, sz = scale
        if abs(sx) < 1e-6 or abs(sy) < 1e-6 or abs(sz) < 1e-6:
            raise EvaluationError(f"Node '{node.id}' has singular scale")

        inv_rotation = _rotation_matrix_xyz(rotate, self.eval_dtype).T if any(abs(v) > 1e-9 for v in rotate) else None
        tx, ty, tz = translate
        out = (tx, ty, tz, sx, sy, sz, inv_rotation)
        self._transform_cache[node.id] = out
        return out

    def _node_spline_polyline(self, node: SceneNode) -> list[tuple[float, float, float]]:
        points_raw = self._node_string_param(
            node,
            "points",
            "0 0 0; 0.4 0.2 0; 0.8 -0.2 0.1; 1.2 0 0",
        )
        closed = self._node_scalar_param(node, "closed", 0.0) >= 0.5
        samples = int(max(4, round(self._node_scalar_param(node, "samples", 20.0))))
        cache_key = (node.id, points_raw, closed, samples)
        if cache_key in self._spline_polyline_cache:
            return self._spline_polyline_cache[cache_key]

        points = _parse_points_string(points_raw)
        if points is None or len(points) < 2:
            raise EvaluationError(
                f"spline primitive '{node.id}' must define at least two 3D control points in 'points'"
            )
        polyline = _sample_catmull_rom_polyline(points, samples, closed)
        if len(polyline) < 2:
            raise EvaluationError(f"spline primitive '{node.id}' could not build a valid curve")
        self._spline_polyline_cache[cache_key] = polyline
        return polyline

    def _node_freeform_grid(self, node: SceneNode) -> np.ndarray:
        heights_raw = self._node_string_param(
            node,
            "heights",
            "0 0 0 0; 0 0.2 0.2 0; 0 0.2 0.2 0; 0 0 0 0",
        )
        xp_name = "cupy" if cp is not None and self.xp is cp else "numpy"
        cache_key = (node.id, heights_raw, self.eval_dtype.name, xp_name)
        if cache_key in self._freeform_grid_cache:
            return self._freeform_grid_cache[cache_key]

        grid = _parse_heights_grid(heights_raw)
        if grid is None:
            raise EvaluationError(
                f"freeform_surface primitive '{node.id}' expects exactly 16 numeric values in 'heights'"
            )
        cast_grid = self.xp.asarray(grid, dtype=self.eval_dtype)
        self._freeform_grid_cache[cache_key] = cast_grid
        return cast_grid

    def evaluate(self, px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
        xp = self.xp
        memo: dict[tuple[str, int, int, int], np.ndarray] = {}

        def to_bool(value: Any) -> bool:
            if hasattr(value, "item"):
                return bool(value.item())
            return bool(value)

        def coerce_eval_array(value: Any, ref_shape: tuple[int, ...]) -> np.ndarray:
            arr = xp.asarray(value, dtype=self.eval_dtype)
            if arr.shape == ref_shape:
                return arr
            try:
                return xp.broadcast_to(arr, ref_shape).astype(self.eval_dtype, copy=False)
            except ValueError as exc:
                raise EvaluationError(
                    f"Node produced non-broadcastable shape {arr.shape}; expected {ref_shape}"
                ) from exc

        def eval_node(node_id: str, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            key = (node_id, id(qx), id(qy), id(qz))
            if key in memo:
                return memo[key]

            node = self.node_lookup.get(node_id)
            if node is None:
                raise EvaluationError(f"Node '{node_id}' is not defined")

            bounds_hint = self._bounds_hint_cache.get(node.id)
            if bounds_hint is not None and node.type in LEAF_TYPES_WITH_AABB_SKIP:
                xmin, xmax, ymin, ymax, zmin, zmax = bounds_hint
                mask = (
                    (qx >= xmin)
                    & (qx <= xmax)
                    & (qy >= ymin)
                    & (qy <= ymax)
                    & (qz >= zmin)
                    & (qz <= zmax)
                )

                if not to_bool(xp.any(mask)):
                    out = coerce_eval_array(_distance_to_aabb(qx, qy, qz, bounds_hint, xp=xp), qx.shape)
                    memo[key] = out
                    return out

                if to_bool(xp.all(mask)):
                    out = coerce_eval_array(eval_node_core(node, qx, qy, qz), qx.shape)
                    memo[key] = out
                    return out

                out = _distance_to_aabb(qx, qy, qz, bounds_hint, xp=xp)
                sub = eval_node_core(node, qx[mask], qy[mask], qz[mask])
                out = out.astype(self.eval_dtype, copy=True)
                out[mask] = sub
                memo[key] = out
                return out

            out = coerce_eval_array(eval_node_core(node, qx, qy, qz), qx.shape)
            memo[key] = out
            return out

        def eval_node_core(node: SceneNode, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            if node.type == "primitive":
                return eval_primitive(node, qx, qy, qz)
            if node.type == "boolean":
                return eval_boolean(node, qx, qy, qz)
            if node.type == "transform":
                return eval_transform(node, qx, qy, qz)
            if node.type == "field_expr":
                return eval_field_expr(node, qx, qy, qz)
            if node.type == "domain_op":
                return eval_domain_op(node, qx, qy, qz)
            if node.type == "lattice":
                return eval_lattice(node, qx, qy, qz)
            if node.type == "turbomachine":
                return eval_turbomachine(node, qx, qy, qz)
            raise EvaluationError(f"Unknown node type '{node.type}'")

        def eval_expr(expr: dict[str, Any], qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            kind = expr.get("kind")
            if kind == "number":
                return float(expr.get("value", 0.0))
            if kind == "param":
                name = expr.get("name")
                if not isinstance(name, str) or name not in self.parameter_values:
                    raise EvaluationError(f"Unknown parameter '{name}' in expression")
                return float(self.parameter_values[name])
            if kind == "var":
                name = expr.get("name")
                if name == "x":
                    return qx
                if name == "y":
                    return qy
                if name == "z":
                    return qz
                raise EvaluationError(f"Unknown variable '{name}' in expression")
            if kind == "node_ref":
                node_id = expr.get("id")
                if not isinstance(node_id, str):
                    raise EvaluationError("Expression node_ref is missing node id")
                return eval_node(node_id, qx, qy, qz)
            if kind == "unary":
                op = expr.get("op")
                arg = eval_expr(expr["arg"], qx, qy, qz)
                if op == "-":
                    return -arg
                raise EvaluationError(f"Unsupported unary op '{op}'")
            if kind == "binary":
                left = eval_expr(expr["left"], qx, qy, qz)
                right = eval_expr(expr["right"], qx, qy, qz)
                op = expr.get("op")
                if op == "+":
                    return left + right
                if op == "-":
                    return left - right
                if op == "*":
                    return left * right
                if op == "/":
                    eps = self.eval_dtype.type(self._small_eps)
                    return left / xp.where(xp.abs(right) < eps, eps, right)
                if op == "^":
                    return xp.power(left, right)
                raise EvaluationError(f"Unsupported binary op '{op}'")
            if kind == "func":
                name = expr.get("name")
                args = [eval_expr(arg, qx, qy, qz) for arg in expr.get("args", [])]
                if name == "sin":
                    if len(args) != 1:
                        raise EvaluationError("sin expects exactly 1 argument")
                    return xp.sin(args[0])
                if name == "cos":
                    if len(args) != 1:
                        raise EvaluationError("cos expects exactly 1 argument")
                    return xp.cos(args[0])
                if name == "tan":
                    if len(args) != 1:
                        raise EvaluationError("tan expects exactly 1 argument")
                    return xp.tan(args[0])
                if name == "abs":
                    if len(args) != 1:
                        raise EvaluationError("abs expects exactly 1 argument")
                    return xp.abs(args[0])
                if name == "sqrt":
                    if len(args) != 1:
                        raise EvaluationError("sqrt expects exactly 1 argument")
                    return xp.sqrt(xp.maximum(args[0], 0.0))
                if name == "exp":
                    if len(args) != 1:
                        raise EvaluationError("exp expects exactly 1 argument")
                    return xp.exp(args[0])
                if name == "log":
                    if len(args) != 1:
                        raise EvaluationError("log expects exactly 1 argument")
                    eps = self.eval_dtype.type(self._small_eps)
                    return xp.log(xp.maximum(args[0], eps))
                if name == "min":
                    if not args:
                        raise EvaluationError("min expects at least 1 argument")
                    out = args[0]
                    for arg in args[1:]:
                        out = xp.minimum(out, arg)
                    return out
                if name == "max":
                    if not args:
                        raise EvaluationError("max expects at least 1 argument")
                    out = args[0]
                    for arg in args[1:]:
                        out = xp.maximum(out, arg)
                    return out
                if name == "clamp":
                    if len(args) != 3:
                        raise EvaluationError("clamp expects exactly 3 arguments")
                    return xp.clip(args[0], args[1], args[2])
                raise EvaluationError(f"Unsupported function '{name}'")

            raise EvaluationError(f"Unsupported expression kind '{kind}'")

        def eval_field_expr(node: SceneNode, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            if node.expr is None:
                raise EvaluationError(f"field_expr node '{node.id}' missing expression payload")
            out = eval_expr(node.expr, qx, qy, qz)
            arr = xp.asarray(out, dtype=self.eval_dtype)
            if arr.shape == qx.shape:
                return arr
            try:
                return xp.broadcast_to(arr, qx.shape).astype(self.eval_dtype, copy=False)
            except ValueError as exc:
                raise EvaluationError(
                    f"field_expr node '{node.id}' produced non-broadcastable shape {arr.shape}"
                ) from exc

        def eval_primitive(node: SceneNode, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            primitive = node.primitive
            if primitive == "sphere":
                r = self._node_scalar_param(node, "r", 1.0)
                return xp.sqrt(qx * qx + qy * qy + qz * qz) - r

            if primitive == "box":
                bx = self._node_scalar_param(node, "x", 0.5)
                by = self._node_scalar_param(node, "y", 0.5)
                bz = self._node_scalar_param(node, "z", 0.5)
                qmx = xp.abs(qx) - bx
                qmy = xp.abs(qy) - by
                qmz = xp.abs(qz) - bz
                ox = xp.maximum(qmx, 0.0)
                oy = xp.maximum(qmy, 0.0)
                oz = xp.maximum(qmz, 0.0)
                outside = xp.sqrt(ox * ox + oy * oy + oz * oz)
                inside = xp.maximum(qmx, qmy)
                inside = xp.maximum(inside, qmz)
                inside = xp.minimum(inside, 0.0)
                return outside + inside

            if primitive == "cylinder":
                r = self._node_scalar_param(node, "r", 0.4)
                h = self._node_scalar_param(node, "h", 1.0)
                d1 = xp.sqrt(qx * qx + qz * qz) - r
                d2 = xp.abs(qy) - h * 0.5
                outside = xp.sqrt(xp.maximum(d1, 0.0) ** 2 + xp.maximum(d2, 0.0) ** 2)
                inside = xp.minimum(xp.maximum(d1, d2), 0.0)
                return outside + inside

            if primitive == "torus":
                major_r = self._node_scalar_param(node, "R", 0.8)
                minor_r = self._node_scalar_param(node, "r", 0.2)
                q = xp.sqrt(qx * qx + qz * qz) - major_r
                return xp.sqrt(q * q + qy * qy) - minor_r

            if primitive == "spline":
                radius = self._node_scalar_param(node, "radius", 0.08)
                curve = self._node_spline_polyline(node)
                if xp is np and NUMBA_AVAILABLE and _spline_min_dist_numba is not None:
                    segments = np.asarray(
                        [
                            [a[0], a[1], a[2], b[0], b[1], b[2]]
                            for a, b in zip(curve[:-1], curve[1:])
                        ],
                        dtype=np.float32,
                    )
                    qx_flat = np.asarray(qx, dtype=np.float32).reshape(-1)
                    qy_flat = np.asarray(qy, dtype=np.float32).reshape(-1)
                    qz_flat = np.asarray(qz, dtype=np.float32).reshape(-1)
                    dist_flat = _spline_min_dist_numba(qx_flat, qy_flat, qz_flat, segments)
                    return dist_flat.reshape(qx.shape).astype(self.eval_dtype, copy=False) - abs(radius)

                dist = xp.full_like(qx, fill_value=xp.inf, dtype=self.eval_dtype)
                for idx in range(len(curve) - 1):
                    dseg = _dist_to_segment(qx, qy, qz, curve[idx], curve[idx + 1], xp=xp)
                    dist = xp.minimum(dist, dseg)
                return dist - abs(radius)

            if primitive == "freeform_surface":
                grid = self._node_freeform_grid(node)
                x_extent = abs(self._node_scalar_param(node, "x", 1.0))
                z_extent = abs(self._node_scalar_param(node, "z", 1.0))
                thickness = abs(self._node_scalar_param(node, "thickness", 0.06))
                if x_extent < 1e-6 or z_extent < 1e-6:
                    raise EvaluationError("freeform_surface x/z extents must be non-zero")

                u = xp.clip(qx / (2.0 * x_extent) + 0.5, 0.0, 1.0)
                v = xp.clip(qz / (2.0 * z_extent) + 0.5, 0.0, 1.0)
                bu0, bu1, bu2, bu3 = _cubic_bezier_basis(u)
                bv0, bv1, bv2, bv3 = _cubic_bezier_basis(v)

                row0 = bv0 * grid[0, 0] + bv1 * grid[0, 1] + bv2 * grid[0, 2] + bv3 * grid[0, 3]
                row1 = bv0 * grid[1, 0] + bv1 * grid[1, 1] + bv2 * grid[1, 2] + bv3 * grid[1, 3]
                row2 = bv0 * grid[2, 0] + bv1 * grid[2, 1] + bv2 * grid[2, 2] + bv3 * grid[2, 3]
                row3 = bv0 * grid[3, 0] + bv1 * grid[3, 1] + bv2 * grid[3, 2] + bv3 * grid[3, 3]
                surface_y = bu0 * row0 + bu1 * row1 + bu2 * row2 + bu3 * row3

                sheet = xp.abs(qy - surface_y) - thickness

                rx = xp.abs(qx) - x_extent
                rz = xp.abs(qz) - z_extent
                outside = xp.sqrt(xp.maximum(rx, 0.0) ** 2 + xp.maximum(rz, 0.0) ** 2)
                inside = xp.minimum(xp.maximum(rx, rz), 0.0)
                footprint = outside + inside
                return xp.maximum(sheet, footprint)

            if primitive == "plane":
                nx = self._node_scalar_param(node, "nx", 0.0)
                ny = self._node_scalar_param(node, "ny", 1.0)
                nz = self._node_scalar_param(node, "nz", 0.0)
                d = self._node_scalar_param(node, "d", 0.0)
                norm = np.sqrt(nx * nx + ny * ny + nz * nz)
                if norm < 1e-9:
                    raise EvaluationError("plane normal cannot be a zero vector")
                nx /= norm
                ny /= norm
                nz /= norm
                return nx * qx + ny * qy + nz * qz + d

            raise EvaluationError(f"Unknown primitive '{primitive}'")

        def eval_boolean(node: SceneNode, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            fields = [eval_node(child_id, qx, qy, qz) for child_id in node.inputs]
            if len(fields) < 2:
                raise EvaluationError(f"Boolean node '{node.id}' requires at least 2 inputs")

            op = node.op
            if op == "union":
                result = fields[0]
                for field in fields[1:]:
                    result = xp.minimum(result, field)
                return result
            if op == "intersection":
                result = fields[0]
                for field in fields[1:]:
                    result = xp.maximum(result, field)
                return result
            if op == "difference":
                result = fields[0]
                for field in fields[1:]:
                    result = xp.maximum(result, -field)
                return result
            if op == "smooth_union":
                k = self._node_scalar_param(node, "k", 0.2)
                result = fields[0]
                for field in fields[1:]:
                    result = _smooth_union(result, field, k, xp=xp)
                return result

            raise EvaluationError(f"Unknown boolean op '{op}'")

        def eval_transform(node: SceneNode, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            if len(node.inputs) != 1:
                raise EvaluationError(f"Transform node '{node.id}' must have exactly one input")

            tx, ty, tz, sx, sy, sz, inv_rotation = self._node_transform(node)
            local_x = qx - tx
            local_y = qy - ty
            local_z = qz - tz

            if inv_rotation is not None:
                rx = inv_rotation[0, 0] * local_x + inv_rotation[0, 1] * local_y + inv_rotation[0, 2] * local_z
                ry = inv_rotation[1, 0] * local_x + inv_rotation[1, 1] * local_y + inv_rotation[1, 2] * local_z
                rz = inv_rotation[2, 0] * local_x + inv_rotation[2, 1] * local_y + inv_rotation[2, 2] * local_z
            else:
                rx, ry, rz = local_x, local_y, local_z

            rx = rx / sx
            ry = ry / sy
            rz = rz / sz

            return eval_node(node.inputs[0], rx, ry, rz)

        def eval_domain_op(node: SceneNode, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            if len(node.inputs) != 1:
                raise EvaluationError(f"Domain op node '{node.id}' must have exactly one child")

            op = node.op

            if op == "repeat":
                pxp = self._node_scalar_param(node, "x", 1.0)
                pyp = self._node_scalar_param(node, "y", 1.0)
                pzp = self._node_scalar_param(node, "z", 1.0)
                rx = _repeat_local(qx, pxp)
                ry = _repeat_local(qy, pyp)
                rz = _repeat_local(qz, pzp)
                return eval_node(node.inputs[0], rx, ry, rz)

            if op == "twist":
                k = self._node_scalar_param(node, "k", 1.0)
                axis = self._node_string_param(node, "axis", "y").lower()
                if axis == "y":
                    angle = k * qy
                    ca = xp.cos(angle)
                    sa = xp.sin(angle)
                    rx = ca * qx - sa * qz
                    rz = sa * qx + ca * qz
                    return eval_node(node.inputs[0], rx, qy, rz)
                if axis == "x":
                    angle = k * qx
                    ca = xp.cos(angle)
                    sa = xp.sin(angle)
                    ry = ca * qy - sa * qz
                    rz = sa * qy + ca * qz
                    return eval_node(node.inputs[0], qx, ry, rz)
                angle = k * qz
                ca = xp.cos(angle)
                sa = xp.sin(angle)
                rx = ca * qx - sa * qy
                ry = sa * qx + ca * qy
                return eval_node(node.inputs[0], rx, ry, qz)

            if op == "bend":
                k = self._node_scalar_param(node, "k", 0.5)
                axis = self._node_string_param(node, "axis", "x").lower()
                if axis == "x":
                    angle = k * qx
                    ca = xp.cos(angle)
                    sa = xp.sin(angle)
                    ry = ca * qy - sa * qz
                    rz = sa * qy + ca * qz
                    return eval_node(node.inputs[0], qx, ry, rz)
                if axis == "y":
                    angle = k * qy
                    ca = xp.cos(angle)
                    sa = xp.sin(angle)
                    rx = ca * qx - sa * qz
                    rz = sa * qx + ca * qz
                    return eval_node(node.inputs[0], rx, qy, rz)
                angle = k * qz
                ca = xp.cos(angle)
                sa = xp.sin(angle)
                rx = ca * qx - sa * qy
                ry = sa * qx + ca * qy
                return eval_node(node.inputs[0], rx, ry, qz)

            if op == "shell":
                thickness = self._node_scalar_param(node, "t", 0.1)
                inner = eval_node(node.inputs[0], qx, qy, qz)
                return xp.abs(inner) - abs(thickness)

            if op == "offset":
                d = self._node_scalar_param(node, "d", 0.0)
                child = eval_node(node.inputs[0], qx, qy, qz)
                return child - d

            if op == "circular_array":
                raw_count = self._node_scalar_param(node, "count", 12.0)
                count = int(max(1, round(raw_count)))
                phase = self._node_scalar_param(node, "phase", 0.0)
                axis = self._node_string_param(node, "axis", "y").lower()

                if count == 1:
                    return eval_node(node.inputs[0], qx, qy, qz)

                step = (2.0 * np.pi) / float(count)
                half_step = 0.5 * step

                if axis == "x":
                    r = xp.sqrt(qy * qy + qz * qz)
                    theta = xp.arctan2(qz, qy)
                    folded = ((theta - phase + half_step) % step) - half_step + phase
                    ry = r * xp.cos(folded)
                    rz = r * xp.sin(folded)
                    return eval_node(node.inputs[0], qx, ry, rz)

                if axis == "z":
                    r = xp.sqrt(qx * qx + qy * qy)
                    theta = xp.arctan2(qy, qx)
                    folded = ((theta - phase + half_step) % step) - half_step + phase
                    rx = r * xp.cos(folded)
                    ry = r * xp.sin(folded)
                    return eval_node(node.inputs[0], rx, ry, qz)

                r = xp.sqrt(qx * qx + qz * qz)
                theta = xp.arctan2(qz, qx)
                folded = ((theta - phase + half_step) % step) - half_step + phase
                rx = r * xp.cos(folded)
                rz = r * xp.sin(folded)
                return eval_node(node.inputs[0], rx, qy, rz)

            raise EvaluationError(f"Unsupported domain op '{op}'")

        def eval_lattice(node: SceneNode, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            op = node.op

            if op in {"gyroid", "schwarz_p", "diamond"}:
                pitch = self._node_scalar_param(node, "pitch", 1.0)
                phase = self._node_scalar_param(node, "phase", 0.0)
                thickness = self._node_scalar_param(node, "thickness", 0.08)
                scale = 2.0 * np.pi / max(abs(pitch), 1e-6)
                u = qx * scale + phase
                v = qy * scale + phase
                w = qz * scale + phase

                if op == "gyroid":
                    base = xp.sin(u) * xp.cos(v) + xp.sin(v) * xp.cos(w) + xp.sin(w) * xp.cos(u)
                elif op == "schwarz_p":
                    base = xp.cos(u) + xp.cos(v) + xp.cos(w)
                else:
                    base = (
                        xp.sin(u) * xp.sin(v) * xp.sin(w)
                        + xp.sin(u) * xp.cos(v) * xp.cos(w)
                        + xp.cos(u) * xp.sin(v) * xp.cos(w)
                        + xp.cos(u) * xp.cos(v) * xp.sin(w)
                    )

                return xp.abs(base) - abs(thickness)

            if op == "strut_lattice":
                kind = self._node_string_param(node, "type", "bcc").lower()
                pitch = self._node_scalar_param(node, "pitch", 1.0)
                radius = self._node_scalar_param(node, "radius", 0.08)

                rx = _repeat_local(qx, pitch) / max(abs(pitch), 1e-6)
                ry = _repeat_local(qy, pitch) / max(abs(pitch), 1e-6)
                rz = _repeat_local(qz, pitch) / max(abs(pitch), 1e-6)

                dist = xp.full_like(rx, fill_value=xp.inf, dtype=self.eval_dtype)
                segments = _strut_segments(kind)
                if xp is np and NUMBA_AVAILABLE and _spline_min_dist_numba is not None:
                    segment_array = np.asarray(
                        [(a[0], a[1], a[2], b[0], b[1], b[2]) for a, b in segments],
                        dtype=self.eval_dtype,
                    )
                    dist = _spline_min_dist_numba(
                        np.asarray(rx, dtype=self.eval_dtype).reshape(-1),
                        np.asarray(ry, dtype=self.eval_dtype).reshape(-1),
                        np.asarray(rz, dtype=self.eval_dtype).reshape(-1),
                        segment_array,
                    ).reshape(rx.shape).astype(self.eval_dtype, copy=False)
                else:
                    for a, b in segments:
                        dseg = _dist_to_segment(rx, ry, rz, a, b, xp=xp)
                        dist = xp.minimum(dist, dseg)

                return dist * abs(pitch) - abs(radius)

            if op == "conformal_fill":
                if len(node.inputs) != 2:
                    raise EvaluationError("conformal_fill requires two child nodes: host and lattice")
                host = eval_node(node.inputs[0], qx, qy, qz)
                lattice = eval_node(node.inputs[1], qx, qy, qz)
                wall = self._node_scalar_param(node, "wall", 0.1)
                offset = self._node_scalar_param(node, "offset", 0.0)
                mode = self._node_string_param(node, "mode", "shell").lower()

                host_shifted = host + offset
                clipped = xp.maximum(lattice, host_shifted)
                shell_band = xp.maximum(lattice, xp.abs(host_shifted) - abs(wall))

                if mode == "clip":
                    return clipped
                if mode == "hybrid":
                    return xp.minimum(clipped, shell_band)
                return shell_band

            raise EvaluationError(f"Unsupported lattice operation '{op}'")

        def eval_turbomachine(node: SceneNode, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            op = node.op
            r = xp.sqrt(qx * qx + qz * qz)
            theta = xp.arctan2(qz, qx)

            if op in {"impeller_centrifugal", "radial_turbine"}:
                r_in = self._node_scalar_param(node, "r_in", 0.25)
                r_out = self._node_scalar_param(node, "r_out", 1.0)
                hub_h = self._node_scalar_param(node, "hub_h", 0.45)
                blade_count = int(max(1, round(self._node_scalar_param(node, "blade_count", 8.0))))
                blade_thickness = self._node_scalar_param(node, "blade_thickness", 0.08)
                blade_twist = self._node_scalar_param(node, "blade_twist", 0.8)

                if op == "impeller_centrifugal":
                    shroud_gap = self._node_scalar_param(node, "shroud_gap", 0.05)
                    blade_twist_dir = blade_twist
                else:
                    shroud_gap = self._node_scalar_param(node, "shroud_gap", 0.03)
                    blade_twist_dir = blade_twist

                radial_span = max(r_out - r_in, 1e-6)
                t = xp.clip((r - r_in) / radial_span, 0.0, 1.0)

                hub = xp.maximum(r - r_out, xp.abs(qy) - hub_h * 0.25)
                shroud = xp.maximum(xp.abs(r - r_out) - abs(shroud_gap), xp.abs(qy) - hub_h * 0.5)

                if xp is np and NUMBA_AVAILABLE and _blade_field_numba is not None:
                    blades = _blade_field_numba(
                        np.asarray(r, dtype=np.float32).reshape(-1),
                        np.asarray(theta, dtype=np.float32).reshape(-1),
                        np.asarray(qy, dtype=np.float32).reshape(-1),
                        float(r_in),
                        float(radial_span),
                        int(blade_count),
                        float(blade_thickness),
                        float(hub_h),
                        float(blade_twist_dir),
                    ).reshape(r.shape).astype(self.eval_dtype, copy=False)
                else:
                    blades = xp.full_like(r, xp.inf, dtype=self.eval_dtype)
                    for idx in range(blade_count):
                        base_angle = 2.0 * np.pi * idx / float(blade_count)
                        target = base_angle + blade_twist_dir * t
                        ang = _angle_wrap(theta - target)
                        arc_dist = xp.abs(ang) * xp.maximum(r, 1e-6)
                        blade_field = xp.maximum(arc_dist - abs(blade_thickness) * 0.5, xp.abs(qy) - hub_h * 0.5)
                        blades = xp.minimum(blades, blade_field)

                solid = xp.minimum(xp.minimum(hub, shroud), blades)
                bore = r_in - r
                return xp.maximum(solid, bore)

            if op == "volute_casing":
                throat_radius = self._node_scalar_param(node, "throat_radius", 0.35)
                outlet_radius = self._node_scalar_param(node, "outlet_radius", 1.4)
                area_growth = self._node_scalar_param(node, "area_growth", 0.9)
                width = self._node_scalar_param(node, "width", 0.6)
                wall = self._node_scalar_param(node, "wall", 0.08)
                tongue_clearance = self._node_scalar_param(node, "tongue_clearance", 0.06)

                theta01 = (theta + np.pi) / (2.0 * np.pi)
                centerline = throat_radius + (outlet_radius - throat_radius) * theta01
                tube_r = 0.12 + area_growth * 0.25 * theta01

                radial = xp.abs(r - centerline) - tube_r
                height = xp.abs(qy) - width * 0.5
                channel = xp.maximum(radial, height)
                wall_field = xp.abs(channel) - abs(wall)

                # Tongue shaping near positive-x side.
                tongue_plane = qx - (throat_radius + tongue_clearance)
                tongue_slot = xp.maximum(xp.abs(qz) - tongue_clearance, tongue_plane)
                return xp.maximum(wall_field, -tongue_slot)

            raise EvaluationError(f"Unsupported turbomachine op '{op}'")

        return xp.asarray(eval_node(self.scene_ir.root_node_id, px, py, pz), dtype=self.eval_dtype)

    def _resolve_bounds_hint(
        self, node: SceneNode
    ) -> tuple[float, float, float, float, float, float] | None:
        if node.bounds_hint is None or len(node.bounds_hint) != 3:
            return None
        try:
            xmin = _resolve_scalar(node.bounds_hint[0][0], self.parameter_values)
            xmax = _resolve_scalar(node.bounds_hint[0][1], self.parameter_values)
            ymin = _resolve_scalar(node.bounds_hint[1][0], self.parameter_values)
            ymax = _resolve_scalar(node.bounds_hint[1][1], self.parameter_values)
            zmin = _resolve_scalar(node.bounds_hint[2][0], self.parameter_values)
            zmax = _resolve_scalar(node.bounds_hint[2][1], self.parameter_values)
        except Exception:
            return None

        if xmin >= xmax or ymin >= ymax or zmin >= zmax:
            return None
        return (xmin, xmax, ymin, ymax, zmin, zmax)



def merge_parameter_values(scene_ir: SceneIR, overrides: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {spec.name: spec.default for spec in scene_ir.parameter_schema}
    for key, value in overrides.items():
        merged[key] = float(value)
    return merged



def evaluate_scene_field(
    scene_ir: SceneIR,
    parameter_values: dict[str, float],
    grid: GridConfig,
    compute_precision: ComputePrecision = "float32",
    compute_backend: ComputeBackend = "auto",
) -> np.ndarray:
    field, _ = evaluate_scene_field_with_backend(
        scene_ir,
        parameter_values,
        grid,
        compute_precision=compute_precision,
        compute_backend=compute_backend,
    )
    return field


def evaluate_scene_field_with_backend(
    scene_ir: SceneIR,
    parameter_values: dict[str, float],
    grid: GridConfig,
    compute_precision: ComputePrecision = "float32",
    compute_backend: ComputeBackend = "auto",
) -> tuple[np.ndarray, Literal["cpu", "cuda"]]:
    resolved_backend = _resolve_compute_backend(compute_backend)
    if resolved_backend == "cuda":
        try:
            return (
                _evaluate_scene_field_cuda(scene_ir, parameter_values, grid, compute_precision=compute_precision),
                "cuda",
            )
        except Exception:
            return (
                _evaluate_scene_field_cpu(scene_ir, parameter_values, grid, compute_precision=compute_precision),
                "cpu",
            )

    return (
        _evaluate_scene_field_cpu(scene_ir, parameter_values, grid, compute_precision=compute_precision),
        "cpu",
    )



def evaluate_scene_field_chunked(
    scene_ir: SceneIR,
    parameter_values: dict[str, float],
    grid: GridConfig,
    chunk_size: int = 72,
    compute_precision: ComputePrecision = "float32",
) -> np.ndarray:
    return _evaluate_scene_field_cpu(
        scene_ir,
        parameter_values,
        grid,
        chunk_size=chunk_size,
        compute_precision=compute_precision,
    )


def _evaluate_scene_field_cpu(
    scene_ir: SceneIR,
    parameter_values: dict[str, float],
    grid: GridConfig,
    chunk_size: int = 72,
    compute_precision: ComputePrecision = "float32",
) -> np.ndarray:
    if scene_ir.root_node_id not in {node.id for node in scene_ir.nodes}:
        raise EvaluationError("Scene root node is not present in nodes list")

    bounds = grid.bounds
    eval_dtype = _resolve_eval_dtype(compute_precision)
    bounds_key = (
        float(bounds[0][0]),
        float(bounds[0][1]),
        float(bounds[1][0]),
        float(bounds[1][1]),
        float(bounds[2][0]),
        float(bounds[2][1]),
    )

    if grid.resolution <= 160:
        x, y, z = _cached_grid(bounds_key, grid.resolution, eval_dtype.name)
        runtime = _FieldRuntime(scene_ir, parameter_values, eval_dtype=eval_dtype, xp=np)
        return runtime.evaluate(x, y, z)

    x_axis, y_axis, z_axis = _cached_axes(bounds_key, grid.resolution, eval_dtype.name)

    runtime = _FieldRuntime(scene_ir, parameter_values, eval_dtype=eval_dtype, xp=np)
    field = np.empty((grid.resolution, grid.resolution, grid.resolution), dtype=eval_dtype)

    for start in range(0, grid.resolution, chunk_size):
        stop = min(grid.resolution, start + chunk_size)
        x = x_axis[start:stop]
        px, py, pz = np.meshgrid(x, y_axis, z_axis, indexing="ij")
        field[start:stop, :, :] = runtime.evaluate(px, py, pz)

    return field


def _evaluate_scene_field_cuda(
    scene_ir: SceneIR,
    parameter_values: dict[str, float],
    grid: GridConfig,
    chunk_size: int = 72,
    compute_precision: ComputePrecision = "float32",
) -> np.ndarray:
    if not CUPY_AVAILABLE or cp is None:
        raise EvaluationError("CUDA evaluation requested but CuPy is not available")

    if scene_ir.root_node_id not in {node.id for node in scene_ir.nodes}:
        raise EvaluationError("Scene root node is not present in nodes list")

    bounds = grid.bounds
    eval_dtype = _resolve_eval_dtype(compute_precision)
    bounds_key = (
        float(bounds[0][0]),
        float(bounds[0][1]),
        float(bounds[1][0]),
        float(bounds[1][1]),
        float(bounds[2][0]),
        float(bounds[2][1]),
    )

    runtime = _FieldRuntime(scene_ir, parameter_values, eval_dtype=eval_dtype, xp=cp)

    if grid.resolution <= 160:
        x, y, z = _cached_grid(bounds_key, grid.resolution, eval_dtype.name)
        field_gpu = runtime.evaluate(cp.asarray(x), cp.asarray(y), cp.asarray(z))
        return cp.asnumpy(field_gpu)

    x_axis, y_axis, z_axis = _cached_axes(bounds_key, grid.resolution, eval_dtype.name)
    y_gpu = cp.asarray(y_axis)
    z_gpu = cp.asarray(z_axis)
    field = np.empty((grid.resolution, grid.resolution, grid.resolution), dtype=eval_dtype)

    for start in range(0, grid.resolution, chunk_size):
        stop = min(grid.resolution, start + chunk_size)
        x_gpu = cp.asarray(x_axis[start:stop])
        px, py, pz = cp.meshgrid(x_gpu, y_gpu, z_gpu, indexing="ij")
        field[start:stop, :, :] = cp.asnumpy(runtime.evaluate(px, py, pz))

    return field



def ensure_scene_valid(scene_ir: SceneIR) -> None:
    node_ids = {node.id for node in scene_ir.nodes}
    if scene_ir.root_node_id not in node_ids:
        raise DslError("root_node_id is not present in nodes")
    for node in scene_ir.nodes:
        for child_id in node.inputs:
            if child_id not in node_ids:
                raise DslError(f"Node '{node.id}' refers to missing child '{child_id}'")
