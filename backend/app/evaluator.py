from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

from .dsl import DslError
from .models import GridConfig, SceneIR, SceneNode

try:
    from numba import vectorize

    @vectorize(["float32(float32,float32,float32)", "float64(float64,float64,float64)"], nopython=True, cache=True)
    def _smin_numba(a: float, b: float, k: float) -> float:
        h = 0.5 + 0.5 * (b - a) / k
        if h < 0.0:
            h = 0.0
        elif h > 1.0:
            h = 1.0
        return (b * (1.0 - h) + a * h) - k * h * (1.0 - h)

    NUMBA_AVAILABLE = True
except Exception:
    _smin_numba = None
    NUMBA_AVAILABLE = False


class EvaluationError(ValueError):
    pass


LEAF_TYPES_WITH_AABB_SKIP = {"primitive", "field_expr", "turbomachine"}


@lru_cache(maxsize=48)
def _cached_grid(bounds_key: tuple[float, ...], resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds_key
    x = np.linspace(xmin, xmax, resolution, dtype=np.float64)
    y = np.linspace(ymin, ymax, resolution, dtype=np.float64)
    z = np.linspace(zmin, zmax, resolution, dtype=np.float64)
    return np.meshgrid(x, y, z, indexing="ij")


@lru_cache(maxsize=48)
def _cached_axes(bounds_key: tuple[float, ...], resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds_key
    x = np.linspace(xmin, xmax, resolution, dtype=np.float64)
    y = np.linspace(ymin, ymax, resolution, dtype=np.float64)
    z = np.linspace(zmin, zmax, resolution, dtype=np.float64)
    return x, y, z


def _rotation_matrix_xyz(deg: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.array(deg, dtype=np.float64))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    mx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    my = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    mz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
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


def _smooth_union(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    if k <= 0:
        return np.minimum(a, b)
    if NUMBA_AVAILABLE and _smin_numba is not None:
        return _smin_numba(a, b, np.array(k, dtype=a.dtype))
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return (b * (1.0 - h) + a * h) - k * h * (1.0 - h)


def _distance_to_aabb(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    dx = np.maximum(np.maximum(xmin - px, 0.0), px - xmax)
    dy = np.maximum(np.maximum(ymin - py, 0.0), py - ymax)
    dz = np.maximum(np.maximum(zmin - pz, 0.0), pz - zmax)
    outside = np.sqrt(dx * dx + dy * dy + dz * dz)

    inside = np.minimum.reduce([
        px - xmin,
        xmax - px,
        py - ymin,
        ymax - py,
        pz - zmin,
        zmax - pz,
    ])
    return outside - np.maximum(inside, 0.0)


def _dist_to_segment(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> np.ndarray:
    ax, ay, az = a
    bx, by, bz = b
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    denom = abx * abx + aby * aby + abz * abz
    if denom < 1e-12:
        return np.sqrt((px - ax) ** 2 + (py - ay) ** 2 + (pz - az) ** 2)

    apx = px - ax
    apy = py - ay
    apz = pz - az
    t = (apx * abx + apy * aby + apz * abz) / denom
    t = np.clip(t, 0.0, 1.0)

    cx = ax + t * abx
    cy = ay + t * aby
    cz = az + t * abz
    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)


def _repeat_local(coord: np.ndarray, pitch: float) -> np.ndarray:
    if pitch <= 1e-6:
        return coord
    return ((coord + pitch * 0.5) % pitch) - pitch * 0.5


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
    def __init__(self, scene_ir: SceneIR, parameter_values: dict[str, float]) -> None:
        self.scene_ir = scene_ir
        self.parameter_values = parameter_values
        self.node_lookup: dict[str, SceneNode] = {node.id: node for node in scene_ir.nodes}

    def evaluate(self, px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
        memo: dict[tuple[str, int, int, int], np.ndarray] = {}

        def eval_node(node_id: str, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            key = (node_id, id(qx), id(qy), id(qz))
            if key in memo:
                return memo[key]

            node = self.node_lookup.get(node_id)
            if node is None:
                raise EvaluationError(f"Node '{node_id}' is not defined")

            bounds_hint = self._resolve_bounds_hint(node)
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

                if not np.any(mask):
                    out = _distance_to_aabb(qx, qy, qz, bounds_hint)
                    memo[key] = out
                    return out

                if np.all(mask):
                    out = eval_node_core(node, qx, qy, qz)
                    memo[key] = out
                    return out

                out = _distance_to_aabb(qx, qy, qz, bounds_hint)
                sub = eval_node_core(node, qx[mask], qy[mask], qz[mask])
                out = out.astype(np.float64, copy=True)
                out[mask] = sub
                memo[key] = out
                return out

            out = eval_node_core(node, qx, qy, qz)
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
                    return left / np.where(np.abs(right) < 1e-12, 1e-12, right)
                if op == "^":
                    return np.power(left, right)
                raise EvaluationError(f"Unsupported binary op '{op}'")
            if kind == "func":
                name = expr.get("name")
                args = [eval_expr(arg, qx, qy, qz) for arg in expr.get("args", [])]
                if name == "sin":
                    if len(args) != 1:
                        raise EvaluationError("sin expects exactly 1 argument")
                    return np.sin(args[0])
                if name == "cos":
                    if len(args) != 1:
                        raise EvaluationError("cos expects exactly 1 argument")
                    return np.cos(args[0])
                if name == "tan":
                    if len(args) != 1:
                        raise EvaluationError("tan expects exactly 1 argument")
                    return np.tan(args[0])
                if name == "abs":
                    if len(args) != 1:
                        raise EvaluationError("abs expects exactly 1 argument")
                    return np.abs(args[0])
                if name == "sqrt":
                    if len(args) != 1:
                        raise EvaluationError("sqrt expects exactly 1 argument")
                    return np.sqrt(np.maximum(args[0], 0.0))
                if name == "exp":
                    if len(args) != 1:
                        raise EvaluationError("exp expects exactly 1 argument")
                    return np.exp(args[0])
                if name == "log":
                    if len(args) != 1:
                        raise EvaluationError("log expects exactly 1 argument")
                    return np.log(np.maximum(args[0], 1e-12))
                if name == "min":
                    if not args:
                        raise EvaluationError("min expects at least 1 argument")
                    out = args[0]
                    for arg in args[1:]:
                        out = np.minimum(out, arg)
                    return out
                if name == "max":
                    if not args:
                        raise EvaluationError("max expects at least 1 argument")
                    out = args[0]
                    for arg in args[1:]:
                        out = np.maximum(out, arg)
                    return out
                if name == "clamp":
                    if len(args) != 3:
                        raise EvaluationError("clamp expects exactly 3 arguments")
                    return np.clip(args[0], args[1], args[2])
                raise EvaluationError(f"Unsupported function '{name}'")

            raise EvaluationError(f"Unsupported expression kind '{kind}'")

        def eval_field_expr(node: SceneNode, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            if node.expr is None:
                raise EvaluationError(f"field_expr node '{node.id}' missing expression payload")
            out = eval_expr(node.expr, qx, qy, qz)
            arr = np.asarray(out, dtype=np.float64)
            if arr.shape == qx.shape:
                return arr
            try:
                return np.broadcast_to(arr, qx.shape).astype(np.float64, copy=False)
            except ValueError as exc:
                raise EvaluationError(
                    f"field_expr node '{node.id}' produced non-broadcastable shape {arr.shape}"
                ) from exc

        def eval_primitive(node: SceneNode, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            primitive = node.primitive
            params = node.params
            if primitive == "sphere":
                r = _resolve_scalar(params.get("r", 1.0), self.parameter_values)
                return np.sqrt(qx * qx + qy * qy + qz * qz) - r

            if primitive == "box":
                bx = _resolve_scalar(params.get("x", 0.5), self.parameter_values)
                by = _resolve_scalar(params.get("y", 0.5), self.parameter_values)
                bz = _resolve_scalar(params.get("z", 0.5), self.parameter_values)
                qmx = np.abs(qx) - bx
                qmy = np.abs(qy) - by
                qmz = np.abs(qz) - bz
                ox = np.maximum(qmx, 0.0)
                oy = np.maximum(qmy, 0.0)
                oz = np.maximum(qmz, 0.0)
                outside = np.sqrt(ox * ox + oy * oy + oz * oz)
                inside = np.minimum(np.maximum.reduce([qmx, qmy, qmz]), 0.0)
                return outside + inside

            if primitive == "cylinder":
                r = _resolve_scalar(params.get("r", 0.4), self.parameter_values)
                h = _resolve_scalar(params.get("h", 1.0), self.parameter_values)
                d1 = np.sqrt(qx * qx + qz * qz) - r
                d2 = np.abs(qy) - h * 0.5
                outside = np.sqrt(np.maximum(d1, 0.0) ** 2 + np.maximum(d2, 0.0) ** 2)
                inside = np.minimum(np.maximum(d1, d2), 0.0)
                return outside + inside

            if primitive == "torus":
                major_r = _resolve_scalar(params.get("R", 0.8), self.parameter_values)
                minor_r = _resolve_scalar(params.get("r", 0.2), self.parameter_values)
                q = np.sqrt(qx * qx + qz * qz) - major_r
                return np.sqrt(q * q + qy * qy) - minor_r

            if primitive == "plane":
                nx = _resolve_scalar(params.get("nx", 0.0), self.parameter_values)
                ny = _resolve_scalar(params.get("ny", 1.0), self.parameter_values)
                nz = _resolve_scalar(params.get("nz", 0.0), self.parameter_values)
                d = _resolve_scalar(params.get("d", 0.0), self.parameter_values)
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
                return np.minimum.reduce(fields)
            if op == "intersection":
                return np.maximum.reduce(fields)
            if op == "difference":
                result = fields[0]
                for field in fields[1:]:
                    result = np.maximum(result, -field)
                return result
            if op == "smooth_union":
                k = _resolve_scalar(node.params.get("k", 0.2), self.parameter_values)
                result = fields[0]
                for field in fields[1:]:
                    result = _smooth_union(result, field, k)
                return result

            raise EvaluationError(f"Unknown boolean op '{op}'")

        def eval_transform(node: SceneNode, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            if len(node.inputs) != 1:
                raise EvaluationError(f"Transform node '{node.id}' must have exactly one input")

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

            tx, ty, tz = translate
            local_x = qx - tx
            local_y = qy - ty
            local_z = qz - tz

            if any(abs(v) > 1e-9 for v in rotate):
                rotation = _rotation_matrix_xyz(rotate)
                inv_rotation = rotation.T
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
            params = node.params

            if op == "repeat":
                pxp = _resolve_scalar(params.get("x", 1.0), self.parameter_values)
                pyp = _resolve_scalar(params.get("y", 1.0), self.parameter_values)
                pzp = _resolve_scalar(params.get("z", 1.0), self.parameter_values)
                rx = _repeat_local(qx, pxp)
                ry = _repeat_local(qy, pyp)
                rz = _repeat_local(qz, pzp)
                return eval_node(node.inputs[0], rx, ry, rz)

            if op == "twist":
                k = _resolve_scalar(params.get("k", 1.0), self.parameter_values)
                axis = _resolve_string(params.get("axis", "y"), "y").lower()
                if axis == "y":
                    angle = k * qy
                    ca = np.cos(angle)
                    sa = np.sin(angle)
                    rx = ca * qx - sa * qz
                    rz = sa * qx + ca * qz
                    return eval_node(node.inputs[0], rx, qy, rz)
                if axis == "x":
                    angle = k * qx
                    ca = np.cos(angle)
                    sa = np.sin(angle)
                    ry = ca * qy - sa * qz
                    rz = sa * qy + ca * qz
                    return eval_node(node.inputs[0], qx, ry, rz)
                angle = k * qz
                ca = np.cos(angle)
                sa = np.sin(angle)
                rx = ca * qx - sa * qy
                ry = sa * qx + ca * qy
                return eval_node(node.inputs[0], rx, ry, qz)

            if op == "bend":
                k = _resolve_scalar(params.get("k", 0.5), self.parameter_values)
                axis = _resolve_string(params.get("axis", "x"), "x").lower()
                if axis == "x":
                    angle = k * qx
                    ca = np.cos(angle)
                    sa = np.sin(angle)
                    ry = ca * qy - sa * qz
                    rz = sa * qy + ca * qz
                    return eval_node(node.inputs[0], qx, ry, rz)
                if axis == "y":
                    angle = k * qy
                    ca = np.cos(angle)
                    sa = np.sin(angle)
                    rx = ca * qx - sa * qz
                    rz = sa * qx + ca * qz
                    return eval_node(node.inputs[0], rx, qy, rz)
                angle = k * qz
                ca = np.cos(angle)
                sa = np.sin(angle)
                rx = ca * qx - sa * qy
                ry = sa * qx + ca * qy
                return eval_node(node.inputs[0], rx, ry, qz)

            if op == "shell":
                thickness = _resolve_scalar(params.get("t", 0.1), self.parameter_values)
                inner = eval_node(node.inputs[0], qx, qy, qz)
                return np.abs(inner) - abs(thickness)

            if op == "offset":
                d = _resolve_scalar(params.get("d", 0.0), self.parameter_values)
                child = eval_node(node.inputs[0], qx, qy, qz)
                return child - d

            if op == "circular_array":
                raw_count = _resolve_scalar(params.get("count", 12.0), self.parameter_values)
                count = int(max(1, round(raw_count)))
                phase = _resolve_scalar(params.get("phase", 0.0), self.parameter_values)
                axis = _resolve_string(params.get("axis", "y"), "y").lower()

                if count == 1:
                    return eval_node(node.inputs[0], qx, qy, qz)

                result = np.full_like(qx, np.inf, dtype=np.float64)
                step = (2.0 * np.pi) / float(count)

                for idx in range(count):
                    angle = phase + step * idx
                    ca = math.cos(angle)
                    sa = math.sin(angle)

                    if axis == "x":
                        ry = ca * qy - sa * qz
                        rz = sa * qy + ca * qz
                        child = eval_node(node.inputs[0], qx, ry, rz)
                    elif axis == "z":
                        rx = ca * qx - sa * qy
                        ry = sa * qx + ca * qy
                        child = eval_node(node.inputs[0], rx, ry, qz)
                    else:
                        rx = ca * qx - sa * qz
                        rz = sa * qx + ca * qz
                        child = eval_node(node.inputs[0], rx, qy, rz)

                    result = np.minimum(result, child)
                return result

            raise EvaluationError(f"Unsupported domain op '{op}'")

        def eval_lattice(node: SceneNode, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            op = node.op
            params = node.params

            if op in {"gyroid", "schwarz_p", "diamond"}:
                pitch = _resolve_scalar(params.get("pitch", 1.0), self.parameter_values)
                phase = _resolve_scalar(params.get("phase", 0.0), self.parameter_values)
                thickness = _resolve_scalar(params.get("thickness", 0.08), self.parameter_values)
                scale = 2.0 * np.pi / max(abs(pitch), 1e-6)
                u = qx * scale + phase
                v = qy * scale + phase
                w = qz * scale + phase

                if op == "gyroid":
                    base = np.sin(u) * np.cos(v) + np.sin(v) * np.cos(w) + np.sin(w) * np.cos(u)
                elif op == "schwarz_p":
                    base = np.cos(u) + np.cos(v) + np.cos(w)
                else:
                    base = (
                        np.sin(u) * np.sin(v) * np.sin(w)
                        + np.sin(u) * np.cos(v) * np.cos(w)
                        + np.cos(u) * np.sin(v) * np.cos(w)
                        + np.cos(u) * np.cos(v) * np.sin(w)
                    )

                return np.abs(base) - abs(thickness)

            if op == "strut_lattice":
                kind = _resolve_string(params.get("type", "bcc"), "bcc").lower()
                pitch = _resolve_scalar(params.get("pitch", 1.0), self.parameter_values)
                radius = _resolve_scalar(params.get("radius", 0.08), self.parameter_values)

                rx = _repeat_local(qx, pitch) / max(abs(pitch), 1e-6)
                ry = _repeat_local(qy, pitch) / max(abs(pitch), 1e-6)
                rz = _repeat_local(qz, pitch) / max(abs(pitch), 1e-6)

                dist = np.full_like(rx, fill_value=np.inf, dtype=np.float64)
                segments = _strut_segments(kind)
                for a, b in segments:
                    dseg = _dist_to_segment(rx, ry, rz, a, b)
                    dist = np.minimum(dist, dseg)

                return dist * abs(pitch) - abs(radius)

            if op == "conformal_fill":
                if len(node.inputs) != 2:
                    raise EvaluationError("conformal_fill requires two child nodes: host and lattice")
                host = eval_node(node.inputs[0], qx, qy, qz)
                lattice = eval_node(node.inputs[1], qx, qy, qz)
                wall = _resolve_scalar(params.get("wall", 0.1), self.parameter_values)
                offset = _resolve_scalar(params.get("offset", 0.0), self.parameter_values)
                mode = _resolve_string(params.get("mode", "shell"), "shell").lower()

                host_shifted = host + offset
                clipped = np.maximum(lattice, host_shifted)
                shell_band = np.maximum(lattice, np.abs(host_shifted) - abs(wall))

                if mode == "clip":
                    return clipped
                if mode == "hybrid":
                    return np.minimum(clipped, shell_band)
                return shell_band

            raise EvaluationError(f"Unsupported lattice operation '{op}'")

        def eval_turbomachine(node: SceneNode, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
            op = node.op
            params = node.params
            r = np.sqrt(qx * qx + qz * qz)
            theta = np.arctan2(qz, qx)

            if op in {"impeller_centrifugal", "radial_turbine"}:
                r_in = _resolve_scalar(params.get("r_in", 0.25), self.parameter_values)
                r_out = _resolve_scalar(params.get("r_out", 1.0), self.parameter_values)
                hub_h = _resolve_scalar(params.get("hub_h", 0.45), self.parameter_values)
                blade_count = int(max(1, round(_resolve_scalar(params.get("blade_count", 8.0), self.parameter_values))))
                blade_thickness = _resolve_scalar(params.get("blade_thickness", 0.08), self.parameter_values)
                blade_twist = _resolve_scalar(params.get("blade_twist", 0.8), self.parameter_values)

                if op == "impeller_centrifugal":
                    shroud_gap = _resolve_scalar(params.get("shroud_gap", 0.05), self.parameter_values)
                    blade_twist_dir = blade_twist
                else:
                    shroud_gap = _resolve_scalar(params.get("shroud_gap", 0.03), self.parameter_values)
                    blade_twist_dir = blade_twist

                radial_span = max(r_out - r_in, 1e-6)
                t = np.clip((r - r_in) / radial_span, 0.0, 1.0)

                hub = np.maximum(r - r_out, np.abs(qy) - hub_h * 0.25)
                shroud = np.maximum(np.abs(r - r_out) - abs(shroud_gap), np.abs(qy) - hub_h * 0.5)

                blades = np.full_like(r, np.inf, dtype=np.float64)
                for idx in range(blade_count):
                    base_angle = 2.0 * np.pi * idx / float(blade_count)
                    target = base_angle + blade_twist_dir * t
                    ang = _angle_wrap(theta - target)
                    arc_dist = np.abs(ang) * np.maximum(r, 1e-6)
                    blade_field = np.maximum(arc_dist - abs(blade_thickness) * 0.5, np.abs(qy) - hub_h * 0.5)
                    blades = np.minimum(blades, blade_field)

                solid = np.minimum(np.minimum(hub, shroud), blades)
                bore = r_in - r
                return np.maximum(solid, bore)

            if op == "volute_casing":
                throat_radius = _resolve_scalar(params.get("throat_radius", 0.35), self.parameter_values)
                outlet_radius = _resolve_scalar(params.get("outlet_radius", 1.4), self.parameter_values)
                area_growth = _resolve_scalar(params.get("area_growth", 0.9), self.parameter_values)
                width = _resolve_scalar(params.get("width", 0.6), self.parameter_values)
                wall = _resolve_scalar(params.get("wall", 0.08), self.parameter_values)
                tongue_clearance = _resolve_scalar(params.get("tongue_clearance", 0.06), self.parameter_values)

                theta01 = (theta + np.pi) / (2.0 * np.pi)
                centerline = throat_radius + (outlet_radius - throat_radius) * theta01
                tube_r = 0.12 + area_growth * 0.25 * theta01

                radial = np.abs(r - centerline) - tube_r
                height = np.abs(qy) - width * 0.5
                channel = np.maximum(radial, height)
                wall_field = np.abs(channel) - abs(wall)

                # Tongue shaping near positive-x side.
                tongue_plane = qx - (throat_radius + tongue_clearance)
                tongue_slot = np.maximum(np.abs(qz) - tongue_clearance, tongue_plane)
                return np.maximum(wall_field, -tongue_slot)

            raise EvaluationError(f"Unsupported turbomachine op '{op}'")

        return eval_node(self.scene_ir.root_node_id, px, py, pz)

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



def evaluate_scene_field(scene_ir: SceneIR, parameter_values: dict[str, float], grid: GridConfig) -> np.ndarray:
    if scene_ir.root_node_id not in {node.id for node in scene_ir.nodes}:
        raise EvaluationError("Scene root node is not present in nodes list")

    bounds = grid.bounds
    bounds_key = (
        float(bounds[0][0]),
        float(bounds[0][1]),
        float(bounds[1][0]),
        float(bounds[1][1]),
        float(bounds[2][0]),
        float(bounds[2][1]),
    )

    if grid.resolution > 160:
        return evaluate_scene_field_chunked(scene_ir, parameter_values, grid)

    x, y, z = _cached_grid(bounds_key, grid.resolution)
    runtime = _FieldRuntime(scene_ir, parameter_values)
    return runtime.evaluate(x, y, z)



def evaluate_scene_field_chunked(
    scene_ir: SceneIR,
    parameter_values: dict[str, float],
    grid: GridConfig,
    chunk_size: int = 72,
) -> np.ndarray:
    bounds = grid.bounds
    bounds_key = (
        float(bounds[0][0]),
        float(bounds[0][1]),
        float(bounds[1][0]),
        float(bounds[1][1]),
        float(bounds[2][0]),
        float(bounds[2][1]),
    )
    x_axis, y_axis, z_axis = _cached_axes(bounds_key, grid.resolution)

    runtime = _FieldRuntime(scene_ir, parameter_values)
    field = np.empty((grid.resolution, grid.resolution, grid.resolution), dtype=np.float64)

    for start in range(0, grid.resolution, chunk_size):
        stop = min(grid.resolution, start + chunk_size)
        x = x_axis[start:stop]
        px, py, pz = np.meshgrid(x, y_axis, z_axis, indexing="ij")
        field[start:stop, :, :] = runtime.evaluate(px, py, pz)

    return field



def ensure_scene_valid(scene_ir: SceneIR) -> None:
    node_ids = {node.id for node in scene_ir.nodes}
    if scene_ir.root_node_id not in node_ids:
        raise DslError("root_node_id is not present in nodes")
    for node in scene_ir.nodes:
        for child_id in node.inputs:
            if child_id not in node_ids:
                raise DslError(f"Node '{node.id}' refers to missing child '{child_id}'")
