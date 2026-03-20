from __future__ import annotations

import base64
import math
import time
from typing import Any

import numpy as np

from .models import GridConfig, QualityProfile, SceneIR, SceneNode, SceneProgramPayload


QUALITY_TO_RAYMARCH = {
    "interactive": {"max_steps": 144, "hit_eps": 0.0025, "normal_eps": 0.0035},
    "medium": {"max_steps": 208, "hit_eps": 0.0018, "normal_eps": 0.0026},
    "high": {"max_steps": 288, "hit_eps": 0.0013, "normal_eps": 0.0020},
    "ultra": {"max_steps": 384, "hit_eps": 0.0010, "normal_eps": 0.0015},
}

SUPPORTED_PRIMITIVES = {"sphere", "box", "cylinder", "torus", "plane"}
SUPPORTED_BOOLEANS = {"union", "intersection", "difference", "smooth_union"}
SUPPORTED_DOMAIN_OPS = {"repeat", "twist", "bend", "shell", "offset"}
SUPPORTED_LATTICE_OPS = {"gyroid", "schwarz_p", "diamond", "conformal_fill"}


def _quality_budget(quality: QualityProfile) -> dict[str, float | int]:
    return QUALITY_TO_RAYMARCH.get(quality, QUALITY_TO_RAYMARCH["high"])


def _encode_f32(values: np.ndarray) -> str:
    payload = np.asarray(values, dtype=np.float32).tobytes(order="C")
    return base64.b64encode(payload).decode("ascii")


def _resolve_scalar(value: Any, parameter_values: dict[str, float], default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and "$param" in value:
        name = value["$param"]
        if name in parameter_values:
            return float(parameter_values[name])
        return float(default)
    return float(default)


def _resolve_vec3(
    values: list[Any] | None,
    parameter_values: dict[str, float],
    fallback: tuple[float, float, float],
) -> tuple[float, float, float]:
    if not isinstance(values, list) or len(values) != 3:
        return fallback
    return (
        _resolve_scalar(values[0], parameter_values, fallback[0]),
        _resolve_scalar(values[1], parameter_values, fallback[1]),
        _resolve_scalar(values[2], parameter_values, fallback[2]),
    )


def _resolve_str(value: Any, default: str) -> str:
    if isinstance(value, str):
        return value
    return default


def _rotation_matrix_xyz(deg: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.array(deg, dtype=np.float64))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    my = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return mz @ my @ mx


def _fmt(value: float) -> str:
    s = f"{float(value):.9g}"
    # GLSL requires float literals to contain a decimal point or exponent.
    # "3" is an integer literal; "3.0" or "3e0" is a float literal.
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s


def _mat3_literal(mat: np.ndarray) -> str:
    vals = [float(v) for v in mat.reshape(-1).tolist()]
    return (
        "mat3("
        + ", ".join(_fmt(v) for v in vals[:3])
        + ", "
        + ", ".join(_fmt(v) for v in vals[3:6])
        + ", "
        + ", ".join(_fmt(v) for v in vals[6:9])
        + ")"
    )


def _safe_ident(node_id: str) -> str:
    out = []
    for ch in node_id:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    ident = "".join(out)
    if not ident:
        ident = "n"
    if ident[0].isdigit():
        ident = f"n_{ident}"
    return ident


def _collect_node_refs(expr: dict[str, Any]) -> list[str]:
    """Recursively collect all node_ref IDs from an expression tree."""
    refs: list[str] = []
    kind = expr.get("kind")
    if kind == "node_ref":
        ref = expr.get("id")
        if isinstance(ref, str):
            refs.append(ref)
    elif kind == "unary":
        refs.extend(_collect_node_refs(expr.get("arg", {})))
    elif kind == "binary":
        refs.extend(_collect_node_refs(expr.get("left", {})))
        refs.extend(_collect_node_refs(expr.get("right", {})))
    elif kind == "func":
        for arg in expr.get("args", []):
            refs.extend(_collect_node_refs(arg))
    return refs


def _compile_expr(
    expr: dict[str, Any],
    node_vars: dict[str, str],
    parameter_values: dict[str, float],
) -> tuple[str, list[str]]:
    kind = expr.get("kind")
    unsupported: list[str] = []

    if kind == "number":
        return _fmt(float(expr.get("value", 0.0))), unsupported
    if kind == "param":
        name = expr.get("name")
        return _fmt(float(parameter_values.get(name, 0.0))), unsupported
    if kind == "var":
        name = expr.get("name")
        if name == "x":
            return "p.x", unsupported
        if name == "y":
            return "p.y", unsupported
        if name == "z":
            return "p.z", unsupported
        unsupported.append(f"field_expr var '{name}'")
        return "0.0", unsupported
    if kind == "node_ref":
        ref = expr.get("id")
        if isinstance(ref, str) and ref in node_vars:
            # node_vars now stores function names; call as a function
            return f"{node_vars[ref]}(p)", unsupported
        unsupported.append("field_expr node_ref")
        return "0.0", unsupported
    if kind == "unary":
        op = expr.get("op")
        arg_src, us = _compile_expr(expr.get("arg", {}), node_vars, parameter_values)
        unsupported.extend(us)
        if op == "-":
            return f"(-({arg_src}))", unsupported
        unsupported.append(f"field_expr unary '{op}'")
        return "0.0", unsupported
    if kind == "binary":
        left_src, ul = _compile_expr(expr.get("left", {}), node_vars, parameter_values)
        right_src, ur = _compile_expr(expr.get("right", {}), node_vars, parameter_values)
        unsupported.extend(ul)
        unsupported.extend(ur)
        op = expr.get("op")
        if op in {"+", "-", "*", "/"}:
            if op == "/":
                return f"(({left_src}) / max(abs({right_src}), 1e-6))", unsupported
            return f"(({left_src}) {op} ({right_src}))", unsupported
        if op == "^":
            return f"pow(({left_src}), ({right_src}))", unsupported
        unsupported.append(f"field_expr binary '{op}'")
        return "0.0", unsupported
    if kind == "func":
        name = str(expr.get("name", ""))
        args = expr.get("args", [])
        srcs: list[str] = []
        for arg in args:
            a_src, a_unsupported = _compile_expr(arg, node_vars, parameter_values)
            srcs.append(a_src)
            unsupported.extend(a_unsupported)
        if name in {"sin", "cos", "tan", "abs", "exp"} and len(srcs) == 1:
            return f"{name}({srcs[0]})", unsupported
        if name == "sqrt" and len(srcs) == 1:
            return f"sqrt(max({srcs[0]}, 0.0))", unsupported
        if name == "log" and len(srcs) == 1:
            return f"log(max({srcs[0]}, 1e-6))", unsupported
        if name in {"min", "max"} and len(srcs) >= 1:
            out = srcs[0]
            for arg in srcs[1:]:
                out = f"{name}({out}, {arg})"
            return out, unsupported
        if name == "clamp" and len(srcs) == 3:
            return f"clamp({srcs[0]}, {srcs[1]}, {srcs[2]})", unsupported
        unsupported.append(f"field_expr func '{name}'")
        return "0.0", unsupported

    unsupported.append(f"field_expr kind '{kind}'")
    return "0.0", unsupported


def compile_scene_program(
    scene_ir: SceneIR,
    parameter_values: dict[str, float],
    grid: GridConfig,
    quality_profile: QualityProfile,
) -> tuple[SceneProgramPayload | None, str | None, float]:
    """Compile a Scene IR to a GLSL SDF function for analytic GPU raymarching.

    Each scene node is emitted as a named GLSL function ``float sdf_nX(vec3 p)``
    so that complex nodes (box, cylinder, transform, repeat, gyroid, …) can use
    local variables and multi-statement bodies without relying on the
    ``{ … }()`` anonymous-block syntax that is **not valid GLSL**.
    """
    compile_start = time.perf_counter()
    nodes = {node.id: node for node in scene_ir.nodes}
    if scene_ir.root_node_id not in nodes:
        return None, "root node missing", 0.0

    # func_defs: ordered list of "float sdf_nX(vec3 p) { … }" strings
    func_defs: list[str] = []
    # node_vars: node_id → GLSL function name (e.g. "sdf_n1")
    node_vars: dict[str, str] = {}
    visiting: set[str] = set()
    visited: set[str] = set()
    unsupported: list[str] = []

    def compile_node(node_id: str) -> None:  # noqa: C901
        if node_id in visited:
            return
        if node_id in visiting:
            unsupported.append(f"cyclic node graph at '{node_id}'")
            return
        visiting.add(node_id)
        node = nodes.get(node_id)
        if node is None:
            unsupported.append(f"missing node '{node_id}'")
            visiting.remove(node_id)
            return

        # Compile children first so their function names are in node_vars.
        for child in node.inputs:
            compile_node(child)

        func_name = f"sdf_{_safe_ident(node.id)}"
        # body: list of GLSL statements (without leading indentation — added below)
        body: list[str] = []

        # ── primitives ────────────────────────────────────────────────────────
        if node.type == "primitive":
            primitive = node.primitive or ""
            if primitive not in SUPPORTED_PRIMITIVES:
                unsupported.append(f"primitive '{primitive}'")
                body.append("return 1e9;")
            elif primitive == "sphere":
                r = _resolve_scalar(node.params.get("r"), parameter_values, 1.0)
                body.append(f"return length(p) - {_fmt(r)};")
            elif primitive == "box":
                bx = abs(_resolve_scalar(node.params.get("x"), parameter_values, 0.5))
                by = abs(_resolve_scalar(node.params.get("y"), parameter_values, 0.5))
                bz = abs(_resolve_scalar(node.params.get("z"), parameter_values, 0.5))
                body.append(f"vec3 q = abs(p) - vec3({_fmt(bx)}, {_fmt(by)}, {_fmt(bz)});")
                body.append("return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);")
            elif primitive == "cylinder":
                r = abs(_resolve_scalar(node.params.get("r"), parameter_values, 0.4))
                h = abs(_resolve_scalar(node.params.get("h"), parameter_values, 1.0))
                body.append(f"vec2 d = abs(vec2(length(p.xz), p.y)) - vec2({_fmt(r)}, {_fmt(h * 0.5)});")
                body.append("return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));")
            elif primitive == "torus":
                major_r = abs(_resolve_scalar(node.params.get("R"), parameter_values, 0.8))
                minor_r = abs(_resolve_scalar(node.params.get("r"), parameter_values, 0.2))
                body.append(f"vec2 q = vec2(length(p.xz) - {_fmt(major_r)}, p.y);")
                body.append(f"return length(q) - {_fmt(minor_r)};")
            elif primitive == "plane":
                nx = _resolve_scalar(node.params.get("nx"), parameter_values, 0.0)
                ny = _resolve_scalar(node.params.get("ny"), parameter_values, 1.0)
                nz = _resolve_scalar(node.params.get("nz"), parameter_values, 0.0)
                d = _resolve_scalar(node.params.get("d"), parameter_values, 0.0)
                norm = math.sqrt(nx * nx + ny * ny + nz * nz)
                if norm < 1e-8:
                    unsupported.append("plane normal is zero")
                    body.append("return 1e9;")
                else:
                    nx /= norm; ny /= norm; nz /= norm
                    body.append(f"return dot(p, vec3({_fmt(nx)}, {_fmt(ny)}, {_fmt(nz)})) + {_fmt(d)};")

        # ── boolean ops ───────────────────────────────────────────────────────
        elif node.type == "boolean":
            op = node.op or ""
            if op not in SUPPORTED_BOOLEANS:
                unsupported.append(f"boolean op '{op}'")
                body.append("return 1e9;")
            elif len(node.inputs) < 2:
                unsupported.append(f"boolean '{node.id}' missing inputs")
                body.append("return 1e9;")
            else:
                child_funcs = [node_vars[c] for c in node.inputs if c in node_vars]
                if len(child_funcs) < 2:
                    unsupported.append(f"boolean '{node.id}' unresolved children")
                    body.append("return 1e9;")
                elif op == "union":
                    result = f"{child_funcs[0]}(p)"
                    for cf in child_funcs[1:]:
                        result = f"min({result}, {cf}(p))"
                    body.append(f"return {result};")
                elif op == "intersection":
                    result = f"{child_funcs[0]}(p)"
                    for cf in child_funcs[1:]:
                        result = f"max({result}, {cf}(p))"
                    body.append(f"return {result};")
                elif op == "difference":
                    result = f"{child_funcs[0]}(p)"
                    for cf in child_funcs[1:]:
                        result = f"max({result}, -({cf}(p)))"
                    body.append(f"return {result};")
                elif op == "smooth_union":
                    k = abs(_resolve_scalar(node.params.get("k"), parameter_values, 0.2))
                    k_str = _fmt(max(k, 1e-6))
                    body.append(f"float su_r = {child_funcs[0]}(p);")
                    for i, cf in enumerate(child_funcs[1:]):
                        body.append(f"float su_b{i} = {cf}(p);")
                        body.append(f"float su_h{i} = clamp(0.5 + 0.5*(su_b{i}-su_r)/{k_str}, 0.0, 1.0);")
                        body.append(f"su_r = mix(su_b{i}, su_r, su_h{i}) - {_fmt(k)}*su_h{i}*(1.0-su_h{i});")
                    body.append("return su_r;")

        # ── transform ─────────────────────────────────────────────────────────
        elif node.type == "transform":
            if len(node.inputs) != 1 or node.inputs[0] not in node_vars:
                unsupported.append(f"transform '{node.id}' unresolved child")
                body.append("return 1e9;")
            else:
                child_func = node_vars[node.inputs[0]]
                payload = node.transform or {}
                tx, ty, tz = _resolve_vec3(payload.get("translate"), parameter_values, (0.0, 0.0, 0.0))
                rx, ry, rz = _resolve_vec3(payload.get("rotate"), parameter_values, (0.0, 0.0, 0.0))
                sx, sy, sz = _resolve_vec3(payload.get("scale"), parameter_values, (1.0, 1.0, 1.0))
                sx = sx if abs(sx) > 1e-6 else 1e-6
                sy = sy if abs(sy) > 1e-6 else 1e-6
                sz = sz if abs(sz) > 1e-6 else 1e-6
                body.append(f"vec3 q = p - vec3({_fmt(tx)}, {_fmt(ty)}, {_fmt(tz)});")
                if any(abs(v) > 1e-9 for v in (rx, ry, rz)):
                    inv_rot = _rotation_matrix_xyz((rx, ry, rz)).T
                    body.append(f"q = {_mat3_literal(inv_rot)} * q;")
                body.append(f"q /= vec3({_fmt(sx)}, {_fmt(sy)}, {_fmt(sz)});")
                body.append(f"return {child_func}(q);")

        # ── domain ops ────────────────────────────────────────────────────────
        elif node.type == "domain_op":
            op = node.op or ""
            if op not in SUPPORTED_DOMAIN_OPS:
                unsupported.append(f"domain op '{op}'")
                body.append("return 1e9;")
            elif len(node.inputs) != 1 or node.inputs[0] not in node_vars:
                unsupported.append(f"domain '{node.id}' unresolved child")
                body.append("return 1e9;")
            else:
                child_func = node_vars[node.inputs[0]]
                if op == "repeat":
                    px = _resolve_scalar(node.params.get("x"), parameter_values, 1.0)
                    py = _resolve_scalar(node.params.get("y"), parameter_values, 1.0)
                    pz = _resolve_scalar(node.params.get("z"), parameter_values, 1.0)
                    body.append("vec3 q = p;")
                    if abs(px) > 1e-6:
                        body.append(f"q.x = mod(q.x + {_fmt(px * 0.5)}, {_fmt(px)}) - {_fmt(px * 0.5)};")
                    if abs(py) > 1e-6:
                        body.append(f"q.y = mod(q.y + {_fmt(py * 0.5)}, {_fmt(py)}) - {_fmt(py * 0.5)};")
                    if abs(pz) > 1e-6:
                        body.append(f"q.z = mod(q.z + {_fmt(pz * 0.5)}, {_fmt(pz)}) - {_fmt(pz * 0.5)};")
                    body.append(f"return {child_func}(q);")
                elif op in {"twist", "bend"}:
                    k = _resolve_scalar(node.params.get("k"), parameter_values, 1.0 if op == "twist" else 0.5)
                    axis = _resolve_str(node.params.get("axis"), "y" if op == "twist" else "x").lower()
                    if axis == "x":
                        body.append(f"float a = {_fmt(k)} * p.x;")
                        body.append("float ca = cos(a); float sa = sin(a);")
                        body.append("vec3 q = vec3(p.x, ca*p.y - sa*p.z, sa*p.y + ca*p.z);")
                    elif axis == "z":
                        body.append(f"float a = {_fmt(k)} * p.z;")
                        body.append("float ca = cos(a); float sa = sin(a);")
                        body.append("vec3 q = vec3(ca*p.x - sa*p.y, sa*p.x + ca*p.y, p.z);")
                    else:  # y
                        body.append(f"float a = {_fmt(k)} * p.y;")
                        body.append("float ca = cos(a); float sa = sin(a);")
                        body.append("vec3 q = vec3(ca*p.x - sa*p.z, p.y, sa*p.x + ca*p.z);")
                    body.append(f"return {child_func}(q);")
                elif op == "shell":
                    t = abs(_resolve_scalar(node.params.get("t"), parameter_values, 0.1))
                    body.append(f"return abs({child_func}(p)) - {_fmt(t)};")
                elif op == "offset":
                    d = _resolve_scalar(node.params.get("d"), parameter_values, 0.0)
                    body.append(f"return {child_func}(p) - {_fmt(d)};")

        # ── lattice ops ───────────────────────────────────────────────────────
        elif node.type == "lattice":
            op = node.op or ""
            if op not in SUPPORTED_LATTICE_OPS:
                unsupported.append(f"lattice op '{op}'")
                body.append("return 1e9;")
            elif op in {"gyroid", "schwarz_p", "diamond"}:
                pitch = _resolve_scalar(node.params.get("pitch"), parameter_values, 1.0)
                phase = _resolve_scalar(node.params.get("phase"), parameter_values, 0.0)
                thickness = abs(_resolve_scalar(node.params.get("thickness"), parameter_values, 0.08))
                scale = 2.0 * math.pi / max(abs(pitch), 1e-6)
                body.append(f"float u = p.x * {_fmt(scale)} + {_fmt(phase)};")
                body.append(f"float v = p.y * {_fmt(scale)} + {_fmt(phase)};")
                body.append(f"float w = p.z * {_fmt(scale)} + {_fmt(phase)};")
                if op == "gyroid":
                    body.append("float base = sin(u)*cos(v) + sin(v)*cos(w) + sin(w)*cos(u);")
                elif op == "schwarz_p":
                    body.append("float base = cos(u) + cos(v) + cos(w);")
                else:  # diamond
                    body.append(
                        "float base = sin(u)*sin(v)*sin(w)"
                        " + sin(u)*cos(v)*cos(w)"
                        " + cos(u)*sin(v)*cos(w)"
                        " + cos(u)*cos(v)*sin(w);"
                    )
                body.append(f"return abs(base) - {_fmt(thickness)};")
            elif op == "conformal_fill":
                if len(node.inputs) != 2:
                    unsupported.append("conformal_fill expects two inputs")
                    body.append("return 1e9;")
                elif node.inputs[0] not in node_vars or node.inputs[1] not in node_vars:
                    unsupported.append("conformal_fill unresolved inputs")
                    body.append("return 1e9;")
                else:
                    host_func = node_vars[node.inputs[0]]
                    lat_func = node_vars[node.inputs[1]]
                    wall = abs(_resolve_scalar(node.params.get("wall"), parameter_values, 0.1))
                    offset = _resolve_scalar(node.params.get("offset"), parameter_values, 0.0)
                    mode = _resolve_str(node.params.get("mode"), "shell").lower()
                    body.append(f"float hs = {host_func}(p) + {_fmt(offset)};")
                    body.append(f"float lat = {lat_func}(p);")
                    if mode == "clip":
                        body.append("return max(lat, hs);")
                    elif mode == "hybrid":
                        body.append("float clipped = max(lat, hs);")
                        body.append(f"float band = max(lat, abs(hs) - {_fmt(wall)});")
                        body.append("return min(clipped, band);")
                    else:  # shell
                        body.append(f"return max(lat, abs(hs) - {_fmt(wall)});")

        # ── field expressions ─────────────────────────────────────────────────
        elif node.type == "field_expr":
            if node.expr is None:
                unsupported.append(f"field_expr '{node.id}' missing expr")
                body.append("return 1e9;")
            else:
                # Compile node_ref dependencies that are not in node.inputs.
                for ref_id in _collect_node_refs(node.expr):
                    compile_node(ref_id)
                expr_str, expr_unsupported = _compile_expr(node.expr, node_vars, parameter_values)
                unsupported.extend(expr_unsupported)
                body.append(f"return {expr_str};")

        else:
            unsupported.append(f"node type '{node.type}'")
            body.append("return 1e9;")

        # Emit the named function definition.
        indented = "\n".join(f"  {line}" for line in body)
        func_defs.append(f"float {func_name}(vec3 p) {{\n{indented}\n}}")
        node_vars[node.id] = func_name
        visiting.remove(node_id)
        visited.add(node_id)

    compile_node(scene_ir.root_node_id)
    if unsupported:
        compile_ms = (time.perf_counter() - compile_start) * 1000.0
        reason = "Unsupported analytic nodes: " + ", ".join(sorted(set(unsupported))[:8])
        return None, reason, compile_ms
    root_func = node_vars.get(scene_ir.root_node_id)
    if not root_func:
        compile_ms = (time.perf_counter() - compile_start) * 1000.0
        return None, "root codegen failed", compile_ms

    # Assemble: forward-declared helper functions + sdfScene entry point.
    glsl = "\n".join(func_defs) + f"\nfloat sdfScene(vec3 p) {{\n  return {root_func}(p);\n}}\n"
    budget = _quality_budget(quality_profile)
    payload = SceneProgramPayload(
        mode="dsl",
        bounds=[[float(axis[0]), float(axis[1])] for axis in grid.bounds],
        glsl_sdf=glsl,
        quality_profile=quality_profile,
        max_steps=int(budget["max_steps"]),
        hit_epsilon=float(budget["hit_eps"]),
        normal_epsilon=float(budget["normal_eps"]),
    )
    compile_ms = (time.perf_counter() - compile_start) * 1000.0
    return payload, None, compile_ms


