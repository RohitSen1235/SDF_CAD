from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any

from lark import Lark, Transformer, v_args

from .models import CompileDiagnostics, ParameterSpec, ScalarValue, SceneIR, SceneNode


GRAMMAR = r"""
start: statement+

statement: param_stmt ";"?        -> param_statement
         | assign_stmt ";"?       -> assign_statement

param_stmt: "param" NAME "default" "=" SIGNED_NUMBER "min" "=" SIGNED_NUMBER "max" "=" SIGNED_NUMBER "step" "=" SIGNED_NUMBER
assign_stmt: NAME "=" expr

?expr: sum

?sum: sum "+" product             -> add
    | sum "-" product             -> sub
    | product

?product: product "*" power       -> mul
        | product "/" power       -> div
        | power

?power: unary "^" power           -> pow
      | unary

?unary: "-" unary                 -> neg
      | atom

?atom: call
     | PARAM                       -> param_ref
     | NAME                        -> name_ref
     | NUMBER                      -> number
     | STRING                      -> string
     | "(" expr ")"

call: NAME "(" [arguments] ")"

arguments: argument ("," argument)*

?argument: NAME "=" expr          -> kw_argument
         | expr                    -> pos_argument

PARAM: /\$[A-Za-z_][A-Za-z0-9_]*/
NAME: /[A-Za-z_][A-Za-z0-9_]*/

%import common.NUMBER
%import common.SIGNED_NUMBER
%import common.ESCAPED_STRING -> STRING
%import common.WS
%ignore WS
%ignore /#[^\n]*/
"""


@dataclass
class NumberExpr:
    value: float


@dataclass
class StringExpr:
    value: str


@dataclass
class NameExpr:
    name: str


@dataclass
class ParamExpr:
    name: str


@dataclass
class UnaryExpr:
    op: str
    value: Any


@dataclass
class BinaryExpr:
    op: str
    left: Any
    right: Any


@dataclass
class CallExpr:
    name: str
    pos_args: list[Any] = field(default_factory=list)
    kw_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParamDecl:
    name: str
    default: float
    min: float
    max: float
    step: float


@dataclass
class AssignStmt:
    name: str
    expr: Any


class _AstBuilder(Transformer):
    def start(self, items: list[Any]) -> list[Any]:
        return items

    def param_statement(self, items: list[Any]) -> Any:
        return items[0]

    def assign_statement(self, items: list[Any]) -> Any:
        return items[0]

    @v_args(inline=True)
    def param_stmt(self, name: str, default: str, minv: str, maxv: str, step: str) -> ParamDecl:
        return ParamDecl(
            name=str(name),
            default=float(default),
            min=float(minv),
            max=float(maxv),
            step=float(step),
        )

    @v_args(inline=True)
    def assign_stmt(self, name: str, expr: Any) -> AssignStmt:
        return AssignStmt(name=str(name), expr=expr)

    @v_args(inline=True)
    def call(self, name: str, args: Any | None = None) -> CallExpr:
        args_dict = args or {"pos": [], "kw": {}}
        return CallExpr(name=str(name), pos_args=args_dict["pos"], kw_args=args_dict["kw"])

    def arguments(self, items: list[tuple[str, Any]]) -> dict[str, Any]:
        pos: list[Any] = []
        kw: dict[str, Any] = {}
        for kind, value in items:
            if kind == "pos":
                pos.append(value)
            else:
                key, arg_value = value
                kw[key] = arg_value
        return {"pos": pos, "kw": kw}

    @v_args(inline=True)
    def kw_argument(self, key: str, value: Any) -> tuple[str, tuple[str, Any]]:
        return ("kw", (str(key), value))

    @v_args(inline=True)
    def pos_argument(self, value: Any) -> tuple[str, Any]:
        return ("pos", value)

    @v_args(inline=True)
    def number(self, value: str) -> NumberExpr:
        return NumberExpr(float(value))

    @v_args(inline=True)
    def string(self, token: str) -> StringExpr:
        parsed = ast.literal_eval(str(token))
        return StringExpr(str(parsed))

    @v_args(inline=True)
    def name_ref(self, name: str) -> NameExpr:
        return NameExpr(str(name))

    @v_args(inline=True)
    def param_ref(self, token: str) -> ParamExpr:
        return ParamExpr(str(token)[1:])

    @v_args(inline=True)
    def neg(self, value: Any) -> UnaryExpr:
        return UnaryExpr(op="-", value=value)

    @v_args(inline=True)
    def add(self, left: Any, right: Any) -> BinaryExpr:
        return BinaryExpr(op="+", left=left, right=right)

    @v_args(inline=True)
    def sub(self, left: Any, right: Any) -> BinaryExpr:
        return BinaryExpr(op="-", left=left, right=right)

    @v_args(inline=True)
    def mul(self, left: Any, right: Any) -> BinaryExpr:
        return BinaryExpr(op="*", left=left, right=right)

    @v_args(inline=True)
    def div(self, left: Any, right: Any) -> BinaryExpr:
        return BinaryExpr(op="/", left=left, right=right)

    @v_args(inline=True)
    def pow(self, left: Any, right: Any) -> BinaryExpr:
        return BinaryExpr(op="^", left=left, right=right)


PRIMITIVES = {"sphere", "box", "cylinder", "torus", "plane", "spline", "freeform_surface", "freeform"}
BOOLEAN_OPS = {"union", "intersection", "difference", "smooth_union"}
TRANSFORMS = {"translate", "rotate", "scale"}
DOMAIN_OPS = {"repeat", "twist", "bend", "shell", "offset", "circular_array"}
LATTICE_OPS = {"gyroid", "schwarz_p", "diamond", "strut_lattice", "conformal_fill"}
TURBOMACHINERY_OPS = {"impeller_centrifugal", "radial_turbine", "volute_casing"}
MATH_FUNCTIONS = {
    "sin",
    "cos",
    "tan",
    "abs",
    "sqrt",
    "exp",
    "log",
    "min",
    "max",
    "clamp",
}
COORDINATE_NAMES = {"x", "y", "z"}


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


def _parse_heights_grid(raw: str) -> list[list[float]] | None:
    values = _parse_float_tokens(raw)
    if values is None or len(values) != 16:
        return None
    return [values[idx * 4 : (idx + 1) * 4] for idx in range(4)]


class DslError(ValueError):
    pass


class SceneCompiler:
    def __init__(self) -> None:
        self._parser = Lark(GRAMMAR, parser="lalr", transformer=_AstBuilder())

    def compile(self, source: str) -> tuple[SceneIR, CompileDiagnostics]:
        if not source.strip():
            raise DslError("DSL source is empty")

        try:
            statements = self._parser.parse(source)
        except Exception as exc:
            raise DslError(f"Failed to parse DSL: {exc}") from exc

        parameter_schema: list[ParameterSpec] = []
        declared_params: dict[str, ParameterSpec] = {}
        symbol_table: dict[str, str] = {}
        nodes: list[SceneNode] = []
        last_assigned: str | None = None
        warnings: list[str] = []
        node_counter = 0

        def new_node_id() -> str:
            nonlocal node_counter
            node_counter += 1
            return f"n{node_counter}"

        def add_node(node: SceneNode) -> str:
            node.id = new_node_id()
            nodes.append(node)
            return node.id

        def ensure_scalar(value: Any, *, allow_string: bool = False) -> ScalarValue:
            if isinstance(value, NumberExpr):
                return float(value.value)
            if isinstance(value, ParamExpr):
                if value.name not in declared_params:
                    raise DslError(f"Parameter '{value.name}' is referenced but not declared")
                return {"$param": value.name}
            if isinstance(value, StringExpr):
                if allow_string:
                    return value.value
                raise DslError("String value is not allowed in this argument")
            if isinstance(value, UnaryExpr) and value.op == "-":
                if isinstance(value.value, NumberExpr):
                    return -float(value.value.value)
                raise DslError("Unary '-' is only supported for numeric literals in scalar arguments")
            raise DslError(f"Expected numeric{'/string' if allow_string else ''} value, got {value!r}")

        def expr_to_payload(expr: Any) -> dict[str, Any]:
            if isinstance(expr, NumberExpr):
                return {"kind": "number", "value": float(expr.value)}

            if isinstance(expr, ParamExpr):
                if expr.name not in declared_params:
                    raise DslError(f"Parameter '{expr.name}' is referenced but not declared")
                return {"kind": "param", "name": expr.name}

            if isinstance(expr, NameExpr):
                if expr.name in symbol_table:
                    return {"kind": "node_ref", "id": symbol_table[expr.name]}
                if expr.name in COORDINATE_NAMES:
                    return {"kind": "var", "name": expr.name}
                if expr.name in declared_params:
                    return {"kind": "param", "name": expr.name}
                raise DslError(
                    f"Name '{expr.name}' is not valid in field expressions. Use x/y/z or declared parameters"
                )

            if isinstance(expr, UnaryExpr):
                if expr.op != "-":
                    raise DslError(f"Unsupported unary op '{expr.op}'")
                return {"kind": "unary", "op": "-", "arg": expr_to_payload(expr.value)}

            if isinstance(expr, BinaryExpr):
                return {
                    "kind": "binary",
                    "op": expr.op,
                    "left": expr_to_payload(expr.left),
                    "right": expr_to_payload(expr.right),
                }

            if isinstance(expr, CallExpr):
                if expr.name not in MATH_FUNCTIONS:
                    raise DslError(
                        f"Function '{expr.name}' is not allowed inside field expressions"
                    )
                if expr.kw_args:
                    raise DslError(
                        f"Function '{expr.name}' does not support keyword args in field expressions"
                    )
                return {
                    "kind": "func",
                    "name": expr.name,
                    "args": [expr_to_payload(item) for item in expr.pos_args],
                }

            raise DslError(f"Unsupported field expression payload: {expr!r}")

        def compile_field_expr(expr: Any) -> str:
            payload = expr_to_payload(expr)
            return add_node(SceneNode(id="", type="field_expr", expr=payload, params={}))

        def resolve_scene_expr(expr: Any) -> str:
            if isinstance(expr, NameExpr):
                if expr.name in symbol_table:
                    return symbol_table[expr.name]
                if expr.name in COORDINATE_NAMES or expr.name in declared_params:
                    return compile_field_expr(expr)
                raise DslError(f"Unknown symbol '{expr.name}'")

            if isinstance(expr, CallExpr):
                fn = expr.name
                if fn in PRIMITIVES:
                    return add_node(compile_primitive(fn, expr))
                if fn in BOOLEAN_OPS:
                    return add_node(compile_boolean(fn, expr))
                if fn in TRANSFORMS:
                    return add_node(compile_transform(fn, expr))
                if fn in DOMAIN_OPS:
                    return add_node(compile_domain_op(fn, expr))
                if fn in LATTICE_OPS:
                    return add_node(compile_lattice(fn, expr))
                if fn in TURBOMACHINERY_OPS:
                    return add_node(compile_turbomachinery(fn, expr))
                if fn in MATH_FUNCTIONS:
                    return compile_field_expr(expr)
                raise DslError(f"Unknown function '{fn}'")

            if isinstance(expr, (NumberExpr, ParamExpr, UnaryExpr, BinaryExpr)):
                return compile_field_expr(expr)

            raise DslError(f"Invalid scene expression {expr!r}")

        def resolve_children(children: list[Any]) -> list[str]:
            return [resolve_scene_expr(item) for item in children]

        def primitive_bounds(name: str, params: dict[str, ScalarValue]) -> list[list[ScalarValue]] | None:
            if name == "sphere":
                r = params.get("r", 1.0)
                if isinstance(r, (float, int)):
                    return [[-float(r), float(r)], [-float(r), float(r)], [-float(r), float(r)]]
                return None

            if name == "box":
                x = params.get("x", 0.5)
                y = params.get("y", 0.5)
                z = params.get("z", 0.5)
                if isinstance(x, (float, int)) and isinstance(y, (float, int)) and isinstance(z, (float, int)):
                    return [[-float(x), float(x)], [-float(y), float(y)], [-float(z), float(z)]]
                return None

            if name == "cylinder":
                r = params.get("r", 0.4)
                h = params.get("h", 1.0)
                if isinstance(r, (float, int)) and isinstance(h, (float, int)):
                    return [[-float(r), float(r)], [-float(h), float(h)], [-float(r), float(r)]]
                return None

            if name == "torus":
                major_r = params.get("R", 0.8)
                minor_r = params.get("r", 0.2)
                if isinstance(major_r, (float, int)) and isinstance(minor_r, (float, int)):
                    reach = float(major_r) + float(minor_r)
                    return [[-reach, reach], [-float(minor_r), float(minor_r)], [-reach, reach]]
                return None

            if name == "spline":
                points = params.get("points")
                radius = params.get("radius", 0.08)
                if isinstance(points, str) and isinstance(radius, (float, int)):
                    parsed = _parse_points_string(points)
                    if parsed is None:
                        return None
                    xs = [point[0] for point in parsed]
                    ys = [point[1] for point in parsed]
                    zs = [point[2] for point in parsed]
                    pad = abs(float(radius))
                    return [
                        [min(xs) - pad, max(xs) + pad],
                        [min(ys) - pad, max(ys) + pad],
                        [min(zs) - pad, max(zs) + pad],
                    ]
                return None

            if name == "freeform_surface":
                heights = params.get("heights")
                x_extent = params.get("x", 1.0)
                z_extent = params.get("z", 1.0)
                thickness = params.get("thickness", 0.05)
                if (
                    isinstance(heights, str)
                    and isinstance(x_extent, (float, int))
                    and isinstance(z_extent, (float, int))
                    and isinstance(thickness, (float, int))
                ):
                    grid = _parse_heights_grid(heights)
                    if grid is None:
                        return None
                    y_values = [item for row in grid for item in row]
                    pad = abs(float(thickness))
                    return [
                        [-abs(float(x_extent)), abs(float(x_extent))],
                        [min(y_values) - pad, max(y_values) + pad],
                        [-abs(float(z_extent)), abs(float(z_extent))],
                    ]
                return None

            return None

        def compile_primitive(fn: str, call: CallExpr) -> SceneNode:
            primitive_name = "freeform_surface" if fn == "freeform" else fn
            defaults: dict[str, ScalarValue] = {}
            key_order: list[str]
            string_keys: set[str] = set()
            if primitive_name == "sphere":
                defaults = {"r": 1.0}
                key_order = ["r"]
            elif primitive_name == "box":
                defaults = {"x": 0.5, "y": 0.5, "z": 0.5}
                key_order = ["x", "y", "z"]
            elif primitive_name == "cylinder":
                defaults = {"r": 0.4, "h": 1.0}
                key_order = ["r", "h"]
            elif primitive_name == "torus":
                defaults = {"R": 0.8, "r": 0.2}
                key_order = ["R", "r"]
            elif primitive_name == "plane":
                defaults = {"nx": 0.0, "ny": 1.0, "nz": 0.0, "d": 0.0}
                key_order = ["nx", "ny", "nz", "d"]
            elif primitive_name == "spline":
                defaults = {
                    "points": "0 0 0; 0.4 0.2 0; 0.8 -0.2 0.1; 1.2 0 0",
                    "radius": 0.08,
                    "samples": 20.0,
                    "closed": 0.0,
                }
                key_order = ["points", "radius", "samples", "closed"]
                string_keys = {"points"}
            elif primitive_name == "freeform_surface":
                defaults = {
                    "heights": "0 0 0 0; 0 0.2 0.2 0; 0 0.2 0.2 0; 0 0 0 0",
                    "x": 1.0,
                    "z": 1.0,
                    "thickness": 0.06,
                }
                key_order = ["heights", "x", "z", "thickness"]
                string_keys = {"heights"}
            else:
                raise DslError(f"Unsupported primitive {primitive_name}")

            params: dict[str, ScalarValue] = defaults.copy()
            if call.pos_args:
                if primitive_name == "box" and len(call.pos_args) == 1:
                    size = ensure_scalar(call.pos_args[0])
                    params["x"] = size
                    params["y"] = size
                    params["z"] = size
                else:
                    if len(call.pos_args) > len(key_order):
                        raise DslError(f"{primitive_name} received too many positional arguments")
                    for idx, pos_value in enumerate(call.pos_args):
                        key = key_order[idx]
                        params[key] = ensure_scalar(pos_value, allow_string=key in string_keys)

            for key, value in call.kw_args.items():
                if key not in defaults:
                    raise DslError(f"{primitive_name} does not support argument '{key}'")
                params[key] = ensure_scalar(value, allow_string=key in string_keys)

            return SceneNode(
                id="",
                type="primitive",
                primitive=primitive_name,
                params=params,
                bounds_hint=primitive_bounds(primitive_name, params),
            )

        def compile_boolean(fn: str, call: CallExpr) -> SceneNode:
            op_params: dict[str, ScalarValue] = {}
            children: list[str] = []

            child_exprs = call.pos_args
            if fn == "smooth_union":
                if "k" in call.kw_args:
                    op_params["k"] = ensure_scalar(call.kw_args["k"])
                elif call.pos_args and not isinstance(call.pos_args[0], (CallExpr, NameExpr, BinaryExpr, UnaryExpr)):
                    op_params["k"] = ensure_scalar(call.pos_args[0])
                    child_exprs = call.pos_args[1:]
                else:
                    op_params["k"] = 0.2

            children = resolve_children(child_exprs)

            if fn in {"union", "intersection", "smooth_union"} and len(children) < 2:
                raise DslError(f"{fn} requires at least 2 child expressions")
            if fn == "difference" and len(children) < 2:
                raise DslError("difference requires at least 2 child expressions")

            return SceneNode(id="", type="boolean", op=fn, inputs=children, params=op_params)

        def compile_transform(fn: str, call: CallExpr) -> SceneNode:
            if not call.pos_args:
                raise DslError(f"{fn} requires a child expression as first argument")

            child = resolve_scene_expr(call.pos_args[0])

            if fn == "translate":
                defaults: list[ScalarValue] = [0.0, 0.0, 0.0]
                mapping = {"x": 0, "y": 1, "z": 2}
            elif fn == "rotate":
                defaults = [0.0, 0.0, 0.0]
                mapping = {"x": 0, "y": 1, "z": 2}
            elif fn == "scale":
                defaults = [1.0, 1.0, 1.0]
                mapping = {"x": 0, "y": 1, "z": 2, "s": -1}
            else:
                raise DslError(f"Unsupported transform {fn}")

            vector: list[ScalarValue] = defaults.copy()
            for key, idx in mapping.items():
                if key in call.kw_args:
                    value = ensure_scalar(call.kw_args[key])
                    if idx == -1:
                        vector = [value, value, value]
                    else:
                        vector[idx] = value

            return SceneNode(
                id="",
                type="transform",
                op=fn,
                inputs=[child],
                transform={fn: vector},
            )

        def compile_domain_op(fn: str, call: CallExpr) -> SceneNode:
            if not call.pos_args:
                raise DslError(f"{fn} requires child expression as first argument")
            child = resolve_scene_expr(call.pos_args[0])

            if fn == "repeat":
                params: dict[str, ScalarValue] = {"x": 1.0, "y": 1.0, "z": 1.0}
                for axis in ("x", "y", "z"):
                    if axis in call.kw_args:
                        params[axis] = ensure_scalar(call.kw_args[axis])
            elif fn == "twist":
                params = {"k": 1.0, "axis": "y"}
                if "k" in call.kw_args:
                    params["k"] = ensure_scalar(call.kw_args["k"])
                if "axis" in call.kw_args:
                    params["axis"] = ensure_scalar(call.kw_args["axis"], allow_string=True)
            elif fn == "bend":
                params = {"k": 0.5, "axis": "x"}
                if "k" in call.kw_args:
                    params["k"] = ensure_scalar(call.kw_args["k"])
                if "axis" in call.kw_args:
                    params["axis"] = ensure_scalar(call.kw_args["axis"], allow_string=True)
            elif fn == "shell":
                params = {"t": 0.1}
                if "t" in call.kw_args:
                    params["t"] = ensure_scalar(call.kw_args["t"])
            elif fn == "offset":
                params = {"d": 0.0}
                if "d" in call.kw_args:
                    params["d"] = ensure_scalar(call.kw_args["d"])
            elif fn == "circular_array":
                params = {"count": 12.0, "axis": "y", "phase": 0.0}
                if "count" in call.kw_args:
                    params["count"] = ensure_scalar(call.kw_args["count"])
                if "axis" in call.kw_args:
                    params["axis"] = ensure_scalar(call.kw_args["axis"], allow_string=True)
                if "phase" in call.kw_args:
                    params["phase"] = ensure_scalar(call.kw_args["phase"])
            else:
                raise DslError(f"Unsupported domain op '{fn}'")

            return SceneNode(id="", type="domain_op", op=fn, inputs=[child], params=params)

        def compile_lattice(fn: str, call: CallExpr) -> SceneNode:
            params: dict[str, ScalarValue]
            children: list[str] = []

            if fn == "conformal_fill":
                if len(call.pos_args) < 2:
                    raise DslError("conformal_fill requires host and lattice child expressions")
                children = [resolve_scene_expr(call.pos_args[0]), resolve_scene_expr(call.pos_args[1])]
                params = {
                    "wall": 0.1,
                    "offset": 0.0,
                    "mode": "shell",
                }
                for key in ("wall", "offset"):
                    if key in call.kw_args:
                        params[key] = ensure_scalar(call.kw_args[key])
                if "mode" in call.kw_args:
                    params["mode"] = ensure_scalar(call.kw_args["mode"], allow_string=True)
            elif fn in {"gyroid", "schwarz_p", "diamond"}:
                params = {"pitch": 1.0, "phase": 0.0, "thickness": 0.08}
                for key in ("pitch", "phase", "thickness"):
                    if key in call.kw_args:
                        params[key] = ensure_scalar(call.kw_args[key])
            elif fn == "strut_lattice":
                params = {"type": "bcc", "pitch": 1.0, "radius": 0.08}
                if "type" in call.kw_args:
                    params["type"] = ensure_scalar(call.kw_args["type"], allow_string=True)
                for key in ("pitch", "radius"):
                    if key in call.kw_args:
                        params[key] = ensure_scalar(call.kw_args[key])
            else:
                raise DslError(f"Unsupported lattice op '{fn}'")

            return SceneNode(id="", type="lattice", op=fn, inputs=children, params=params)

        def compile_turbomachinery(fn: str, call: CallExpr) -> SceneNode:
            defaults: dict[str, ScalarValue]
            if fn == "impeller_centrifugal":
                defaults = {
                    "r_in": 0.25,
                    "r_out": 1.0,
                    "hub_h": 0.45,
                    "blade_count": 8.0,
                    "blade_thickness": 0.08,
                    "blade_twist": 0.8,
                    "shroud_gap": 0.05,
                }
            elif fn == "radial_turbine":
                defaults = {
                    "r_in": 0.2,
                    "r_out": 1.05,
                    "hub_h": 0.5,
                    "blade_count": 10.0,
                    "blade_thickness": 0.07,
                    "blade_twist": -0.7,
                }
            elif fn == "volute_casing":
                defaults = {
                    "throat_radius": 0.35,
                    "outlet_radius": 1.4,
                    "area_growth": 0.9,
                    "width": 0.6,
                    "wall": 0.08,
                    "tongue_clearance": 0.06,
                }
            else:
                raise DslError(f"Unsupported turbomachine op '{fn}'")

            params = defaults.copy()
            for key, value in call.kw_args.items():
                if key not in defaults:
                    raise DslError(f"{fn} does not support argument '{key}'")
                params[key] = ensure_scalar(value)

            bounds_hint: list[list[ScalarValue]] | None = None
            if fn in {"impeller_centrifugal", "radial_turbine"}:
                r_out = params.get("r_out")
                hub_h = params.get("hub_h")
                if isinstance(r_out, (float, int)) and isinstance(hub_h, (float, int)):
                    bounds_hint = [
                        [-float(r_out), float(r_out)],
                        [-float(hub_h), float(hub_h)],
                        [-float(r_out), float(r_out)],
                    ]
            elif fn == "volute_casing":
                outlet = params.get("outlet_radius")
                width = params.get("width")
                if isinstance(outlet, (float, int)) and isinstance(width, (float, int)):
                    reach = float(outlet) + 0.8
                    bounds_hint = [[-reach, reach], [-float(width), float(width)], [-reach, reach]]

            return SceneNode(
                id="",
                type="turbomachine",
                op=fn,
                params=params,
                bounds_hint=bounds_hint,
            )

        for statement in statements:
            if isinstance(statement, ParamDecl):
                if statement.name in declared_params:
                    raise DslError(f"Parameter '{statement.name}' declared multiple times")
                spec = ParameterSpec(
                    name=statement.name,
                    default=statement.default,
                    min=statement.min,
                    max=statement.max,
                    step=statement.step,
                )
                declared_params[statement.name] = spec
                parameter_schema.append(spec)
                continue

            if isinstance(statement, AssignStmt):
                node_id = resolve_scene_expr(statement.expr)
                symbol_table[statement.name] = node_id
                last_assigned = statement.name
                continue

            raise DslError(f"Unexpected statement {statement!r}")

        if not symbol_table:
            raise DslError("No scene nodes were defined. Add assignments like root = sphere(r=1)")

        root_symbol = "root" if "root" in symbol_table else last_assigned
        if root_symbol is None:
            raise DslError("Unable to determine root node")

        scene_ir = SceneIR(
            nodes=nodes,
            root_node_id=symbol_table[root_symbol],
            parameter_schema=parameter_schema,
            source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        )

        defaults = {spec.name: spec.default for spec in parameter_schema}
        inferred_bounds = self._infer_bounds(scene_ir, defaults)
        if inferred_bounds is None:
            warnings.append(
                "Could not infer finite bounds for the root node; use grid bounds to control preview domain"
            )

        diagnostics = CompileDiagnostics(warnings=warnings, inferred_bounds=inferred_bounds)
        return scene_ir, diagnostics

    def _infer_bounds(
        self, scene_ir: SceneIR, default_params: dict[str, float]
    ) -> list[list[float]] | None:
        node_by_id = {node.id: node for node in scene_ir.nodes}
        memo: dict[str, list[list[float]] | None] = {}

        def to_num(value: ScalarValue) -> float | None:
            if isinstance(value, (float, int)):
                return float(value)
            if isinstance(value, dict) and "$param" in value:
                return default_params.get(value["$param"])
            return None

        def get_bounds(node_id: str) -> list[list[float]] | None:
            if node_id in memo:
                return memo[node_id]

            node = node_by_id.get(node_id)
            if node is None:
                return None

            out: list[list[float]] | None = None

            if node.bounds_hint is not None:
                try:
                    out = [
                        [to_num(node.bounds_hint[0][0]), to_num(node.bounds_hint[0][1])],
                        [to_num(node.bounds_hint[1][0]), to_num(node.bounds_hint[1][1])],
                        [to_num(node.bounds_hint[2][0]), to_num(node.bounds_hint[2][1])],
                    ]
                except Exception:
                    out = None
                if out is not None and any(v is None for axis in out for v in axis):
                    out = None

            if out is None and node.type == "boolean":
                child_bounds = [get_bounds(child_id) for child_id in node.inputs]
                if node.op == "union" and all(item is not None for item in child_bounds):
                    out = [
                        [min(item[0][0] for item in child_bounds if item), max(item[0][1] for item in child_bounds if item)],
                        [min(item[1][0] for item in child_bounds if item), max(item[1][1] for item in child_bounds if item)],
                        [min(item[2][0] for item in child_bounds if item), max(item[2][1] for item in child_bounds if item)],
                    ]
                elif node.op in {"intersection", "difference", "smooth_union"} and child_bounds:
                    out = child_bounds[0]

            if out is None and node.type == "transform" and node.inputs:
                child = get_bounds(node.inputs[0])
                if child is not None and node.transform:
                    out = [axis[:] for axis in child]
                    translate = node.transform.get("translate") or [0.0, 0.0, 0.0]
                    scale = node.transform.get("scale") or [1.0, 1.0, 1.0]
                    tv = [to_num(translate[0]), to_num(translate[1]), to_num(translate[2])]
                    sv = [to_num(scale[0]), to_num(scale[1]), to_num(scale[2])]
                    if all(item is not None for item in tv) and all(item is not None for item in sv):
                        out = [
                            [out[0][0] * sv[0] + tv[0], out[0][1] * sv[0] + tv[0]],
                            [out[1][0] * sv[1] + tv[1], out[1][1] * sv[1] + tv[1]],
                            [out[2][0] * sv[2] + tv[2], out[2][1] * sv[2] + tv[2]],
                        ]
                        for axis in out:
                            if axis[0] > axis[1]:
                                axis[0], axis[1] = axis[1], axis[0]
                    else:
                        out = None

            if out is None and node.type == "domain_op" and node.inputs:
                child = get_bounds(node.inputs[0])
                if child is not None:
                    if node.op in {"shell", "offset"}:
                        key = "t" if node.op == "shell" else "d"
                        pad = to_num(node.params.get(key, 0.0))
                        if pad is not None:
                            pad_abs = abs(pad)
                            out = [
                                [child[0][0] - pad_abs, child[0][1] + pad_abs],
                                [child[1][0] - pad_abs, child[1][1] + pad_abs],
                                [child[2][0] - pad_abs, child[2][1] + pad_abs],
                            ]
                    elif node.op == "circular_array":
                        axis_val = node.params.get("axis", "y")
                        axis = axis_val.lower() if isinstance(axis_val, str) else "y"
                        if axis == "y":
                            reach = max(abs(child[0][0]), abs(child[0][1]), abs(child[2][0]), abs(child[2][1]))
                            out = [[-reach, reach], [child[1][0], child[1][1]], [-reach, reach]]
                        elif axis == "x":
                            reach = max(abs(child[1][0]), abs(child[1][1]), abs(child[2][0]), abs(child[2][1]))
                            out = [[child[0][0], child[0][1]], [-reach, reach], [-reach, reach]]
                        else:
                            reach = max(abs(child[0][0]), abs(child[0][1]), abs(child[1][0]), abs(child[1][1]))
                            out = [[-reach, reach], [-reach, reach], [child[2][0], child[2][1]]]
                    elif node.op in {"twist", "bend", "repeat"}:
                        out = None

            if out is None and node.type == "lattice" and node.op == "conformal_fill" and node.inputs:
                out = get_bounds(node.inputs[0])

            memo[node_id] = out
            return out

        return get_bounds(scene_ir.root_node_id)


_compiler = SceneCompiler()


def compile_source(source: str) -> SceneIR:
    scene_ir, _ = _compiler.compile(source)
    return scene_ir


def compile_source_with_diagnostics(source: str) -> tuple[SceneIR, CompileDiagnostics]:
    return _compiler.compile(source)
