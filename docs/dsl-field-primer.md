# DSL Field Expressions Primer

This primer is for writing `field_expr` style DSL by hand, from simple formulas to multi-stage models.

## 1) Mental model

Think of every expression as a scalar field `f(x, y, z)`:

- `f < 0`: inside solid
- `f = 0`: surface
- `f > 0`: outside solid

In this DSL, `x`, `y`, `z` are sampled over a 3D grid. Most built-in primitives assume `y` is the vertical axis (for example, `cylinder(h=...)` extends along `y`).

Your final assignment should be a field-producing node, typically named `root`:

```txt
root = sphere(r=1.0)
```

or

```txt
root = sin(x * 3.0) + cos(y * 3.0) - 0.25
```

If `root` is not present, the compiler uses the last assigned symbol as the scene root.

## 2) Expression grammar you can use

Inside field expressions you can use:

- Numbers: `1`, `-0.25`, `2.0`
- Coordinates: `x`, `y`, `z`
- Declared params: `$scale` (or `scale` after declaration)
- Operators: `+ - * / ^`
- Unary minus: `-expr`
- Functions: `sin`, `cos`, `tan`, `abs`, `sqrt`, `exp`, `log`, `min`, `max`, `clamp`
- Previously assigned symbols (including primitives/booleans/transforms/domain ops)

Example:

```txt
param scale default=1.5 min=0.5 max=3.0 step=0.1
a = sin(x * $scale) + cos(y * $scale) + sin(z * $scale)
root = abs(a) - 0.45
```

Notes:

- Field-expression functions accept positional args only. Example: `sin(x)` is valid, `sin(v=x)` is not.
- Parameters must be declared via `param ...` before use.

## 3) How compiler/runtime interpret expressions

The compiler turns raw expressions into `field_expr` nodes and expression payloads:

- `x`, `y`, `z` become variable nodes (`var` payload).
- `$param` (or declared param name) becomes `param`.
- Prior symbols become `node_ref`, so expressions can reuse any earlier node.
- Math calls become `func` payloads.

Example with symbol reuse:

```txt
core = sphere(r=0.8)
ripple = abs(sin(x * 8.0) + sin(z * 8.0)) * 0.03
root = core + ripple
```

At runtime:

- `node_ref` pulls the referenced node field and combines it with your math.
- Scalars are broadcast to the full grid shape automatically.
- Safety clamps are applied:
- Division uses `1e-12` when denominator is near zero.
- `sqrt(v)` becomes `sqrt(max(v, 0))`.
- `log(v)` becomes `log(max(v, 1e-12))`.

Practical constraints and common failures:

- Undeclared param: `Parameter 'name' is referenced but not declared`
- Unknown symbol: `Unknown symbol 'name'`
- Invalid field name inside expression: `Name 'name' is not valid in field expressions...`
- Unsupported function in expression: `Function 'foo' is not allowed inside field expressions`

## 4) Design-a-shape workflow

Use this repeatable workflow:

1. Intent: Decide what should be inside/outside.
2. Coordinate remap: Center, scale, or warp coordinates.
3. Distance core: Build core implicit distance logic.
4. Combine pieces: `min`/`max` (or boolean nodes).
5. Finishers: `offset`, `shell`, smooth blend, pattern modulation.
6. Parameterize: expose key dimensions with `param`.

Template:

```txt
param size default=1.0 min=0.2 max=3.0 step=0.05
param fillet default=0.1 min=0.0 max=0.4 step=0.01

# 1-2) Remap
qx = x / $size
qy = y / $size
qz = z / $size

# 3) Core
core = sqrt(qx^2 + qy^2 + qz^2) - 1.0

# 4-5) Finish
root = core - $fillet
```

## 5) Progressive recipes (copy/paste DSL)

### A) Periodic field (abs(sin/cos) style)

```txt
param freq default=3.0 min=0.5 max=8.0 step=0.1
param iso default=0.45 min=0.1 max=1.0 step=0.01

waves = sin(x * $freq) + cos(y * $freq) + sin(z * $freq)
root = abs(waves) - $iso
```

### B) Custom cuboid via outside + inside

```txt
param sx default=0.9 min=0.2 max=2.0 step=0.05
param sy default=0.45 min=0.2 max=1.5 step=0.05
param sz default=0.6 min=0.2 max=2.0 step=0.05

qx = abs(x) - $sx
qy = abs(y) - $sy
qz = abs(z) - $sz

outside = sqrt(max(qx, 0)^2 + max(qy, 0)^2 + max(qz, 0)^2)
inside = min(max(qx, max(qy, qz)), 0)
root = outside + inside
```

### C) Rounded variation via offset + smoothstep-style modulation

```txt
param r default=0.16 min=0.02 max=0.4 step=0.01

qx = abs(x) - 0.8
qy = abs(y) - 0.35
qz = abs(z) - 0.55
outside = sqrt(max(qx, 0)^2 + max(qy, 0)^2 + max(qz, 0)^2)
inside = min(max(qx, max(qy, qz)), 0)
base = outside + inside

# Rounded by constant offset of SDF
rounded = offset(base, d=$r)

# Optional smoothstep-like remap near surface: t^2 * (3 - 2t)
t = clamp((rounded + $r) / max($r * 2.0, 0.0001), 0, 1)
smooth = t * t * (3 - 2 * t)
root = rounded + (smooth - t) * 0.03
```

### D) Ring/tube style field (torus from first principles)

```txt
param major default=0.85 min=0.3 max=2.0 step=0.05
param minor default=0.18 min=0.05 max=0.6 step=0.01

q = sqrt(x^2 + z^2) - $major
root = sqrt(q^2 + y^2) - $minor
```

### E) Band/slot carving using difference logic

`difference(a, b)` is `max(a, -b)` in field math.

```txt
param band_r default=0.8 min=0.3 max=2.0 step=0.05
param band_t default=0.09 min=0.02 max=0.4 step=0.01
param slot_h default=0.08 min=0.02 max=0.3 step=0.01

band = abs(sqrt(x^2 + z^2) - $band_r) - $band_t
slot = abs(y) - $slot_h

# Keep band but carve slot out
root = max(band, -slot)
```

### F) Layered composition using prior symbols

```txt
param ripple default=9.0 min=1.0 max=20.0 step=0.5
param blend default=0.10 min=0.01 max=0.3 step=0.01

core = sphere(r=0.72)
ripple_field = abs(sin(x * $ripple) + sin(z * $ripple)) * 0.03
textured_core = core + ripple_field

cap = translate(cylinder(r=0.22, h=0.9), y=0.45)
neck = smooth_union($blend, textured_core, cap)
root = shell(neck, t=0.04)
```

## 6) Complex-shape composition patterns

Use this order to keep models predictable:

1. Build local parts (primitives or custom fields).
2. Combine forms (`union` / `intersection` / `difference` / `smooth_union`).
3. Apply object transforms (`translate`, `rotate`, `scale`).
4. Apply domain ops for structure (`repeat`, `twist`, `bend`, `circular_array`, `shell`, `offset`).
5. End with final trimming or shelling.

When to use:

- `union(a, b)`: keep either part (`min`)
- `intersection(a, b)`: keep overlap (`max`)
- `difference(a, b)`: carve `b` from `a` (`max(a, -b)`)
- `smooth_union(k, ...)`: softened blend transitions

Compact multi-stage example:

```txt
param twist_k default=0.75 min=0.1 max=1.5 step=0.05
param blend_k default=0.08 min=0.02 max=0.2 step=0.01

hub = cylinder(r=0.42, h=0.32)

blade_single = intersection(
  translate(box(x=0.18, y=0.22, z=0.03), x=0.72),
  difference(cylinder(r=1.0, h=0.28), cylinder(r=0.45, h=0.28))
)

blades = circular_array(blade_single, count=10, axis="y")
twisted = twist(blades, k=$twist_k, axis="y")
rim = rotate(translate(torus(R=0.74, r=0.06), y=0.16), x=90)

stage = smooth_union($blend_k, hub, twisted, rim)
skin = shell(stage, t=0.035)

trim = translate(box(x=0.12, y=0.6, z=2.0), x=-0.05)
root = difference(skin, trim)
```

## 7) Debugging checklist

If a field is not behaving as expected:

1. Parameter declarations:
- Every `$param` must be declared before use.
- Check ranges so slider defaults are valid.

2. Numerical safety:
- Near-zero denominator is clamped to `1e-12`.
- `sqrt` and `log` clamp negative/zero inputs internally.
- `^` (`pow`) on negative bases with non-integer exponents can produce unstable/NaN behavior.

3. Bounds and resolution:
- If mesh appears clipped, expand preview/export bounds.
- Start at low resolution (`interactive`) while shaping fields, then raise quality.
- Complex trigonometric fields often need tighter bounds around the object.

4. Expression legality:
- Only allowed functions work in field expressions.
- No keyword arguments inside expression functions.
- Unknown symbols usually mean ordering issue or typo.

5. Composition sanity checks:
- `difference` can erase your model if subtractor is too large.
- Very large `smooth_union` `k` can over-soften and collapse details.
- `repeat`/`twist`/`bend` are domain warps; debug them on a simple child first.

Common compiler/runtime errors and fixes:

- `Parameter 'x' is referenced but not declared`: add `param x ...`.
- `Unknown symbol 'foo'`: define `foo` earlier or fix typo.
- `Function 'foo' is not allowed inside field expressions`: use supported math functions only.
- `... does not support keyword args in field expressions`: pass positional args.
- `Could not infer finite bounds...`: set explicit preview grid bounds.

## 8) Quick reference

### Operators and identities

- `a + b`: add/modulate fields
- `min(a, b)`: union equivalent
- `max(a, b)`: intersection equivalent
- `max(a, -b)`: difference equivalent
- `abs(a) - t`: shell/band around surface
- `a - d`: offset (positive `d` expands)

### Core function list

- Trig: `sin`, `cos`, `tan`
- Basic: `abs`, `sqrt`, `exp`, `log`
- Combine: `min`, `max`, `clamp(v, lo, hi)`

### Common custom SDF patterns

Sphere:

```txt
param r default=0.8 min=0.1 max=2.0 step=0.05
root = sqrt(x^2 + y^2 + z^2) - $r
```

Torus:

```txt
param R default=0.8 min=0.2 max=2.0 step=0.05
param r default=0.2 min=0.05 max=0.8 step=0.01
q = sqrt(x^2 + z^2) - $R
root = sqrt(q^2 + y^2) - $r
```

Box (axis-aligned):

```txt
param sx default=0.7 min=0.1 max=2.0 step=0.05
param sy default=0.4 min=0.1 max=2.0 step=0.05
param sz default=0.5 min=0.1 max=2.0 step=0.05
qx = abs(x) - $sx
qy = abs(y) - $sy
qz = abs(z) - $sz
outside = sqrt(max(qx, 0)^2 + max(qy, 0)^2 + max(qz, 0)^2)
inside = min(max(qx, max(qy, qz)), 0)
root = outside + inside
```

### Composition helpers (non-expression nodes)

- Boolean: `union`, `intersection`, `difference`, `smooth_union`
- Transform: `translate`, `rotate`, `scale`
- Domain: `repeat`, `twist`, `bend`, `shell`, `offset`, `circular_array`

Use them together with expression fields to move from single formulas to production-ready models.
