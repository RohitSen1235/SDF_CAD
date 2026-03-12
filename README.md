# SDF CAD

Local-first implicit modeling tool with a Python backend and React 3D frontend.

## Features

- Math-first DSL for implicit expressions (`x,y,z`, `+ - * / ^`, `sin/cos/tan/abs/sqrt/exp/log/min/max/clamp`)
- Boolean composition: `union`, `intersection`, `difference`, `smooth_union`
- Primitives: `sphere`, `box`, `cylinder`, `torus`, `plane`, `spline`, `freeform_surface` (`freeform` alias)
- Domain operations: `repeat`, `twist`, `bend`, `shell`, `offset`
- Lattice/TPMS: `gyroid`, `schwarz_p`, `diamond`, `strut_lattice`, `conformal_fill`
- Turbomachinery concepts: `impeller_centrifugal`, `radial_turbine`, `volute_casing`
- Mesh workflow: upload `.stl/.obj`, hollow with inward shell thickness `t`, fill cavity with TPMS (`gyroid`, `schwarz_p`, `diamond`)
- Interactive 3D viewport with zoom/pan/rotate, transform gizmo, fit view, section cut, wireframe
- Interactive field-volume preview mode (raymarched WebGL2 isosurface) for fast DSL edits
- Preview quality tiers: `interactive (64)`, `medium (128)`, `high (192)`, `ultra (256)`
- STL and OBJ export

## Quick Start

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## API

- `POST /api/v1/scene/compile`
- `POST /api/v1/preview/field`
- `POST /api/v1/preview/mesh`
- `POST /api/v1/export`
- `POST /api/v1/mesh/preview` (multipart)
- `POST /api/v1/mesh/export` (multipart)
- `WS /api/v1/preview`

## Example DSL

Need help writing custom implicit formulas? See the [DSL Field Expressions Primer](docs/dsl-field-primer.md).

```txt
param shell default=0.08 min=0.03 max=0.2 step=0.01
host = sphere(r=1.0)
lat = gyroid(pitch=0.45, thickness=0.1)
root = conformal_fill(host, lat, wall=$shell, mode="hybrid")
```

```txt
path = spline(points="0 0 0; 0.3 0.25 0.0; 0.9 -0.2 0.2; 1.2 0.0 0.0", radius=0.08, samples=24)
patch = freeform_surface(heights="0 0 0 0; 0 0.25 0.25 0; 0 0.3 0.3 0; 0 0 0 0", x=0.9, z=0.8, thickness=0.05)
root = union(path, patch)
```
