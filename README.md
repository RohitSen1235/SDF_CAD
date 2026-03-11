# SDF CAD

Local-first implicit modeling tool with a Python backend and React 3D frontend.

## Features

- Math-first DSL for implicit expressions (`x,y,z`, `+ - * / ^`, `sin/cos/tan/abs/sqrt/exp/log/min/max/clamp`)
- Boolean composition: `union`, `intersection`, `difference`, `smooth_union`
- Primitives: `sphere`, `box`, `cylinder`, `torus`, `plane`
- Domain operations: `repeat`, `twist`, `bend`, `shell`, `offset`
- Lattice/TPMS: `gyroid`, `schwarz_p`, `diamond`, `strut_lattice`, `conformal_fill`
- Turbomachinery concepts: `impeller_centrifugal`, `radial_turbine`, `volute_casing`
- Interactive 3D viewport with zoom/pan/rotate, transform gizmo, fit view, section cut, wireframe
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
- `POST /api/v1/preview/mesh`
- `POST /api/v1/export`
- `WS /api/v1/preview`

## Example DSL

```txt
param shell default=0.08 min=0.03 max=0.2 step=0.01
host = sphere(r=1.0)
lat = gyroid(pitch=0.45, thickness=0.1)
root = conformal_fill(host, lat, wall=$shell, mode="hybrid")
```
