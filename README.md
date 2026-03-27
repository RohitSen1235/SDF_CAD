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
- Analytic raymarch preview mode (GPU SDF evaluation from compiled program) with automatic fallback to field/mesh previews
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

### Docker Compose

From the repository root:

```bash
docker compose up --build
```

The backend container is configured to request NVIDIA GPU access, so Docker will only expose the GPU if the host has a compatible NVIDIA driver and the NVIDIA Container Toolkit installed.

If Docker returns an error like `could not select device driver "" with capabilities: [[gpu]]`, the host still needs NVIDIA Container Toolkit setup. On Linux, configure Docker with `sudo nvidia-ctk runtime configure --runtime=docker` and restart Docker with `sudo systemctl restart docker`. For rootless Docker, use the user daemon config path described in the NVIDIA Container Toolkit docs.

This starts:

- Frontend: `http://127.0.0.1:5173`
- Backend API: `http://127.0.0.1:8000`
- Redis: `127.0.0.1:6379`
- Celery worker

The frontend in Docker Compose is configured with `VITE_API_BASE=http://127.0.0.1:8000` so browser requests resolve to the backend published on the host port.

Compose now starts the queue stack by default so structural optimization works without extra flags.

For low-memory environments, the worker runs with `--concurrency=1 --pool=solo` in Compose to reduce memory pressure.

To stop and remove containers:

```bash
docker compose down
```

If you hit stale-container conflicts from previous runs, clean up with:

```bash
docker compose down --remove-orphans
```

## API

- `POST /api/v1/scene/compile`
- `POST /api/v1/preview/field`
- `POST /api/v1/preview/mesh`
- `POST /api/v1/preview/program`
- `POST /api/v1/export`
- `POST /api/v1/mesh/preview` (multipart)
- `POST /api/v1/mesh/program` (multipart)
- `POST /api/v1/mesh/export` (multipart)
- `WS /api/v1/preview`
- `WS /api/v1/mesh/preview/ws` (phased uploaded mesh preview: `field` then `mesh`)

## Example DSL

Need help writing custom implicit formulas? See the [DSL Field Expressions Primer](docs/dsl-field-primer.md).

For the planned structural optimization module, see the [Generative Design Physics and PDE Design Note](docs/generative-design-physics-pde.md).

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
