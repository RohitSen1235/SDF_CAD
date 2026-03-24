# Option C: Eager Backend Pre-Processing — Implementation Plan

## Problem Statement

When a user selects a large STL file (e.g., 169 MB), the frontend calls
`parseLocalMeshPreview()` immediately on file selection. This function runs
entirely on the browser's main thread and takes **~65 seconds** for a 169 MB
file due to:

1. A pure-JS loop over 3.54M triangles creating 10.6M tiny `number[]` arrays
2. O(N²) string concatenation in `bytesToBase64()` on 296 MB of typed array data

The server-side timing metrics (`server_metadata_resolve_ms`, etc.) only start
**after** the user clicks "Preview Field" — which they cannot do until the
browser unfreezes.

## Solution: Option C — Eager Backend Pre-Processing

Replace the local parse with an immediate backend upload on file selection.
The backend:
1. Parses and validates the mesh (fast, vectorized NumPy)
2. Builds the host SDF field (voxelization + EDT — the expensive step)
3. Returns the raw outer mesh as a binary payload for immediate display
4. Caches both metadata and host field so "Preview Field" is a near-instant cache hit

## Architecture

```
User selects file
      │
      ▼
POST /api/v1/mesh/preprocess  ──────────────────────────────────────────────┐
      │                                                                       │
      │  (async, non-blocking — UI stays responsive)                         │
      │                                                                       │
      ▼                                                                       │
Backend:                                                                      │
  1. _resolve_uploaded_mesh_metadata()  → warms metadata cache               │
  2. _resolve_uploaded_host_field()     → warms host field cache             │
  3. Returns _pack_mesh_binary(outer_mesh)                                    │
      │                                                                       │
      ▼                                                                       │
Frontend: setMesh(decodeBinaryMeshPacket(packet))                            │
  → Raw geometry visible in viewer                                            │
      │                                                                       │
User adjusts lattice params                                                   │
      │                                                                       │
      ▼                                                                       │
User clicks "Preview Field"                                                   │
      │                                                                       │
      ▼                                                                       │
POST /api/v1/mesh/field.binary                                               │
      │                                                                       │
      ▼                                                                       │
Backend: metadata cache HIT + host field cache HIT → near-instant            │
  → Composed field returned in seconds                                        │
```

## Files to Change

| File | Change Type | Description |
|------|-------------|-------------|
| `backend/app/main.py` | Add | New `/api/v1/mesh/preprocess` endpoint |
| `frontend/src/lib/api.ts` | Add + Export | `preprocessUploadedMesh()` function; export `decodeBinaryMeshPacket` |
| `frontend/src/App.tsx` | Modify | Replace `parseLocalMeshPreview` call with `preprocessUploadedMesh` |
| `frontend/src/lib/localMeshPreview.ts` | Delete | No longer needed |
| `frontend/src/types.ts` | No change | Existing `MeshPayloadBinary` type covers the response |
| `backend/tests/test_api.py` | Add | Tests for the new preprocess endpoint |
| `frontend/src/App.test.tsx` | Modify | Update test for file selection behavior |

---

## Step 1: Backend — Add `/api/v1/mesh/preprocess` Endpoint

**File:** `backend/app/main.py`

**Where to insert:** After the existing `/api/v1/mesh/field.binary` endpoint
(around line 2180), before the telemetry endpoint.

**What it does:**
- Accepts the same file + lattice params as other mesh endpoints
- Calls `_resolve_uploaded_mesh_metadata()` to warm the metadata cache
- Calls `_resolve_uploaded_host_field()` to warm the host field cache
- Returns `_pack_mesh_binary(outer_mesh)` — the raw geometry without lattice

**Exact code to add:**

```python
@app.post("/api/v1/mesh/preprocess", response_model=None)
async def preprocess_uploaded_mesh(
    request: Request,
    file: UploadFile = File(...),
    lattice_pitch: float = Form(...),
    voxels_per_lattice_period: int = Form(6),
    compute_backend: ComputeBackend = Form("auto"),
    field_storage_mode: UploadedFieldStorageMode = Form("auto"),
) -> Response:
    """
    Upload a mesh file and warm the metadata + host SDF field caches.

    Returns the raw outer mesh (no lattice infill) as a binary payload so the
    frontend can display the geometry immediately. The expensive voxelization
    and EDT steps run here so that the subsequent /mesh/field or /mesh/preview
    requests hit the cache and return in seconds.

    This endpoint should be called immediately on file selection, before the
    user clicks "Preview Field".
    """
    await _reject_legacy_uploaded_mesh_quality_profile(request)
    file_bytes, extension = await _read_uploaded_mesh(file)
    file_hash = _hash_uploaded_mesh_bytes(file_bytes)

    try:
        # Step 1: Resolve metadata (parse + validate + normals) — warms metadata cache
        metadata = await asyncio.to_thread(
            _resolve_uploaded_mesh_metadata,
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
        )
        resolution, _ = _resolve_mesh_resolution(
            lattice_pitch, metadata.mesh_span, voxels_per_lattice_period
        )

        # Step 2: Build host SDF field (voxelize + EDT) — warms host field cache.
        # This is the expensive step; doing it now means Preview Field is a cache hit.
        await asyncio.to_thread(
            _resolve_uploaded_host_field,
            file_bytes=file_bytes,
            file_hash=file_hash,
            extension=extension,
            resolution=resolution,
            compute_backend=compute_backend,
            parsed=metadata.parsed,
            field_storage_mode=field_storage_mode,
        )
    except (MeshUploadError, MeshingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        cleanup_gpu_memory(reason="preprocess_uploaded_mesh_inline")

    # Return the raw outer mesh so the frontend can display it immediately.
    outer = metadata.outer_mesh
    headers = {
        "X-SDF-Vertex-Count": str(int(outer.vertices.shape[0])),
        "X-SDF-Face-Count": str(int(outer.faces.shape[0])),
    }
    return Response(
        content=_pack_mesh_binary(outer),
        media_type="application/octet-stream",
        headers=headers,
    )
```

**Notes:**
- Uses `asyncio.to_thread()` for both blocking operations — consistent with all
  other mesh endpoints
- Error handling mirrors `/api/v1/mesh/field.binary`
- `_hash_uploaded_mesh_bytes` already exists in the recently updated `main.py`
- `_pack_mesh_binary` already exists at line ~546
- No new imports needed — all dependencies already imported

---

## Step 2: Frontend — Export `decodeBinaryMeshPacket` from `api.ts`

**File:** `frontend/src/lib/api.ts`

**Change:** Add `export` keyword to the existing `decodeBinaryMeshPacket`
function (currently at line 130, it is not exported).

```typescript
// Change line 130 from:
function decodeBinaryMeshPacket(buffer: ArrayBuffer): MeshPayloadBinary {

// To:
export function decodeBinaryMeshPacket(buffer: ArrayBuffer): MeshPayloadBinary {
```

---

## Step 3: Frontend — Add `preprocessUploadedMesh` to `api.ts`

**File:** `frontend/src/lib/api.ts`

**Where to insert:** After the `previewUploadedMeshField` function (around
line 548), before `submitUploadedFieldPreviewTelemetry`.

**What it does:**
- Sends the file + lattice params to `/api/v1/mesh/preprocess`
- Returns a `MeshPayloadBinary` (the raw outer mesh)
- Handles errors gracefully — failure is non-fatal (user can still click
  "Preview Field" which will do the work inline)

```typescript
export async function preprocessUploadedMesh(
  file: File,
  params: MeshWorkflowParams,
  computeBackend: ComputeBackend = "auto",
  voxelsPerLatticePeriod: number = 6,
  signal?: AbortSignal
): Promise<MeshPayloadBinary> {
  const body = new FormData();
  body.append("file", file, file.name);
  body.append("lattice_pitch", String(params.latticePitch));
  body.append("voxels_per_lattice_period", String(voxelsPerLatticePeriod));
  body.append("compute_backend", computeBackend);
  // shell_thickness, lattice_type, lattice_thickness, lattice_phase are NOT
  // needed for preprocess — only lattice_pitch and voxels_per_lattice_period
  // affect the resolution (and therefore the host field cache key).

  const response = await fetch(`${API_BASE}/api/v1/mesh/preprocess`, {
    method: "POST",
    body,
    signal
  });
  if (!response.ok) {
    await parseErrorResponse(response, "Mesh preprocess failed");
  }
  const packet = await response.arrayBuffer();
  return decodeBinaryMeshPacket(packet);
}
```

**Import update in `App.tsx`:** Add `preprocessUploadedMesh` to the import
from `./lib/api`.

---

## Step 4: Frontend — Replace `parseLocalMeshPreview` in `App.tsx`

**File:** `frontend/src/App.tsx`

### 4a. Update imports

Remove `parseLocalMeshPreview` import (line 14):
```typescript
// Remove:
import { parseLocalMeshPreview } from "./lib/localMeshPreview";

// Add to the api import:
import {
  ...
  preprocessUploadedMesh,
  ...
} from "./lib/api";
```

### 4b. Add preprocess abort controller ref

After the existing `meshFieldPreviewControllerRef` (around line 225), add:
```typescript
const preprocessControllerRef = useRef<AbortController | null>(null);
```

### 4c. Replace the file selection handler

**Current code** (lines 906–926):
```typescript
onChange={(event) => {
  const selected = event.target.files?.[0] ?? null;
  setMeshFile(selected);
  setField(null);
  setStats(null);
  setUploadedFieldPreviewTrace(null);
  setMeshCommitted(false);
  setError(null);
  if (!selected) {
    setMesh(null);
    return;
  }
  void parseLocalMeshPreview(selected)
    .then((payload) => {
      setMesh(payload);
    })
    .catch((parseError) => {
      setMesh(null);
      setError((parseError as Error).message);
    });
}}
```

**New code:**
```typescript
onChange={(event) => {
  const selected = event.target.files?.[0] ?? null;

  // Abort any in-flight preprocess for the previous file
  preprocessControllerRef.current?.abort();
  preprocessControllerRef.current = null;

  setMeshFile(selected);
  setField(null);
  setMesh(null);
  setStats(null);
  setUploadedFieldPreviewTrace(null);
  setMeshCommitted(false);
  setError(null);

  if (!selected) {
    return;
  }

  // Upload to backend immediately:
  //   1. Warms metadata + host field caches for the subsequent "Preview Field"
  //   2. Returns the raw outer mesh for immediate display in the viewer
  const controller = new AbortController();
  preprocessControllerRef.current = controller;
  setIsPreviewing(true);

  preprocessUploadedMesh(
    selected,
    meshWorkflowParams,
    computeBackend,
    voxelsPerLatticePeriod,
    controller.signal
  )
    .then((outerMesh) => {
      if (preprocessControllerRef.current !== controller) {
        return; // Superseded by a newer file selection
      }
      setMesh(outerMesh);
    })
    .catch((preprocessError) => {
      if ((preprocessError as Error).name === "AbortError") {
        return;
      }
      if (preprocessControllerRef.current !== controller) {
        return;
      }
      // Preprocess failure is non-fatal — user can still click "Preview Field"
      // which will run the full pipeline inline
      setMesh(null);
      setError((preprocessError as Error).message);
    })
    .finally(() => {
      if (preprocessControllerRef.current === controller) {
        preprocessControllerRef.current = null;
        setIsPreviewing(false);
      }
    });
}}
```

### 4d. Cleanup on unmount

Add to the existing cleanup `useEffect` (around line 706):
```typescript
useEffect(() => {
  return () => {
    preprocessControllerRef.current?.abort();
    cancelPendingMeshFieldPreview();
  };
}, [cancelPendingMeshFieldPreview]);
```

### 4e. Update `runMeshFieldPreview` dependencies

The `runMeshFieldPreview` callback (line 263) already depends on
`meshWorkflowParams`, `computeBackend`, `voxelsPerLatticePeriod` — no change
needed there.

However, the `onChange` handler now also uses `meshWorkflowParams`,
`computeBackend`, and `voxelsPerLatticePeriod`. Since `onChange` is an inline
function in JSX (not a `useCallback`), these are captured from the closure
correctly. No additional memoization needed.

---

## Step 5: Delete `localMeshPreview.ts`

**File:** `frontend/src/lib/localMeshPreview.ts`

Delete this file entirely. It is no longer imported anywhere after Step 4a.

---

## Step 6: No Changes to `types.ts`

The response from `/api/v1/mesh/preprocess` is decoded by
`decodeBinaryMeshPacket()` which returns `MeshPayloadBinary` — already defined
in `types.ts` at line 66. No new types needed.

---

## Step 7: Update Tests

### 7a. Backend test — `backend/tests/test_api.py`

Add a test for the new endpoint. The test should verify:
- Returns 200 with `application/octet-stream` content type
- Response body is a valid binary mesh packet (starts with `SDFMESH1`)
- Vertex count and face count headers are present
- A second identical request returns faster (cache hit — though this is hard
  to assert in unit tests without timing)

```python
def test_preprocess_uploaded_mesh_returns_binary_outer_mesh(client):
    """POST /api/v1/mesh/preprocess returns a binary mesh packet for the outer geometry."""
    obj_bytes = _tetra_obj_bytes()  # reuse existing fixture
    response = client.post(
        "/api/v1/mesh/preprocess",
        data={
            "lattice_pitch": "0.45",
            "voxels_per_lattice_period": "6",
            "compute_backend": "cpu",
            "field_storage_mode": "dense",
        },
        files={"file": ("tetra.obj", obj_bytes, "application/octet-stream")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert "X-SDF-Vertex-Count" in response.headers
    assert "X-SDF-Face-Count" in response.headers
    body = response.content
    assert body[:8] == b"SDFMESH1"
    vertex_count = int.from_bytes(body[8:12], "little")
    face_count = int.from_bytes(body[12:16], "little")
    assert vertex_count > 0
    assert face_count > 0


def test_preprocess_uploaded_mesh_rejects_invalid_file(client):
    """POST /api/v1/mesh/preprocess returns 400 for invalid mesh data."""
    response = client.post(
        "/api/v1/mesh/preprocess",
        data={"lattice_pitch": "0.45"},
        files={"file": ("bad.obj", b"not a mesh", "application/octet-stream")},
    )
    assert response.status_code == 400
```

### 7b. Frontend test — `frontend/src/App.test.tsx`

**Update the mock** to include `preprocessUploadedMesh`:

```typescript
const preprocessUploadedMesh = vi.fn();

vi.mock("./lib/api", () => ({
  ...existing mocks...,
  preprocessUploadedMesh: (...args: unknown[]) => preprocessUploadedMesh(...args),
}));
```

**In `beforeEach`**, set a default mock return value:
```typescript
preprocessUploadedMesh.mockResolvedValue({
  encoding: "mesh-f32-u32-binary-v1",
  vertex_count: 3,
  face_count: 1,
  vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
  indices: new Uint32Array([0, 1, 2]),
  normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1])
});
```

**Update the existing test** "renders uploaded source mesh locally right after
file selection" (line 461) — it currently tests that `parseLocalMeshPreview`
is called. With the new approach, it should test that `preprocessUploadedMesh`
is called and the mesh appears:

```typescript
it("calls preprocessUploadedMesh and shows outer mesh after file selection", async () => {
  render(<App />);
  fireEvent.click(screen.getByRole("tab", { name: "Mesh" }));

  const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", {
    type: "text/plain"
  });
  const fileInput = screen.getByLabelText("Mesh file upload") as HTMLInputElement;
  fireEvent.change(fileInput, { target: { files: [file] } });

  await waitFor(() => {
    expect(preprocessUploadedMesh).toHaveBeenCalledTimes(1);
  });

  // Verify the outer mesh is displayed
  await waitFor(() => {
    const latestViewerProps = viewerMock.mock.calls[viewerMock.mock.calls.length - 1]?.[0] as Record<string, unknown>;
    expect(latestViewerProps.mesh).not.toBeNull();
  });
});
```

**Add a test** for preprocess failure being non-fatal:
```typescript
it("shows error but allows Preview Field after preprocess failure", async () => {
  preprocessUploadedMesh.mockRejectedValueOnce(new Error("Preprocess failed"));

  render(<App />);
  fireEvent.click(screen.getByRole("tab", { name: "Mesh" }));

  const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", {
    type: "text/plain"
  });
  fireEvent.change(screen.getByLabelText("Mesh file upload"), { target: { files: [file] } });

  await waitFor(() => {
    expect(screen.getByText("Preprocess failed")).toBeInTheDocument();
  });

  // User can still click Preview Field
  previewUploadedMeshField.mockClear();
  fireEvent.click(screen.getByRole("button", { name: "Preview Field" }));
  await waitFor(() => {
    expect(previewUploadedMeshField).toHaveBeenCalledTimes(1);
  });
});
```

**Update the test** "does not auto-preview uploaded field when mesh preview
parameters change" (line 248) — it currently fires `parseLocalMeshPreview`
implicitly. With the new approach, `preprocessUploadedMesh` is called on file
selection. The test should mock it and verify `previewUploadedMeshField` is
still not called on param changes:

```typescript
it("does not auto-preview uploaded field when mesh preview parameters change", async () => {
  render(<App />);
  fireEvent.click(screen.getByRole("tab", { name: "Mesh" }));

  previewUploadedMesh.mockClear();
  previewUploadedMeshField.mockClear();

  const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
  const fileInput = screen.getByLabelText("Mesh file upload") as HTMLInputElement;
  fireEvent.change(fileInput, { target: { files: [file] } });

  // Wait for preprocess to complete
  await waitFor(() => expect(preprocessUploadedMesh).toHaveBeenCalledTimes(1));

  expect(previewUploadedMeshField).not.toHaveBeenCalled();
  expect(previewUploadedMesh).not.toHaveBeenCalled();

  // Changing params should NOT trigger a new preview
  fireEvent.change(screen.getByLabelText("Lattice pitch"), { target: { value: "0.55" } });
  fireEvent.change(screen.getByLabelText("Lattice thickness"), { target: { value: "0.12" } });
  // ... etc

  expect(previewUploadedMeshField).not.toHaveBeenCalled();
});
```

---

## Cache Invalidation Consideration

The preprocess endpoint warms the host field cache for a specific
`(file_hash, resolution, compute_backend, field_storage_mode)` key. The
`resolution` is derived from `lattice_pitch` and `voxels_per_lattice_period`.

If the user changes `lattice_pitch` or `voxels_per_lattice_period` **after**
selecting the file but **before** clicking "Preview Field", the resolution
changes and the host field cache misses. In this case, "Preview Field" runs
the full pipeline inline (same as today).

**Mitigation options (future work):**
- Re-trigger preprocess when `lattice_pitch` or `voxels_per_lattice_period`
  changes (with debouncing to avoid excessive uploads)
- Accept the cache miss — the user still benefits from the preprocess on the
  first "Preview Field" click with the original params

For the initial implementation, accept the cache miss on param changes. The
common case (user uploads file, clicks "Preview Field" with default params) is
fully optimized.

---

## Expected Performance Impact

| Scenario | Before | After |
|----------|--------|-------|
| File selection (169 MB STL) | 65s UI freeze | ~0s (async upload starts) |
| Time to raw geometry visible | 65s | Upload time + backend parse (~10–30s) |
| "Preview Field" click (same params) | 30–120s cold | ~2–5s (cache hit) |
| "Preview Field" click (changed params) | 30–120s cold | 30–120s cold (same as before) |
| UI responsiveness during processing | Frozen | Fully responsive |

---

## Implementation Order

1. **Backend** (`main.py`): Add the endpoint — self-contained, no frontend changes needed to test
2. **Frontend `api.ts`**: Export `decodeBinaryMeshPacket` + add `preprocessUploadedMesh`
3. **Frontend `App.tsx`**: Replace file selection handler
4. **Delete** `localMeshPreview.ts`
5. **Tests**: Backend API test + frontend App test updates
