# Plan: Lattice Parameters — MM Units, Min-Thickness Feedback, Half-Thickness Clarification

## Background

All lattice parameters (`lattice_pitch`, `shell_thickness`, `lattice_thickness`) are already in **world units**, which equal **mm** when the uploaded mesh is in mm. The backend math is unit-agnostic — no backend changes are needed. All changes are confined to the frontend UI.

---

## Changes Required

### 1. Update Default State Values to mm-Scale Defaults

**File:** [`frontend/src/App.tsx`](../frontend/src/App.tsx:191)

Current defaults are calibrated for a normalised unit-cube world. For real mm-scale parts (50–200 mm), they are far too small:

| State variable | Current default | New default | Rationale |
|---|---|---|---|
| `meshShellThickness` | `0.08` | `2.0` | 2 mm outer shell — printable on FDM/SLA |
| `meshLatticePitch` | `0.45` | `5.0` | 5 mm unit cell — visible and printable |
| `meshLatticeThickness` | `0.09` | `0.5` | 0.5 mm half-thickness → 1 mm strut total |

```tsx
// Before
const [meshShellThickness, setMeshShellThickness] = useState(0.08);
const [meshLatticePitch, setMeshLatticePitch] = useState(0.45);
const [meshLatticeThickness, setMeshLatticeThickness] = useState(0.09);

// After
const [meshShellThickness, setMeshShellThickness] = useState(2.0);
const [meshLatticePitch, setMeshLatticePitch] = useState(5.0);
const [meshLatticeThickness, setMeshLatticeThickness] = useState(0.5);
```

---

### 2. Rename "Lattice pitch" Label to "Unit cell size (mm)"

**File:** [`frontend/src/App.tsx`](../frontend/src/App.tsx:1077)

The term "pitch" is opaque to non-engineers. "Unit cell size" is the physical meaning — the edge length of one repeating lattice cell.

```tsx
// Before
<span>Lattice pitch (world units)</span>

// After
<span>Unit cell size (mm)</span>
```

Also update the `step` and `min` attributes to be mm-appropriate:

```tsx
// Before
<input type="number" step={0.01} min={0.001} value={meshLatticePitch} ... />

// After
<input type="number" step={0.5} min={0.5} value={meshLatticePitch} ... />
```

---

### 3. Add Computed Minimum Shell Thickness Feedback

**File:** [`frontend/src/App.tsx`](../frontend/src/App.tsx:1051)

Add a new computed value `minShellThickness` derived from the voxel spacing formula:

```
minShellThickness = 2 × (lattice_pitch / voxels_per_lattice_period)
```

This is the minimum shell thickness that can be faithfully represented at the current grid resolution. If the user sets `shell_thickness` below this value, the lattice will protrude through the shell.

**New helper function** (add near [`computeRequiredResolution()`](../frontend/src/App.tsx:161)):

```tsx
function computeMinShellThickness(latticePitch: number, voxelsPerPeriod: number): number {
  // Shell must be at least 2 voxels thick to be faithfully represented.
  // Voxel spacing ≈ lattice_pitch / voxels_per_period
  return 2.0 * (latticePitch / voxelsPerPeriod);
}
```

**New computed value** (add inside the `App` component, near [`meshMemoryRisk`](../frontend/src/App.tsx:246)):

```tsx
const minShellThickness = useMemo(
  () => computeMinShellThickness(meshLatticePitch, voxelsPerLatticePeriod),
  [meshLatticePitch, voxelsPerLatticePeriod]
);
const shellTooThin = meshShellThickness < minShellThickness;
```

**UI feedback** (add below the shell thickness `<input>` at line 1060):

```tsx
<label className="slider-row">
  <span>Shell thickness (mm)</span>
  <input
    type="number"
    step={0.5}
    min={0.1}
    value={meshShellThickness}
    aria-label="Shell thickness"
    onChange={(event) => setMeshShellThickness(Number(event.target.value))}
  />
</label>
<p className="muted">
  Min recommended shell thickness at current settings:{" "}
  <strong>{minShellThickness.toFixed(2)} mm</strong>
  {" "}(= 2 × unit cell size / sampling quality)
</p>
{shellTooThin ? (
  <p className="warning">
    Shell thickness {meshShellThickness.toFixed(2)} mm is below the minimum{" "}
    {minShellThickness.toFixed(2)} mm. The lattice may protrude outside the shell surface.
    Increase shell thickness or reduce unit cell size.
  </p>
) : null}
```

---

### 4. Rename "Lattice thickness" to "Lattice half-thickness (mm)" and Add Hint

**File:** [`frontend/src/App.tsx`](../frontend/src/App.tsx:1088)

`lattice_thickness` is the **half-thickness** of the TPMS strut walls. The total strut width is `2 × lattice_thickness`. This must be made explicit.

```tsx
// Before
<label className="slider-row">
  <span>Lattice thickness</span>
  <input type="number" step={0.01} min={0.001} value={meshLatticeThickness} ... />
</label>

// After
<label className="slider-row">
  <span>Lattice half-thickness (mm)</span>
  <input type="number" step={0.1} min={0.05} value={meshLatticeThickness} ... />
</label>
<p className="muted">
  Strut total width = 2 × half-thickness ={" "}
  <strong>{(meshLatticeThickness * 2).toFixed(2)} mm</strong>
</p>
```

---

### 5. Add Computed Minimum Lattice Thickness Feedback

**File:** [`frontend/src/App.tsx`](../frontend/src/App.tsx:1088)

The minimum resolvable lattice half-thickness is one voxel spacing:

```
minLatticeThickness = lattice_pitch / voxels_per_lattice_period
```

Below this, the strut walls are thinner than one voxel and will not render correctly.

**New computed value** (add near `minShellThickness`):

```tsx
const minLatticeThickness = useMemo(
  () => meshLatticePitch / voxelsPerLatticePeriod,
  [meshLatticePitch, voxelsPerLatticePeriod]
);
const latticeTooThin = meshLatticeThickness < minLatticeThickness;
```

**UI feedback** (add below the lattice half-thickness `<input>`):

```tsx
<p className="muted">
  Min resolvable half-thickness at current settings:{" "}
  <strong>{minLatticeThickness.toFixed(2)} mm</strong>
  {" "}(= 1 voxel = unit cell size / sampling quality)
</p>
{latticeTooThin ? (
  <p className="warning">
    Lattice half-thickness {meshLatticeThickness.toFixed(2)} mm is below the minimum{" "}
    {minLatticeThickness.toFixed(2)} mm. Strut walls may not render correctly.
    Increase half-thickness or reduce unit cell size.
  </p>
) : null}
```

---

### 6. Update Shell Thickness Label

**File:** [`frontend/src/App.tsx`](../frontend/src/App.tsx:1052)

```tsx
// Before
<span>Shell thickness</span>

// After
<span>Shell thickness (mm)</span>
```

Also update `step` and `min` to mm-appropriate values:

```tsx
// Before
<input type="number" step={0.01} min={0.001} ... />

// After
<input type="number" step={0.5} min={0.1} ... />
```

---

### 7. Update App.test.tsx to Reflect New Defaults

**File:** [`frontend/src/App.test.tsx`](../frontend/src/App.test.tsx:395)

The test at line 395 checks that `previewUploadedMeshField` is called with `latticePitch: 0.45` and `latticeThickness: 0.09`. These must be updated to the new defaults:

```tsx
// Before
expect(previewUploadedMeshField.mock.calls[0]?.[1]).toMatchObject({
  latticePitch: 0.45,
  latticeThickness: 0.09,
  latticePhase: 0
});

// After
expect(previewUploadedMeshField.mock.calls[0]?.[1]).toMatchObject({
  latticePitch: 5.0,
  latticeThickness: 0.5,
  latticePhase: 0
});
```

Also check for any other tests that reference the old default values and update them.

---

### 8. Update Documentation

**File:** [`docs/dsl-field-primer.md`](../docs/dsl-field-primer.md)

Add a section explaining:
- Parameters are in the same units as the uploaded mesh (mm if mesh is in mm)
- `lattice_pitch` = unit cell size (edge length of one repeating cell)
- `lattice_thickness` = **half**-thickness of strut walls; total strut width = `2 × lattice_thickness`
- `shell_thickness` minimum = `2 × lattice_pitch / voxels_per_lattice_period`
- `lattice_thickness` minimum = `lattice_pitch / voxels_per_lattice_period`

---

## Summary of All File Changes

| File | Change type | Description |
|---|---|---|
| [`frontend/src/App.tsx`](../frontend/src/App.tsx) | Modify | New defaults, new labels, new helper functions, new feedback UI |
| [`frontend/src/App.test.tsx`](../frontend/src/App.test.tsx) | Modify | Update expected default values in tests |
| [`docs/dsl-field-primer.md`](../docs/dsl-field-primer.md) | Modify | Add mm-units and parameter meaning documentation |

**No backend changes required.** The math is already unit-correct.

---

## Derived Formulas Reference

```
voxel_spacing = lattice_pitch / voxels_per_lattice_period

min_shell_thickness = 2 × voxel_spacing
                    = 2 × lattice_pitch / voxels_per_lattice_period

min_lattice_half_thickness = 1 × voxel_spacing
                           = lattice_pitch / voxels_per_lattice_period

total_strut_width = 2 × lattice_half_thickness

grid_resolution = ceil((mesh_span × 1.24 / lattice_pitch) × voxels_per_lattice_period)
```

### Example for a 100 mm part, 5 mm cell, 6 voxels/period:

```
voxel_spacing         = 5.0 / 6 = 0.83 mm
min_shell_thickness   = 2 × 0.83 = 1.67 mm  → recommend ≥ 2.0 mm
min_lattice_half_t    = 0.83 mm              → recommend ≥ 1.0 mm (2 mm strut)
grid_resolution       = ceil(100 × 1.24 / 5 × 6) = ceil(148.8) = 149³
```
