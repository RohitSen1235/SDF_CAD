import struct
from types import SimpleNamespace

import numpy as np
import pytest

from app import mesh_upload
from app.meshing import build_mesh
from app.mesh_upload import MeshUploadError, parse_mesh_bytes, validate_triangle_mesh


def _tetra_obj_bytes() -> bytes:
    return b"""
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
"""


def _tetra_ascii_stl_bytes() -> bytes:
    return b"""solid tetra
facet normal 0 0 -1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
facet normal 0 -1 0
  outer loop
    vertex 0 0 0
    vertex 0 0 1
    vertex 1 0 0
  endloop
endfacet
facet normal 1 1 1
  outer loop
    vertex 1 0 0
    vertex 0 0 1
    vertex 0 1 0
  endloop
endfacet
facet normal -1 0 0
  outer loop
    vertex 0 0 0
    vertex 0 1 0
    vertex 0 0 1
  endloop
endfacet
endsolid tetra
"""


def _cube_obj_bytes() -> bytes:
    return b"""
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""


def _single_tri_binary_stl_bytes() -> bytes:
    header = b"binary-stl-test".ljust(80, b" ")
    count = struct.pack("<I", 1)
    facet = bytearray(50)
    struct.pack_into("<3f", facet, 12, 0.0, 0.0, 0.0)
    struct.pack_into("<3f", facet, 24, 1.0, 0.0, 0.0)
    struct.pack_into("<3f", facet, 36, 0.0, 1.0, 0.0)
    return header + count + bytes(facet)


def test_parse_obj_triangulates_polygon_face() -> None:
    payload = b"""
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"""
    mesh = parse_mesh_bytes(payload, ".obj")
    assert mesh.faces.shape == (2, 3)


def test_parse_ascii_stl_and_validate_watertight() -> None:
    mesh = parse_mesh_bytes(_tetra_ascii_stl_bytes(), ".stl")
    validate_triangle_mesh(mesh)
    assert mesh.vertices.shape[0] >= 4
    assert mesh.faces.shape[0] == 4


def test_parse_binary_stl_single_triangle() -> None:
    mesh = parse_mesh_bytes(_single_tri_binary_stl_bytes(), ".stl")
    assert mesh.vertices.shape[0] == 3
    assert mesh.faces.shape[0] == 1


def test_validate_rejects_non_watertight_mesh() -> None:
    payload = b"""
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 2 3
"""
    mesh = parse_mesh_bytes(payload, ".obj")
    with pytest.raises(MeshUploadError, match="watertight"):
        validate_triangle_mesh(mesh)


def test_validate_accepts_closed_obj_mesh() -> None:
    mesh = parse_mesh_bytes(_tetra_obj_bytes(), ".obj")
    validate_triangle_mesh(mesh)
    assert np.all(np.isfinite(mesh.vertices))


def test_numba_rasterization_topology_equivalent_to_python() -> None:
    if not mesh_upload.NUMBA_AVAILABLE:
        pytest.skip("Numba is not available in this environment")

    mesh = parse_mesh_bytes(_tetra_obj_bytes(), ".obj")
    bounds = mesh_upload._build_bounds(mesh)
    resolution = 48
    mins = np.asarray([bounds[0][0], bounds[1][0], bounds[2][0]], dtype=np.float64)
    scales = np.asarray(
        [
            (resolution - 1) / (bounds[0][1] - bounds[0][0]),
            (resolution - 1) / (bounds[1][1] - bounds[1][0]),
            (resolution - 1) / (bounds[2][1] - bounds[2][0]),
        ],
        dtype=np.float64,
    )
    verts_grid = np.ascontiguousarray((mesh.vertices - mins) * scales, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)

    py_surface = mesh_upload._rasterize_surface_python(verts_grid, faces, resolution).astype(bool)
    nb_surface = mesh_upload._rasterize_surface_numba(verts_grid, faces, resolution).astype(bool)
    union = np.logical_or(py_surface, nb_surface)
    if not np.any(union):
        raise AssertionError("Expected non-empty rasterized surface for comparison")
    overlap = float(np.count_nonzero(np.logical_and(py_surface, nb_surface))) / float(np.count_nonzero(union))
    assert overlap >= 0.995


def test_voxelize_falls_back_when_numba_kernel_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = parse_mesh_bytes(_tetra_obj_bytes(), ".obj")
    bounds = mesh_upload._build_bounds(mesh)
    called = {"python": False}
    python_impl = mesh_upload._rasterize_surface_python

    def fail_numba(_verts_grid: np.ndarray, _faces: np.ndarray, _resolution: int) -> np.ndarray:
        raise RuntimeError("numba failure")

    def track_python(verts_grid: np.ndarray, faces: np.ndarray, resolution: int) -> np.ndarray:
        called["python"] = True
        return python_impl(verts_grid, faces, resolution)

    monkeypatch.setattr(mesh_upload, "_rasterize_surface_numba", fail_numba)
    monkeypatch.setattr(mesh_upload, "_rasterize_surface_python", track_python)
    monkeypatch.setattr(mesh_upload, "NUMBA_AVAILABLE", True)

    filled = mesh_upload._voxelize_and_fill(mesh, bounds, resolution=40)
    assert called["python"] is True
    assert np.any(filled)


def test_build_host_field_defaults_to_dense_host_sdf_metadata() -> None:
    mesh = parse_mesh_bytes(_tetra_obj_bytes(), ".obj")
    host = mesh_upload.build_host_field(mesh, resolution=48)
    assert host.host_sdf.shape == (48, 48, 48)
    assert host.host_sdf.dtype == np.float32
    assert host.block_size is None
    assert host.active_blocks is None


def test_octree_sparse_host_sdf_returns_active_blocks_for_large_sparse_volume() -> None:
    resolution = 128
    occupancy = np.zeros((resolution, resolution, resolution), dtype=bool)
    occupancy[56:72, 56:72, 56:72] = True
    bounds = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

    host_sdf, block_size, active_blocks, sparse_background_value, decision_reason, host_backend = mesh_upload._build_host_sdf_octree_sparse(
        None,
        occupancy,
        bounds,
        resolution,
    )

    assert host_sdf.shape == (resolution, resolution, resolution)
    assert block_size is not None
    assert active_blocks is not None
    assert len(active_blocks) > 0
    assert sparse_background_value is not None
    assert decision_reason.startswith("sparse_selected") or decision_reason.startswith("dense_sparse")
    assert host_backend == "cpu"
    assert host_sdf[0, 0, 0] > 0.0
    assert host_sdf[63, 63, 63] < 0.0


def test_octree_sparse_host_sdf_scan_conversion_matches_cube_face_distance() -> None:
    mesh = parse_mesh_bytes(_cube_obj_bytes(), ".obj")
    bounds = [[-2.0, 3.0], [-2.0, 3.0], [-2.0, 3.0]]
    resolution = 128
    occupancy = mesh_upload._voxelize_and_fill(mesh, bounds, resolution)
    host_sdf, block_size, active_blocks, sparse_background_value, decision_reason, host_backend = mesh_upload._build_host_sdf_octree_sparse(
        mesh,
        occupancy,
        bounds,
        resolution,
    )

    x_axis = np.linspace(bounds[0][0], bounds[0][1], resolution, dtype=np.float64)
    y_axis = np.linspace(bounds[1][0], bounds[1][1], resolution, dtype=np.float64)
    z_axis = np.linspace(bounds[2][0], bounds[2][1], resolution, dtype=np.float64)

    ix_inside = int(np.argmin(np.abs(x_axis - 0.9)))
    ix_outside = int(np.argmin(np.abs(x_axis - 1.2)))
    iy = int(np.argmin(np.abs(y_axis - 0.5)))
    iz = int(np.argmin(np.abs(z_axis - 0.5)))

    inside_point = np.array([x_axis[ix_inside], y_axis[iy], z_axis[iz]], dtype=np.float64)
    outside_point = np.array([x_axis[ix_outside], y_axis[iy], z_axis[iz]], dtype=np.float64)
    spacing = float(np.max(mesh_upload._bounds_spacing(bounds, resolution)))

    expected_inside = -(1.0 - inside_point[0])
    expected_outside = outside_point[0] - 1.0

    assert block_size is not None
    assert active_blocks is not None
    assert len(active_blocks) > 0
    assert sparse_background_value is not None
    assert decision_reason.startswith("sparse_selected")
    assert host_backend == "cpu"
    assert host_sdf[ix_inside, iy, iz] < 0.0
    assert host_sdf[ix_outside, iy, iz] > 0.0
    assert abs(float(host_sdf[ix_inside, iy, iz]) - expected_inside) <= (1.5 * spacing)
    assert abs(float(host_sdf[ix_outside, iy, iz]) - expected_outside) <= (1.5 * spacing)


def test_build_host_field_auto_short_circuits_before_triangle_candidate_work(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = parse_mesh_bytes(_cube_obj_bytes(), ".obj")
    called = {"candidates": 0}

    def fail_if_called(*_args, **_kwargs):
        called["candidates"] += 1
        raise AssertionError("triangle candidate generation should not run for this auto-gated case")

    monkeypatch.setattr(mesh_upload, "_triangle_scan_convert_candidates", fail_if_called)

    host = mesh_upload.build_host_field(mesh, resolution=128, field_storage_mode="auto")

    assert called["candidates"] == 0
    assert host.field_storage_mode == "dense"
    assert host.host_build_strategy == "dense"
    assert host.host_decision_reason.startswith("dense_auto_gate")


def test_build_host_field_explicit_sparse_attempts_scan_conversion_even_when_auto_gate_would_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = parse_mesh_bytes(_cube_obj_bytes(), ".obj")
    called = {"candidates": 0}
    original = mesh_upload._triangle_scan_convert_candidates

    def track_candidates(*args, **kwargs):
        called["candidates"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mesh_upload, "_triangle_scan_convert_candidates", track_candidates)
    monkeypatch.setattr(
        mesh_upload,
        "_estimate_sparse_profitability",
        lambda *_args, **_kwargs: (False, "dense_auto_gate_near_ratio", 1.0, 10_000_000, 9999.0),
    )

    host = mesh_upload.build_host_field(mesh, resolution=128, field_storage_mode="octree_sparse")

    assert called["candidates"] > 0
    assert host.host_decision_reason.startswith("sparse_selected") or host.host_decision_reason.startswith("dense_sparse")


def test_build_host_field_populates_sparse_metadata_when_sparse_path_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = parse_mesh_bytes(_tetra_obj_bytes(), ".obj")
    resolution = 128
    synthetic = np.zeros((resolution, resolution, resolution), dtype=bool)
    synthetic[56:72, 56:72, 56:72] = True

    monkeypatch.setattr(mesh_upload, "_voxelize_and_fill", lambda *_args, **_kwargs: synthetic)
    monkeypatch.setattr(mesh_upload, "_build_bounds", lambda *_args, **_kwargs: [[-2.0, 3.0], [-2.0, 3.0], [-2.0, 3.0]])

    host = mesh_upload.build_host_field(mesh, resolution=resolution)

    assert host.host_sdf.shape == (resolution, resolution, resolution)
    assert host.block_size is not None
    assert host.active_blocks is not None
    assert len(host.active_blocks) > 0
    assert host.field_storage_mode == "octree_sparse"
    assert host.sparse_bricks is not None


def test_build_host_field_supports_explicit_dense_mode() -> None:
    mesh = parse_mesh_bytes(_tetra_obj_bytes(), ".obj")
    host = mesh_upload.build_host_field(mesh, resolution=96, field_storage_mode="dense")
    assert host.host_sdf.dtype == np.float32
    assert host.field_storage_mode == "dense"
    assert host.block_size is None
    assert host.active_blocks is None


def test_build_host_field_reports_gpu_backend_when_cuda_dense_path_is_selected(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mesh = parse_mesh_bytes(_tetra_obj_bytes(), ".obj")
    bounds = mesh_upload._build_bounds(mesh)
    occupancy = mesh_upload._voxelize_and_fill(mesh, bounds, resolution=48)
    cpu_baseline = mesh_upload._build_host_sdf_dense_cpu(occupancy, bounds, 48)

    monkeypatch.setattr(mesh_upload, "_resolve_compute_backend", lambda _requested: "cuda")
    monkeypatch.setattr(
        mesh_upload,
        "_build_host_sdf_dense_cuda",
        lambda occ, bnds, res: mesh_upload._build_host_sdf_dense_cpu(occ, bnds, res),
    )

    host = mesh_upload.build_host_field(
        mesh,
        resolution=48,
        compute_backend="cuda",
        field_storage_mode="dense",
    )

    captured = capsys.readouterr().out
    assert "Host field SDF computation used GPU backend" in captured
    assert host.host_compute_backend == "cuda"
    assert np.allclose(host.host_sdf, cpu_baseline)


def test_build_host_field_reports_cpu_backend_when_cuda_dense_path_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mesh = parse_mesh_bytes(_tetra_obj_bytes(), ".obj")
    bounds = mesh_upload._build_bounds(mesh)
    occupancy = mesh_upload._voxelize_and_fill(mesh, bounds, resolution=48)
    cpu_baseline = mesh_upload._build_host_sdf_dense_cpu(occupancy, bounds, 48)

    monkeypatch.setattr(mesh_upload, "_resolve_compute_backend", lambda _requested: "cuda")

    def fail_cuda(*_args, **_kwargs) -> np.ndarray:
        raise RuntimeError("cuda failed")

    monkeypatch.setattr(
        mesh_upload,
        "_build_host_sdf_dense_cuda",
        fail_cuda,
    )

    host = mesh_upload.build_host_field(
        mesh,
        resolution=48,
        compute_backend="cuda",
        field_storage_mode="dense",
    )

    captured = capsys.readouterr().out
    assert "Host field SDF computation used CPU backend" in captured
    assert host.host_compute_backend == "cpu"
    assert np.allclose(host.host_sdf, cpu_baseline)


def test_uploaded_host_field_threads_compute_backend_to_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main as main_module

    main_module.uploaded_host_field_cache.clear()

    mesh = parse_mesh_bytes(_tetra_obj_bytes(), ".obj")
    observed: dict[str, str] = {}

    def fake_build_host_field(
        parsed,
        resolution: int,
        compute_backend: str = "auto",
        field_storage_mode: str = "auto",
    ):
        observed["compute_backend"] = compute_backend
        return SimpleNamespace(
            mesh=parsed,
            bounds=[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
            host_sdf=np.zeros((resolution, resolution, resolution), dtype=np.float32),
            host_compute_backend="cuda",
            field_storage_mode="dense",
            block_size=None,
            active_blocks=None,
            sparse_background_value=None,
            sparse_bricks=None,
            octree_node_min=None,
            octree_node_max=None,
            octree_node_depth=None,
            octree_node_kind=None,
            host_build_strategy="dense",
            host_decision_reason="dense_requested",
        )

    monkeypatch.setattr(main_module, "build_host_field", fake_build_host_field)

    result = main_module._resolve_uploaded_host_field(
        file_bytes=_tetra_obj_bytes(),
        extension=".obj",
        resolution=48,
        compute_backend="cuda",
        parsed=mesh,
        field_storage_mode="dense",
    )

    assert observed["compute_backend"] == "cuda"
    assert result.host_compute_backend == "cuda"


def _legacy_compose_field(
    host_sdf: np.ndarray,
    bounds: list[list[float]],
    *,
    shell_thickness: float,
    lattice_type: str,
    lattice_pitch: float,
    lattice_thickness: float,
    lattice_phase: float,
) -> np.ndarray:
    host64 = np.asarray(host_sdf, dtype=np.float64)
    shell_field = np.maximum(host64, -host64 - abs(shell_thickness))
    cavity = host64 + abs(shell_thickness)
    lattice_clipped = cavity.copy()

    mask = cavity < 0.0
    if np.any(mask):
        resolution = host64.shape[0]
        x_axis = np.linspace(bounds[0][0], bounds[0][1], resolution, dtype=np.float64)
        y_axis = np.linspace(bounds[1][0], bounds[1][1], resolution, dtype=np.float64)
        z_axis = np.linspace(bounds[2][0], bounds[2][1], resolution, dtype=np.float64)
        ix, iy, iz = np.nonzero(mask)
        lattice_values = mesh_upload._tpms_field(
            x_axis[ix],
            y_axis[iy],
            z_axis[iz],
            lattice_type=lattice_type,
            lattice_pitch=lattice_pitch,
            lattice_thickness=lattice_thickness,
            lattice_phase=lattice_phase,
        )
        lattice_clipped[mask] = np.maximum(lattice_values, cavity[mask])

    return np.minimum(shell_field, lattice_clipped)


def test_uploaded_compose_normalizes_to_float32_and_preserves_mesh_parity() -> None:
    mesh = parse_mesh_bytes(_tetra_obj_bytes(), ".obj")
    host = mesh_upload.build_host_field(mesh, resolution=48, field_storage_mode="dense")

    field = mesh_upload.compose_hollow_lattice_field(
        host.host_sdf,
        host.bounds,
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
    )
    legacy_field = _legacy_compose_field(
        host.host_sdf,
        host.bounds,
        shell_thickness=0.08,
        lattice_type="gyroid",
        lattice_pitch=0.45,
        lattice_thickness=0.09,
        lattice_phase=0.0,
    )

    spacing = np.array(mesh_upload._bounds_spacing(host.bounds, int(field.shape[0])), dtype=np.float64)
    max_spacing = float(np.max(spacing))

    assert field.dtype == np.float32
    assert np.max(np.abs(field.astype(np.float64) - legacy_field)) <= (0.5 * max_spacing)

    mesh_new = build_mesh(field, host.bounds, backend="cpu")
    mesh_legacy = build_mesh(np.asarray(legacy_field, dtype=np.float32), host.bounds, backend="cpu")

    new_min = np.min(mesh_new.vertices, axis=0)
    new_max = np.max(mesh_new.vertices, axis=0)
    legacy_min = np.min(mesh_legacy.vertices, axis=0)
    legacy_max = np.max(mesh_legacy.vertices, axis=0)

    assert np.max(np.abs(new_min - legacy_min)) <= max_spacing
    assert np.max(np.abs(new_max - legacy_max)) <= max_spacing

    tri_delta = abs(int(mesh_new.faces.shape[0]) - int(mesh_legacy.faces.shape[0])) / float(
        max(1, int(mesh_legacy.faces.shape[0]))
    )
    assert tri_delta <= 0.02
