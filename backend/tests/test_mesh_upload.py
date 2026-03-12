import struct

import numpy as np
import pytest

from app import mesh_upload
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
