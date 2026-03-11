import struct

import numpy as np
import pytest

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
