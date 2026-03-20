from app import main as main_module


def test_main_has_no_local_compute_vertex_normals() -> None:
    assert not hasattr(main_module, "_compute_vertex_normals")
