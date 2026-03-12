import numpy as np
import pytest
from app import meshing


def test_mesh_backend_auto_uses_runtime_cuda_check(monkeypatch) -> None:
    monkeypatch.setattr(meshing, "_cuda_runtime_ready", lambda: False)
    assert meshing._resolve_mesh_backend("auto") == "cpu"
    assert meshing._resolve_mesh_backend("cuda") == "cpu"

    monkeypatch.setattr(meshing, "_cuda_runtime_ready", lambda: True)
    assert meshing._resolve_mesh_backend("auto") == "cuda"
    assert meshing._resolve_mesh_backend("cuda") == "cuda"
    assert meshing._resolve_mesh_backend("cpu") == "cpu"


def test_cuda_kernel_does_not_depend_on_host_math_header() -> None:
    assert "#include <math.h>" not in meshing._MC_GENERATE_KERNEL
    assert "fabsf(" not in meshing._MC_GENERATE_KERNEL
    assert "sqrtf(" not in meshing._MC_GENERATE_KERNEL


def test_explicit_cuda_backend_does_not_silently_fallback(monkeypatch) -> None:
    field = np.array(
        [[[-1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
        dtype=np.float32,
    )
    bounds = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

    monkeypatch.setattr(meshing, "_resolve_mesh_backend", lambda _: "cuda")
    monkeypatch.setattr(meshing, "_mesh_single_cuda", lambda _f, _b: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(meshing.MeshingError, match="CUDA meshing failed"):
        meshing.build_mesh_with_backend(field, bounds, backend="cuda")


def test_auto_backend_can_fallback_to_cpu_when_cuda_fails(monkeypatch) -> None:
    field = np.array(
        [[[-1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
        dtype=np.float32,
    )
    bounds = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
    fake_mesh = meshing.MeshData(
        vertices=np.zeros((0, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int32),
        normals=np.zeros((0, 3), dtype=np.float64),
    )

    monkeypatch.setattr(meshing, "_resolve_mesh_backend", lambda _: "cuda")
    monkeypatch.setattr(meshing, "_mesh_single_cuda", lambda _f, _b: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(meshing, "_mesh_single_cpu", lambda _f, _b: fake_mesh)

    mesh, backend = meshing.build_mesh_with_backend(field, bounds, backend="auto")
    assert mesh is fake_mesh
    assert backend == "cpu"
