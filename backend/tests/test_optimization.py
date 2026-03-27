from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app import optimization
from app import worker_tasks
from app.models import (
    ConstraintRegion,
    FieldPayload,
    LoadRegion,
    OptimizationHistoryEntry,
    SelectionPoint,
    StructuralMaterial,
    StructuralOptimizationConfig,
    StructuralOptimizationIterationResult,
    StructuralOptimizationIterationWebhookRequest,
    StructuralOptimizationRequest,
    StructuralOptimizationResultResponse,
)
from app.optimization import MeshUploadError, PreparedOptimizationDomain


EMPTY_MESH_PAYLOAD = {
    "encoding": "mesh-f32-u32-base64-v1",
    "vertex_count": 0,
    "face_count": 0,
    "vertices_b64": "",
    "indices_b64": "",
    "normals_b64": "",
}


def _field(shape: tuple[int, int, int]) -> FieldPayload:
    return FieldPayload(
        resolution_xyz=list(shape),
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        data="",
    )


def _request(config: StructuralOptimizationConfig) -> StructuralOptimizationRequest:
    return StructuralOptimizationRequest(
        design_space_file_name="design.stl",
        design_space_file_data_base64="AA==",
        non_design_space_file_name="keep.stl",
        non_design_space_file_data_base64="AA==",
        compute_backend="cpu",
        mesh_backend="cpu",
        execution_mode="inline",
        constraints=[ConstraintRegion(points=[SelectionPoint(point_xyz=[0.0, 0.0, 0.0])])],
        loads=[
            LoadRegion(
                points=[SelectionPoint(point_xyz=[1.0, 0.0, 0.0])],
                direction_xyz=[1.0, 0.0, 0.0],
                magnitude=1.0,
            )
        ],
        material=StructuralMaterial(),
        config=config,
    )


def test_element_stiffness_is_symmetric_and_positive_semidefinite() -> None:
    spacing = (1.0, 1.0, 1.0)
    material = StructuralMaterial()
    ke = optimization._build_element_stiffness_matrix(spacing, material)

    assert ke.shape == (24, 24)
    assert np.allclose(ke, ke.T, atol=1e-8)
    eigenvalues = np.linalg.eigvalsh(ke)
    assert np.min(eigenvalues) >= -1e-6


def test_matrix_free_operator_returns_zero_for_zero_displacement() -> None:
    material = StructuralMaterial()
    density = np.ones((2, 2, 2), dtype=np.float32)
    stiffness = optimization._build_element_stiffness_matrix((1.0, 1.0, 1.0), material)
    blocks = [[np.asarray(block, dtype=np.float32) for block in row] for row in optimization._element_stiffness_blocks(stiffness)]
    displacement = np.zeros((3, 3, 3, 3), dtype=np.float32)
    fixed = (
        np.zeros((3, 3, 3), dtype=bool),
        np.zeros((3, 3, 3), dtype=bool),
        np.zeros((3, 3, 3), dtype=bool),
    )

    result = optimization._apply_structured_fem_operator(displacement, density, blocks, fixed, np)

    assert result.shape == displacement.shape
    assert np.allclose(result, 0.0)


def test_oc_update_respects_target_volume_fraction_and_bounds() -> None:
    density = np.full((3, 3, 3), 0.6, dtype=np.float32)
    design_mask = np.ones_like(density, dtype=bool)
    sensitivities = -np.linspace(0.5, 2.0, density.size, dtype=np.float32).reshape(density.shape)
    config = StructuralOptimizationConfig(
        target_volume_fraction=0.35,
        min_density=1e-3,
        oc_move_limit=1.0,
    )

    updated = optimization._update_design_density(density, sensitivities, design_mask, config)

    assert np.all(updated >= config.min_density)
    assert np.all(updated <= 1.0)
    assert abs(float(np.mean(updated[design_mask])) - config.target_volume_fraction) < 5e-3


def test_filter_sensitivities_smooths_design_values() -> None:
    density = np.ones((5, 5, 5), dtype=np.float32)
    design_mask = np.ones_like(density, dtype=bool)
    sensitivities = np.zeros_like(density, dtype=np.float32)
    sensitivities[2, 2, 2] = -10.0

    filtered = optimization._filter_sensitivities(sensitivities, density, design_mask, filter_radius_voxels=1.5)

    assert filtered[2, 2, 2] > sensitivities[2, 2, 2]
    assert np.count_nonzero(filtered) > 1


def test_structural_optimization_reports_density_converged(monkeypatch: pytest.MonkeyPatch) -> None:
    design = np.ones((3, 2, 2), dtype=bool)
    keep = np.zeros_like(design)
    support_nodes = np.zeros((4, 3, 3), dtype=bool)
    support_nodes[0, 0, 0] = True
    load_nodes = np.zeros((4, 3, 3), dtype=bool)
    load_nodes[-1, 0, 0] = True

    prepared = PreparedOptimizationDomain(
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        resolution_xyz=(3, 2, 2),
        design_mask=design,
        keep_mask=keep,
        diagnostics=[],
        design_host_sdf=np.zeros((3, 2, 2), dtype=np.float32),
        keep_host_sdf=np.zeros((3, 2, 2), dtype=np.float32),
        node_resolution_xyz=(4, 3, 3),
    )

    monkeypatch.setattr(optimization, "_decode_mesh_upload", lambda *_args, **_kwargs: (SimpleNamespace(), ".stl"))
    monkeypatch.setattr(optimization, "prepare_structural_domain", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        optimization,
        "_build_force_and_constraint_masks",
        lambda *_args, **_kwargs: (
            np.zeros(prepared.node_resolution_xyz, dtype=np.float32),
            np.zeros(prepared.node_resolution_xyz, dtype=np.float32),
            np.zeros(prepared.node_resolution_xyz, dtype=np.float32),
            support_nodes.copy(),
            support_nodes.copy(),
            support_nodes.copy(),
            support_nodes.copy(),
            [load_nodes.copy()],
        ),
    )
    monkeypatch.setattr(
        optimization,
        "_cg_solve_elasticity",
        lambda *_args, **_kwargs: (
            np.zeros(prepared.node_resolution_xyz, dtype=np.float32),
            np.zeros(prepared.node_resolution_xyz, dtype=np.float32),
            np.zeros(prepared.node_resolution_xyz, dtype=np.float32),
            "cpu",
        ),
    )
    monkeypatch.setattr(
        optimization,
        "_compute_element_compliance_energy",
        lambda *_args, **_kwargs: np.ones(prepared.resolution_xyz, dtype=np.float32),
    )
    monkeypatch.setattr(
        optimization,
        "_compute_element_strain_stress",
        lambda *_args, **_kwargs: (
            np.ones(prepared.resolution_xyz, dtype=np.float32),
            np.ones(prepared.resolution_xyz, dtype=np.float32),
            np.ones(prepared.resolution_xyz, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        optimization,
        "_build_iteration_mesh",
        lambda *_args, **_kwargs: (EMPTY_MESH_PAYLOAD, "cpu"),
    )
    monkeypatch.setattr(
        optimization,
        "_update_design_density",
        lambda density, *_args, **_kwargs: density,
    )

    result = optimization.run_structural_optimization(
        _request(
                StructuralOptimizationConfig(
                    max_iterations=8,
                    target_volume_fraction=0.35,
                    optimization_tolerance=1e-3,
                    density_iso_threshold=0.2,
                )
            )
        )

    assert result.stop_reason == "density_converged"
    assert result.final_iteration.density_field is not None
    assert result.final_iteration.stress_field is not None


def test_structural_optimization_rejects_non_finite_iteration_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    design = np.ones((2, 2, 2), dtype=bool)
    keep = np.zeros_like(design)
    support_nodes = np.zeros((3, 3, 3), dtype=bool)
    support_nodes[0, 0, 0] = True
    load_nodes = np.zeros((3, 3, 3), dtype=bool)
    load_nodes[-1, -1, -1] = True

    prepared = PreparedOptimizationDomain(
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        resolution_xyz=(2, 2, 2),
        design_mask=design,
        keep_mask=keep,
        diagnostics=[],
        design_host_sdf=np.zeros((2, 2, 2), dtype=np.float32),
        keep_host_sdf=np.zeros((2, 2, 2), dtype=np.float32),
        node_resolution_xyz=(3, 3, 3),
    )

    monkeypatch.setattr(optimization, "_decode_mesh_upload", lambda *_args, **_kwargs: (SimpleNamespace(), ".stl"))
    monkeypatch.setattr(optimization, "prepare_structural_domain", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        optimization,
        "_build_force_and_constraint_masks",
        lambda *_args, **_kwargs: (
            np.zeros(prepared.node_resolution_xyz, dtype=np.float32),
            np.zeros(prepared.node_resolution_xyz, dtype=np.float32),
            np.zeros(prepared.node_resolution_xyz, dtype=np.float32),
            support_nodes.copy(),
            support_nodes.copy(),
            support_nodes.copy(),
            support_nodes.copy(),
            [load_nodes.copy()],
        ),
    )
    monkeypatch.setattr(
        optimization,
        "_cg_solve_elasticity",
        lambda *_args, **_kwargs: (
            np.zeros(prepared.node_resolution_xyz, dtype=np.float32),
            np.zeros(prepared.node_resolution_xyz, dtype=np.float32),
            np.zeros(prepared.node_resolution_xyz, dtype=np.float32),
            "cpu",
        ),
    )
    monkeypatch.setattr(
        optimization,
        "_compute_element_compliance_energy",
        lambda *_args, **_kwargs: np.full(prepared.resolution_xyz, np.nan, dtype=np.float32),
    )
    monkeypatch.setattr(
        optimization,
        "_compute_element_strain_stress",
        lambda *_args, **_kwargs: (
            np.ones(prepared.resolution_xyz, dtype=np.float32),
            np.ones(prepared.resolution_xyz, dtype=np.float32),
            np.ones(prepared.resolution_xyz, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(optimization, "_build_iteration_mesh", lambda *_args, **_kwargs: (EMPTY_MESH_PAYLOAD, "cpu"))

    with pytest.raises(MeshUploadError, match="non-finite objective value"):
        optimization.run_structural_optimization(_request(StructuralOptimizationConfig(max_iterations=1)))


def test_structural_worker_posts_iteration_callbacks_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    posted_iterations: list[int] = []

    class DummyStore:
        def mark_running(self, _job_id: str) -> None:
            return None

        def mark_failed(self, _job_id: str, _detail: str) -> None:
            raise AssertionError("mark_failed should not be called on successful runs")

    def fake_run_structural_optimization(_request, *, iteration_callback):
        first = StructuralOptimizationIterationWebhookRequest(
            iteration_result=StructuralOptimizationIterationResult(
                iteration=1,
                objective_value=2.0,
                active_volume_fraction=0.9,
                removed_voxels=0,
                density_field=_field((4, 4, 4)),
            ),
            history_entry=OptimizationHistoryEntry(
                iteration=1,
                objective_value=2.0,
                active_volume_fraction=0.9,
                removed_voxels=0,
                max_displacement=0.1,
            ),
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            resolution_xyz=[8, 8, 8],
            compute_backend_used="cpu",
            mesh_backend_used="cpu",
        )
        second = StructuralOptimizationIterationWebhookRequest(
            iteration_result=StructuralOptimizationIterationResult(
                iteration=2,
                objective_value=1.0,
                active_volume_fraction=0.7,
                removed_voxels=1,
                density_field=_field((4, 4, 4)),
            ),
            history_entry=OptimizationHistoryEntry(
                iteration=2,
                objective_value=1.0,
                active_volume_fraction=0.7,
                removed_voxels=1,
                max_displacement=0.05,
            ),
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            resolution_xyz=[8, 8, 8],
            compute_backend_used="cpu",
            mesh_backend_used="cpu",
            is_final=True,
            stop_reason="objective_converged",
        )
        iteration_callback(first)
        iteration_callback(second)
        return StructuralOptimizationResultResponse(
            history=[first.history_entry, second.history_entry],
            final_iteration=second.iteration_result,
            bounds=second.bounds or [],
            resolution_xyz=second.resolution_xyz or [],
            compute_backend_used="cpu",
            mesh_backend_used="cpu",
            stop_reason="objective_converged",
        )

    monkeypatch.setattr(worker_tasks, "get_structural_progress_store", lambda: DummyStore())
    monkeypatch.setattr(worker_tasks, "run_structural_optimization", fake_run_structural_optimization)
    monkeypatch.setattr(
        worker_tasks,
        "_post_structural_iteration_callback",
        lambda **kwargs: posted_iterations.append(kwargs["payload"].iteration_result.iteration),
    )

    result = worker_tasks._run_structural_optimization_job(
        {
            **_request(StructuralOptimizationConfig()).model_dump(mode="json"),
            "_callback_token": "token-success",
        },
        job_id="job-worker-success",
    )

    assert posted_iterations == [1, 2]
    assert result["payload"]["stop_reason"] == "objective_converged"
