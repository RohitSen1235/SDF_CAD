from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from scipy import ndimage

from .mesh_upload import MeshUploadError, ParsedMesh, build_host_field, parse_mesh_bytes, validate_triangle_mesh
from .meshing import MeshData, build_mesh_with_backend
from .models import (
    ComputeBackend,
    ConstraintRegion,
    FieldPayload,
    LoadRegion,
    MeshBackend,
    MeshPayload,
    OptimizationHistoryEntry,
    SelectionPoint,
    StructuralMaterial,
    StructuralOptimizationConfig,
    StructuralOptimizationIterationResult,
    StructuralOptimizationIterationWebhookRequest,
    StructuralOptimizationPreprocessResponse,
    StructuralOptimizationRequest,
    StructuralOptimizationResultResponse,
    StructuralOptimizationStopReason,
)

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False


_GAUSS_POINTS = np.array(
    [
        -math.sqrt(3.0 / 5.0),
        0.0,
        math.sqrt(3.0 / 5.0),
    ],
    dtype=np.float64,
)
_GAUSS_WEIGHTS = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)
_NODE_OFFSETS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
_CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(3, 1)


def _cuda_runtime_available() -> bool:
    if not CUPY_AVAILABLE or cp is None:
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def _backend_xp(compute_backend: ComputeBackend) -> tuple[Any, Literal["cpu", "cuda"]]:
    if compute_backend == "cuda":
        if not _cuda_runtime_available():
            raise MeshUploadError("CUDA backend requested for structural optimization but CUDA runtime is unavailable")
        return cp, "cuda"
    if compute_backend == "auto" and _cuda_runtime_available():
        return cp, "cuda"
    return np, "cpu"


def _field_to_numpy(field: Any) -> np.ndarray:
    if CUPY_AVAILABLE and cp is not None and isinstance(field, cp.ndarray):
        return cp.asnumpy(field)
    return np.asarray(field)


def _decode_mesh_upload(file_name: str, file_data_base64: str) -> tuple[ParsedMesh, str]:
    extension = Path(file_name or "").suffix.lower()
    if extension not in {".stl", ".obj"}:
        raise MeshUploadError("Only .stl and .obj uploads are supported")
    try:
        raw = base64.b64decode(file_data_base64, validate=True)
    except Exception as exc:
        raise MeshUploadError("Uploaded mesh payload is not valid base64") from exc
    parsed = parse_mesh_bytes(raw, extension)
    validate_triangle_mesh(parsed)
    return parsed, extension


def _encode_field(field: object) -> str:
    array = _field_to_numpy(field).astype(np.float32, copy=False)
    return base64.b64encode(array.tobytes(order="C")).decode("ascii")


def _encode_mesh_payload(mesh: MeshData) -> MeshPayload:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    normals = np.asarray(mesh.normals, dtype=np.float32)
    return MeshPayload(
        encoding="mesh-f32-u32-base64-v1",
        vertex_count=int(vertices.shape[0]),
        face_count=int(faces.shape[0]),
        vertices_b64=base64.b64encode(vertices.tobytes(order="C")).decode("ascii"),
        indices_b64=base64.b64encode(faces.tobytes(order="C")).decode("ascii"),
        normals_b64=base64.b64encode(normals.tobytes(order="C")).decode("ascii"),
    )


def _field_payload(field: object, bounds: list[list[float]]) -> FieldPayload:
    arr = _field_to_numpy(field).astype(np.float32, copy=False)
    return FieldPayload(
        encoding="f32-base64",
        resolution_xyz=[int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])],
        bounds=[[float(axis[0]), float(axis[1])] for axis in bounds],
        data=_encode_field(arr),
    )


def _ensure_finite_iteration_field(name: str, field: np.ndarray, mask: np.ndarray, iteration: int) -> None:
    sampled = np.asarray(field[mask] if field.shape == mask.shape else field)
    if sampled.size == 0:
        return
    if not np.all(np.isfinite(sampled)):
        raise MeshUploadError(f"Structural optimization produced non-finite {name} values at iteration {iteration}")


def _bounds_from_meshes(design_mesh: ParsedMesh, keep_mesh: ParsedMesh) -> list[list[float]]:
    vertices = np.concatenate([design_mesh.vertices, keep_mesh.vertices], axis=0)
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    extents = np.maximum(maxs - mins, 1e-3)
    pad = max(float(np.max(extents)) * 0.08, 1e-3)
    return [
        [float(mins[0] - pad), float(maxs[0] + pad)],
        [float(mins[1] - pad), float(maxs[1] + pad)],
        [float(mins[2] - pad), float(maxs[2] + pad)],
    ]


def _resolution_xyz(bounds: list[list[float]], resolution: int) -> tuple[int, int, int]:
    spans = np.array(
        [bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0], bounds[2][1] - bounds[2][0]],
        dtype=np.float64,
    )
    max_span = float(np.max(spans))
    if max_span <= 0.0:
        return (resolution, resolution, resolution)
    scaled = np.maximum(24, np.round((spans / max_span) * float(resolution)).astype(np.int32))
    return (int(scaled[0]), int(scaled[1]), int(scaled[2]))


def _signed_distance_from_mask(mask: np.ndarray, bounds: list[list[float]]) -> np.ndarray:
    spacing = (
        (bounds[0][1] - bounds[0][0]) / max(mask.shape[0] - 1, 1),
        (bounds[1][1] - bounds[1][0]) / max(mask.shape[1] - 1, 1),
        (bounds[2][1] - bounds[2][0]) / max(mask.shape[2] - 1, 1),
    )
    outside = np.logical_not(mask)
    return (
        ndimage.distance_transform_edt(outside, sampling=spacing)
        - ndimage.distance_transform_edt(mask, sampling=spacing)
    ).astype(np.float32)


def _combine_mesh_from_masks(
    design_mask: np.ndarray,
    keep_mask: np.ndarray,
    bounds: list[list[float]],
    mesh_backend: MeshBackend,
) -> tuple[MeshData, Literal["cpu", "cuda"]]:
    field = _signed_distance_from_mask(np.logical_or(design_mask, keep_mask), bounds)
    mesh, mesh_backend_used, _ = build_mesh_with_backend(field, bounds, backend=mesh_backend, meshing_mode="uniform")
    return mesh, mesh_backend_used


def _shape_func_gradient(node: int, axis: int, x: float, y: float, z: float) -> float:
    if node == 0:
        values = (-(1.0 - y) * (1.0 - z), -(1.0 - x) * (1.0 - z), -(1.0 - x) * (1.0 - y))
    elif node == 1:
        values = ((1.0 - y) * (1.0 - z), -x * (1.0 - z), -x * (1.0 - y))
    elif node == 2:
        values = (-y * (1.0 - z), (1.0 - x) * (1.0 - z), -(1.0 - x) * y)
    elif node == 3:
        values = (y * (1.0 - z), x * (1.0 - z), -x * y)
    elif node == 4:
        values = (-(1.0 - y) * z, -(1.0 - x) * z, (1.0 - x) * (1.0 - y))
    elif node == 5:
        values = ((1.0 - y) * z, -x * z, x * (1.0 - y))
    elif node == 6:
        values = (-y * z, (1.0 - x) * z, (1.0 - x) * y)
    elif node == 7:
        values = (y * z, x * z, x * y)
    else:  # pragma: no cover - internal invariant
        raise ValueError(f"Invalid hexahedral node index: {node}")
    return float(values[axis])


def _elasticity_matrix(material: StructuralMaterial) -> np.ndarray:
    lam = float(
        material.youngs_modulus
        * material.poissons_ratio
        / ((1.0 + material.poissons_ratio) * (1.0 - 2.0 * material.poissons_ratio))
    )
    mu = float(material.youngs_modulus / (2.0 * (1.0 + material.poissons_ratio)))
    return np.array(
        [
            [lam + 2.0 * mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2.0 * mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2.0 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ],
        dtype=np.float64,
    )


def _build_center_strain_blocks(spacing_xyz: tuple[float, float, float]) -> list[np.ndarray]:
    hx, hy, hz = spacing_xyz
    x = y = z = 0.5
    blocks: list[np.ndarray] = []
    for node in range(8):
        dnx = _shape_func_gradient(node, 0, x, y, z) / max(hx, 1e-12)
        dny = _shape_func_gradient(node, 1, x, y, z) / max(hy, 1e-12)
        dnz = _shape_func_gradient(node, 2, x, y, z) / max(hz, 1e-12)
        blocks.append(
            np.array(
                [
                    [dnx, 0.0, 0.0],
                    [0.0, dny, 0.0],
                    [0.0, 0.0, dnz],
                    [dny, dnx, 0.0],
                    [dnz, 0.0, dnx],
                    [0.0, dnz, dny],
                ],
                dtype=np.float64,
            )
        )
    return blocks


def _build_element_stiffness_matrix(
    spacing_xyz: tuple[float, float, float],
    material: StructuralMaterial,
) -> np.ndarray:
    hx, hy, hz = spacing_xyz
    dmat = _elasticity_matrix(material)
    ke = np.zeros((24, 24), dtype=np.float64)
    det_j = hx * hy * hz
    for ix, wx in zip(_GAUSS_POINTS, _GAUSS_WEIGHTS):
        x = 0.5 * (ix + 1.0)
        for iy, wy in zip(_GAUSS_POINTS, _GAUSS_WEIGHTS):
            y = 0.5 * (iy + 1.0)
            for iz, wz in zip(_GAUSS_POINTS, _GAUSS_WEIGHTS):
                z = 0.5 * (iz + 1.0)
                weight = 0.125 * wx * wy * wz * det_j
                b = np.zeros((6, 24), dtype=np.float64)
                for node in range(8):
                    dnx = _shape_func_gradient(node, 0, x, y, z) / max(hx, 1e-12)
                    dny = _shape_func_gradient(node, 1, x, y, z) / max(hy, 1e-12)
                    dnz = _shape_func_gradient(node, 2, x, y, z) / max(hz, 1e-12)
                    col = node * 3
                    b[:, col : col + 3] = np.array(
                        [
                            [dnx, 0.0, 0.0],
                            [0.0, dny, 0.0],
                            [0.0, 0.0, dnz],
                            [dny, dnx, 0.0],
                            [dnz, 0.0, dnx],
                            [0.0, dnz, dny],
                        ],
                        dtype=np.float64,
                    )
                ke += (b.T @ dmat @ b) * weight
    return ke


def _element_stiffness_blocks(ke: np.ndarray) -> list[list[np.ndarray]]:
    return [[ke[a * 3 : (a + 1) * 3, b * 3 : (b + 1) * 3] for b in range(8)] for a in range(8)]


@dataclass
class PreparedOptimizationDomain:
    bounds: list[list[float]]
    resolution_xyz: tuple[int, int, int]
    design_mask: np.ndarray
    keep_mask: np.ndarray
    diagnostics: list[str]
    design_host_sdf: np.ndarray
    keep_host_sdf: np.ndarray
    node_resolution_xyz: tuple[int, int, int]


def prepare_structural_domain(
    design_mesh: ParsedMesh,
    keep_mesh: ParsedMesh,
    *,
    resolution: int,
    compute_backend: ComputeBackend,
) -> PreparedOptimizationDomain:
    bounds = _bounds_from_meshes(design_mesh, keep_mesh)
    resolution_xyz = _resolution_xyz(bounds, resolution)
    design_host = build_host_field(
        design_mesh,
        resolution_xyz,
        bounds=bounds,
        compute_backend=compute_backend,
        field_storage_mode="dense",
    )
    keep_host = build_host_field(
        keep_mesh,
        resolution_xyz,
        bounds=bounds,
        compute_backend=compute_backend,
        field_storage_mode="dense",
    )

    design_mask = _field_to_numpy(design_host.host_sdf) < 0.0
    keep_mask = _field_to_numpy(keep_host.host_sdf) < 0.0
    diagnostics: list[str] = []
    if not np.any(design_mask):
        raise MeshUploadError("Design space voxelization produced no solid volume")
    if not np.any(keep_mask):
        raise MeshUploadError("Non-design space voxelization produced no solid volume")
    if not np.any(np.logical_and(design_mask, keep_mask)):
        diagnostics.append(
            "Non-design space does not overlap the design space; it will still be preserved as an added protected body."
        )
    if np.any(np.logical_and(~design_mask, keep_mask)):
        diagnostics.append("Protected non-design voxels extend outside the design space and will be unioned into the final solid.")
    return PreparedOptimizationDomain(
        bounds=bounds,
        resolution_xyz=resolution_xyz,
        design_mask=design_mask,
        keep_mask=keep_mask,
        diagnostics=diagnostics,
        design_host_sdf=_field_to_numpy(design_host.host_sdf).astype(np.float32, copy=False),
        keep_host_sdf=_field_to_numpy(keep_host.host_sdf).astype(np.float32, copy=False),
        node_resolution_xyz=(int(resolution_xyz[0] + 1), int(resolution_xyz[1] + 1), int(resolution_xyz[2] + 1)),
    )


def preprocess_structural_optimization(
    design_file_name: str,
    design_file_data_base64: str,
    non_design_file_name: str,
    non_design_file_data_base64: str,
    *,
    resolution: int,
    compute_backend: ComputeBackend,
    mesh_backend: MeshBackend,
) -> StructuralOptimizationPreprocessResponse:
    design_mesh, _ = _decode_mesh_upload(design_file_name, design_file_data_base64)
    keep_mesh, _ = _decode_mesh_upload(non_design_file_name, non_design_file_data_base64)
    prepared = prepare_structural_domain(design_mesh, keep_mesh, resolution=resolution, compute_backend=compute_backend)
    combined_mesh, _ = _combine_mesh_from_masks(prepared.design_mask, prepared.keep_mask, prepared.bounds, mesh_backend)
    design_outer, _, _ = build_mesh_with_backend(prepared.design_host_sdf, prepared.bounds, backend=mesh_backend, meshing_mode="uniform")
    keep_outer, _, _ = build_mesh_with_backend(prepared.keep_host_sdf, prepared.bounds, backend=mesh_backend, meshing_mode="uniform")
    return StructuralOptimizationPreprocessResponse(
        design_mesh=_encode_mesh_payload(design_outer),
        non_design_mesh=_encode_mesh_payload(keep_outer),
        combined_mesh=_encode_mesh_payload(combined_mesh),
        bounds=[[float(axis[0]), float(axis[1])] for axis in prepared.bounds],
        resolution_xyz=[int(v) for v in prepared.resolution_xyz],
        diagnostics=prepared.diagnostics,
    )


def _node_slice(offset: tuple[int, int, int], element_shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    return (
        slice(offset[0], offset[0] + element_shape[0]),
        slice(offset[1], offset[1] + element_shape[1]),
        slice(offset[2], offset[2] + element_shape[2]),
    )


def _apply_force_points(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    solid_nodes: np.ndarray,
    points: list[SelectionPoint],
    radius: float,
    default_radius: float,
) -> np.ndarray:
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    if not points:
        return np.zeros_like(solid_nodes)
    region = np.zeros_like(solid_nodes)
    active_radius = max(float(radius), default_radius)
    r2 = active_radius * active_radius
    for selection in points:
        px, py, pz = selection.point_xyz
        region |= ((xg - px) ** 2 + (yg - py) ** 2 + (zg - pz) ** 2) <= r2
    return np.logical_and(region, solid_nodes)


def _build_force_and_constraint_masks(
    domain: PreparedOptimizationDomain,
    constraints: list[ConstraintRegion],
    loads: list[LoadRegion],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    nx, ny, nz = domain.node_resolution_xyz
    xs = np.linspace(domain.bounds[0][0], domain.bounds[0][1], nx, dtype=np.float32)
    ys = np.linspace(domain.bounds[1][0], domain.bounds[1][1], ny, dtype=np.float32)
    zs = np.linspace(domain.bounds[2][0], domain.bounds[2][1], nz, dtype=np.float32)
    solid_elements = np.logical_or(domain.design_mask, domain.keep_mask)
    solid_nodes = np.zeros(domain.node_resolution_xyz, dtype=bool)
    for offset in _NODE_OFFSETS:
        solid_nodes[_node_slice(offset, domain.resolution_xyz)] |= solid_elements

    fx = np.zeros(domain.node_resolution_xyz, dtype=np.float32)
    fy = np.zeros(domain.node_resolution_xyz, dtype=np.float32)
    fz = np.zeros(domain.node_resolution_xyz, dtype=np.float32)
    fixed = np.zeros(domain.node_resolution_xyz, dtype=bool)
    load_masks: list[np.ndarray] = []

    span = max(
        domain.bounds[0][1] - domain.bounds[0][0],
        domain.bounds[1][1] - domain.bounds[1][0],
        domain.bounds[2][1] - domain.bounds[2][0],
    )
    default_radius = max(span / max(max(nx, ny, nz), 1), span * 0.02)

    for constraint in constraints:
        fixed |= _apply_force_points(xs, ys, zs, solid_nodes, constraint.points, constraint.radius, default_radius)

    for load in loads:
        region = _apply_force_points(xs, ys, zs, solid_nodes, load.points, load.radius, default_radius)
        count = int(np.count_nonzero(region))
        if count == 0:
            continue
        direction = np.asarray(load.direction_xyz, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            continue
        direction /= norm
        if abs(float(load.magnitude)) > 0.0:
            load_masks.append(np.array(region, copy=True))
        weight = float(load.magnitude) / float(count)
        fx[region] += float(direction[0] * weight)
        fy[region] += float(direction[1] * weight)
        fz[region] += float(direction[2] * weight)

    if not np.any(fixed):
        raise MeshUploadError("At least one fixed support point is required before solving the structural optimization problem")
    if not np.any(np.abs(fx) + np.abs(fy) + np.abs(fz) > 0.0):
        raise MeshUploadError("At least one non-zero load point or surface is required before solving the structural optimization problem")

    return fx, fy, fz, fixed.copy(), fixed.copy(), fixed.copy(), fixed.copy(), load_masks


def _ensure_support_load_connectivity(
    solid_mask: np.ndarray,
    support_mask: np.ndarray,
    load_masks: list[np.ndarray],
    *,
    stage: str,
) -> None:
    support_voxels = np.logical_and(solid_mask, support_mask)
    if not np.any(support_voxels):
        raise MeshUploadError(f"Fixed supports are not attached to the solid during {stage}.")
    if not load_masks:
        return

    labels, component_count = ndimage.label(solid_mask, structure=_CONNECTIVITY_STRUCTURE)
    if component_count <= 0:
        raise MeshUploadError(f"No solid material remains during {stage}.")

    support_labels = {int(v) for v in np.unique(labels[support_voxels]).tolist() if int(v) > 0}
    if not support_labels:
        raise MeshUploadError(f"Fixed supports are not attached to a valid solid component during {stage}.")

    for idx, load_mask in enumerate(load_masks, start=1):
        load_voxels = np.logical_and(solid_mask, load_mask)
        if not np.any(load_voxels):
            raise MeshUploadError(f"Load region {idx} is not attached to the solid during {stage}.")
        load_labels = {int(v) for v in np.unique(labels[load_voxels]).tolist() if int(v) > 0}
        if support_labels.isdisjoint(load_labels):
            raise MeshUploadError(f"Load region {idx} is disconnected from fixed supports during {stage}.")

    anchor_mask = np.array(support_voxels, copy=True)
    for load_mask in load_masks:
        anchor_mask |= np.logical_and(solid_mask, load_mask)
    anchor_labels = {int(v) for v in np.unique(labels[anchor_mask]).tolist() if int(v) > 0}
    if len(anchor_labels) > 1:
        raise MeshUploadError(f"Selected supports and loads are split across disconnected components during {stage}.")


def _build_density_field(
    design_density: np.ndarray,
    keep_mask: np.ndarray,
    design_mask: np.ndarray,
    density_floor: float,
) -> np.ndarray:
    density = np.zeros_like(design_density, dtype=np.float32)
    density[design_mask] = design_density[design_mask]
    density[keep_mask] = 1.0
    density[np.logical_and(~design_mask, ~keep_mask)] = float(density_floor)
    return density


def _node_mask_to_element_mask(node_mask: np.ndarray, element_shape: tuple[int, int, int]) -> np.ndarray:
    element_mask = np.zeros(element_shape, dtype=bool)
    for offset in _NODE_OFFSETS:
        element_mask |= node_mask[_node_slice(offset, element_shape)]
    return element_mask


def _density_scale_field(density: Any, material: StructuralMaterial, xp: Any) -> Any:
    return float(material.stiffness_floor_ratio) + xp.power(density, float(material.simp_penalty)) * (
        1.0 - float(material.stiffness_floor_ratio)
    )


def _apply_structured_fem_operator(
    displacement: Any,
    density_scale: Any,
    stiffness_blocks: list[list[Any]],
    fixed_masks: tuple[Any, Any, Any],
    xp: Any,
) -> Any:
    element_shape = tuple(int(v) for v in density_scale.shape)
    result = xp.zeros_like(displacement)
    for a, offset_a in enumerate(_NODE_OFFSETS):
        slice_a = _node_slice(offset_a, element_shape)
        accum = xp.zeros((*element_shape, 3), dtype=displacement.dtype)
        for b, offset_b in enumerate(_NODE_OFFSETS):
            slice_b = _node_slice(offset_b, element_shape)
            accum = accum + xp.einsum("...j,ij->...i", displacement[slice_b], stiffness_blocks[a][b])
        result[slice_a] = result[slice_a] + density_scale[..., None] * accum
    result[..., 0][fixed_masks[0]] = displacement[..., 0][fixed_masks[0]]
    result[..., 1][fixed_masks[1]] = displacement[..., 1][fixed_masks[1]]
    result[..., 2][fixed_masks[2]] = displacement[..., 2][fixed_masks[2]]
    return result


def _build_preconditioner(
    density_scale: np.ndarray,
    stiffness_blocks: list[list[np.ndarray]],
    node_shape: tuple[int, int, int],
) -> np.ndarray:
    diagonal = np.zeros((*node_shape, 3), dtype=np.float32)
    for a, offset_a in enumerate(_NODE_OFFSETS):
        slice_a = _node_slice(offset_a, density_scale.shape)
        diagonal[slice_a] += density_scale[..., None] * np.diag(stiffness_blocks[a][a]).astype(np.float32)
    return np.maximum(diagonal, 1e-6)


def _cg_solve_elasticity(
    density: np.ndarray,
    fx_np: np.ndarray,
    fy_np: np.ndarray,
    fz_np: np.ndarray,
    fixed_masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    material: StructuralMaterial,
    config: StructuralOptimizationConfig,
    compute_backend: ComputeBackend,
    stiffness_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Literal["cpu", "cuda"]]:
    xp, backend_used = _backend_xp(compute_backend)
    density_xp = xp.asarray(density, dtype=xp.float32)
    force = xp.stack(
        [
            xp.asarray(fx_np, dtype=xp.float32),
            xp.asarray(fy_np, dtype=xp.float32),
            xp.asarray(fz_np, dtype=xp.float32),
        ],
        axis=-1,
    )
    fixed_x = xp.asarray(fixed_masks[0])
    fixed_y = xp.asarray(fixed_masks[1])
    fixed_z = xp.asarray(fixed_masks[2])
    density_scale = _density_scale_field(density_xp, material, xp)
    stiffness_blocks = [[xp.asarray(block, dtype=xp.float32) for block in row] for row in _element_stiffness_blocks(stiffness_matrix)]
    node_shape = fx_np.shape
    preconditioner = xp.asarray(
        _build_preconditioner(_field_to_numpy(density_scale), _element_stiffness_blocks(stiffness_matrix), node_shape),
        dtype=xp.float32,
    )
    preconditioner[..., 0][fixed_x] = 1.0
    preconditioner[..., 1][fixed_y] = 1.0
    preconditioner[..., 2][fixed_z] = 1.0

    def apply(vec: Any) -> Any:
        return _apply_structured_fem_operator(vec, density_scale, stiffness_blocks, (fixed_x, fixed_y, fixed_z), xp)

    def dot(lhs: Any, rhs: Any) -> float:
        return float(xp.sum(lhs * rhs).item())

    u = xp.zeros((*node_shape, 3), dtype=xp.float32)
    b = force.copy()
    b[..., 0][fixed_x] = 0.0
    b[..., 1][fixed_y] = 0.0
    b[..., 2][fixed_z] = 0.0
    r = b - apply(u)
    z = r / preconditioner
    p = z.copy()
    rz_old = dot(r, z)
    rhs_norm = max(math.sqrt(max(dot(b, b), 1e-20)), 1e-10)

    for _ in range(int(config.cg_max_iterations)):
        q = apply(p)
        denom = max(dot(p, q), 1e-20)
        alpha = rz_old / denom
        u = u + alpha * p
        r = r - alpha * q
        rel_res = math.sqrt(max(dot(r, r), 0.0)) / rhs_norm
        if rel_res <= float(config.cg_tolerance):
            break
        z = r / preconditioner
        rz_new = dot(r, z)
        beta = rz_new / max(rz_old, 1e-20)
        p = z + beta * p
        rz_old = rz_new

    u[..., 0][fixed_x] = 0.0
    u[..., 1][fixed_y] = 0.0
    u[..., 2][fixed_z] = 0.0
    solved = _field_to_numpy(u).astype(np.float32, copy=False)
    return solved[..., 0], solved[..., 1], solved[..., 2], backend_used


def _compute_element_compliance_energy(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    stiffness_matrix: np.ndarray,
    element_shape: tuple[int, int, int],
) -> np.ndarray:
    displacement = np.stack([ux, uy, uz], axis=-1).astype(np.float32, copy=False)
    base_energy = np.zeros(element_shape, dtype=np.float32)
    blocks = _element_stiffness_blocks(stiffness_matrix)
    for a, offset_a in enumerate(_NODE_OFFSETS):
        ua = displacement[_node_slice(offset_a, element_shape)]
        for b, offset_b in enumerate(_NODE_OFFSETS):
            ub = displacement[_node_slice(offset_b, element_shape)]
            base_energy += np.einsum("...i,ij,...j->...", ua, blocks[a][b], ub, optimize=True).astype(np.float32)
    return base_energy


def _compute_element_strain_stress(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    density_scale: np.ndarray,
    center_strain_blocks: list[np.ndarray],
    dmat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    displacement = np.stack([ux, uy, uz], axis=-1).astype(np.float32, copy=False)
    strain = np.zeros((*density_scale.shape, 6), dtype=np.float32)
    for node, offset in enumerate(_NODE_OFFSETS):
        strain += np.einsum(
            "ij,...j->...i",
            center_strain_blocks[node],
            displacement[_node_slice(offset, density_scale.shape)],
            optimize=True,
        ).astype(np.float32)
    stress = np.einsum("...j,ij->...i", strain, dmat.astype(np.float32), optimize=True).astype(np.float32)
    stress *= density_scale[..., None].astype(np.float32)
    strain_mag = np.sqrt(
        np.maximum(
            0.0,
            strain[..., 0] ** 2
            + strain[..., 1] ** 2
            + strain[..., 2] ** 2
            + 0.5 * (strain[..., 3] ** 2 + strain[..., 4] ** 2 + strain[..., 5] ** 2),
        )
    ).astype(np.float32)
    sxx, syy, szz, sxy, sxz, syz = [stress[..., idx] for idx in range(6)]
    von_mises = np.sqrt(
        np.maximum(
            0.0,
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy * sxy + sxz * sxz + syz * syz),
        )
    ).astype(np.float32)
    energy = (0.5 * np.sum(strain * stress, axis=-1)).astype(np.float32)
    return strain_mag, von_mises, energy


def _filter_sensitivities(
    sensitivities: np.ndarray,
    density: np.ndarray,
    design_mask: np.ndarray,
    filter_radius_voxels: float,
) -> np.ndarray:
    if filter_radius_voxels <= 0.0:
        filtered = np.array(sensitivities, copy=True)
    else:
        sigma = max(float(filter_radius_voxels) * 0.5, 1e-3)
        weighted = ndimage.gaussian_filter((density * sensitivities * design_mask).astype(np.float32), sigma=sigma)
        norm = ndimage.gaussian_filter((density * design_mask).astype(np.float32), sigma=sigma)
        filtered = weighted / np.maximum(norm, 1e-8)
    filtered[~design_mask] = 0.0
    return filtered.astype(np.float32, copy=False)


def _update_design_density(
    density: np.ndarray,
    sensitivities: np.ndarray,
    design_mask: np.ndarray,
    config: StructuralOptimizationConfig,
) -> np.ndarray:
    updated = np.array(density, copy=True, dtype=np.float32)
    if not np.any(design_mask):
        return updated
    current = density[design_mask].astype(np.float64)
    gradient = np.minimum(sensitivities[design_mask].astype(np.float64), -1e-12)
    target_sum = float(config.target_volume_fraction) * float(current.size)
    move = float(config.oc_move_limit)
    min_density = float(config.min_density)
    l1 = 1e-12
    l2 = 1e12
    candidate = current
    for _ in range(100):
        mid = 0.5 * (l1 + l2)
        candidate = np.clip(
            np.maximum(
                min_density,
                np.maximum(current - move, np.minimum(current + move, current * np.sqrt(np.maximum(-gradient / mid, 1e-12)))),
            ),
            min_density,
            1.0,
        )
        if float(np.sum(candidate)) > target_sum:
            l1 = mid
        else:
            l2 = mid
        if (l2 - l1) / max(l2 + l1, 1e-12) < 1e-4:
            break
    updated[design_mask] = candidate.astype(np.float32)
    return updated


def _build_iteration_mesh(
    density: np.ndarray,
    keep_mask: np.ndarray,
    bounds: list[list[float]],
    mesh_backend: MeshBackend,
    density_iso_threshold: float,
) -> tuple[MeshPayload, Literal["cpu", "cuda"]]:
    mask = np.logical_or(density >= float(density_iso_threshold), keep_mask)
    field = _signed_distance_from_mask(mask, bounds)
    mesh, mesh_backend_used, _ = build_mesh_with_backend(field, bounds, backend=mesh_backend, meshing_mode="uniform")
    return _encode_mesh_payload(mesh), mesh_backend_used


def run_structural_optimization(
    request: StructuralOptimizationRequest,
    *,
    iteration_callback: Callable[[StructuralOptimizationIterationWebhookRequest], None] | None = None,
) -> StructuralOptimizationResultResponse:
    design_mesh, _ = _decode_mesh_upload(request.design_space_file_name, request.design_space_file_data_base64)
    keep_mesh, _ = _decode_mesh_upload(request.non_design_space_file_name, request.non_design_space_file_data_base64)
    prepared = prepare_structural_domain(
        design_mesh,
        keep_mesh,
        resolution=request.config.resolution,
        compute_backend=request.compute_backend,
    )
    spacing_xyz = (
        (prepared.bounds[0][1] - prepared.bounds[0][0]) / max(prepared.resolution_xyz[0], 1),
        (prepared.bounds[1][1] - prepared.bounds[1][0]) / max(prepared.resolution_xyz[1], 1),
        (prepared.bounds[2][1] - prepared.bounds[2][0]) / max(prepared.resolution_xyz[2], 1),
    )
    stiffness_matrix = _build_element_stiffness_matrix(spacing_xyz, request.material)
    center_strain_blocks = _build_center_strain_blocks(spacing_xyz)
    dmat = _elasticity_matrix(request.material)

    fx, fy, fz, mfx, mfy, mfz, support_mask, load_masks = _build_force_and_constraint_masks(prepared, request.constraints, request.loads)
    _ensure_support_load_connectivity(
        np.logical_or(prepared.design_mask, prepared.keep_mask),
        _node_mask_to_element_mask(support_mask, prepared.resolution_xyz),
        [_node_mask_to_element_mask(mask, prepared.resolution_xyz) for mask in load_masks],
        stage="initial validation",
    )

    design_density = np.zeros(prepared.resolution_xyz, dtype=np.float32)
    design_density[prepared.design_mask] = float(request.config.target_volume_fraction)
    design_density[prepared.keep_mask] = 1.0
    design_density[~np.logical_or(prepared.design_mask, prepared.keep_mask)] = float(request.material.density_floor)

    history: list[OptimizationHistoryEntry] = []
    compute_backend_used: Literal["cpu", "cuda"] = "cpu"
    mesh_backend_used: Literal["cpu", "cuda"] = "cpu"
    final_iteration: StructuralOptimizationIterationResult | None = None
    stop_reason: StructuralOptimizationStopReason = "max_iterations"
    previous_objective: float | None = None
    previous_threshold_mask = np.logical_or(
        design_density >= float(request.config.density_iso_threshold),
        prepared.keep_mask,
    )

    for iteration in range(1, int(request.config.max_iterations) + 1):
        density = _build_density_field(design_density, prepared.keep_mask, prepared.design_mask, request.material.density_floor)
        ux, uy, uz, compute_backend_used = _cg_solve_elasticity(
            density,
            fx,
            fy,
            fz,
            (mfx, mfy, mfz),
            request.material,
            request.config,
            request.compute_backend,
            stiffness_matrix,
        )
        displacement_mag = np.sqrt(ux * ux + uy * uy + uz * uz).astype(np.float32)
        density_scale = _density_scale_field(density, request.material, np).astype(np.float32)
        base_energy = _compute_element_compliance_energy(ux, uy, uz, stiffness_matrix, prepared.resolution_xyz)
        compliance_field = (density_scale * base_energy).astype(np.float32)
        strain_field, stress_field, strain_energy_field = _compute_element_strain_stress(
            ux,
            uy,
            uz,
            density_scale,
            center_strain_blocks,
            dmat,
        )
        objective_value = float(np.sum(compliance_field[np.logical_or(prepared.design_mask, prepared.keep_mask)]))
        active_volume_fraction = float(np.mean(design_density[prepared.design_mask])) if np.any(prepared.design_mask) else 0.0
        max_displacement = float(np.max(displacement_mag))
        _ensure_finite_iteration_field("density", density, np.logical_or(prepared.design_mask, prepared.keep_mask), iteration)
        _ensure_finite_iteration_field("displacement", displacement_mag, prepared.design_mask, iteration)
        _ensure_finite_iteration_field("strain", strain_field, prepared.design_mask, iteration)
        _ensure_finite_iteration_field("stress", stress_field, prepared.design_mask, iteration)
        _ensure_finite_iteration_field("strain energy", strain_energy_field, prepared.design_mask, iteration)
        if not math.isfinite(objective_value):
            raise MeshUploadError(f"Structural optimization produced a non-finite objective value at iteration {iteration}")
        if not math.isfinite(active_volume_fraction):
            raise MeshUploadError(f"Structural optimization produced a non-finite volume fraction at iteration {iteration}")
        if not math.isfinite(max_displacement):
            raise MeshUploadError(f"Structural optimization produced a non-finite maximum displacement at iteration {iteration}")

        sensitivities = np.zeros_like(design_density, dtype=np.float32)
        penal_scale = float(request.material.simp_penalty) * (1.0 - float(request.material.stiffness_floor_ratio))
        design_rho = np.maximum(design_density, float(request.config.min_density))
        sensitivities[prepared.design_mask] = (
            -penal_scale
            * np.power(design_rho[prepared.design_mask], float(request.material.simp_penalty) - 1.0)
            * base_energy[prepared.design_mask]
        ).astype(np.float32)
        filtered_sensitivities = _filter_sensitivities(
            sensitivities,
            np.clip(design_density, float(request.config.min_density), 1.0),
            prepared.design_mask,
            request.config.filter_radius_voxels,
        )
        updated_density = _update_design_density(design_density, filtered_sensitivities, prepared.design_mask, request.config)
        density_change = float(np.max(np.abs(updated_density[prepared.design_mask] - design_density[prepared.design_mask]))) if np.any(prepared.design_mask) else 0.0

        threshold_mask = np.logical_or(updated_density >= float(request.config.density_iso_threshold), prepared.keep_mask)
        removed_voxels = max(
            int(np.count_nonzero(previous_threshold_mask)) - int(np.count_nonzero(threshold_mask)),
            0,
        )
        previous_threshold_mask = threshold_mask
        design_density = updated_density

        iteration_mesh, mesh_backend_used = _build_iteration_mesh(
            design_density,
            prepared.keep_mask,
            prepared.bounds,
            request.mesh_backend,
            request.config.density_iso_threshold,
        )
        is_final_iteration = iteration == int(request.config.max_iterations)
        iteration_result = StructuralOptimizationIterationResult(
            iteration=iteration,
            objective_value=objective_value,
            active_volume_fraction=active_volume_fraction,
            removed_voxels=removed_voxels,
            mesh=iteration_mesh,
            density_field=_field_payload(design_density, prepared.bounds),
            displacement_field=_field_payload(displacement_mag, prepared.bounds) if is_final_iteration else None,
            stress_field=_field_payload(stress_field, prepared.bounds) if is_final_iteration else None,
            strain_field=_field_payload(strain_field, prepared.bounds) if is_final_iteration else None,
        )
        history_entry = OptimizationHistoryEntry(
            iteration=iteration,
            objective_value=objective_value,
            active_volume_fraction=active_volume_fraction,
            removed_voxels=removed_voxels,
            max_displacement=max_displacement,
        )
        history.append(history_entry)
        final_iteration = iteration_result

        objective_delta = math.inf if previous_objective is None else abs(previous_objective - objective_value) / max(abs(previous_objective), 1.0)
        previous_objective = objective_value
        threshold_design_fraction = (
            float(np.mean(design_density[prepared.design_mask] >= float(request.config.density_iso_threshold)))
            if np.any(prepared.design_mask)
            else 0.0
        )

        iteration_stop_reason: StructuralOptimizationStopReason | None = None
        if threshold_design_fraction <= float(request.config.target_volume_fraction) + float(request.config.optimization_tolerance):
            iteration_stop_reason = "target_volume_reached"
        elif objective_delta <= float(request.config.optimization_tolerance):
            iteration_stop_reason = "objective_converged"
        elif density_change <= float(request.config.optimization_tolerance):
            iteration_stop_reason = "density_converged"
        elif iteration == int(request.config.max_iterations):
            iteration_stop_reason = "max_iterations"

        if iteration_stop_reason is not None:
            final_mask = np.logical_or(design_density >= float(request.config.density_iso_threshold), prepared.keep_mask)
            _ensure_support_load_connectivity(
                final_mask,
                _node_mask_to_element_mask(support_mask, prepared.resolution_xyz),
                [_node_mask_to_element_mask(mask, prepared.resolution_xyz) for mask in load_masks],
                stage="final validation",
            )

        if iteration_callback is not None:
            iteration_callback(
                StructuralOptimizationIterationWebhookRequest(
                    iteration_result=iteration_result,
                    history_entry=history_entry,
                    bounds=[[float(axis[0]), float(axis[1])] for axis in prepared.bounds],
                    resolution_xyz=[int(v) for v in prepared.resolution_xyz],
                    compute_backend_used=compute_backend_used,
                    mesh_backend_used=mesh_backend_used,
                    is_final=iteration_stop_reason is not None,
                    stop_reason=iteration_stop_reason,
                )
            )

        if iteration_stop_reason is not None:
            stop_reason = iteration_stop_reason
            if iteration_stop_reason != "max_iterations":
                final_iteration.displacement_field = _field_payload(displacement_mag, prepared.bounds)
                final_iteration.stress_field = _field_payload(stress_field, prepared.bounds)
                final_iteration.strain_field = _field_payload(strain_field, prepared.bounds)
            break

    if final_iteration is None:
        raise MeshUploadError("Structural optimization did not produce any iterations")

    return StructuralOptimizationResultResponse(
        history=history,
        final_iteration=final_iteration,
        bounds=[[float(axis[0]), float(axis[1])] for axis in prepared.bounds],
        resolution_xyz=[int(v) for v in prepared.resolution_xyz],
        compute_backend_used=compute_backend_used,
        mesh_backend_used=mesh_backend_used,
        stop_reason=stop_reason,
    )
