import numpy as np

from app.dsl import compile_source
from app.evaluator import CUPY_AVAILABLE, evaluate_scene_field, evaluate_scene_field_with_backend, merge_parameter_values
from app.meshing import build_mesh
from app.models import GridConfig


def test_sphere_field_signs() -> None:
    scene = compile_source("root = sphere(r=0.6)")
    params = merge_parameter_values(scene, {})
    field = evaluate_scene_field(scene, params, GridConfig(bounds=[[-1, 1], [-1, 1], [-1, 1]], resolution=33))

    center = field[16, 16, 16]
    corner = field[0, 0, 0]
    assert center < 0
    assert corner > 0


def test_tpms_conformal_fill_builds_mesh() -> None:
    source = """
    host = sphere(r=1.0)
    lat = gyroid(pitch=0.5, thickness=0.12)
    root = conformal_fill(host, lat, wall=0.08, mode="shell")
    """
    scene = compile_source(source)
    params = merge_parameter_values(scene, {})
    grid = GridConfig(bounds=[[-1.2, 1.2], [-1.2, 1.2], [-1.2, 1.2]], resolution=64)
    field = evaluate_scene_field(scene, params, grid)
    mesh = build_mesh(field, grid.bounds)

    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0
    assert np.all(np.isfinite(mesh.vertices))


def test_turbomachinery_union_mesh_non_empty() -> None:
    source = """
    imp = impeller_centrifugal(r_in=0.2, r_out=0.9, blade_count=9)
    vol = volute_casing(throat_radius=0.35, outlet_radius=1.1, width=0.45)
    root = union(imp, vol)
    """
    scene = compile_source(source)
    params = merge_parameter_values(scene, {})
    grid = GridConfig(bounds=[[-1.5, 1.5], [-1.0, 1.0], [-1.5, 1.5]], resolution=64)
    field = evaluate_scene_field(scene, params, grid)
    mesh = build_mesh(field, grid.bounds)

    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0


def test_strut_lattice_field_is_finite() -> None:
    scene = compile_source("root = strut_lattice(type=\"octet\", pitch=0.7, radius=0.08)")
    params = merge_parameter_values(scene, {})
    field = evaluate_scene_field(
        scene,
        params,
        GridConfig(bounds=[[-0.7, 0.7], [-0.7, 0.7], [-0.7, 0.7]], resolution=40),
    )

    assert np.all(np.isfinite(field))
    assert float(np.min(field)) < float(np.max(field))


def test_field_expression_symbol_reference_evaluates() -> None:
    source = """
    a = sin(x * 3.0) + cos(y * 3.0) + sin(z * 3.0)
    root = abs(a) - 0.45
    """
    scene = compile_source(source)
    params = merge_parameter_values(scene, {})
    field = evaluate_scene_field(
        scene,
        params,
        GridConfig(bounds=[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], resolution=36),
    )

    assert np.all(np.isfinite(field))
    assert float(np.min(field)) < 0.0


def test_field_expression_cuboid_min_max_with_scalars() -> None:
    source = """
    qx = abs(x) - 2.5
    qy = abs(y) - 1.0
    qz = abs(z) - 1.0
    outside = sqrt(max(qx, 0)^2 + max(qy, 0)^2 + max(qz, 0)^2)
    inside = min(max(qx, max(qy, qz)), 0)
    root = outside + inside
    """
    scene = compile_source(source)
    params = merge_parameter_values(scene, {})
    field = evaluate_scene_field(
        scene,
        params,
        GridConfig(bounds=[[-3.0, 3.0], [-2.0, 2.0], [-2.0, 2.0]], resolution=36),
    )

    assert field.shape == (36, 36, 36)
    assert np.all(np.isfinite(field))


def test_circular_array_domain_op_produces_periodic_blades() -> None:
    source = """
    single = intersection(
      translate(box(x=0.18, y=0.22, z=0.03), x=0.75),
      difference(cylinder(r=1.1, h=0.45), cylinder(r=0.45, h=0.45))
    )
    root = circular_array(single, count=12, axis="y")
    """
    scene = compile_source(source)
    params = merge_parameter_values(scene, {})
    field = evaluate_scene_field(
        scene,
        params,
        GridConfig(bounds=[[-1.2, 1.2], [-0.35, 0.35], [-1.2, 1.2]], resolution=64),
    )
    mesh = build_mesh(field, [[-1.2, 1.2], [-0.35, 0.35], [-1.2, 1.2]])

    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0
    ring_angle = np.linspace(-np.pi, np.pi, 180, endpoint=False)
    ring_x = 0.78 * np.cos(ring_angle)
    ring_y = np.zeros_like(ring_angle)
    ring_z = 0.78 * np.sin(ring_angle)
    # Use nearest samples from the computed field by quick projection to indices.
    ix = np.clip(((ring_x + 1.2) / 2.4 * 63).astype(int), 0, 63)
    iy = np.clip(((ring_y + 0.35) / 0.7 * 63).astype(int), 0, 63)
    iz = np.clip(((ring_z + 1.2) / 2.4 * 63).astype(int), 0, 63)
    samples = field[ix, iy, iz]
    assert float(np.std(samples)) > 1e-3


def test_spline_primitive_builds_tubular_mesh() -> None:
    source = """
    root = spline(
      points="0 0 0; 0.3 0.25 0.0; 0.75 -0.2 0.18; 1.2 0.0 0.0",
      radius=0.09,
      samples=24
    )
    """
    scene = compile_source(source)
    params = merge_parameter_values(scene, {})
    grid = GridConfig(bounds=[[-0.3, 1.4], [-0.6, 0.6], [-0.6, 0.6]], resolution=64)
    field = evaluate_scene_field(scene, params, grid)
    mesh = build_mesh(field, grid.bounds)

    assert np.all(np.isfinite(field))
    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0


def test_freeform_surface_primitive_builds_mesh() -> None:
    source = """
    root = freeform_surface(
      heights="0 0 0 0; 0 0.26 0.32 0; 0 0.24 0.28 0; 0 0 0 0",
      x=0.9,
      z=0.8,
      thickness=0.05
    )
    """
    scene = compile_source(source)
    params = merge_parameter_values(scene, {})
    grid = GridConfig(bounds=[[-1.0, 1.0], [-0.3, 0.6], [-1.0, 1.0]], resolution=64)
    field = evaluate_scene_field(scene, params, grid)
    mesh = build_mesh(field, grid.bounds)

    assert np.all(np.isfinite(field))
    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0


def test_evaluator_respects_requested_compute_precision() -> None:
    scene = compile_source("root = sphere(r=0.6)")
    params = merge_parameter_values(scene, {})
    grid = GridConfig(bounds=[[-1, 1], [-1, 1], [-1, 1]], resolution=33)

    field32 = evaluate_scene_field(scene, params, grid, compute_precision="float32")
    field16 = evaluate_scene_field(scene, params, grid, compute_precision="float16")

    assert field32.dtype == np.float32
    assert field16.dtype == np.float16
    assert np.all(np.isfinite(field16))


def test_cuda_backend_selection_reports_fallback_or_cuda() -> None:
    scene = compile_source("root = sphere(r=0.55)")
    params = merge_parameter_values(scene, {})
    grid = GridConfig(bounds=[[-1, 1], [-1, 1], [-1, 1]], resolution=24)
    field, backend = evaluate_scene_field_with_backend(
        scene,
        params,
        grid,
        compute_precision="float32",
        compute_backend="cuda",
    )

    assert field.dtype == np.float32
    if CUPY_AVAILABLE:
        assert backend in {"cpu", "cuda"}
    else:
        assert backend == "cpu"
