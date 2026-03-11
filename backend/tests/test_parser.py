from app.dsl import DslError, compile_source


def test_compile_basic_scene() -> None:
    source = """
    param r default=0.8 min=0.2 max=1.5 step=0.1
    a = sphere(r=$r)
    b = translate(box(0.2), x=0.4)
    root = union(a, b)
    """
    scene = compile_source(source)

    assert scene.root_node_id
    assert len(scene.nodes) == 4
    assert scene.parameter_schema[0].name == "r"


def test_compile_field_expression_node() -> None:
    source = """
    param scale default=1.5 min=0.5 max=3.0 step=0.1
    root = sin(x * $scale) + cos(y) - 0.25
    """
    scene = compile_source(source)
    assert scene.nodes[-1].type == "field_expr"
    assert scene.nodes[-1].expr is not None


def test_compile_field_expression_can_reference_previous_symbol() -> None:
    source = """
    a = sin(x * 3.0) + cos(y * 3.0) + sin(z * 3.0)
    root = abs(a) - 0.45
    """
    scene = compile_source(source)
    assert len(scene.nodes) == 2
    assert scene.nodes[1].type == "field_expr"


def test_compile_lattice_and_turbomachinery_nodes() -> None:
    source = """
    lat = gyroid(pitch=0.6, thickness=0.08)
    imp = impeller_centrifugal(r_in=0.25, r_out=1.0, blade_count=7)
    root = union(conformal_fill(imp, lat, wall=0.06, mode="hybrid"), volute_casing())
    """
    scene = compile_source(source)
    kinds = {node.type for node in scene.nodes}
    assert "lattice" in kinds
    assert "turbomachine" in kinds


def test_compile_circular_array_domain_op() -> None:
    source = """
    blade = translate(box(x=0.18, y=0.25, z=0.04), x=0.7)
    root = circular_array(blade, count=9, axis="y", phase=0.1)
    """
    scene = compile_source(source)
    domain_nodes = [node for node in scene.nodes if node.type == "domain_op" and node.op == "circular_array"]
    assert len(domain_nodes) == 1
    assert scene.root_node_id == domain_nodes[0].id


def test_compile_raises_for_unknown_symbol() -> None:
    source = "root = union(a, sphere(r=1.0))"
    try:
        compile_source(source)
    except DslError as exc:
        assert "Unknown symbol" in str(exc)
    else:
        raise AssertionError("Expected DslError")


def test_compile_raises_for_undeclared_parameter() -> None:
    source = "root = sphere(r=$radius)"
    try:
        compile_source(source)
    except DslError as exc:
        assert "referenced but not declared" in str(exc)
    else:
        raise AssertionError("Expected DslError")
