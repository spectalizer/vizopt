"""Tests for vizopt.scene and the scene_configuration hook"""

import json

import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from pydantic import ValidationError

from vizopt.base import ObjectiveTerm, OptimConfig, OptimizationProblemTemplate
from vizopt.scene import Circle, DragBinding, Line, Polygon, Scene, Style, Text
from vizopt.templates.layered_graph import LayeredGraphOptimizer


def _NO_PRINT(*_):
    pass


def _element[E](scene: Scene, element_id: str, kind: type[E]) -> E:
    """The element with this id, checked to be of the given class."""
    element = next(e for e in scene.elements if e.id == element_id)
    assert isinstance(element, kind)
    return element


def _example_scene() -> Scene:
    return Scene(
        elements=[
            Line(id="l", x1=0, y1=0, x2=1, y2=1, arrow_end=True),
            Circle(
                id="c",
                cx=0.5,
                cy=0.5,
                r=4,
                radius_units="px",
                style=Style(fill="red"),
                drag=DragBinding(var="xy", index=2),
            ),
            Polygon(id="p", points=[(0, 0), (1, 0), (0, 1)]),
            Text(id="t", x=0, y=0, text="hello"),
        ]
    )


# --- models ---


def test_json_round_trip():
    scene = _example_scene()
    payload = json.loads(json.dumps(scene.to_json_dict()))
    assert Scene.model_validate(payload) == scene


def test_to_json_dict_omits_unset_optional_fields():
    payload = _example_scene().to_json_dict()
    line = payload["elements"][0]
    assert "drag" not in line and "tooltip" not in line
    assert line["style"] == {}


def test_elements_are_discriminated_by_kind():
    scene = Scene.model_validate(
        {"elements": [{"kind": "circle", "id": "a", "cx": 0, "cy": 0, "r": 1}]}
    )
    assert isinstance(scene.elements[0], Circle)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Circle.model_validate({"id": "a", "cx": 0, "cy": 0, "r": 1, "radius": 2})


def test_numpy_scalars_are_accepted():
    circle = Circle.model_validate(
        {"id": "a", "cx": np.float32(1.5), "cy": np.float64(2.0), "r": 1}
    )
    assert circle.cx == 1.5 and isinstance(circle.cx, float)


def test_json_schema_covers_all_element_kinds():
    schema = Scene.model_json_schema()
    assert {"Circle", "Line", "Polygon", "Text", "DragBinding", "Style"} <= set(
        schema["$defs"]
    )


# --- scene_configuration hook ---


def _template(scene_configuration=None):
    return OptimizationProblemTemplate(
        terms=[ObjectiveTerm("sq", lambda v, p: jnp.sum(v["xy"] ** 2))],
        initialize=lambda p, seed: {"xy": jnp.ones((2, 2))},
        scene_configuration=scene_configuration,
    )


def _points_scene(optim_vars, input_params):
    return Scene(
        elements=[
            Circle(id=f"p/{k}", cx=x, cy=y, r=1, drag=DragBinding(var="xy", index=k))
            for k, (x, y) in enumerate(np.asarray(optim_vars["xy"]))
        ]
    )


def test_problem_scene_from_explicit_vars():
    problem = _template(_points_scene).instantiate({})
    scene = problem.scene({"xy": np.array([[1.0, 2.0], [3.0, 4.0]])})
    circles = [_element(scene, f"p/{k}", Circle) for k in range(2)]
    assert [(c.cx, c.cy) for c in circles] == [(1.0, 2.0), (3.0, 4.0)]


def test_problem_scene_defaults_to_result():
    problem = _template(_points_scene).instantiate({})
    with pytest.raises(ValueError, match="optimize"):
        problem.scene()
    problem.optimize(OptimConfig(n_iters=2), callback=_NO_PRINT)
    assert len(problem.scene().elements) == 2


def test_problem_scene_without_configuration_raises():
    with pytest.raises(ValueError, match="scene_configuration"):
        _template().instantiate({}).scene({"xy": np.zeros((2, 2))})


def test_session_scene_reflects_current_vars():
    session = _template(_points_scene).instantiate({}).session(OptimConfig())
    session.pin("xy", 1, value=[5.0, 6.0])
    circle = _element(session.scene(), "p/1", Circle)
    assert (circle.cx, circle.cy) == (5.0, 6.0)


# --- layered graph ---


@pytest.fixture
def layered_session():
    graph = nx.DiGraph([("a", "b"), ("a", "c"), ("b", "d")])
    return LayeredGraphOptimizer(graph).session(OptimConfig(learning_rate=0.05))


def test_layered_graph_scene_structure(layered_session):
    layered_session.step(5)
    elements = layered_session.scene().elements
    kinds = [e.kind for e in elements]
    assert kinds.count("line") == 3
    assert kinds.count("circle") == 4
    assert kinds.count("text") == 4
    assert len({e.id for e in elements}) == len(elements)
    # Edges are painted below nodes.
    assert kinds.index("circle") > max(i for i, k in enumerate(kinds) if k == "line")


def test_layered_graph_drag_bindings_point_at_node_positions(layered_session):
    layered_session.step(5)
    node_xys = np.asarray(layered_session.vars["node_xys"])
    for element in layered_session.scene().elements:
        if isinstance(element, Circle):
            assert element.drag is not None and element.drag.var == "node_xys"
            np.testing.assert_allclose(
                [element.cx, element.cy], node_xys[element.drag.index], rtol=1e-6
            )


def test_layered_graph_drag_via_binding(layered_session):
    """A frontend drag: read the binding off the scene, pin to the cursor."""
    circle = _element(layered_session.scene(), "node/2", Circle)
    assert circle.drag is not None
    layered_session.pin(circle.drag.var, circle.drag.index, value=[3.0, -1.0])
    layered_session.step(10)
    moved = _element(layered_session.scene(), "node/2", Circle)
    assert (moved.cx, moved.cy) == pytest.approx((3.0, -1.0))


def test_layered_graph_edges_follow_nodes(layered_session):
    scene = layered_session.scene()
    edge = _element(scene, "edge/0", Line)
    source = _element(scene, "node/0", Circle)
    assert (edge.x1, edge.y1) == (source.cx, source.cy)
