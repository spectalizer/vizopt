"""Tests for vizopt.server"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vizopt.base import ObjectiveTerm, OptimConfig, OptimizationProblemTemplate
from vizopt.scene import Circle
from vizopt.server import LiveSession, create_app
from vizopt.server.protocol import (
    DragEndMessage,
    DragMessage,
    DragStartMessage,
    PauseMessage,
    ReheatMessage,
    ResetMessage,
    ResumeMessage,
    SetWeightMessage,
    UnpinMessage,
    client_message_adapter,
    protocol_json_schema,
)
from vizopt.templates.layered_graph import LayeredGraphOptimizer


def _session(n_iters=100):
    graph = nx.DiGraph([("a", "b"), ("a", "c"), ("b", "d")])
    return LayeredGraphOptimizer(graph).session(
        OptimConfig(n_iters=n_iters, learning_rate=0.05)
    )


@pytest.fixture
def live():
    return LiveSession(_session(), steps_per_frame=5, settle_iters=20)


def _node(frame, k):
    circle = next(e for e in frame.scene.elements if e.id == f"node/{k}")
    assert isinstance(circle, Circle)
    return circle


# --- protocol ---


def test_client_messages_parse_by_type():
    message = client_message_adapter.validate_python(
        {"type": "drag", "var": "node_xys", "index": 1, "x": 0.5, "y": 2}
    )
    assert message == DragMessage(var="node_xys", index=1, x=0.5, y=2.0)


def test_committed_protocol_schema_is_up_to_date():
    """The frontend's types are generated from this file; regenerate it with
    `uv run python scripts/export_protocol_schema.py` after model changes."""
    path = (
        Path(__file__).parent.parent
        / "frontend"
        / "src"
        / "protocol"
        / "protocol.schema.json"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == protocol_json_schema()


def test_protocol_schema_defines_all_messages():
    defs = protocol_json_schema()["$defs"]
    assert {"HelloMessage", "FrameMessage", "DragMessage", "Scene", "Circle"} <= set(
        defs
    )


# --- LiveSession state machine ---


def test_requires_scene_configuration():
    template = OptimizationProblemTemplate(
        terms=[ObjectiveTerm("t", lambda v, p: v["x"] ** 2)],
        initialize=lambda p, seed: {"x": np.float32(1.0)},
    )
    with pytest.raises(ValueError, match="scene_configuration"):
        LiveSession(template.instantiate({}).session())


def test_hello_lists_terms(live):
    hello = live.hello()
    assert hello.terms == [
        "edge_direction",
        "edge_vector",
        "node_separation",
        "sibling_separation",
    ]
    assert hello.adjustable_terms == hello.terms
    assert hello.steps_per_frame == 5


def test_first_frame_has_no_metrics(live):
    frame = live.frame()
    assert frame.iteration == 0 and frame.metrics is None and frame.running


def test_tick_steps_and_reports_metrics(live):
    assert live.tick()
    frame = live.frame()
    assert frame.iteration == 5
    assert frame.metrics is not None
    assert set(frame.metrics.terms) == set(live.hello().terms)


def test_settles_after_settle_iters(live):
    while live.tick():
        pass
    assert live.session.iteration == 20
    assert not live.running


def test_interaction_wakes_settled_run(live):
    while live.tick():
        pass
    live.apply(ReheatMessage())
    assert live.running
    assert live.tick()


def test_drag_moves_and_pins_node(live):
    live.apply(DragStartMessage(var="node_xys", index=1))
    live.apply(DragMessage(var="node_xys", index=1, x=4.0, y=-2.0))
    live.tick()
    frame = live.frame()
    assert (_node(frame, 1).cx, _node(frame, 1).cy) == pytest.approx((4.0, -2.0))
    assert frame.pinned == {"node_xys": [1]}


def test_drag_end_releases_node(live):
    live.apply(DragStartMessage(var="node_xys", index=1))
    live.apply(DragEndMessage(var="node_xys", index=1))
    assert live.frame().pinned == {}


def test_drag_end_can_keep_node_pinned(live):
    live.apply(DragStartMessage(var="node_xys", index=2))
    live.apply(DragEndMessage(var="node_xys", index=2, keep_pinned=True))
    assert live.frame().pinned == {"node_xys": [2]}
    live.apply(UnpinMessage(var="node_xys", index=2))
    assert live.frame().pinned == {}


def test_pause_and_resume(live):
    live.apply(PauseMessage())
    assert not live.tick()
    assert live.frame().paused
    live.apply(ResumeMessage())
    assert live.tick()


def test_set_weight(live):
    live.apply(SetWeightMessage(name="edge_vector", value=3.0))
    assert live.frame().weights["edge_vector"] == 3.0


def test_reset_starts_fresh_session(live):
    live.tick()
    before = live.session
    live.apply(ResetMessage(seed=7))
    assert live.session is not before
    assert live.session.iteration == 0
    # The compiled step is reused across resets.
    assert live.session._step_function is before._step_function


@pytest.mark.parametrize(
    "message",
    [
        DragStartMessage(var="nope", index=0),
        DragMessage(var="node_xys", index=99, x=0, y=0),
        SetWeightMessage(name="nope", value=1.0),
    ],
)
def test_validate_rejects_bad_messages(live, message):
    with pytest.raises(ValueError):
        live.validate(message)


def test_worker_thread_publishes_frames(live):
    published = []
    live.subscribe(lambda: published.append(live.latest()[0]))
    live.start()
    try:
        live.submit(DragMessage(var="node_xys", index=0, x=1.0, y=1.0))
        while live.running or live._commands.qsize():
            live._stop.wait(0.01)
    finally:
        live.stop()
    version, frame = live.latest()
    assert frame is not None and version == published[-1] >= 2
    assert frame["iteration"] == live.session.iteration


# --- app ---


@pytest.fixture
def client(live):
    with TestClient(create_app(live, static_dir=None)) as client:
        yield client


def _receive_until(ws, predicate, limit=500):
    for _ in range(limit):
        message = ws.receive_json()
        if predicate(message):
            return message
    raise AssertionError("expected message not received")


def test_index_without_frontend_explains_build(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "npm run build" in response.text


def test_websocket_hello_then_frames(client):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        frame = ws.receive_json()
        assert frame["type"] == "frame"
        assert any(e["kind"] == "circle" for e in frame["scene"]["elements"])


def test_websocket_drag_round_trip(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # hello
        ws.send_json({"type": "drag_start", "var": "node_xys", "index": 0})
        ws.send_json({"type": "drag", "var": "node_xys", "index": 0, "x": 5, "y": 5})
        frame = _receive_until(
            ws,
            lambda m: m["type"] == "frame"
            and any(
                e["id"] == "node/0" and (e["cx"], e["cy"]) == (5.0, 5.0)
                for e in m["scene"]["elements"]
            ),
        )
        assert frame["pinned"] == {"node_xys": [0]}


def test_websocket_reports_invalid_messages(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # hello
        ws.send_json({"type": "drag", "var": "node_xys"})
        error = _receive_until(ws, lambda m: m["type"] == "error")
        assert "index" in error["message"]
        ws.send_json({"type": "set_weight", "name": "nope", "value": 1})
        error = _receive_until(ws, lambda m: m["type"] == "error")
        assert "nope" in error["message"]


# --- history ---


def test_publish_records_history_for_new_clients(live):
    live.publish()  # before any step: no metrics, no history
    assert live.hello().history == []
    for _ in range(3):
        live.tick()
        live.publish()
    live.publish()  # same iteration again: not recorded twice
    history = live.hello().history
    assert [p.iteration for p in history] == [5, 10, 15]
    assert set(history[0].terms) == set(live.hello().terms)


def test_history_is_thinned_to_its_bound():
    live = LiveSession(_session(), steps_per_frame=1, settle_iters=50, max_history=8)
    while live.tick():
        live.publish()
    history = live.hello().history
    assert len(history) <= 8
    assert history[-1].iteration == 50
    iterations = [p.iteration for p in history]
    assert iterations == sorted(iterations)


def test_reset_clears_history(live):
    live.tick()
    live.publish()
    live.apply(ResetMessage())
    assert live.hello().history == []
