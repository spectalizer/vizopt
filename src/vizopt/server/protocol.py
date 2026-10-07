"""WebSocket protocol between the live optimization server and its frontends.

Every message is a JSON object with a `type` field. The server sends one
`HelloMessage` when a client connects, then a `FrameMessage` whenever the
state changes (at most once per server tick; a slow client simply misses
intermediate frames). Clients send any `ClientMessage`.

The JSON Schema of both directions (`protocol_json_schema()`) is exported
for the frontend's generated TypeScript types by
`scripts/export_protocol_schema.py`.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ..scene import Scene, _mark_discriminators_required


class _Message(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra=_mark_discriminators_required
    )


# --- server → client ---


class Metrics(_Message):
    """Loss values at the frame's iteration.

    Attributes:
        total: Total (schedule-weighted) loss.
        terms: Weighted value per term name.
    """

    total: float
    terms: dict[str, float]


class HelloMessage(_Message):
    """Sent once on connection: what can be steered.

    Attributes:
        terms: Names of all objective terms, in definition order.
        weights: Current multiplier per term.
        adjustable_terms: Terms whose weight can be changed live (those
            with a non-zero multiplier when the problem was built).
        learning_rate: Current peak learning rate.
        steps_per_frame: Optimization steps between two frames.
    """

    type: Literal["hello"] = "hello"
    terms: list[str]
    weights: dict[str, float]
    adjustable_terms: list[str]
    learning_rate: float
    steps_per_frame: int


class FrameMessage(_Message):
    """The current state of the optimization.

    Attributes:
        iteration: Number of optimization steps performed so far.
        running: Whether the optimizer is currently stepping (neither paused
            by the user nor settled after its learning-rate decay).
        paused: Whether the user paused the optimizer.
        scene: The current configuration, ready to render.
        metrics: Loss values, or `None` before the first step.
        pinned: Per variable name, the indices of pinned entries.
        weights: Current multiplier per term.
    """

    type: Literal["frame"] = "frame"
    iteration: int
    running: bool
    paused: bool
    scene: Scene
    metrics: Metrics | None = None
    pinned: dict[str, list[int]] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)


class ErrorMessage(_Message):
    """A client message could not be applied.

    Attributes:
        message: Human-readable reason.
    """

    type: Literal["error"] = "error"
    message: str


ServerMessage = Annotated[
    HelloMessage | FrameMessage | ErrorMessage, Field(discriminator="type")
]
"""Any message sent by the server."""


# --- client → server ---


class DragStartMessage(_Message):
    """The user grabbed a draggable element: pin it where it is.

    Attributes:
        var: Variable name, from the element's `DragBinding`.
        index: Entry index, from the element's `DragBinding`.
    """

    type: Literal["drag_start"] = "drag_start"
    var: str
    index: int


class DragMessage(_Message):
    """The user moved a grabbed element to `(x, y)`, in data coordinates.

    Attributes:
        var: Variable name, from the element's `DragBinding`.
        index: Entry index, from the element's `DragBinding`.
        x: New x position.
        y: New y position.
    """

    type: Literal["drag"] = "drag"
    var: str
    index: int
    x: float
    y: float


class DragEndMessage(_Message):
    """The user released a grabbed element.

    Attributes:
        var: Variable name, from the element's `DragBinding`.
        index: Entry index, from the element's `DragBinding`.
        keep_pinned: Leave the element pinned where it was dropped instead
            of releasing it back to the optimizer.
    """

    type: Literal["drag_end"] = "drag_end"
    var: str
    index: int
    keep_pinned: bool = False


class UnpinMessage(_Message):
    """Release a pinned entry, or every entry of `var` when `index` is `None`.

    Attributes:
        var: Variable name.
        index: Entry index, or `None` for the whole variable.
    """

    type: Literal["unpin"] = "unpin"
    var: str
    index: int | None = None


class PauseMessage(_Message):
    """Stop stepping until resumed."""

    type: Literal["pause"] = "pause"


class ResumeMessage(_Message):
    """Resume stepping after a pause."""

    type: Literal["resume"] = "resume"


class ReheatMessage(_Message):
    """Restart the learning-rate decay so the layout re-settles."""

    type: Literal["reheat"] = "reheat"


class SetWeightMessage(_Message):
    """Change a term's multiplier.

    Attributes:
        name: Term name.
        value: New multiplier.
    """

    type: Literal["set_weight"] = "set_weight"
    name: str
    value: float


class ResetMessage(_Message):
    """Start over from a fresh initialization.

    Attributes:
        seed: Seed for the problem's `initialize`; `None` uses the next seed.
    """

    type: Literal["reset"] = "reset"
    seed: int | None = None


ClientMessage = Annotated[
    DragStartMessage
    | DragMessage
    | DragEndMessage
    | UnpinMessage
    | PauseMessage
    | ResumeMessage
    | ReheatMessage
    | SetWeightMessage
    | ResetMessage,
    Field(discriminator="type"),
]
"""Any message sent by a client."""

client_message_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)
"""Validates incoming JSON into the matching `ClientMessage` class."""


class _ProtocolRoot(_Message):
    """Root holding both directions, so one schema defines every message."""

    server_message: ServerMessage
    client_message: ClientMessage


def protocol_json_schema() -> dict:
    """JSON Schema covering every server and client message (and `Scene`).

    Returns:
        A JSON Schema whose root object has the properties `server_message`
        and `client_message`, with every message class under `$defs`.
    """
    schema = _ProtocolRoot.model_json_schema()
    schema["title"] = "Protocol"
    return schema
