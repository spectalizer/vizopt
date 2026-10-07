"""Declarative scene descriptions for rendering outside of Python.

A `Scene` describes one configuration of an optimization problem as a flat
list of drawing primitives (circles, lines, polygons, text), as plain data
that serializes to JSON. It is the contract between vizopt and non-Python
frontends (e.g. a browser app rendering a live optimization with D3), which
therefore need no knowledge of any particular template.

Conventions:

- Positions are in data coordinates; the frontend chooses the viewport.
- Sizes (radii, stroke widths, offsets) are in data units unless their field
  name ends in `_px`, or a `*_units` field says `"px"`: those are screen
  pixels, for marker-like elements that should not scale with zoom.
- Every element has an `id` that is stable across frames of the same
  problem, so frontends can key their data joins on it.
- An element with a `drag` binding can be dragged: its anchor point (circle
  center, text position) is the entry `optim_vars[drag.var][drag.index]`,
  a 2D `[x, y]` position, so a frontend can turn a drag into
  `session.pin(drag.var, drag.index, value=[x, y])`.

The JSON Schema of `Scene` is exported, as part of the live-server protocol
(`vizopt.server.protocol`), for the frontend's generated TypeScript types by
`scripts/export_protocol_schema.py`.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema

_DISCRIMINATORS = ("type", "kind")


def _mark_discriminators_required(schema: dict) -> None:
    """List `type` / `kind` as required in a model's JSON Schema.

    They have defaults (so Python callers can omit them), which would make
    them optional in the schema and in the generated TypeScript types, where
    discriminated unions can then no longer be narrowed. They are always
    present in serialized messages.
    """
    required = schema.setdefault("required", [])
    for key in _DISCRIMINATORS:
        if key in schema.get("properties", {}) and key not in required:
            required.insert(0, key)


class _SceneModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra=_mark_discriminators_required
    )


Point = Annotated[
    tuple[float, float],
    # Plain-array form; `prefixItems` (pydantic's default for tuples) is not
    # understood by json-schema-to-typescript.
    WithJsonSchema(
        {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
    ),
]
"""An `[x, y]` pair in data coordinates."""


class Style(_SceneModel):
    """Paint attributes of an element; `None` leaves the frontend default.

    Attributes:
        fill: CSS color of the interior, or `"none"`.
        stroke: CSS color of the outline, or `"none"`.
        stroke_width_px: Outline width in screen pixels.
        opacity: Overall opacity in `[0, 1]`.
    """

    fill: str | None = None
    stroke: str | None = None
    stroke_width_px: float | None = None
    opacity: float | None = None


class DragBinding(_SceneModel):
    """Ties an element's anchor point to a 2D position optimization variable.

    Attributes:
        var: Name of the optimization variable, e.g. `"node_xys"`.
        index: Index of the `[x, y]` row in that variable.
    """

    var: str
    index: int


class _Element(_SceneModel):
    id: str
    style: Style = Field(default_factory=Style)
    drag: DragBinding | None = None
    tooltip: str | None = None


class Circle(_Element):
    """A circle (or point marker).

    Attributes:
        cx: Center x, data coordinates.
        cy: Center y, data coordinates.
        r: Radius, in `radius_units`.
        radius_units: `"data"` (scales with zoom) or `"px"` (fixed marker size).
    """

    kind: Literal["circle"] = "circle"
    cx: float
    cy: float
    r: float
    radius_units: Literal["data", "px"] = "data"


class Line(_Element):
    """A straight segment, optionally with an arrowhead at its end.

    Attributes:
        x1: Start x, data coordinates.
        y1: Start y, data coordinates.
        x2: End x, data coordinates.
        y2: End y, data coordinates.
        arrow_end: Draw an arrowhead at `(x2, y2)`.
        shorten_start_px: Screen pixels to trim from the start, e.g. the
            radius of a pixel-sized node marker the line starts from.
        shorten_end_px: Screen pixels to trim from the end.
    """

    kind: Literal["line"] = "line"
    x1: float
    y1: float
    x2: float
    y2: float
    arrow_end: bool = False
    shorten_start_px: float = 0.0
    shorten_end_px: float = 0.0


class Polygon(_Element):
    """A closed polygon, e.g. a sampled star-shaped or band boundary.

    Attributes:
        points: Vertices as `[x, y]` pairs, data coordinates.
    """

    kind: Literal["polygon"] = "polygon"
    points: list[Point]


class Text(_Element):
    """A text label.

    Attributes:
        x: Anchor x, data coordinates.
        y: Anchor y, data coordinates.
        text: The label content.
        dx_px: Horizontal offset from the anchor, screen pixels.
        dy_px: Vertical offset from the anchor, screen pixels (positive is up).
        anchor: Horizontal alignment relative to the offset anchor.
        font_size_px: Font size in screen pixels.
    """

    kind: Literal["text"] = "text"
    x: float
    y: float
    text: str
    dx_px: float = 0.0
    dy_px: float = 0.0
    anchor: Literal["start", "middle", "end"] = "start"
    font_size_px: float = 12.0


Element = Annotated[Circle | Line | Polygon | Text, Field(discriminator="kind")]
"""Any scene element, discriminated by its `kind` field."""


class Scene(_SceneModel):
    """A renderable description of one configuration.

    Attributes:
        elements: Drawing primitives, painted in list order (later on top).
        y_axis: `"up"` for mathematical orientation (as in matplotlib), or
            `"down"` for screen orientation.
        equal_aspect: Whether x and y data units must have the same length
            on screen.
    """

    elements: list[Element] = Field(default_factory=list)
    y_axis: Literal["up", "down"] = "up"
    equal_aspect: bool = True

    def to_json_dict(self) -> dict:
        """Serialize to a JSON-compatible dict, omitting unset optional fields."""
        return self.model_dump(mode="json", exclude_none=True)
