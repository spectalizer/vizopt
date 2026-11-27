/* Generated from protocol.schema.json by json-schema-to-typescript; run 'npm run codegen' to update. */

export type ServerMessage = HelloMessage | FrameMessage | ErrorMessage;
export type Type = "hello";
export type Terms = string[];
export type AdjustableTerms = string[];
export type LearningRate = number;
export type StepsPerFrame = number;
export type Total = number;
export type Iteration = number;
export type History = HistoryPoint[];
export type Type1 = "frame";
export type Iteration1 = number;
export type Running = boolean;
export type Paused = boolean;
export type Id = string;
export type Fill = string | null;
export type Stroke = string | null;
export type StrokeWidthPx = number | null;
export type Opacity = number | null;
export type FillOpacity = number | null;
export type Var = string;
export type Index = number;
export type Tooltip = string | null;
export type Kind = "circle";
export type Cx = number;
export type Cy = number;
export type R = number;
export type RadiusUnits = "data" | "px";
export type Id1 = string;
export type Tooltip1 = string | null;
export type Kind1 = "rect";
export type X = number;
export type Y = number;
export type Width = number;
export type Height = number;
export type Origin = "center" | "min_corner";
export type Id2 = string;
export type Tooltip2 = string | null;
export type Kind2 = "line";
export type X1 = number;
export type Y1 = number;
export type X2 = number;
export type Y2 = number;
export type ArrowEnd = boolean;
export type ShortenStartPx = number;
export type ShortenEndPx = number;
export type Id3 = string;
export type Tooltip3 = string | null;
export type Kind3 = "polygon";
export type Points = [number, number][];
export type Anchor = [number, number] | null;
export type Id4 = string;
export type Tooltip4 = string | null;
export type Kind4 = "text";
export type X3 = number;
export type Y3 = number;
export type Text1 = string;
export type DxPx = number;
export type DyPx = number;
export type Anchor1 = "start" | "middle" | "end";
export type FontSizePx = number;
export type Elements = (Circle | Rect | Line | Polygon | Text)[];
export type YAxis = "up" | "down";
export type EqualAspect = boolean;
export type Total1 = number;
export type Type2 = "error";
export type Message = string;
export type ClientMessage =
  | DragStartMessage
  | DragMessage
  | DragEndMessage
  | UnpinMessage
  | PauseMessage
  | ResumeMessage
  | ReheatMessage
  | SetWeightMessage
  | ResetMessage;
export type Type3 = "drag_start";
export type Var1 = string;
export type Index1 = number;
export type Type4 = "drag";
export type Var2 = string;
export type Index2 = number;
export type X4 = number;
export type Y4 = number;
export type Type5 = "drag_end";
export type Var3 = string;
export type Index3 = number;
export type KeepPinned = boolean;
export type Type6 = "unpin";
export type Var4 = string;
export type Index4 = number | null;
export type Type7 = "pause";
export type Type8 = "resume";
export type Type9 = "reheat";
export type Type10 = "set_weight";
export type Name = string;
export type Value = number;
export type Type11 = "reset";
export type Seed = number | null;

/**
 * Root holding both directions, so one schema defines every message.
 */
export interface Protocol {
  server_message: ServerMessage;
  client_message: ClientMessage;
}
/**
 * Sent once on connection: what can be steered, and what happened so far.
 *
 * Attributes:
 *     terms: Names of all objective terms, in definition order.
 *     weights: Current multiplier per term.
 *     adjustable_terms: Terms whose weight can be changed live (those
 *         with a non-zero multiplier when the problem was built).
 *     learning_rate: Current peak learning rate.
 *     steps_per_frame: Optimization steps between two frames.
 *     history: Loss values of the current run at past published frames,
 *         oldest first (thinned to a bounded length), so that a client
 *         joining late, or reloading, sees the whole curve.
 */
export interface HelloMessage {
  type: Type;
  terms: Terms;
  weights: Weights;
  adjustable_terms: AdjustableTerms;
  learning_rate: LearningRate;
  steps_per_frame: StepsPerFrame;
  history?: History;
}
export interface Weights {
  [k: string]: number;
}
/**
 * Loss values at a past iteration.
 *
 * Attributes:
 *     iteration: The iteration the values were recorded at.
 */
export interface HistoryPoint {
  total: Total;
  terms: Terms1;
  iteration: Iteration;
}
export interface Terms1 {
  [k: string]: number;
}
/**
 * The current state of the optimization.
 *
 * Attributes:
 *     iteration: Number of optimization steps performed so far.
 *     running: Whether the optimizer is currently stepping (neither paused
 *         by the user nor settled after its learning-rate decay).
 *     paused: Whether the user paused the optimizer.
 *     scene: The current configuration, ready to render.
 *     metrics: Loss values, or `None` before the first step.
 *     pinned: Per variable name, the indices of pinned entries.
 *     weights: Current multiplier per term.
 */
export interface FrameMessage {
  type: Type1;
  iteration: Iteration1;
  running: Running;
  paused: Paused;
  scene: Scene;
  metrics?: Metrics | null;
  pinned?: Pinned;
  weights?: Weights1;
}
/**
 * A renderable description of one configuration.
 *
 * Attributes:
 *     elements: Drawing primitives, painted in list order (later on top).
 *     y_axis: `"up"` for mathematical orientation (as in matplotlib), or
 *         `"down"` for screen orientation.
 *     equal_aspect: Whether x and y data units must have the same length
 *         on screen.
 */
export interface Scene {
  elements?: Elements;
  y_axis?: YAxis;
  equal_aspect?: EqualAspect;
}
/**
 * A circle (or point marker).
 *
 * Attributes:
 *     cx: Center x, data coordinates.
 *     cy: Center y, data coordinates.
 *     r: Radius, in `radius_units`.
 *     radius_units: `"data"` (scales with zoom) or `"px"` (fixed marker size).
 */
export interface Circle {
  id: Id;
  style?: Style;
  drag?: DragBinding | null;
  tooltip?: Tooltip;
  kind: Kind;
  cx: Cx;
  cy: Cy;
  r: R;
  radius_units?: RadiusUnits;
}
/**
 * Paint attributes of an element; `None` leaves the frontend default.
 *
 * Attributes:
 *     fill: CSS color of the interior, or `"none"`.
 *     stroke: CSS color of the outline, or `"none"`.
 *     stroke_width_px: Outline width in screen pixels.
 *     opacity: Overall opacity in `[0, 1]`.
 *     fill_opacity: Opacity of the interior only, in `[0, 1]`, e.g. for
 *         translucent regions with a solid outline.
 */
export interface Style {
  fill?: Fill;
  stroke?: Stroke;
  stroke_width_px?: StrokeWidthPx;
  opacity?: Opacity;
  fill_opacity?: FillOpacity;
}
/**
 * Ties an element's anchor point to a 2D position optimization variable.
 *
 * Attributes:
 *     var: Name of the optimization variable, e.g. `"node_xys"`.
 *     index: Index of the `[x, y]` row in that variable.
 */
export interface DragBinding {
  var: Var;
  index: Index;
}
/**
 * An axis-aligned rectangle.
 *
 * Attributes:
 *     x: Anchor x, data coordinates (see `origin`).
 *     y: Anchor y, data coordinates (see `origin`).
 *     width: Width, data units.
 *     height: Height, data units.
 *     origin: What `(x, y)` is: the rectangle's `"center"`, or its
 *         `"min_corner"` (smallest x and y).
 */
export interface Rect {
  id: Id1;
  style?: Style;
  drag?: DragBinding | null;
  tooltip?: Tooltip1;
  kind: Kind1;
  x: X;
  y: Y;
  width: Width;
  height: Height;
  origin?: Origin;
}
/**
 * A straight segment, optionally with an arrowhead at its end.
 *
 * Attributes:
 *     x1: Start x, data coordinates.
 *     y1: Start y, data coordinates.
 *     x2: End x, data coordinates.
 *     y2: End y, data coordinates.
 *     arrow_end: Draw an arrowhead at `(x2, y2)`.
 *     shorten_start_px: Screen pixels to trim from the start, e.g. the
 *         radius of a pixel-sized node marker the line starts from.
 *     shorten_end_px: Screen pixels to trim from the end.
 */
export interface Line {
  id: Id2;
  style?: Style;
  drag?: DragBinding | null;
  tooltip?: Tooltip2;
  kind: Kind2;
  x1: X1;
  y1: Y1;
  x2: X2;
  y2: Y2;
  arrow_end?: ArrowEnd;
  shorten_start_px?: ShortenStartPx;
  shorten_end_px?: ShortenEndPx;
}
/**
 * A closed polygon, e.g. a sampled star-shaped or band boundary.
 *
 * Attributes:
 *     points: Vertices as `[x, y]` pairs, data coordinates.
 *     anchor: The point a `drag` binding refers to (e.g. the center of a
 *         star-shaped region); required for a draggable polygon.
 */
export interface Polygon {
  id: Id3;
  style?: Style;
  drag?: DragBinding | null;
  tooltip?: Tooltip3;
  kind: Kind3;
  points: Points;
  anchor?: Anchor;
}
/**
 * A text label.
 *
 * Attributes:
 *     x: Anchor x, data coordinates.
 *     y: Anchor y, data coordinates.
 *     text: The label content.
 *     dx_px: Horizontal offset from the anchor, screen pixels.
 *     dy_px: Vertical offset from the anchor, screen pixels (positive is up).
 *     anchor: Horizontal alignment relative to the offset anchor.
 *     font_size_px: Font size in screen pixels.
 */
export interface Text {
  id: Id4;
  style?: Style;
  drag?: DragBinding | null;
  tooltip?: Tooltip4;
  kind: Kind4;
  x: X3;
  y: Y3;
  text: Text1;
  dx_px?: DxPx;
  dy_px?: DyPx;
  anchor?: Anchor1;
  font_size_px?: FontSizePx;
}
/**
 * Loss values at the frame's iteration.
 *
 * Attributes:
 *     total: Total (schedule-weighted) loss.
 *     terms: Weighted value per term name.
 */
export interface Metrics {
  total: Total1;
  terms: Terms2;
}
export interface Terms2 {
  [k: string]: number;
}
export interface Pinned {
  [k: string]: number[];
}
export interface Weights1 {
  [k: string]: number;
}
/**
 * A client message could not be applied.
 *
 * Attributes:
 *     message: Human-readable reason.
 */
export interface ErrorMessage {
  type: Type2;
  message: Message;
}
/**
 * The user grabbed a draggable element: pin it where it is.
 *
 * Attributes:
 *     var: Variable name, from the element's `DragBinding`.
 *     index: Entry index, from the element's `DragBinding`.
 */
export interface DragStartMessage {
  type: Type3;
  var: Var1;
  index: Index1;
}
/**
 * The user moved a grabbed element to `(x, y)`, in data coordinates.
 *
 * Attributes:
 *     var: Variable name, from the element's `DragBinding`.
 *     index: Entry index, from the element's `DragBinding`.
 *     x: New x position.
 *     y: New y position.
 */
export interface DragMessage {
  type: Type4;
  var: Var2;
  index: Index2;
  x: X4;
  y: Y4;
}
/**
 * The user released a grabbed element.
 *
 * Attributes:
 *     var: Variable name, from the element's `DragBinding`.
 *     index: Entry index, from the element's `DragBinding`.
 *     keep_pinned: Leave the element pinned where it was dropped instead
 *         of releasing it back to the optimizer.
 */
export interface DragEndMessage {
  type: Type5;
  var: Var3;
  index: Index3;
  keep_pinned?: KeepPinned;
}
/**
 * Release a pinned entry, or every entry of `var` when `index` is `None`.
 *
 * Attributes:
 *     var: Variable name.
 *     index: Entry index, or `None` for the whole variable.
 */
export interface UnpinMessage {
  type: Type6;
  var: Var4;
  index?: Index4;
}
/**
 * Stop stepping until resumed.
 */
export interface PauseMessage {
  type: Type7;
}
/**
 * Resume stepping after a pause.
 */
export interface ResumeMessage {
  type: Type8;
}
/**
 * Restart the learning-rate decay so the layout re-settles.
 */
export interface ReheatMessage {
  type: Type9;
}
/**
 * Change a term's multiplier.
 *
 * Attributes:
 *     name: Term name.
 *     value: New multiplier.
 */
export interface SetWeightMessage {
  type: Type10;
  name: Name;
  value: Value;
}
/**
 * Start over from a fresh initialization.
 *
 * Attributes:
 *     seed: Seed for the problem's `initialize`; `None` uses the next seed.
 */
export interface ResetMessage {
  type: Type11;
  seed?: Seed;
}
