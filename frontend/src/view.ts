import { drag, type D3DragEvent } from "d3-drag";
import { scaleLinear, type ScaleLinear } from "d3-scale";
import { select, type Selection } from "d3-selection";
import { zoom, zoomIdentity, type ZoomBehavior, type ZoomTransform } from "d3-zoom";

import type {
  Circle,
  DragBinding,
  FrameMessage,
  Line,
  Polygon,
  Scene,
  Text,
} from "./protocol/protocol";

export type SceneElement = Circle | Line | Polygon | Text;

/** Callbacks for user interactions with draggable elements. */
export interface DragHandlers {
  start(binding: DragBinding): void;
  /** `(x, y)` is the element's new anchor position, in data coordinates. */
  move(binding: DragBinding, x: number, y: number): void;
  end(binding: DragBinding, keepPinned: boolean): void;
  unpin(binding: DragBinding): void;
  /** The user started interacting with the view (zoom, pan or drag). */
  interacted(): void;
}

const SVG_NS = "http://www.w3.org/2000/svg";
const TAGS: Record<SceneElement["kind"], string> = {
  circle: "circle",
  line: "line",
  polygon: "polygon",
  text: "text",
};
const PADDING_PX = 40;

type Scale = ScaleLinear<number, number>;
type Bounds = { x0: number; x1: number; y0: number; y1: number };

/**
 * Renders scenes into an SVG with D3, in data coordinates mapped to the
 * screen (fit-to-view, zoom and pan), and turns drags on bound elements into
 * data-space positions.
 */
export class SceneView {
  readonly svg: Selection<SVGSVGElement, unknown, null, undefined>;
  private readonly layer: Selection<SVGGElement, unknown, null, undefined>;
  private readonly zoomBehavior: ZoomBehavior<SVGSVGElement, unknown>;
  private transform: ZoomTransform = zoomIdentity;
  private baseX: Scale = scaleLinear();
  private baseY: Scale = scaleLinear();
  private frame: FrameMessage | null = null;
  private pinned = new Set<string>();

  constructor(
    container: HTMLElement,
    private readonly handlers: DragHandlers,
  ) {
    this.svg = select(container).append("svg").attr("class", "scene");
    this.svg
      .append("defs")
      .append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 9)
      .attr("refY", 5)
      .attr("markerWidth", 7)
      .attr("markerHeight", 7)
      .attr("orient", "auto-start-reverse")
      .append("path")
      .attr("d", "M0,0 L10,5 L0,10 z")
      .attr("class", "arrow-head");
    this.layer = this.svg.append("g");

    this.zoomBehavior = zoom<SVGSVGElement, unknown>().on("zoom", (event) => {
      this.transform = event.transform;
      if (event.sourceEvent) this.handlers.interacted();
      this.redraw();
    });
    this.svg.call(this.zoomBehavior).on("dblclick.zoom", null);
    new ResizeObserver(() => this.fit()).observe(container);
  }

  private get x(): Scale {
    return this.transform.rescaleX(this.baseX);
  }

  private get y(): Scale {
    return this.transform.rescaleY(this.baseY);
  }

  /** Show a new frame. */
  update(frame: FrameMessage): void {
    this.frame = frame;
    this.pinned = new Set(
      Object.entries(frame.pinned ?? {}).flatMap(([name, indices]) =>
        indices.map((index) => `${name}/${index}`),
      ),
    );
    this.redraw();
  }

  /** Fit the current scene into the view and reset zoom/pan. */
  fit(): void {
    const scene = this.frame?.scene;
    const node = this.svg.node();
    if (!scene || !node) return;
    const { width, height } = node.getBoundingClientRect();
    const bounds = sceneBounds(scene);
    if (!bounds || width === 0 || height === 0) return;

    const w = Math.max(width - 2 * PADDING_PX, 1);
    const h = Math.max(height - 2 * PADDING_PX, 1);
    let { x0, x1, y0, y1 } = bounds;
    // A single point (or a flat scene) still needs a non-empty domain.
    if (x1 === x0) [x0, x1] = [x0 - 0.5, x1 + 0.5];
    if (y1 === y0) [y0, y1] = [y0 - 0.5, y1 + 0.5];
    if (scene.equal_aspect ?? true) {
      const unit = Math.min(w / (x1 - x0), h / (y1 - y0)); // pixels per data unit
      const cx = (x0 + x1) / 2;
      const cy = (y0 + y1) / 2;
      [x0, x1] = [cx - w / (2 * unit), cx + w / (2 * unit)];
      [y0, y1] = [cy - h / (2 * unit), cy + h / (2 * unit)];
    }
    const yUp = (scene.y_axis ?? "up") === "up";
    this.baseX = scaleLinear().domain([x0, x1]).range([PADDING_PX, PADDING_PX + w]);
    this.baseY = scaleLinear()
      .domain([y0, y1])
      .range(yUp ? [PADDING_PX + h, PADDING_PX] : [PADDING_PX, PADDING_PX + h]);
    // Resets the transform, which triggers a redraw via the zoom listener.
    this.svg.call(this.zoomBehavior.transform, zoomIdentity);
  }

  private redraw(): void {
    const elements = this.frame?.scene.elements ?? [];
    const join = this.layer
      .selectChildren<SVGElement, SceneElement>()
      .data(elements, (d) => d.id);
    join.exit().remove();
    const entered = join
      .enter()
      .append((d) => document.createElementNS(SVG_NS, TAGS[d.kind]) as SVGElement)
      .attr("class", (d) => `el el-${d.kind}`)
      .each((d, i, nodes) => {
        if (d.drag) this.makeDraggable(select(nodes[i]));
      });
    const all = entered.merge(join).order();
    all.each((d, i, nodes) => this.draw(nodes[i], d));
  }

  private draw(node: SVGElement, d: SceneElement): void {
    const x = this.x;
    const y = this.y;
    switch (d.kind) {
      case "circle": {
        node.setAttribute("cx", String(x(d.cx)));
        node.setAttribute("cy", String(y(d.cy)));
        const r = (d.radius_units ?? "data") === "px" ? d.r : Math.abs(x(d.r) - x(0));
        node.setAttribute("r", String(r));
        break;
      }
      case "line": {
        let [x1, y1, x2, y2] = [x(d.x1), y(d.y1), x(d.x2), y(d.y2)];
        const length = Math.hypot(x2 - x1, y2 - y1) || 1;
        const [ux, uy] = [(x2 - x1) / length, (y2 - y1) / length];
        const start = d.shorten_start_px ?? 0;
        const end = d.shorten_end_px ?? 0;
        [x1, y1] = [x1 + ux * start, y1 + uy * start];
        [x2, y2] = [x2 - ux * end, y2 - uy * end];
        node.setAttribute("x1", String(x1));
        node.setAttribute("y1", String(y1));
        node.setAttribute("x2", String(x2));
        node.setAttribute("y2", String(y2));
        if (d.arrow_end) node.setAttribute("marker-end", "url(#arrow)");
        else node.removeAttribute("marker-end");
        break;
      }
      case "polygon":
        node.setAttribute("points", d.points.map(([px, py]) => `${x(px)},${y(py)}`).join(" "));
        break;
      case "text":
        node.setAttribute("x", String(x(d.x) + (d.dx_px ?? 0)));
        node.setAttribute("y", String(y(d.y) - (d.dy_px ?? 0)));
        node.setAttribute("text-anchor", d.anchor ?? "start");
        node.style.fontSize = `${d.font_size_px ?? 12}px`;
        node.textContent = d.text;
        break;
    }
    applyStyle(node, d);
    node.classList.toggle("draggable", Boolean(d.drag));
    node.classList.toggle(
      "pinned",
      Boolean(d.drag && this.pinned.has(`${d.drag.var}/${d.drag.index}`)),
    );
    setTooltip(node, d.tooltip ?? null);
  }

  private makeDraggable(selection: Selection<SVGElement, SceneElement, null, undefined>): void {
    let offset: [number, number] = [0, 0];
    type DragEvent = D3DragEvent<SVGElement, SceneElement, SceneElement>;
    const pointer = (event: DragEvent): [number, number] => [
      this.x.invert(event.x),
      this.y.invert(event.y),
    ];
    selection
      .call(
        drag<SVGElement, SceneElement>()
          .on("start", (event: DragEvent, d) => {
            if (!d.drag) return;
            const [px, py] = pointer(event);
            const [ax, ay] = anchor(d) ?? [px, py];
            offset = [ax - px, ay - py];
            this.svg.classed("dragging", true);
            this.handlers.interacted();
            this.handlers.start(d.drag);
          })
          .on("drag", (event: DragEvent, d) => {
            if (!d.drag) return;
            const [px, py] = pointer(event);
            this.handlers.move(d.drag, px + offset[0], py + offset[1]);
          })
          .on("end", (event: DragEvent, d) => {
            if (!d.drag) return;
            this.svg.classed("dragging", false);
            const keepPinned = Boolean((event.sourceEvent as MouseEvent | undefined)?.shiftKey);
            this.handlers.end(d.drag, keepPinned);
          }),
      )
      .on("dblclick", (_event: MouseEvent, d) => {
        if (d.drag) this.handlers.unpin(d.drag);
      });
  }
}

/** The data-space point a drag binding refers to, if the element has one. */
function anchor(d: SceneElement): [number, number] | null {
  switch (d.kind) {
    case "circle":
      return [d.cx, d.cy];
    case "text":
      return [d.x, d.y];
    default:
      return null;
  }
}

function sceneBounds(scene: Scene): Bounds | null {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const d of scene.elements ?? []) {
    switch (d.kind) {
      case "circle": {
        const r = (d.radius_units ?? "data") === "data" ? d.r : 0;
        xs.push(d.cx - r, d.cx + r);
        ys.push(d.cy - r, d.cy + r);
        break;
      }
      case "line":
        xs.push(d.x1, d.x2);
        ys.push(d.y1, d.y2);
        break;
      case "polygon":
        for (const [px, py] of d.points) {
          xs.push(px);
          ys.push(py);
        }
        break;
      case "text":
        xs.push(d.x);
        ys.push(d.y);
        break;
    }
  }
  if (xs.length === 0) return null;
  return {
    x0: Math.min(...xs),
    x1: Math.max(...xs),
    y0: Math.min(...ys),
    y1: Math.max(...ys),
  };
}

/** Inline styles beat the CSS defaults; unset fields fall back to them. */
function applyStyle(node: SVGElement, d: SceneElement): void {
  const style = d.style ?? {};
  node.style.fill = style.fill ?? "";
  node.style.stroke = style.stroke ?? "";
  node.style.strokeWidth = style.stroke_width_px != null ? `${style.stroke_width_px}px` : "";
  node.style.opacity = style.opacity != null ? String(style.opacity) : "";
}

function setTooltip(node: SVGElement, tooltip: string | null): void {
  let title = node.querySelector(":scope > title");
  if (tooltip === null) {
    title?.remove();
    return;
  }
  if (!title) {
    title = document.createElementNS(SVG_NS, "title");
    node.appendChild(title);
  }
  title.textContent = tooltip;
}
