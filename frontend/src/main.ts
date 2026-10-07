import "./style.css";

import { formatValue, LossChart, type Series } from "./chart";
import { Connection, defaultSocketUrl, type ConnectionState } from "./connection";
import type {
  ClientMessage,
  DragBinding,
  FrameMessage,
  HelloMessage,
  ServerMessage,
} from "./protocol/protocol";
import { SceneView } from "./view";

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Partial<HTMLElementTagNameMap[K]> = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

/** Validated categorical slots (see style.css); a term keeps its slot while plotted. */
const SERIES_SLOTS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
const TOTAL_COLOR = "var(--chart-total)";
/** The weight slider spans this many decades either side of the initial weight. */
const WEIGHT_DECADES = 2;

// --- layout ---

const connectionDot = element("span", { className: "dot", title: "connecting" });
const status = element("span", { className: "status" }, "connecting…");
const pauseButton = element("button", { type: "button" }, "Pause");
const reheatButton = element("button", { type: "button", title: "Restart the learning-rate decay" }, "Reheat");
const resetButton = element("button", { type: "button", title: "Start over from a new initialization" }, "Reset");
const fitButton = element("button", { type: "button", title: "Fit the scene into the view" }, "Fit");
const autoFit = element("input", { type: "checkbox", checked: true });
const toolbar = element(
  "header",
  { className: "toolbar" },
  connectionDot,
  status,
  pauseButton,
  reheatButton,
  resetButton,
  element("span", { className: "spacer" }),
  fitButton,
  element("label", { title: "Keep the scene fitted until you zoom, pan or drag" }, autoFit, "Auto-fit"),
);
const stage = element("main", { className: "stage" });
const chartContainer = element("div");
const terms = element("table", { className: "terms" });
const hint = element(
  "p",
  { className: "hint" },
  "Drag elements to move them; the rest re-flows. Shift-drop keeps one pinned; double-click unpins. " +
    "Click a term's swatch to plot it (up to 3). Weight sliders are logarithmic; double-click one to reset it.",
);
const sidebar = element(
  "aside",
  { className: "sidebar" },
  element("h2", {}, "Loss"),
  chartContainer,
  element("h2", {}, "Terms"),
  terms,
  hint,
);
const toast = element("div", { className: "toast" });
document.querySelector("#app")!.append(toolbar, element("div", { className: "body" }, stage, sidebar), toast);

// --- state ---

interface TermRow {
  swatch: HTMLButtonElement;
  value: HTMLTableCellElement;
  slider: HTMLInputElement;
  weight: HTMLSpanElement;
  initialWeight: number;
}

let latest: FrameMessage | null = null;
let fitNext = true;
let renderScheduled = false;
let pendingMove: ClientMessage | null = null;
let pendingWeights = new Map<string, number>();
let activeSlider: string | null = null;
let toastTimer: number | undefined;
const termRows = new Map<string, TermRow>();
/** Plotted terms → their slot index in SERIES_SLOTS. */
const plotted = new Map<string, number>();
let totalValue: HTMLTableCellElement | null = null;

const chart = new LossChart(chartContainer);

const view = new SceneView(stage, {
  start: (binding) => send({ type: "drag_start", ...binding }),
  move: (binding, x, y) => {
    // Coalesce to one message per animation frame.
    if (!pendingMove) requestAnimationFrame(flushMove);
    pendingMove = { type: "drag", ...binding, x, y };
  },
  end: (binding, keepPinned) => {
    flushMove();
    send({ type: "drag_end", ...binding, keep_pinned: keepPinned });
  },
  unpin: (binding: DragBinding) => send({ type: "unpin", ...binding }),
  interacted: () => {
    autoFit.checked = false;
  },
});

const connection = new Connection(defaultSocketUrl(), onMessage, onConnectionState);

function send(message: ClientMessage): void {
  if (!connection.send(message)) showToast("Not connected.");
}

function flushMove(): void {
  if (pendingMove) send(pendingMove);
  pendingMove = null;
}

function flushWeights(): void {
  for (const [name, value] of pendingWeights) send({ type: "set_weight", name, value });
  pendingWeights = new Map();
}

function onConnectionState(state: ConnectionState): void {
  connectionDot.className = `dot ${state}`;
  connectionDot.title = state;
  if (state !== "open") status.textContent = state === "closed" ? "disconnected — retrying…" : "connecting…";
  if (state === "open") fitNext = true;
}

function onMessage(message: ServerMessage): void {
  switch (message.type) {
    case "hello":
      chart.clear();
      for (const point of message.history ?? []) chart.push(point.iteration, point);
      buildTermsPanel(message);
      break;
    case "frame":
      latest = message;
      if (message.metrics) chart.push(message.iteration, message.metrics);
      if (!renderScheduled) {
        renderScheduled = true;
        requestAnimationFrame(render);
      }
      break;
    case "error":
      showToast(message.message);
      break;
  }
}

function render(): void {
  renderScheduled = false;
  if (!latest) return;
  view.update(latest);
  if (fitNext || autoFit.checked) {
    view.fit();
    fitNext = false;
  }
  status.textContent = `${latest.running ? "running" : latest.paused ? "paused" : "settled"} · iteration ${latest.iteration}`;
  pauseButton.textContent = latest.paused ? "Resume" : "Pause";
  updateTermsPanel(latest);
  chart.setSeries(chartSeries());
}

// --- terms panel ---

function chartSeries(): Series[] {
  return [
    { name: "total", color: TOTAL_COLOR },
    ...[...plotted].map(([name, slot]) => ({ name, color: SERIES_SLOTS[slot] })),
  ];
}

function togglePlotted(name: string): void {
  if (plotted.has(name)) {
    plotted.delete(name);
  } else {
    const used = new Set(plotted.values());
    const slot = SERIES_SLOTS.findIndex((_, i) => !used.has(i));
    if (slot < 0) {
      showToast(`At most ${SERIES_SLOTS.length} terms can be plotted; remove one first.`);
      return;
    }
    plotted.set(name, slot);
  }
  updateSwatches();
  chart.setSeries(chartSeries());
}

function updateSwatches(): void {
  for (const [name, row] of termRows) {
    const slot = plotted.get(name);
    row.swatch.style.background = slot === undefined ? "" : SERIES_SLOTS[slot];
    row.swatch.classList.toggle("on", slot !== undefined);
    row.swatch.setAttribute("aria-pressed", String(slot !== undefined));
  }
}

function buildTermsPanel(message: HelloMessage): void {
  termRows.clear();
  for (const name of [...plotted.keys()]) if (!message.terms.includes(name)) plotted.delete(name);
  const adjustable = new Set(message.adjustable_terms);

  totalValue = element("td", { className: "value" }, "–");
  const totalSwatch = element("span", { className: "swatch on", title: "Total loss (always plotted)" });
  totalSwatch.style.background = TOTAL_COLOR;
  const rows: HTMLTableRowElement[] = [
    element("tr", { className: "total" }, element("td", {}, totalSwatch), element("th", {}, "total"), totalValue),
  ];

  for (const name of message.terms) {
    const initialWeight = message.weights[name] ?? 0;
    const canAdjust = adjustable.has(name) && initialWeight > 0;
    const swatch = element("button", {
      type: "button",
      className: "swatch",
      title: `Plot ${name}`,
    });
    swatch.addEventListener("click", () => togglePlotted(name));
    const value = element("td", { className: "value" }, "–");
    const slider = element("input", {
      type: "range",
      min: String(-WEIGHT_DECADES),
      max: String(WEIGHT_DECADES),
      step: "0.01",
      value: "0",
      disabled: !canAdjust,
      title: canAdjust
        ? "Weight (log scale); double-click to reset"
        : "This term was disabled when the problem was built",
    });
    const weight = element("span", { className: "weight" }, formatValue(initialWeight));
    if (canAdjust) {
      slider.addEventListener("pointerdown", () => (activeSlider = name));
      slider.addEventListener("pointerup", () => (activeSlider = null));
      slider.addEventListener("input", () => {
        const w = initialWeight * 10 ** Number(slider.value);
        weight.textContent = formatValue(w);
        if (pendingWeights.size === 0) requestAnimationFrame(flushWeights);
        pendingWeights.set(name, w);
      });
      slider.addEventListener("dblclick", () => {
        slider.value = "0";
        weight.textContent = formatValue(initialWeight);
        send({ type: "set_weight", name, value: initialWeight });
      });
    }
    termRows.set(name, { swatch, value, slider, weight, initialWeight });
    rows.push(
      element("tr", {}, element("td", {}, swatch), element("th", {}, name), value),
      element(
        "tr",
        { className: "weight-row" },
        element("td"),
        element("td", { colSpan: 2 }, element("div", { className: "weight-control" }, slider, weight)),
      ),
    );
  }
  terms.replaceChildren(...rows);
  updateSwatches();
}

function updateTermsPanel(frame: FrameMessage): void {
  if (totalValue) totalValue.textContent = formatValue(frame.metrics?.total);
  for (const [name, row] of termRows) {
    row.value.textContent = formatValue(frame.metrics?.terms[name]);
    const w = frame.weights?.[name];
    if (w === undefined || activeSlider === name || pendingWeights.has(name)) continue;
    row.weight.textContent = formatValue(w);
    if (row.initialWeight > 0 && w > 0) row.slider.value = String(Math.log10(w / row.initialWeight));
  }
}

function showToast(text: string): void {
  toast.textContent = text;
  toast.classList.add("visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 4000);
}

// --- controls ---

pauseButton.addEventListener("click", () => send({ type: latest?.paused ? "resume" : "pause" }));
reheatButton.addEventListener("click", () => send({ type: "reheat" }));
resetButton.addEventListener("click", () => {
  fitNext = true;
  send({ type: "reset" });
});
fitButton.addEventListener("click", () => view.fit());
