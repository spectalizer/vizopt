import "./style.css";

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

function formatNumber(value: number): string {
  return Math.abs(value) >= 1e4 || (value !== 0 && Math.abs(value) < 1e-3)
    ? value.toExponential(2)
    : value.toPrecision(4);
}

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
const metrics = element("table", { className: "metrics" });
const hint = element(
  "p",
  { className: "hint" },
  "Drag a node to move it; others re-flow. Shift-drop keeps it pinned; double-click unpins.",
);
const sidebar = element("aside", { className: "sidebar" }, element("h2", {}, "Loss"), metrics, hint);
const toast = element("div", { className: "toast" });
document.querySelector("#app")!.append(toolbar, element("div", { className: "body" }, stage, sidebar), toast);

// --- state ---

let hello: HelloMessage | null = null;
let latest: FrameMessage | null = null;
let fitNext = true;
let renderScheduled = false;
let pendingMove: ClientMessage | null = null;
let toastTimer: number | undefined;

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

function onConnectionState(state: ConnectionState): void {
  connectionDot.className = `dot ${state}`;
  connectionDot.title = state;
  if (state !== "open") status.textContent = state === "closed" ? "disconnected — retrying…" : "connecting…";
  if (state === "open") fitNext = true;
}

function onMessage(message: ServerMessage): void {
  switch (message.type) {
    case "hello":
      hello = message;
      break;
    case "frame":
      latest = message;
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
  renderMetrics(latest);
}

function renderMetrics(frame: FrameMessage): void {
  const terms = hello?.terms ?? Object.keys(frame.metrics?.terms ?? {});
  const rows = [
    ["total", frame.metrics?.total],
    ...terms.map((name): [string, number | undefined] => [name, frame.metrics?.terms[name]]),
  ] as const;
  metrics.replaceChildren(
    ...rows.map(([name, value]) =>
      element(
        "tr",
        { className: name === "total" ? "total" : "" },
        element("th", {}, name),
        element("td", {}, value === undefined ? "–" : formatNumber(value)),
      ),
    ),
  );
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
