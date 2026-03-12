import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

const DATA_URL = "../data/optimized_layout.json";

// ── Colour palette ────────────────────────────────────────────────────────────
const palette = d3.schemeTableau10.concat(d3.schemePastel1, d3.schemeSet3);
function setColor(i) { return palette[i % palette.length]; }

// ── Load data ─────────────────────────────────────────────────────────────────
const data = await fetch(DATA_URL).then(r => r.json());
const { circles, sets } = data;

// ── Layout constants ──────────────────────────────────────────────────────────
const canvasEl = document.getElementById("canvas");
const W = canvasEl.clientWidth || window.innerWidth - 220;
const H = canvasEl.clientHeight || window.innerHeight;

// Compute data extent (union of circle bboxes + set boundaries)
const allX = circles.flatMap(c => [c.x - c.r, c.x + c.r])
  .concat(sets.flatMap(s => s.boundary.map(p => p[0])));
const allY = circles.flatMap(c => [c.y - c.r, c.y + c.r])
  .concat(sets.flatMap(s => s.boundary.map(p => p[1])));
const pad = 0.05;
const [x0, x1] = d3.extent(allX);
const [y0, y1] = d3.extent(allY);
const dataW = (x1 - x0) * (1 + 2 * pad);
const dataH = (y1 - y0) * (1 + 2 * pad);
const pxPerUnit = Math.min(W / dataW, H / dataH);
const offsetX = (W - dataW * pxPerUnit) / 2;
const offsetY = (H - dataH * pxPerUnit) / 2;

const scaleX = d3.scaleLinear()
  .domain([x0 - pad * (x1 - x0), x1 + pad * (x1 - x0)])
  .range([offsetX, offsetX + dataW * pxPerUnit]);
const scaleY = d3.scaleLinear()
  .domain([y0 - pad * (y1 - y0), y1 + pad * (y1 - y0)])
  .range([offsetY + dataH * pxPerUnit, offsetY]);

// ── SVG setup ─────────────────────────────────────────────────────────────────
const svg = d3.select("#canvas").append("svg")
  .attr("width", W)
  .attr("height", H);

// Zoom
const zoomG = svg.append("g");
svg.call(
  d3.zoom()
    .scaleExtent([0.3, 20])
    .on("zoom", e => zoomG.attr("transform", e.transform))
);

// ── Active-set state ──────────────────────────────────────────────────────────
const activeSetIds = new Set(sets.map((_, i) => i));  // all on by default

function isCircleVisible(ci) {
  return sets.some((s, si) => activeSetIds.has(si) && s.memberIndices.includes(ci));
}

// Precompute memberIndices (we need them to filter in legend clicks)
// The JSON "sets" don't store membership, but we can recover it from the
// Python notebook's `sets` variable.  Instead we re-derive it from the
// boundary/circle spatial test — actually the simplest approach is to just
// parse the sets from the JSON as-is and expose all circles regardless.
// We'll keep all circles visible unless a set is toggled.
// For highlight-on-hover we map circle → sets that contain it.
// Re-derive membership: load sets array from notebook export if present,
// otherwise fall back to spatial containment via a quick point-in-polygon.

// Since the notebook exports `sets` as just boundary polygons (not membership),
// we use a pip test to find which circles belong to each set boundary.
function pointInPolygon(px, py, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

sets.forEach(s => {
  s.memberIndices = circles
    .map((c, i) => ({ c, i }))
    .filter(({ c }) => pointInPolygon(c.x, c.y, s.boundary))
    .map(({ i }) => i);
});

// ── Draw set boundaries (back layer) ─────────────────────────────────────────
const lineGen = d3.line()
  .x(p => scaleX(p[0]))
  .y(p => scaleY(p[1]))
  .curve(d3.curveCatmullRomClosed);

const boundaryG = zoomG.append("g").attr("class", "boundaries");

boundaryG.selectAll("path.set-boundary")
  .data(sets)
  .join("path")
  .attr("class", "set-boundary")
  .attr("data-idx", (_, i) => i)
  .attr("d", s => lineGen(s.boundary))
  .attr("fill", (_, i) => setColor(i))
  .attr("fill-opacity", 0.12)
  .attr("stroke", (_, i) => setColor(i))
  .attr("stroke-opacity", 0.7)
  .attr("stroke-width", 1.5);

// ── Draw circles (front layer) ────────────────────────────────────────────────
const circleG = zoomG.append("g").attr("class", "circles");

const tooltip = document.getElementById("tooltip");

circleG.selectAll("circle.circle-node")
  .data(circles)
  .join("circle")
  .attr("class", "circle-node")
  .attr("cx", c => scaleX(c.x))
  .attr("cy", c => scaleY(c.y))
  .attr("r", c => Math.abs(scaleX(c.r) - scaleX(0)))
  .attr("fill", "#7eb8d4")
  .attr("fill-opacity", 0.55)
  .attr("stroke", "#5a9ab8")
  .attr("stroke-width", 0.8)
  .on("mousemove", (event, c) => {
    tooltip.style.display = "block";
    tooltip.style.left = (event.clientX + 12) + "px";
    tooltip.style.top = (event.clientY - 8) + "px";
    const memberOf = sets
      .filter(s => s.memberIndices.includes(circles.indexOf(c)))
      .map(s => s.name).join(", ");
    tooltip.textContent = `${c.name}${memberOf ? " · " + memberOf : ""}`;
  })
  .on("mouseleave", () => { tooltip.style.display = "none"; });

// ── Circle labels (shown at high zoom) ───────────────────────────────────────
const labelG = zoomG.append("g").attr("class", "labels").attr("pointer-events", "none");

labelG.selectAll("text")
  .data(circles)
  .join("text")
  .attr("x", c => scaleX(c.x))
  .attr("y", c => scaleY(c.y))
  .attr("text-anchor", "middle")
  .attr("dominant-baseline", "middle")
  .attr("font-size", c => Math.min(9, Math.max(2, Math.abs(scaleX(c.r) - scaleX(0)) * 0.35)))
  .attr("fill", "#fff")
  .attr("fill-opacity", 0.9)
  .text(c => c.name);

// ── Legend / sidebar ──────────────────────────────────────────────────────────
const legend = document.getElementById("legend");

sets.forEach((s, i) => {
  const item = document.createElement("div");
  item.className = "legend-item";
  item.dataset.idx = i;

  const swatch = document.createElement("div");
  swatch.className = "legend-swatch";
  swatch.style.background = setColor(i);

  const label = document.createElement("span");
  label.textContent = s.name;

  item.append(swatch, label);
  legend.appendChild(item);

  item.addEventListener("click", () => {
    if (activeSetIds.has(i)) {
      activeSetIds.delete(i);
      item.classList.add("dimmed");
    } else {
      activeSetIds.add(i);
      item.classList.remove("dimmed");
    }
    updateVisibility();
  });
});

function updateVisibility() {
  boundaryG.selectAll("path.set-boundary")
    .attr("display", (_, i) => activeSetIds.has(i) ? null : "none");

  // Dim circles not covered by any active set
  circleG.selectAll("circle.circle-node")
    .attr("fill-opacity", (c, ci) => isCircleVisible(ci) ? 0.55 : 0.15)
    .attr("stroke-opacity", (c, ci) => isCircleVisible(ci) ? 1.0 : 0.3);

  labelG.selectAll("text")
    .attr("fill-opacity", (c, ci) => isCircleVisible(ci) ? 0.9 : 0.2);
}
