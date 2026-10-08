import { axisBottom, axisLeft } from "d3-axis";
import { scaleLinear, scaleLog, type ScaleContinuousNumeric } from "d3-scale";
import { pointer, select, type Selection } from "d3-selection";
import { line } from "d3-shape";

import type { Metrics } from "./protocol/protocol";

/** One plotted series: "total" or a term name, with its CSS color. */
export interface Series {
  name: string;
  color: string;
}

const MAX_POINTS = 2000;
const HEIGHT = 170;
const MARGIN = { top: 8, right: 10, bottom: 22, left: 44 };

type YScale = ScaleContinuousNumeric<number, number>;

/**
 * Loss history over iterations, with a crosshair tooltip. The y-axis is
 * logarithmic while every plotted value is positive (losses usually span
 * orders of magnitude), and linear otherwise (objectives with negative
 * terms).
 */
export class LossChart {
  private iterations: number[] = [];
  private values = new Map<string, number[]>();
  private series: Series[] = [];
  private readonly svg: Selection<SVGSVGElement, unknown, null, undefined>;
  private readonly tooltip: HTMLDivElement;
  private hoverX: number | null = null;
  private logScale = true;

  constructor(private readonly container: HTMLElement) {
    container.classList.add("chart");
    this.svg = select(container)
      .append("svg")
      .attr("height", HEIGHT)
      .attr("role", "img")
      .attr("aria-label", "Loss over iterations, log scale");
    this.tooltip = container.appendChild(document.createElement("div"));
    this.tooltip.className = "chart-tooltip";
    this.svg
      .on("pointermove", (event: PointerEvent) => {
        this.hoverX = pointer(event)[0];
        this.draw();
      })
      .on("pointerleave", () => {
        this.hoverX = null;
        this.draw();
      });
    new ResizeObserver(() => this.draw()).observe(container);
  }

  /** Append one sample; a smaller iteration than the last starts a new run. */
  push(iteration: number, metrics: Metrics): void {
    const last = this.iterations.at(-1);
    if (last !== undefined && iteration < last) this.clear();
    if (iteration === last) return;
    this.iterations.push(iteration);
    this.sample("total").push(metrics.total);
    for (const [name, value] of Object.entries(metrics.terms)) this.sample(name).push(value);
    if (this.iterations.length > MAX_POINTS) this.thin();
  }

  clear(): void {
    this.iterations = [];
    this.values.clear();
  }

  /** Choose which series to plot, in painting order. */
  setSeries(series: Series[]): void {
    this.series = series;
    this.draw();
  }

  draw(): void {
    const width = this.container.clientWidth;
    this.svg.attr("width", width).selectAll("*").remove();
    this.tooltip.classList.remove("visible");
    const n = this.iterations.length;
    if (n < 2 || width === 0) return;

    const plotted = this.series.filter((s) => this.values.has(s.name));
    const finite = plotted.flatMap((s) => this.values.get(s.name)!.filter(Number.isFinite));
    if (finite.length === 0) return;
    this.logScale = finite.every((v) => v > 0);
    this.svg.attr("aria-label", `Loss over iterations, ${this.logScale ? "log" : "linear"} scale`);
    let [lo, hi] = [Math.min(...finite), Math.max(...finite)];
    if (lo === hi) [lo, hi] = this.logScale ? [lo / 2, hi * 2] : [lo - 1, hi + 1];

    const x = scaleLinear()
      .domain([this.iterations[0], this.iterations[n - 1]])
      .range([MARGIN.left, width - MARGIN.right]);
    const y: YScale = (this.logScale ? scaleLog() : scaleLinear())
      .domain([lo, hi])
      .range([HEIGHT - MARGIN.bottom, MARGIN.top])
      .nice();

    const yAxis = this.svg
      .append("g")
      .attr("class", "axis")
      .attr("transform", `translate(${MARGIN.left},0)`)
      .call(
        axisLeft(y)
          .ticks(4, this.logScale ? "~e" : "~g")
          .tickSize(-(width - MARGIN.left - MARGIN.right)),
      );
    yAxis.select(".domain").remove();
    yAxis.selectAll(".tick line").attr("class", "grid");
    const xAxis = this.svg
      .append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${HEIGHT - MARGIN.bottom})`)
      .call(axisBottom(x).ticks(4, "~s").tickSizeOuter(0));
    xAxis.selectAll(".tick line").remove();

    for (const s of plotted) {
      const values = this.values.get(s.name)!;
      const path = line<number>()
        .defined((i) => this.plottable(values[i]))
        .x((i) => x(this.iterations[i]))
        .y((i) => y(values[i]));
      this.svg
        .append("path")
        .attr("class", "series")
        .style("stroke", s.color)
        .attr("d", path(this.iterations.map((_, i) => i)));
    }

    if (this.hoverX !== null) this.drawHover(x, y, plotted);
  }

  private drawHover(
    x: ReturnType<typeof scaleLinear<number, number>>,
    y: YScale,
    plotted: Series[],
  ): void {
    const target = x.invert(this.hoverX!);
    let i = this.iterations.findIndex((it) => it >= target);
    if (i < 0) i = this.iterations.length - 1;
    if (i > 0 && target - this.iterations[i - 1] < this.iterations[i] - target) i -= 1;
    const px = x(this.iterations[i]);

    this.svg
      .append("line")
      .attr("class", "crosshair")
      .attr("x1", px)
      .attr("x2", px)
      .attr("y1", MARGIN.top)
      .attr("y2", HEIGHT - MARGIN.bottom);
    for (const s of plotted) {
      const v = this.values.get(s.name)![i];
      if (!this.plottable(v)) continue;
      this.svg
        .append("circle")
        .attr("class", "hover-dot")
        .attr("cx", px)
        .attr("cy", y(v))
        .attr("r", 4)
        .style("fill", s.color);
    }

    this.tooltip.replaceChildren(
      Object.assign(document.createElement("div"), {
        className: "chart-tooltip-title",
        textContent: `iteration ${this.iterations[i]}`,
      }),
      ...plotted.map((s) => {
        const row = document.createElement("div");
        const swatch = row.appendChild(document.createElement("span"));
        swatch.className = "swatch";
        swatch.style.background = s.color;
        row.append(`${s.name}  ${formatValue(this.values.get(s.name)![i])}`);
        return row;
      }),
    );
    this.tooltip.classList.add("visible");
    const left = px + 12 + this.tooltip.offsetWidth > this.container.clientWidth;
    this.tooltip.style.left = `${left ? px - 12 - this.tooltip.offsetWidth : px + 12}px`;
  }

  /** Whether a value can be drawn on the current y scale. */
  private plottable(value: number): boolean {
    return this.logScale ? value > 0 : Number.isFinite(value);
  }

  private sample(name: string): number[] {
    let values = this.values.get(name);
    if (!values) {
      // A series first seen mid-run is padded so indices line up.
      values = new Array(Math.max(this.iterations.length - 1, 0)).fill(Number.NaN);
      this.values.set(name, values);
    }
    return values;
  }

  /** Halve the resolution once the history gets long, keeping the latest sample. */
  private thin(): void {
    const keep = (_: unknown, i: number, all: unknown[]) => i % 2 === 0 || i === all.length - 1;
    this.iterations = this.iterations.filter(keep);
    for (const [name, values] of this.values) this.values.set(name, values.filter(keep));
  }
}

export function formatValue(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) return "–";
  return Math.abs(value) >= 1e4 || (value !== 0 && Math.abs(value) < 1e-3)
    ? value.toExponential(2)
    : value.toPrecision(4);
}
