<script lang="ts">
  // Small dependency-free canvas line plot: this page only ever needs two or
  // three short series (a few thousand points at this prototype's grid caps),
  // which doesn't justify pulling in a charting library.
  interface Series {
    label: string;
    color: string;
    values: number[];
    dashed?: boolean;
  }

  let {
    xValues,
    series,
    valueDivisor = 1,
    valueMax,
    xLabel = "time [µs]",
    yLabel = "",
    title = "",
  }: {
    xValues: number[];
    // Raw, unscaled values -- divided by valueDivisor only for the
    // decimated points actually drawn (see draw()), not the whole series on
    // every redraw (issue #19 W7).
    series: Series[];
    valueDivisor?: number;
    // Raw (unscaled) peak across all series and all points, not just the
    // decimated ones drawn -- pass this whenever the caller already tracks
    // it incrementally (as +page.svelte's carrierPeak/recombinedPeak do).
    // Without it, the y-axis scan below falls back to a full, undecimated
    // pass so the scale stays correct; scanning only the decimated points
    // would under-report a peak that falls between sampled indices,
    // clipping the line and mislabelling the axis (PR #20 review).
    valueMax?: number;
    xLabel?: string;
    yLabel?: string;
    title?: string;
  } = $props();

  let canvas: HTMLCanvasElement | undefined = $state();

  const WIDTH = 640;
  const HEIGHT = 280;
  const PAD_LEFT = 56;
  const PAD_BOTTOM = 34;
  const PAD_TOP = 24;
  const PAD_RIGHT = 12;

  function draw(): void {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, WIDTH, HEIGHT);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, WIDTH, HEIGHT);

    const plotWidth = WIDTH - PAD_LEFT - PAD_RIGHT;
    const plotHeight = HEIGHT - PAD_TOP - PAD_BOTTOM;

    // Not Math.max(...xValues, 1e-12): spreading into a call is bounded by
    // the engine's argument limit (~128k on V8) and MAX_TOTAL_TIME_STEPS is
    // 200k, so a long run would throw mid-draw (issue #19 W2). xValues is
    // monotonically increasing simulation time, so the last element is
    // already the max.
    const xMax = xValues.length > 0 ? Math.max(xValues[xValues.length - 1] ?? 0, 1e-12) : 1;

    // Strided down to at most one point per plotted pixel, computed once and
    // reused below for both the y-axis scan and the draw itself -- without
    // this, a long run repaints (and rescans for yMax) every one of up to
    // MAX_TOTAL_TIME_STEPS points on every ~150 ms progress chunk, making a
    // single redraw's cost grow with the whole run instead of staying flat
    // (issue #19 W7). Always keeps the last point, so the live edge of the
    // plot is never more than one stride stale.
    const stride = Math.max(1, Math.ceil(xValues.length / plotWidth));
    const indices: number[] = [];
    for (let i = 0; i < xValues.length; i += stride) indices.push(i);
    if (indices.length > 0 && indices[indices.length - 1] !== xValues.length - 1) {
      indices.push(xValues.length - 1);
    }

    let yMax = 0;
    if (valueMax !== undefined) {
      yMax = valueMax / valueDivisor;
    } else {
      // No precomputed peak given -- fall back to a correct (undecimated)
      // scan rather than one over just `indices`, which would silently
      // under-report a peak sitting between strided samples.
      for (const s of series) {
        for (const v of s.values) {
          const scaled = v / valueDivisor;
          if (scaled > yMax) yMax = scaled;
        }
      }
    }
    if (yMax <= 0) yMax = 1;

    const xToPx = (x: number) => PAD_LEFT + (x / xMax) * plotWidth;
    const yToPx = (y: number) => PAD_TOP + plotHeight - (y / yMax) * plotHeight;

    // axes
    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD_LEFT, PAD_TOP);
    ctx.lineTo(PAD_LEFT, PAD_TOP + plotHeight);
    ctx.lineTo(PAD_LEFT + plotWidth, PAD_TOP + plotHeight);
    ctx.stroke();

    ctx.fillStyle = "#475569";
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) {
      const y = (yMax * i) / 4;
      const py = yToPx(y);
      ctx.fillText(y.toFixed(2), PAD_LEFT - 6, py);
      ctx.strokeStyle = "#e2e8f0";
      ctx.beginPath();
      ctx.moveTo(PAD_LEFT, py);
      ctx.lineTo(PAD_LEFT + plotWidth, py);
      ctx.stroke();
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i <= 4; i++) {
      const x = (xMax * i) / 4;
      ctx.fillText(x.toFixed(1), xToPx(x), PAD_TOP + plotHeight + 6);
    }

    for (const s of series) {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      if (s.dashed) ctx.setLineDash([5, 3]);
      else ctx.setLineDash([]);
      ctx.beginPath();
      indices.forEach((i, n) => {
        const x = xToPx(xValues[i] ?? 0);
        const y = yToPx((s.values[i] ?? 0) / valueDivisor);
        if (n === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    ctx.setLineDash([]);

    ctx.fillStyle = "#0f172a";
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText(xLabel, PAD_LEFT + plotWidth / 2, HEIGHT - 4);

    ctx.save();
    ctx.translate(12, PAD_TOP + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }

  $effect(() => {
    // Re-run whenever the data changes.
    void xValues;
    void series;
    void valueDivisor;
    void valueMax;
    draw();
  });
</script>

<figure>
  {#if title}
    <figcaption>{title}</figcaption>
  {/if}
  <canvas bind:this={canvas} width={WIDTH} height={HEIGHT}></canvas>
  <div class="legend">
    {#each series as s (s.label)}
      <span class="legend-item">
        <span class="swatch" style:background={s.color}></span>
        {s.label}
      </span>
    {/each}
  </div>
</figure>

<style>
  figure {
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  figcaption {
    font-weight: 600;
    font-size: 0.9rem;
    color: #1e293b;
  }
  canvas {
    max-width: 100%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
  }
  .legend {
    display: flex;
    gap: 1rem;
    font-size: 0.8rem;
    color: #475569;
  }
  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }
</style>
