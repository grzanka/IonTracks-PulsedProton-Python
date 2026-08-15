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
    xLabel = "time [µs]",
    yLabel = "",
    title = "",
  }: {
    xValues: number[];
    series: Series[];
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

    const xMax = xValues.length > 0 ? Math.max(...xValues, 1e-12) : 1;
    let yMax = 0;
    for (const s of series) {
      for (const v of s.values) if (v > yMax) yMax = v;
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
      for (let i = 0; i < s.values.length; i++) {
        const x = xToPx(xValues[i] ?? 0);
        const y = yToPx(s.values[i] ?? 0);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
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
