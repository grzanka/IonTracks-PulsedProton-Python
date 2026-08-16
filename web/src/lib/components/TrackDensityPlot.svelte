<script lang="ts">
  // Track areal density cross-section: a full (no_xy x no_xy) field, not a
  // scalar time series, so it's rendered once at run completion rather than
  // streamed at the live-plot cadence -- see pulsed_ion_chamber/plots.py's
  // "4. track areal density cross-section" figure, which this ports (issue
  // #6 milestone 5: the original web port left this view out entirely).
  //
  // `density` is `no_xy * no_xy`, flattened row-major as
  // `density[i * no_xy + j]` -- i is the x-voxel, j the y-voxel, matching
  // core/src/solver.rs's `track_density_xy` (itself a port of
  // `pulsed_ion_chamber/state.py`'s `Diagnostics.count_track`).

  interface Props {
    density: number[];
    noXy: number;
    unitLengthCm: number;
    // Scored-disc radius, in voxels -- this crate fixes
    // chamber_fill_fraction at 1.0, so the sampling and scored radii always
    // coincide (unlike the Python CLI, which can draw two circles).
    innerRadiusVoxels: number;
    title?: string;
  }

  let { density, noXy, unitLengthCm, innerRadiusVoxels, title = "" }: Props = $props();

  let canvas: HTMLCanvasElement | undefined = $state();

  const DISPLAY_SIZE = 320;

  // A small hand-picked sequential ramp (light -> dark blue), not a
  // colormap library -- this is one static image per run, not worth a
  // dependency for.
  const STOPS: [number, [number, number, number]][] = [
    [0.0, [247, 251, 255]],
    [0.25, [198, 219, 239]],
    [0.5, [107, 174, 214]],
    [0.75, [33, 113, 181]],
    [1.0, [8, 48, 107]],
  ];

  const LAST_STOP = STOPS[STOPS.length - 1] ?? [1.0, [8, 48, 107]];

  function colorFor(t: number): [number, number, number] {
    const clamped = Math.min(1, Math.max(0, t));
    let previous = STOPS[0] ?? LAST_STOP;
    for (const stop of STOPS) {
      const [t1, c1] = stop;
      if (clamped <= t1) {
        const [t0, c0] = previous;
        const frac = t1 === t0 ? 0 : (clamped - t0) / (t1 - t0);
        return [
          Math.round(c0[0] + (c1[0] - c0[0]) * frac),
          Math.round(c0[1] + (c1[1] - c0[1]) * frac),
          Math.round(c0[2] + (c1[2] - c0[2]) * frac),
        ];
      }
      previous = stop;
    }
    return LAST_STOP[1];
  }

  function draw(): void {
    if (!canvas || noXy <= 0 || density.length < noXy * noXy) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let peak = 0;
    for (const v of density) if (v > peak) peak = v;

    // Render at native no_xy resolution on an offscreen canvas -- one pixel
    // per voxel, exactly what was actually resolved -- then scale that
    // bitmap up onto the display canvas with smoothing off, so voxel edges
    // stay crisp instead of blurring into something that looks more
    // resolved than the grid actually is.
    const offscreen = document.createElement("canvas");
    offscreen.width = noXy;
    offscreen.height = noXy;
    const octx = offscreen.getContext("2d");
    if (!octx) return;
    const image = octx.createImageData(noXy, noXy);
    for (let i = 0; i < noXy; i++) {
      for (let j = 0; j < noXy; j++) {
        const value = density[i * noXy + j] ?? 0;
        const [r, g, b] = colorFor(peak > 0 ? value / peak : 0);
        // Flip j so larger y is drawn higher up (origin="lower", matching
        // plots.py's imshow convention) -- canvas rows increase downward.
        const row = noXy - 1 - j;
        const idx = (row * noXy + i) * 4;
        image.data[idx] = r;
        image.data[idx + 1] = g;
        image.data[idx + 2] = b;
        image.data[idx + 3] = 255;
      }
    }
    octx.putImageData(image, 0, 0);

    ctx.clearRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, 0, 0, DISPLAY_SIZE, DISPLAY_SIZE);

    // Overlay the scored-disc circle at native display resolution, so it
    // stays a smooth curve regardless of how coarse the voxel grid is.
    const scale = DISPLAY_SIZE / noXy;
    const centre = DISPLAY_SIZE / 2;
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.arc(centre, centre, innerRadiusVoxels * scale, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  $effect(() => {
    void density;
    void noXy;
    void innerRadiusVoxels;
    draw();
  });

  let halfWidthUm = $derived((noXy / 2) * unitLengthCm * 1e4);
</script>

<figure>
  {#if title}
    <figcaption>{title}</figcaption>
  {/if}
  <canvas
    bind:this={canvas}
    width={DISPLAY_SIZE}
    height={DISPLAY_SIZE}
    aria-label="Track areal density cross-section"
  ></canvas>
  <p class="axis-note">
    ±{halfWidthUm.toFixed(0)} µm from centre; dashed circle marks the scored disc.
  </p>
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
    aspect-ratio: 1 / 1;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
  }
  .axis-note {
    margin: 0;
    font-size: 0.78rem;
    color: #64748b;
  }
</style>
