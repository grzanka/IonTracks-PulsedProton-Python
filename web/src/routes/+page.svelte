<script lang="ts">
  import { onMount } from "svelte";
  import LinePlot from "$lib/components/LinePlot.svelte";
  import UnitField from "$lib/components/UnitField.svelte";
  import { estimate, loadWasm, type Estimate, type SimParams } from "$lib/wasm-core/loader";
  import { SimulationController } from "$lib/sim/controller";
  import type { ProgressChunk } from "$lib/sim/protocol";
  import { formatBytes, formatCount, formatSeconds, siScale } from "$lib/format";
  import {
    DOSE_RATE_UNITS,
    ENERGY_UNITS,
    GAP_LENGTH_UNITS,
    GRID_SIZE_UNITS,
    MARKUS_FULL_RADIUS_CM,
    PULSE_TIME_UNITS,
    RADIUS_LENGTH_UNITS,
    VOLTAGE_UNITS,
  } from "$lib/units";

  type Phase = "setup" | "running" | "done" | "cancelled" | "error";

  let wasmReady = $state(false);
  // All in the base unit each UnitField below converts to/from -- see
  // $lib/units.ts's per-table doc comments.
  let eMevU = $state(56.2);
  let voltageV = $state(300);
  let electrodeGapCm = $state(0.2);
  let doseRateGyS = $state(8.91);
  let pulseDurationS = $state(540e-6);
  let sampledRadiusCm = $state(0.008);
  let gridSizeUm = $state(10);

  let phase = $state<Phase>("setup");
  let errorMessage = $state("");
  let stepIndex = $state(0);
  let totalSteps = $state(0);
  let runningKs = $state(1);
  let finalKs = $state<number | undefined>(undefined);

  let timeUs: number[] = $state([]);
  let nPositive: number[] = $state([]);
  let nNegative: number[] = $state([]);
  let recombined: number[] = $state([]);

  const controller = new SimulationController();

  function currentParams(): SimParams {
    return {
      eMevU,
      voltageV,
      electrodeGapCm,
      doseRateGyS,
      pulseDurationS,
      sampledRadiusCm,
      gridSizeUm,
    };
  }

  // Recomputed synchronously on every field change -- estimate() is instant
  // and allocation-free (core/src/lib.rs), so this is safe to call directly
  // from a derived value rather than routing it through the worker.
  let sizing = $derived.by<Estimate | undefined>(() => {
    if (!wasmReady) return undefined;
    // Referencing every input so this recomputes when any of them change.
    void eMevU;
    void voltageV;
    void electrodeGapCm;
    void doseRateGyS;
    void pulseDurationS;
    void sampledRadiusCm;
    void gridSizeUm;
    return estimate(currentParams());
  });

  let radiusPercentOfMarkus = $derived((sampledRadiusCm / MARKUS_FULL_RADIUS_CM) * 100);

  type TimeEstimateState = "idle" | "measuring" | "done" | "error";
  let timeEstimateState = $state<TimeEstimateState>("idle");
  let timeEstimateSeconds = $state<number | undefined>(undefined);
  let timeEstimateError = $state("");
  // Bumped on every param change (invalidates a stale in-flight measurement)
  // and on every new measurement (so a late reply to an earlier one is
  // ignored rather than overwriting a newer result).
  let timeEstimateGeneration = 0;

  $effect(() => {
    // Any config change makes a previous measurement meaningless.
    void eMevU;
    void voltageV;
    void electrodeGapCm;
    void doseRateGyS;
    void pulseDurationS;
    void sampledRadiusCm;
    void gridSizeUm;
    timeEstimateGeneration++;
    timeEstimateState = "idle";
    timeEstimateSeconds = undefined;
  });

  onMount(() => {
    loadWasm().then(() => {
      wasmReady = true;
    });
    return () => controller.dispose();
  });

  function runTimeEstimate(): void {
    if (!sizing?.ok) return;
    const generation = ++timeEstimateGeneration;
    timeEstimateState = "measuring";
    timeEstimateSeconds = undefined;
    timeEstimateError = "";
    const seed = Math.floor(Math.random() * 1_000_000_000);
    controller.estimateRunningTime(currentParams(), seed, 3000, {
      onInvalid: (error) => {
        if (generation !== timeEstimateGeneration) return;
        timeEstimateState = "error";
        timeEstimateError = error;
      },
      onResult: (seconds) => {
        if (generation !== timeEstimateGeneration) return;
        timeEstimateState = "done";
        timeEstimateSeconds = seconds;
      },
      onCancelled: () => {
        if (generation !== timeEstimateGeneration) return;
        timeEstimateState = "idle";
      },
      onError: (message) => {
        if (generation !== timeEstimateGeneration) return;
        timeEstimateState = "error";
        timeEstimateError = message;
      },
    });
  }

  function startRun(): void {
    if (!sizing?.ok) return;
    phase = "running";
    errorMessage = "";
    stepIndex = 0;
    totalSteps = sizing.totalTimeSteps;
    runningKs = 1;
    finalKs = undefined;
    timeUs = [];
    nPositive = [];
    nNegative = [];
    recombined = [];

    // A fresh, unexposed seed per run: statistical noise between runs of the
    // same config is real physics (docs/PHYSICS.md sec. 14), not something
    // to hide behind a fixed default -- see repeated-run behaviour there.
    const seed = Math.floor(Math.random() * 1_000_000_000);
    controller.start(currentParams(), seed, {
      onInvalid: (error) => {
        phase = "error";
        errorMessage = error;
      },
      onProgress: (chunk: ProgressChunk, step, total, ks) => {
        for (const t of chunk.timeS) timeUs.push(t * 1e6);
        for (const v of chunk.nPositive) nPositive.push(v);
        for (const v of chunk.nNegative) nNegative.push(v);
        for (const v of chunk.recombined) recombined.push(v);
        stepIndex = step;
        totalSteps = total;
        runningKs = ks;
      },
      onDone: (ks) => {
        phase = "done";
        finalKs = ks;
      },
      onCancelled: () => {
        phase = "cancelled";
      },
      onError: (message) => {
        phase = "error";
        errorMessage = message;
      },
    });
  }

  function cancelRun(): void {
    controller.cancel();
  }

  function backToSetup(): void {
    controller.dispose();
    phase = "setup";
  }

  let carrierScale = $derived(siScale([...nPositive, ...nNegative]));
  let recombinedScale = $derived(siScale(recombined));
  let progressPct = $derived(totalSteps > 0 ? Math.min(100, (100 * stepIndex) / totalSteps) : 0);
</script>

<svelte:head>
  <title>Pulsed-proton ion-chamber recombination</title>
</svelte:head>

<main>
  <header>
    <h1>Pulsed-proton ion-chamber recombination</h1>
    <p class="subtitle">
      Solves the coupled drift-diffusion-recombination equations for a small column of
      ionisation-chamber gas under a pulsed proton beam, and reports the general recombination
      correction factor k<sub>s</sub> -- entirely client-side (Rust compiled to WebAssembly; nothing
      is sent to a server). A single averaged ion-pair species and a small sampled sub-volume of the
      chamber keep it fast enough to run live in a browser tab; see
      <a
        href="https://github.com/grzanka/IonTracks-PulsedProton-Python"
        target="_blank"
        rel="noreferrer">pulsed_ion_chamber</a
      > for the full two-species model and the physics behind every simplification here.
    </p>
  </header>

  {#if phase === "setup"}
    <section class="panel">
      <h2>1. Configure a run</h2>
      <div class="grid">
        <UnitField
          label="Proton energy"
          units={ENERGY_UNITS}
          defaultUnitSymbol="MeV"
          bind:value={eMevU}
          min={10}
          max={250}
        />
        <UnitField
          label="Voltage"
          units={VOLTAGE_UNITS}
          defaultUnitSymbol="V"
          bind:value={voltageV}
          min={50}
          max={1000}
        />
        <UnitField
          label="Electrode gap"
          units={GAP_LENGTH_UNITS}
          defaultUnitSymbol="cm"
          bind:value={electrodeGapCm}
          min={0.05}
          max={1.0}
        />
        <UnitField
          label="Dose rate (to air)"
          units={DOSE_RATE_UNITS}
          defaultUnitSymbol="Gy/s"
          bind:value={doseRateGyS}
          min={0.1}
          max={100}
        />
        <UnitField
          label="Pulse duration"
          units={PULSE_TIME_UNITS}
          defaultUnitSymbol="µs"
          bind:value={pulseDurationS}
          min={50e-6}
          max={2000e-6}
        />
        <UnitField
          label="Sampled column radius"
          units={RADIUS_LENGTH_UNITS}
          defaultUnitSymbol="mm"
          bind:value={sampledRadiusCm}
          hint={`≈ ${radiusPercentOfMarkus.toFixed(1)}% of the full Markus PTW 23343 chamber radius (2.65 mm)`}
        />
        <UnitField
          label="Grid spacing"
          units={GRID_SIZE_UNITS}
          defaultUnitSymbol="µm"
          bind:value={gridSizeUm}
          hint="10 µm is the validated default (resolves the ~20 µm track core); finer is much more expensive -- see the estimate below."
        />
      </div>

      <h2>2. Estimated cost</h2>
      {#if !wasmReady}
        <p>Loading WebAssembly module...</p>
      {:else if sizing}
        {#if sizing.ok}
          <div class="estimate-columns">
            <div class="estimate-group">
              <h3>Memory (RAM)</h3>
              <dl class="estimate">
                <div>
                  <dt>Grid</dt>
                  <dd>{sizing.noXy}&times;{sizing.noXy}&times;{sizing.noZ} voxels</dd>
                </div>
                <div>
                  <dt>Estimated memory</dt>
                  <dd>{formatBytes(sizing.estimatedMemoryBytes)}</dd>
                </div>
                <div>
                  <dt>% of full Markus radius</dt>
                  <dd>{radiusPercentOfMarkus.toFixed(1)}%</dd>
                </div>
              </dl>
            </div>
            <div class="estimate-group">
              <h3>CPU (time)</h3>
              <dl class="estimate">
                <div>
                  <dt>Tracks per pulse</dt>
                  <dd>{formatCount(sizing.numberOfTracksPerPulse)}</dd>
                </div>
                <div>
                  <dt>Time steps</dt>
                  <dd>{formatCount(sizing.totalTimeSteps)}</dd>
                </div>
                <div>
                  <dt>Time step (dt)</dt>
                  <dd>{sizing.dtNs.toFixed(1)} ns</dd>
                </div>
                <div>
                  <dt>Rough guess</dt>
                  <dd>{formatSeconds(sizing.estimatedWallSecondsRough)}</dd>
                </div>
              </dl>
              <p class="cpu-hint">
                Rough guess assumes a fast desktop; actual time varies a lot by device.
              </p>
              <button
                type="button"
                onclick={runTimeEstimate}
                disabled={timeEstimateState === "measuring"}
              >
                {timeEstimateState === "measuring" ? "Measuring... (~3s)" : "Estimate running time"}
              </button>
              {#if timeEstimateState === "done" && timeEstimateSeconds !== undefined}
                <p class="cpu-measured">
                  Measured on this device: ~{formatSeconds(timeEstimateSeconds)}
                </p>
              {:else if timeEstimateState === "error"}
                <p class="error-note">{timeEstimateError}</p>
              {/if}
            </div>
          </div>
          <p class="ok-note">Within this prototype's browser safety limits.</p>
          <button type="button" class="primary" onclick={startRun}>Run simulation</button>
        {:else}
          <p class="error-note">{sizing.error}</p>
          <button type="button" class="primary" disabled>Run simulation</button>
        {/if}
      {/if}
    </section>
  {:else}
    <section class="panel">
      <div class="run-header">
        <h2>
          {#if phase === "running"}
            Running... {progressPct.toFixed(0)}%
          {:else if phase === "done"}
            Finished
          {:else if phase === "cancelled"}
            Cancelled
          {:else}
            Error
          {/if}
        </h2>
        {#if phase === "running"}
          <button type="button" onclick={cancelRun}>Cancel</button>
        {:else}
          <button type="button" onclick={backToSetup}>Back to setup</button>
        {/if}
      </div>

      {#if phase === "running"}
        <progress value={progressPct} max="100"></progress>
        <p class="k-s-live">k<sub>s</sub> so far: {runningKs.toFixed(4)}</p>
      {:else if phase === "done" && finalKs !== undefined}
        <p class="k-s-final">Final k<sub>s</sub> = {finalKs.toFixed(6)}</p>
      {:else if phase === "error"}
        <p class="error-note">{errorMessage}</p>
      {/if}

      <div class="plots">
        <LinePlot
          title="Charge-carrier evolution"
          xValues={timeUs}
          yLabel={`carriers present [${carrierScale.prefix}ion pairs]`}
          series={[
            {
              label: "positive",
              color: "#2563eb",
              values: nPositive.map((v) => v / carrierScale.divisor),
            },
            {
              label: "negative",
              color: "#dc2626",
              dashed: true,
              values: nNegative.map((v) => v / carrierScale.divisor),
            },
          ]}
        />
        <LinePlot
          title="Recombination rate"
          xValues={timeUs}
          yLabel={`recombination [${recombinedScale.prefix}ion pairs / step]`}
          series={[
            {
              label: "recombined",
              color: "#b91c1c",
              values: recombined.map((v) => v / recombinedScale.divisor),
            },
          ]}
        />
      </div>
    </section>
  {/if}

  <footer>
    <a
      href="https://github.com/grzanka/IonTracks-PulsedProton-Python"
      target="_blank"
      rel="noreferrer"
    >
      github.com/grzanka/IonTracks-PulsedProton-Python
    </a>
  </footer>
</main>

<style>
  :global(body) {
    margin: 0;
    background: #f8fafc;
    color: #0f172a;
    font-family:
      system-ui,
      -apple-system,
      "Segoe UI",
      sans-serif;
  }

  main {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem 1.25rem 4rem;
  }

  header h1 {
    font-size: 1.6rem;
    margin-bottom: 0.25rem;
  }

  .subtitle {
    color: #475569;
    font-size: 0.9rem;
    line-height: 1.5;
  }

  .panel {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 1.5rem;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem 1.5rem;
    margin-bottom: 1rem;
  }

  .estimate-columns {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1.5rem;
    margin: 0.75rem 0;
  }

  .estimate-group h3 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    margin: 0 0 0.5rem;
  }

  .cpu-hint {
    font-size: 0.78rem;
    color: #64748b;
    margin: 0.4rem 0 0.6rem;
  }

  .cpu-measured {
    font-size: 0.85rem;
    color: #15803d;
    margin-top: 0.5rem;
  }

  .estimate {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.5rem 1rem;
    margin: 0;
  }

  .estimate div {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    border-bottom: 1px dashed #e2e8f0;
    padding-bottom: 0.3rem;
  }

  .estimate dt {
    color: #64748b;
    font-size: 0.8rem;
  }

  .estimate dd {
    margin: 0;
    font-weight: 600;
  }

  .ok-note {
    color: #15803d;
    font-size: 0.85rem;
  }

  .error-note {
    color: #b91c1c;
    font-size: 0.9rem;
    white-space: pre-wrap;
  }

  button {
    border: 1px solid #cbd5e1;
    background: #f1f5f9;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
    cursor: pointer;
  }

  button.primary {
    background: #2563eb;
    color: white;
    border-color: #2563eb;
    font-weight: 600;
  }

  button.primary:disabled {
    background: #94a3b8;
    border-color: #94a3b8;
    cursor: not-allowed;
  }

  .run-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  progress {
    width: 100%;
    margin: 0.5rem 0;
  }

  .k-s-live,
  .k-s-final {
    font-variant-numeric: tabular-nums;
  }

  .plots {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    margin-top: 1rem;
  }

  footer {
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
    font-size: 0.8rem;
  }

  footer a {
    color: #64748b;
  }
</style>
