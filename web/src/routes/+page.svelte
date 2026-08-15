<script lang="ts">
  import { onMount } from "svelte";
  import LinePlot from "$lib/components/LinePlot.svelte";
  import { estimate, loadWasm, type Estimate, type SimParams } from "$lib/wasm-core/loader";
  import { SimulationController } from "$lib/sim/controller";
  import type { ProgressChunk } from "$lib/sim/protocol";
  import { formatBytes, formatCount, siScale } from "$lib/format";

  type Phase = "setup" | "running" | "done" | "cancelled" | "error";

  let wasmReady = $state(false);
  let eMevU = $state(56.2);
  let voltageV = $state(300);
  let electrodeGapCm = $state(0.2);
  let doseRateGyS = $state(8.91);
  let pulseDurationUs = $state(540);
  let sampledRadiusMm = $state(0.08);
  let seed = $state(1);

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
      pulseDurationS: pulseDurationUs * 1e-6,
      sampledRadiusCm: sampledRadiusMm * 0.1,
    };
  }

  // Recomputed synchronously on every slider change -- estimate() is
  // instant and allocation-free (core/src/lib.rs), so this is safe to call
  // directly from a derived value rather than routing it through the worker.
  let sizing = $derived.by<Estimate | undefined>(() => {
    if (!wasmReady) return undefined;
    // Referencing every input so this recomputes when any of them change.
    void eMevU;
    void voltageV;
    void electrodeGapCm;
    void doseRateGyS;
    void pulseDurationUs;
    void sampledRadiusMm;
    return estimate(currentParams());
  });

  onMount(() => {
    loadWasm().then(() => {
      wasmReady = true;
    });
    return () => controller.dispose();
  });

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

  function reroll(): void {
    seed = Math.floor(Math.random() * 1_000_000_000);
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
      Runs entirely in your browser (Rust compiled to WebAssembly, no server involved) -- a
      browser-scale port of
      <a
        href="https://github.com/grzanka/IonTracks-PulsedProton-Python"
        target="_blank"
        rel="noreferrer">pulsed_ion_chamber</a
      >. See
      <a
        href="https://github.com/grzanka/IonTracks-PulsedProton-Python/issues/6"
        target="_blank"
        rel="noreferrer">issue #6</a
      >
      for what this prototype narrows relative to the full Python package.
    </p>
  </header>

  {#if phase === "setup"}
    <section class="panel">
      <h2>1. Configure a run</h2>
      <div class="grid">
        <label>
          Proton energy
          <input type="range" min="10" max="250" step="1" bind:value={eMevU} />
          <span class="value">{eMevU.toFixed(0)} MeV</span>
        </label>
        <label>
          Voltage
          <input type="range" min="50" max="1000" step="10" bind:value={voltageV} />
          <span class="value">{voltageV.toFixed(0)} V</span>
        </label>
        <label>
          Electrode gap
          <input type="range" min="0.05" max="1.0" step="0.01" bind:value={electrodeGapCm} />
          <span class="value">{electrodeGapCm.toFixed(2)} cm</span>
        </label>
        <label>
          Dose rate (to air)
          <input type="range" min="0.1" max="100" step="0.1" bind:value={doseRateGyS} />
          <span class="value">{doseRateGyS.toFixed(1)} Gy/s</span>
        </label>
        <label>
          Pulse duration
          <input type="range" min="50" max="2000" step="10" bind:value={pulseDurationUs} />
          <span class="value">{pulseDurationUs.toFixed(0)} µs</span>
        </label>
        <label>
          Sampled column radius
          <input type="range" min="0.03" max="0.25" step="0.01" bind:value={sampledRadiusMm} />
          <span class="value">{sampledRadiusMm.toFixed(2)} mm</span>
        </label>
      </div>
      <div class="seed-row">
        <label class="seed-label">
          Seed
          <input type="number" bind:value={seed} />
        </label>
        <button type="button" onclick={reroll}>Reroll</button>
      </div>

      <h2>2. Estimated cost</h2>
      {#if !wasmReady}
        <p>Loading WebAssembly module...</p>
      {:else if sizing}
        {#if sizing.ok}
          <dl class="estimate">
            <div>
              <dt>Grid</dt>
              <dd>{sizing.noXy}&times;{sizing.noXy}&times;{sizing.noZ} voxels</dd>
            </div>
            <div>
              <dt>Tracks per pulse</dt>
              <dd>{formatCount(sizing.numberOfTracksPerPulse)}</dd>
            </div>
            <div>
              <dt>Time steps</dt>
              <dd>{formatCount(sizing.totalTimeSteps)}</dd>
            </div>
            <div>
              <dt>Estimated memory</dt>
              <dd>{formatBytes(sizing.estimatedMemoryBytes)}</dd>
            </div>
            <div>
              <dt>Time step (dt)</dt>
              <dd>{sizing.dtNs.toFixed(1)} ns</dd>
            </div>
          </dl>
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
    max-width: 68ch;
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

  label {
    display: flex;
    flex-direction: column;
    font-size: 0.85rem;
    color: #334155;
    gap: 0.25rem;
  }

  .value {
    font-variant-numeric: tabular-nums;
    color: #0f172a;
    font-weight: 600;
  }

  .seed-row {
    display: flex;
    align-items: flex-end;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .seed-label input {
    width: 8rem;
  }

  .estimate {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.5rem 1rem;
    margin: 0.75rem 0;
  }

  .estimate div {
    display: flex;
    justify-content: space-between;
    border-bottom: 1px dashed #e2e8f0;
    padding-bottom: 0.15rem;
  }

  .estimate dt {
    color: #64748b;
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
</style>
