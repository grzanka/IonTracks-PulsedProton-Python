<script lang="ts">
  import { onMount } from "svelte";
  import LinePlot from "$lib/components/LinePlot.svelte";
  import TrackDensityPlot from "$lib/components/TrackDensityPlot.svelte";
  import UnitField from "$lib/components/UnitField.svelte";
  import { estimate, loadWasm, type Estimate, type SimParams } from "$lib/wasm-core/loader";
  import { SimulationController } from "$lib/sim/controller";
  import type { ProgressChunk } from "$lib/sim/protocol";
  import { formatBytes, formatCount, formatSeconds, scaleForPeak } from "$lib/format";
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

  // Per-field validity, surfaced from each UnitField below (issue #19 W3) --
  // without this, an out-of-range edit leaves the field showing rejected
  // text while `sizing`/Run still reflect the last *valid* value, so the run
  // that starts silently doesn't match what's on screen.
  let eMevUInvalid = $state(false);
  let voltageVInvalid = $state(false);
  let electrodeGapCmInvalid = $state(false);
  let doseRateGySInvalid = $state(false);
  let pulseDurationSInvalid = $state(false);
  let sampledRadiusCmInvalid = $state(false);
  let gridSizeUmInvalid = $state(false);
  let anyFieldInvalid = $derived(
    eMevUInvalid ||
      voltageVInvalid ||
      electrodeGapCmInvalid ||
      doseRateGySInvalid ||
      pulseDurationSInvalid ||
      sampledRadiusCmInvalid ||
      gridSizeUmInvalid,
  );

  let phase = $state<Phase>("setup");
  let errorMessage = $state("");
  let stepIndex = $state(0);
  let totalSteps = $state(0);
  let runningKs = $state(1);
  let finalKs = $state<number | undefined>(undefined);
  // Read once at run completion, not streamed -- see protocol.ts's "done"
  // doc comment (issue #6 milestone 5).
  let trackDensityXy: number[] = $state([]);
  let densityNoXy = $state(0);

  let timeUs: number[] = $state([]);
  let nPositive: number[] = $state([]);
  let nNegative: number[] = $state([]);
  let recombined: number[] = $state([]);
  // Injected charge, converted to a rate (ion pairs / us) same as
  // pulsed_ion_chamber.plots' injection-rate figure -- dividing by dt here
  // (once per point, as each chunk arrives) rather than at draw time keeps
  // LinePlot's valueDivisor purely an SI-prefix scale, matching how the
  // other two plots use it (issue #6 milestone 3: this prototype originally
  // shipped without an injection-rate plot at all).
  let injectedRate: number[] = $state([]);
  let runDtUs = $state(1); // set from sizing.dtNs at startRun(); >0 always
  // Also captured once at startRun(), for the track-density cross-section
  // view (issue #6 milestone 5) -- config fields aren't editable while a
  // run is in progress, but capturing keeps this view's inputs pinned to
  // the run that produced the data regardless.
  let runUnitLengthCm = $state(0);
  let runInnerRadiusVoxels = $state(0);
  let runSampledRadiusCm = $state(0); // for the k_inf extrapolation panel below
  // Running peaks, updated incrementally in onProgress below (each new
  // chunk's own few points, not a rescan of the whole series) rather than
  // recomputed from siScale([...nPositive, ...nNegative]) on every chunk --
  // that spread-and-scan is O(run length) per chunk, so cost over a long run
  // grew with the run itself instead of staying flat (issue #19 W7).
  let carrierPeak = $state(0);
  let recombinedPeak = $state(0);
  let injectedRatePeak = $state(0);

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
    if (!sizing?.ok || anyFieldInvalid) return;
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
    if (!sizing?.ok || anyFieldInvalid) return;
    phase = "running";
    errorMessage = "";
    stepIndex = 0;
    totalSteps = sizing.totalTimeSteps;
    runningKs = 1;
    finalKs = undefined;
    trackDensityXy = [];
    densityNoXy = 0;
    timeUs = [];
    nPositive = [];
    nNegative = [];
    recombined = [];
    injectedRate = [];
    carrierPeak = 0;
    recombinedPeak = 0;
    injectedRatePeak = 0;
    runDtUs = sizing.dtNs / 1000;
    runUnitLengthCm = gridSizeUm * 1e-4;
    runInnerRadiusVoxels = sizing.innerRadiusVoxels;
    runSampledRadiusCm = sampledRadiusCm;
    extrapolationState = "idle";
    extrapolationSecondRadiusCm = Math.min(sampledRadiusCm * 1.5, 1.0);
    extrapolationKInfinity = undefined;
    extrapolationA = undefined;
    extrapolationSecondKs = undefined;

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
        for (const v of chunk.nPositive) {
          nPositive.push(v);
          if (v > carrierPeak) carrierPeak = v;
        }
        for (const v of chunk.nNegative) {
          nNegative.push(v);
          if (v > carrierPeak) carrierPeak = v;
        }
        for (const v of chunk.recombined) {
          recombined.push(v);
          if (v > recombinedPeak) recombinedPeak = v;
        }
        for (const v of chunk.injected) {
          const rate = v / runDtUs;
          injectedRate.push(rate);
          if (rate > injectedRatePeak) injectedRatePeak = rate;
        }
        stepIndex = step;
        totalSteps = total;
        runningKs = ks;
      },
      onDone: (ks, _stepsCompleted, densityXy, noXy) => {
        trackDensityXy = densityXy;
        densityNoXy = noXy;
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

  // --- k_inf (infinite-column) extrapolation, docs/PHYSICS.md sec. 14 ---
  // (issue #6 sec. 1/milestone 5: "why is my radius capped" answered as a
  // feature, not just a limit). k_s(r) = k_inf - A/r is fit from the
  // finished run above (r1, k1) plus one more run at a second radius
  // (r2, k2), both solved from those two points -- not streamed live like
  // the run itself, since it's two closed-form numbers from two already-
  // computed k_s values.
  type ExtrapolationState = "idle" | "running" | "done" | "error";
  let extrapolationState = $state<ExtrapolationState>("idle");
  let extrapolationSecondRadiusCm = $state(0.012);
  let extrapolationSecondRadiusInvalid = $state(false);
  let extrapolationSecondKs = $state<number | undefined>(undefined);
  let extrapolationKInfinity = $state<number | undefined>(undefined);
  let extrapolationA = $state<number | undefined>(undefined);
  let extrapolationError = $state("");

  /** Solves k_s(r) = k_inf - A/r for (k_inf, A) given two (radius, k_s)
   * points. Undefined if the two radii coincide (the system is singular). */
  function fitKInfinity(
    r1Cm: number,
    k1: number,
    r2Cm: number,
    k2: number,
  ): { kInfinity: number; aCm: number } | undefined {
    if (r1Cm === r2Cm) return undefined;
    const invR1 = 1 / r1Cm;
    const invR2 = 1 / r2Cm;
    const aCm = (k2 - k1) / (invR1 - invR2);
    return { kInfinity: k1 + aCm * invR1, aCm };
  }

  function runExtrapolation(): void {
    if (
      phase !== "done" ||
      finalKs === undefined ||
      extrapolationSecondRadiusInvalid ||
      extrapolationState === "running"
    ) {
      return;
    }
    extrapolationState = "running";
    extrapolationError = "";
    extrapolationKInfinity = undefined;
    extrapolationA = undefined;
    const firstKs = finalKs; // narrowed here; the callback below closes over the reactive (possibly-undefined) binding otherwise
    const secondParams: SimParams = {
      ...currentParams(),
      sampledRadiusCm: extrapolationSecondRadiusCm,
    };
    const secondSizing = estimate(secondParams);
    if (!secondSizing.ok) {
      extrapolationState = "error";
      extrapolationError = secondSizing.error ?? "Configuration rejected.";
      return;
    }
    const seed = Math.floor(Math.random() * 1_000_000_000);
    // A separate, short-lived run through the same controller -- it only
    // reuses the primary run's finished (r1, k1), never its live state
    // (timeUs, phase, ...), so the plots above stay exactly as the primary
    // run left them while this one computes in the background.
    controller.start(secondParams, seed, {
      onInvalid: (error) => {
        extrapolationState = "error";
        extrapolationError = error;
      },
      onProgress: () => {},
      onDone: (ks2) => {
        extrapolationSecondKs = ks2;
        const fit = fitKInfinity(runSampledRadiusCm, firstKs, extrapolationSecondRadiusCm, ks2);
        if (fit) {
          extrapolationKInfinity = fit.kInfinity;
          extrapolationA = fit.aCm * 1e4; // cm -> um, matching docs/PHYSICS.md sec. 14's units
          extrapolationState = "done";
        } else {
          extrapolationState = "error";
          extrapolationError = "Second radius must differ from the first.";
        }
      },
      onCancelled: () => {
        extrapolationState = "idle";
      },
      onError: (message) => {
        extrapolationState = "error";
        extrapolationError = message;
      },
    });
  }

  let carrierScale = $derived(scaleForPeak(carrierPeak));
  let recombinedScale = $derived(scaleForPeak(recombinedPeak));
  let injectedScale = $derived(scaleForPeak(injectedRatePeak));
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
          bind:invalid={eMevUInvalid}
          min={10}
          max={250}
        />
        <UnitField
          label="Voltage"
          units={VOLTAGE_UNITS}
          defaultUnitSymbol="V"
          bind:value={voltageV}
          bind:invalid={voltageVInvalid}
          min={50}
          max={1000}
        />
        <UnitField
          label="Electrode gap"
          units={GAP_LENGTH_UNITS}
          defaultUnitSymbol="cm"
          bind:value={electrodeGapCm}
          bind:invalid={electrodeGapCmInvalid}
          min={0.05}
          max={1.0}
        />
        <UnitField
          label="Dose rate (to air)"
          units={DOSE_RATE_UNITS}
          defaultUnitSymbol="Gy/s"
          bind:value={doseRateGyS}
          bind:invalid={doseRateGySInvalid}
          min={0.1}
          max={100}
        />
        <UnitField
          label="Pulse duration"
          units={PULSE_TIME_UNITS}
          defaultUnitSymbol="µs"
          bind:value={pulseDurationS}
          bind:invalid={pulseDurationSInvalid}
          min={50e-6}
          max={2000e-6}
        />
        <UnitField
          label="Sampled column radius"
          units={RADIUS_LENGTH_UNITS}
          defaultUnitSymbol="mm"
          bind:value={sampledRadiusCm}
          bind:invalid={sampledRadiusCmInvalid}
          min={1e-4}
          max={1.0}
          hint={`≈ ${radiusPercentOfMarkus.toFixed(1)}% of the full Markus PTW 23343 chamber radius (2.65 mm)`}
        />
        <UnitField
          label="Grid spacing"
          units={GRID_SIZE_UNITS}
          defaultUnitSymbol="µm"
          bind:value={gridSizeUm}
          bind:invalid={gridSizeUmInvalid}
          min={1}
          max={1000}
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
                disabled={timeEstimateState === "measuring" || anyFieldInvalid}
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
          {#if anyFieldInvalid}
            <p class="error-note">
              One or more fields above are out of range -- fix them before running (issue #19 W3: a
              rejected edit used to leave the old value in effect with no visible warning here).
            </p>
          {/if}
          <button type="button" class="primary" onclick={startRun} disabled={anyFieldInvalid}>
            Run simulation
          </button>
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

        <div class="extrapolation">
          <h3>Extrapolate to an infinite column</h3>
          <p class="extrapolation-hint">
            k<sub>s</sub> rises with the sampled column radius and never fully converges within this
            prototype's browser-safe caps (<a
              href="https://github.com/grzanka/IonTracks-PulsedProton-Python/blob/master/docs/PHYSICS.md#14-known-systematics-and-convergence"
              target="_blank"
              rel="noreferrer">docs/PHYSICS.md §14</a
            >). The shortfall falls as 1/r, so two runs at different radii are enough to fit k<sub
              >∞</sub
            >
            (the disc, not the ring) without paying for a larger grid.
          </p>
          <div class="extrapolation-row">
            <UnitField
              label="Second radius"
              units={RADIUS_LENGTH_UNITS}
              defaultUnitSymbol="mm"
              bind:value={extrapolationSecondRadiusCm}
              bind:invalid={extrapolationSecondRadiusInvalid}
              min={1e-4}
              max={1.0}
            />
            <button
              type="button"
              onclick={runExtrapolation}
              disabled={extrapolationState === "running" || extrapolationSecondRadiusInvalid}
            >
              {extrapolationState === "running" ? "Running second radius…" : "Run & extrapolate"}
            </button>
          </div>
          {#if extrapolationState === "error"}
            <p class="error-note">{extrapolationError}</p>
          {:else if extrapolationState === "done" && extrapolationKInfinity !== undefined && extrapolationA !== undefined}
            <dl class="estimate">
              <div>
                <dt>r₁, k<sub>s</sub>(r₁)</dt>
                <dd>
                  {(runSampledRadiusCm * 1e4).toFixed(0)} µm, {finalKs.toFixed(6)}
                </dd>
              </div>
              <div>
                <dt>r₂, k<sub>s</sub>(r₂)</dt>
                <dd>
                  {(extrapolationSecondRadiusCm * 1e4).toFixed(0)} µm, {extrapolationSecondKs?.toFixed(
                    6,
                  )}
                </dd>
              </div>
              <div>
                <dt>k<sub>∞</sub> = k<sub>s</sub>(r) + A/r</dt>
                <dd>{extrapolationKInfinity.toFixed(6)}</dd>
              </div>
              <div>
                <dt>A</dt>
                <dd>{extrapolationA.toFixed(3)} µm</dd>
              </div>
            </dl>
          {/if}
        </div>
      {:else if phase === "error"}
        <p class="error-note">{errorMessage}</p>
      {/if}

      <div class="plots">
        <LinePlot
          title="Injection rate"
          xValues={timeUs}
          valueDivisor={injectedScale.divisor}
          valueMax={injectedRatePeak}
          yLabel={`injection rate [${injectedScale.prefix}ion pairs / µs]`}
          series={[{ label: "injected", color: "#0891b2", values: injectedRate }]}
        />
        <LinePlot
          title="Charge-carrier evolution"
          xValues={timeUs}
          valueDivisor={carrierScale.divisor}
          valueMax={carrierPeak}
          yLabel={`carriers present [${carrierScale.prefix}ion pairs]`}
          series={[
            { label: "positive", color: "#2563eb", values: nPositive },
            { label: "negative", color: "#dc2626", dashed: true, values: nNegative },
          ]}
        />
        <LinePlot
          title="Recombination rate"
          xValues={timeUs}
          valueDivisor={recombinedScale.divisor}
          valueMax={recombinedPeak}
          yLabel={`recombination [${recombinedScale.prefix}ion pairs / step]`}
          series={[{ label: "recombined", color: "#b91c1c", values: recombined }]}
        />
        {#if phase === "done" && trackDensityXy.length > 0}
          <!-- A full 2D field, not a scalar time series -- rendered once at
               completion, not streamed with the three plots above (issue #6
               milestone 5, see protocol.ts's "done" doc comment). -->
          <TrackDensityPlot
            title="Track areal density cross-section"
            density={trackDensityXy}
            noXy={densityNoXy}
            unitLengthCm={runUnitLengthCm}
            innerRadiusVoxels={runInnerRadiusVoxels}
          />
        {/if}
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

  .extrapolation {
    margin-top: 1rem;
    padding-top: 0.75rem;
    border-top: 1px dashed #e2e8f0;
  }

  .extrapolation h3 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    margin: 0 0 0.4rem;
  }

  .extrapolation-hint {
    font-size: 0.82rem;
    color: #475569;
    line-height: 1.4;
    margin: 0 0 0.6rem;
  }

  .extrapolation-hint a {
    color: #2563eb;
  }

  .extrapolation-row {
    display: flex;
    align-items: flex-end;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
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
