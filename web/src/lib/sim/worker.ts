// Runs the actual step loop off the main thread, so the UI stays responsive
// and Cancel works (worker.terminate() from the controller, or the
// cooperative check below). Drives WasmSimulation the same way
// wasm-bindgen's own "Game of Life" example drives its universe: construct
// once, call step() in a loop, read scalar getters back after each call --
// see core/src/lib.rs's module docs.
import { estimate, loadWasm, WasmSimulation } from "$lib/wasm-core/loader";
import type { MainToWorkerMessage, ProgressChunk, WorkerToMainMessage } from "./protocol";

const FLUSH_INTERVAL_MS = 150;

let cancelRequested = false;

function post(message: WorkerToMainMessage): void {
  self.postMessage(message);
}

function emptyChunk(): ProgressChunk {
  return { timeS: [], nPositive: [], nNegative: [], injected: [], recombined: [] };
}

async function run(request: MainToWorkerMessage & { type: "start" }): Promise<void> {
  cancelRequested = false;
  await loadWasm();

  const { params, seed } = request.request;

  // Re-validate here rather than trusting the main thread's last estimate()
  // call -- it's the same cheap, allocation-free check, and the worker is
  // the thing about to actually allocate the grid.
  const sizing = estimate(params);
  if (!sizing.ok) {
    post({ type: "invalid", error: sizing.error ?? "Configuration rejected." });
    return;
  }

  const sim = new WasmSimulation(
    params.eMevU,
    params.voltageV,
    params.electrodeGapCm,
    params.doseRateGyS,
    params.pulseDurationS,
    params.sampledRadiusCm,
    params.gridSizeUm,
    seed,
  );
  if (!sim.ok()) {
    post({ type: "invalid", error: sim.error() ?? "Configuration rejected." });
    return;
  }

  const totalSteps = sim.total_steps();
  let chunk = emptyChunk();
  let lastFlush = performance.now();

  while (!sim.is_finished()) {
    if (cancelRequested) {
      post({ type: "cancelled" });
      return;
    }

    sim.step();

    chunk.timeS.push(sim.time_s());
    chunk.nPositive.push(sim.last_total_positive());
    chunk.nNegative.push(sim.last_total_negative());
    chunk.injected.push(sim.last_injected());
    chunk.recombined.push(sim.last_recombined());

    const now = performance.now();
    if (now - lastFlush >= FLUSH_INTERVAL_MS) {
      post({ type: "progress", chunk, stepIndex: sim.step_index(), totalSteps, ks: sim.ks() });
      chunk = emptyChunk();
      lastFlush = now;
      // The loop above never awaits anything, so the worker never drains its
      // message queue and the `{type:"cancel"}` message posted by
      // controller.cancel() is never delivered -- the run goes to completion
      // regardless of Cancel (issue #19 W1). A zero-delay macrotask yield,
      // on the same cadence as the flush above (so it costs nothing extra),
      // gives the event loop a chance to deliver it.
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  if (chunk.timeS.length > 0) {
    post({ type: "progress", chunk, stepIndex: sim.step_index(), totalSteps, ks: sim.ks() });
  }
  post({
    type: "done",
    ks: sim.ks(),
    stepsCompleted: sim.step_index(),
    // Read once, here, rather than streamed -- see protocol.ts's doc
    // comment on "done" (issue #6 milestone 5).
    trackDensityXy: Array.from(sim.track_density_xy()),
    noXy: sizing.noXy,
  });
}

/// Runs the *real* backend on the *real* grid for a bounded wall-clock
/// sample, then extrapolates -- the wasm analogue of
/// `pulsed_ion_chamber.benchmark.estimate_full_runtime_empirical`
/// (docs/PERFORMANCE.md sec. 7). Deliberately not the instant analytical
/// `estimate()`: this allocates the grid and pays real deposition/sweep
/// cost, so it measures the actual device instead of guessing at it.
async function estimateTime(
  request: MainToWorkerMessage & { type: "estimate_time" },
): Promise<void> {
  cancelRequested = false;
  await loadWasm();

  const { params, seed } = request.request;
  const sizing = estimate(params);
  if (!sizing.ok) {
    post({ type: "invalid", error: sizing.error ?? "Configuration rejected." });
    return;
  }

  const sim = new WasmSimulation(
    params.eMevU,
    params.voltageV,
    params.electrodeGapCm,
    params.doseRateGyS,
    params.pulseDurationS,
    params.sampledRadiusCm,
    params.gridSizeUm,
    seed,
  );
  if (!sim.ok()) {
    post({ type: "invalid", error: sim.error() ?? "Configuration rejected." });
    return;
  }

  const totalSteps = sim.total_steps();
  // In-pulse steps (1..pulseTimeSteps) also deposit tracks on top of the
  // sweep; clearance steps (pulseTimeSteps+1..totalSteps) skip deposition
  // entirely and are measurably cheaper. A single blended rate from a sample
  // taken at the start of the run is therefore biased towards the more
  // expensive in-pulse cost -- sampled separately below so each phase's
  // remaining steps extrapolate from their own measured rate, not each
  // other's (issue #19 W5).
  const pulseTimeSteps = sim.pulse_time_steps();
  const start = performance.now();
  let stepsSampled = 0;
  let pulseStepsSampled = 0;
  let pulseElapsedMs = 0;
  let clearanceStepsSampled = 0;
  let clearanceElapsedMs = 0;

  while (!sim.is_finished()) {
    if (cancelRequested) {
      post({ type: "cancelled" });
      return;
    }
    const stepStart = performance.now();
    sim.step();
    const stepElapsedMs = performance.now() - stepStart;
    stepsSampled++;
    if (sim.step_index() <= pulseTimeSteps) {
      pulseStepsSampled++;
      pulseElapsedMs += stepElapsedMs;
    } else {
      clearanceStepsSampled++;
      clearanceElapsedMs += stepElapsedMs;
    }
    if (performance.now() - start >= request.sampleMs) break;
    // Same reasoning as run()'s loop: without a periodic yield, cancelEstimate()'s
    // message is never delivered while this loop is still running (issue #19 W1).
    if (stepsSampled % 64 === 0) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  const sampleElapsedMs = performance.now() - start;
  let estimatedTotalSeconds: number;
  if (sim.is_finished()) {
    // If the whole run finished inside the sample window, that elapsed time
    // *is* the answer -- no extrapolation needed.
    estimatedTotalSeconds = sampleElapsedMs / 1000;
  } else {
    const remainingPulseSteps = Math.max(0, pulseTimeSteps - pulseStepsSampled);
    const remainingClearanceSteps = Math.max(
      0,
      totalSteps - pulseTimeSteps - clearanceStepsSampled,
    );
    // If a phase wasn't sampled at all (the whole budget stayed in the
    // other one), fall back to that phase's rate for it -- never worse than
    // the old single-rate extrapolation, and exact once the sample spans
    // both phases (a 3 s default sample usually does: a few thousand steps
    // easily covers a several-hundred-microsecond pulse).
    const pulseRateMsPerStep =
      pulseStepsSampled > 0
        ? pulseElapsedMs / pulseStepsSampled
        : clearanceElapsedMs / Math.max(1, clearanceStepsSampled);
    const clearanceRateMsPerStep =
      clearanceStepsSampled > 0 ? clearanceElapsedMs / clearanceStepsSampled : pulseRateMsPerStep;
    const remainingMs =
      remainingPulseSteps * pulseRateMsPerStep + remainingClearanceSteps * clearanceRateMsPerStep;
    estimatedTotalSeconds = (sampleElapsedMs + remainingMs) / 1000;
  }
  post({ type: "time_estimate", estimatedTotalSeconds, stepsSampled, totalSteps, sampleElapsedMs });
}

self.onmessage = (event: MessageEvent<MainToWorkerMessage>) => {
  const message = event.data;
  if (message.type === "cancel") {
    cancelRequested = true;
    return;
  }
  if (message.type === "start") {
    run(message).catch((err: unknown) => {
      post({ type: "error", message: err instanceof Error ? err.message : String(err) });
    });
    return;
  }
  if (message.type === "estimate_time") {
    estimateTime(message).catch((err: unknown) => {
      post({ type: "error", message: err instanceof Error ? err.message : String(err) });
    });
  }
};
