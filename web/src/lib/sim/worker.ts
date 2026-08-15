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
    }
  }

  if (chunk.timeS.length > 0) {
    post({ type: "progress", chunk, stepIndex: sim.step_index(), totalSteps, ks: sim.ks() });
  }
  post({ type: "done", ks: sim.ks(), stepsCompleted: sim.step_index() });
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
  }
};
