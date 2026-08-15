// Main-thread wrapper around sim.worker.ts: owns the Worker instance and
// turns its messages into plain callbacks, so +page.svelte doesn't need to
// know the wire protocol.
import type { SimParams } from "$lib/wasm-core/loader";
import type { ProgressChunk, WorkerToMainMessage } from "./protocol";

export interface SimulationCallbacks {
  onInvalid: (error: string) => void;
  onProgress: (chunk: ProgressChunk, stepIndex: number, totalSteps: number, ks: number) => void;
  onDone: (ks: number, stepsCompleted: number) => void;
  onCancelled: () => void;
  onError: (message: string) => void;
}

export interface TimeEstimateCallbacks {
  onInvalid: (error: string) => void;
  onResult: (estimatedTotalSeconds: number, stepsSampled: number, totalSteps: number) => void;
  onCancelled: () => void;
  onError: (message: string) => void;
}

export class SimulationController {
  private worker: Worker | undefined;
  private estimateWorker: Worker | undefined;

  start(params: SimParams, seed: number, callbacks: SimulationCallbacks): void {
    this.dispose();
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (event: MessageEvent<WorkerToMainMessage>) => {
      const message = event.data;
      switch (message.type) {
        case "invalid":
          callbacks.onInvalid(message.error);
          break;
        case "progress":
          callbacks.onProgress(message.chunk, message.stepIndex, message.totalSteps, message.ks);
          break;
        case "done":
          callbacks.onDone(message.ks, message.stepsCompleted);
          break;
        case "cancelled":
          callbacks.onCancelled();
          break;
        case "error":
          callbacks.onError(message.message);
          break;
        default:
          break;
      }
    };
    this.worker = worker;
    worker.postMessage({ type: "start", request: { params, seed } });
  }

  /** Cooperative cancel: the worker checks between steps, so this can take up
   * to one step's wall time to take effect -- fine at this prototype's scale
   * (single steps are microseconds to low milliseconds). */
  cancel(): void {
    this.worker?.postMessage({ type: "cancel" });
  }

  /** Runs the real backend on the real grid for `sampleMs` of wall time and
   * extrapolates -- a separate, short-lived worker from `start()`'s, so an
   * in-progress full run (or another estimate) is never disturbed. */
  estimateRunningTime(
    params: SimParams,
    seed: number,
    sampleMs: number,
    callbacks: TimeEstimateCallbacks,
  ): void {
    this.estimateWorker?.terminate();
    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (event: MessageEvent<WorkerToMainMessage>) => {
      const message = event.data;
      switch (message.type) {
        case "invalid":
          callbacks.onInvalid(message.error);
          worker.terminate();
          break;
        case "time_estimate":
          callbacks.onResult(
            message.estimatedTotalSeconds,
            message.stepsSampled,
            message.totalSteps,
          );
          worker.terminate();
          break;
        case "cancelled":
          callbacks.onCancelled();
          worker.terminate();
          break;
        case "error":
          callbacks.onError(message.message);
          worker.terminate();
          break;
        default:
          break;
      }
    };
    this.estimateWorker = worker;
    worker.postMessage({ type: "estimate_time", request: { params, seed }, sampleMs });
  }

  cancelEstimate(): void {
    this.estimateWorker?.postMessage({ type: "cancel" });
  }

  /** Hard stop -- used when leaving the page or starting a new run. */
  dispose(): void {
    this.worker?.terminate();
    this.worker = undefined;
    this.estimateWorker?.terminate();
    this.estimateWorker = undefined;
  }
}
