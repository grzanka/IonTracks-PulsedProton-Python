// Message protocol between the main thread and sim.worker.ts. Kept in its own
// module so both sides import the same types instead of duplicating them.
import type { SimParams } from "$lib/wasm-core/loader";

export interface RunRequest {
  params: SimParams;
  seed: number;
}

export type MainToWorkerMessage = { type: "start"; request: RunRequest } | { type: "cancel" };

/** A batch of newly-completed steps' scalars -- appended to the caller's own
 * growing arrays, never resent in full (see sim.worker.ts's flush cadence). */
export interface ProgressChunk {
  timeS: number[];
  nPositive: number[];
  nNegative: number[];
  injected: number[];
  recombined: number[];
}

export type WorkerToMainMessage =
  | { type: "invalid"; error: string }
  | { type: "progress"; chunk: ProgressChunk; stepIndex: number; totalSteps: number; ks: number }
  | { type: "done"; ks: number; stepsCompleted: number }
  | { type: "cancelled" }
  | { type: "error"; message: string };
