#!/usr/bin/env bash
# py-spy flamegraphs at a low and a high thread count. --native is
# attempted first (needed to see OpenMP/Numba runtime worker threads, not
# just the single Python driver thread) and falls back to a Python-only
# flamegraph if this py-spy build/kernel doesn't support it.
set -uo pipefail
cd "$(dirname "$0")/.."

module load GCCcore/13.3.0 Python/3.12.3
source venv/bin/activate
mkdir -p profiling/data

for n in 1 96; do
  echo "=== py-spy record, threads=$n ===" >&2
  out="profiling/data/flamegraph_${n}threads.svg"
  rm -f "$out"
  cmd=(python -c "from profiling.common import run_once; print(run_once(${n}))")
  # --native stack-walking across many OS threads is expensive enough at a
  # high sample rate to perturb the very thing being measured (measured:
  # 200 Hz/96 threads inflated wall_s 1s -> 102s, with py-spy logging
  # "behind in sampling" throughout) -- drop the rate as thread count
  # grows so the flamegraph stays a *qualitative* call-stack picture, not
  # an additional (and by then unreliable) timing source; thread_sweep.csv
  # is the timing evidence, this is not.
  rate=$((200 / n)); rate=$((rate < 20 ? 20 : rate))
  # py-spy under srun sometimes exits non-zero at teardown ("No child
  # process") even after successfully writing the SVG -- so treat the
  # output file's existence as the real success signal, not the exit code.
  srun --overlap --ntasks=1 --cpus-per-task="$n" --cpu-bind=none \
    py-spy record --native --rate "$rate" -o "$out" -- "${cmd[@]}" || true
  if [[ ! -s "$out" ]]; then
    echo "  --native produced no output, retrying without it" >&2
    srun --overlap --ntasks=1 --cpus-per-task="$n" --cpu-bind=none \
      py-spy record --rate 200 -o "$out" -- "${cmd[@]}" || true
  fi
  if [[ -s "$out" ]]; then
    echo "  wrote $out" >&2
  else
    echo "  FAILED to record flamegraph for threads=$n" >&2
  fi
done
