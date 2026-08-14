#!/usr/bin/env bash
# Thread scaling of the full-electrode tier, as whole runs rather than kernels.
#
# The point of measuring the real run and not just the kernels (see
# profiling/bench_kernels.py) is that a run also pays for the phases that were
# never threaded: track sampling, the per-track Gaussian factors, the row index,
# the diagnostics histogram. Those are what set the ceiling now.
#
# Each thread count is its own `srun` step: an interactive Slurm shell is itself
# cpuset-restricted to one CPU no matter how many the allocation holds, so a
# bare `python ...` would report an affinity of 1 and silently run
# single-threaded. See docs/HELIOS.md.
#
# Usage, from inside an allocation with >= 190 cores on the node:
#   bash profiling/run_full_electrode_sweep.sh [thread counts...]

set -uo pipefail
cd "$(dirname "$0")/.."

THREADS=("${@:-190 96 48 24 8}")
# shellcheck disable=SC2206
THREADS=(${THREADS[*]})
OUT=profiling/data/full_electrode_sweep
mkdir -p "$OUT"

for n in "${THREADS[@]}"; do
  echo "=== full_electrode, ${n} thread(s) ==="
  srun --overlap --ntasks=1 --cpus-per-task="$n" --cpu-bind=none \
    python examples/ifj_aic144/run_markus_2mm.py full_electrode \
      --threads "$n" --json "$OUT/threads_${n}.json" \
    2>&1 | grep -Ev "^  step " | tail -12
done

echo
echo "=== collected ==="
python - "$OUT" <<'PY'
import glob, json, sys
rows = sorted((json.load(open(p)) for p in glob.glob(f"{sys.argv[1]}/threads_*.json")),
              key=lambda r: r["threads"])
base = next((r["wall_s"] for r in rows if r["threads"] == 1), None)
print(f"{'threads':>7} {'wall_s':>9} {'speedup':>8} {'s/step':>8}  k_s")
for r in rows:
    speedup = f"{base / r['wall_s']:.1f}x" if base else "-"
    print(f"{r['threads']:>7} {r['wall_s']:>9.1f} {speedup:>8} "
          f"{r['wall_s'] / r['total_time_steps'] * 1e3:>7.1f}ms  {r['ks']:.6f}")
PY
