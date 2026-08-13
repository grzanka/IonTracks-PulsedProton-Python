#!/usr/bin/env bash
# Sweeps thread count x Numba threading layer for the converged-grid
# scenario. Each point is its own Slurm step / OS process (see
# profiling/sweep.py docstring for why), launched with --cpu-bind=none so
# it actually gets the requested number of CPUs instead of inheriting the
# interactive shell's own 1-CPU cpuset.
set -euo pipefail
cd "$(dirname "$0")/.."

module load GCCcore/13.3.0 Python/3.12.3
source venv/bin/activate

THREADS=(1 16 64 128 190)
LAYERS=(omp workqueue)
OUT=profiling/data/thread_sweep.csv
mkdir -p profiling/data
rm -f "$OUT"

for layer in "${LAYERS[@]}"; do
  for n in "${THREADS[@]}"; do
    echo "=== layer=$layer threads=$n ===" >&2
    NUMBA_THREADING_LAYER="$layer" srun --overlap --ntasks=1 --cpus-per-task="$n" --cpu-bind=none \
      python -m profiling.sweep --threads "$n" --out "$OUT"
  done
done
