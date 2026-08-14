#!/usr/bin/env bash
# Submit the Helios thread-scaling study: the full-electrode grid at 1, 2, 4, 8,
# 16, 32, 64 and 128 threads, at two dose rates (10 and 50 Gy/s to water).
#
# Run it from a Helios access node -- it only submits, it does not compute:
#
#     bash profiling/helios_scaling/submit_all.sh
#     squeue -u $USER
#     python profiling/helios_scaling/collect.py     # once the queue drains
#
# One job per thread count, each requesting exactly the cores it uses. Not a
# job array, because an array shares one --cpus-per-task across all its tasks,
# and asking for 128 cores to run a 1-thread job would waste 127 of them for
# twelve minutes and queue far longer than it needs to.
#
# WALLTIMES are ~2x the measured run time, which is the right margin here: too
# short kills a run at 97 %, and too long only delays scheduling. The estimates
# come from the measured 10 Gy/s curve in docs/HELIOS.md plus the ~5x track
# count at 50 Gy/s; adjust if the grid or the tier changes.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="${REPO}/profiling/data/helios_scaling"
LOGDIR="${OUTDIR}/logs"
DOSE_RATES="${DOSE_RATES:-50 10}"

mkdir -p "$LOGDIR"

# threads : walltime. Both dose rates run inside one job, so the walltime
# covers both, plus ~1 min of JIT warm-up and interpreter start.
#
#   threads     est. 50 Gy/s   est. 10 Gy/s   sum      walltime
#      1           ~12 min        ~11 min     ~23 min   0:50
#      2            ~7             ~6         ~13       0:35
#      4            ~4.5           ~4          ~9       0:25
#      8            ~5             ~4.5        ~10      0:25
#     16            ~3.7           ~3          ~7       0:20
#     32            ~2.5           ~2.3        ~5       0:15
#     64            ~1.9           ~1.5        ~3.5     0:15
#    128            ~1.7           ~1.4        ~3       0:15
#
# Only the 1- and 2-thread jobs exceed the ~15 min per job target, and they do
# so because one core cannot be made faster by splitting the work differently.
declare -A WALLTIME=(
  [1]="00:50:00"
  [2]="00:35:00"
  [4]="00:25:00"
  [8]="00:25:00"
  [16]="00:20:00"
  [32]="00:15:00"
  [64]="00:15:00"
  [128]="00:15:00"
)

THREAD_COUNTS="${THREAD_COUNTS:-1 2 4 8 16 32 64 128}"

echo "repo    : ${REPO}"
echo "results : ${OUTDIR}"
echo "rates   : ${DOSE_RATES} Gy/s to water"
echo

for n in $THREAD_COUNTS; do
  walltime="${WALLTIME[$n]:-00:60:00}"
  jobid=$(sbatch --parsable \
    --job-name="ks-n${n}" \
    --cpus-per-task="${n}" \
    --time="${walltime}" \
    --output="${LOGDIR}/ks-n${n}-%j.out" \
    --export=ALL,THREADS="${n}",DOSE_RATES="${DOSE_RATES}",OUTDIR="${OUTDIR}",REPO="${REPO}" \
    "${REPO}/profiling/helios_scaling/scaling_job.sbatch")
  echo "submitted ${jobid}  threads=${n}  cpus=${n}  walltime=${walltime}"
done

echo
echo "watch:    squeue -u \$USER"
echo "collect:  python ${REPO}/profiling/helios_scaling/collect.py"
