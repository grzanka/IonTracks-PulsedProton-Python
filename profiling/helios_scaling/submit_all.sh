#!/usr/bin/env bash
# Submit the Helios thread-scaling study: the full-electrode grid at 1, 2, 4, 8,
# 16, 32, 64 and 128 threads, at two dose rates (10 and 50 Gy/s to water).
#
# Called by ../../submit.sh; that is the interface. Run this directly only to
# override something submit.sh does not expose.
#
# ---------------------------------------------------------------------------
# One job per (thread count, dose rate) -- 16 independent jobs
# ---------------------------------------------------------------------------
# The two dose rates share nothing but the grid dimensions, so running them in
# one job only serialises them. Split, the whole study's turnaround is the
# longest *single* run (~13 min, the 1-thread 50 Gy/s case) rather than the
# longest *pair* (~23 min), and every job fits inside a 15-minute slot except
# the two single-core ones.
#
# Not a job array, either: an array shares one --cpus-per-task across all its
# tasks, so the 1-thread runs would sit inside a 128-core reservation, wasting
# 127 cores and queueing far longer than they need to. Separate submissions each
# ask for exactly what they use.
#
# ---------------------------------------------------------------------------
# Memory must be explicit
# ---------------------------------------------------------------------------
# Helios's plgrid partition has DefMemPerCPU=1536 MB, i.e. memory is handed out
# *per core*. This study's peak RSS is ~2.1 GiB regardless of thread count -- it
# is one grid, not one grid per thread -- so a 1-core job would default to
# 1.5 GB and be OOM-killed, and a 2-core job would run on a 1.4x margin. The
# --mem below is per job and generous; it costs nothing at these sizes.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="${REPO}/profiling/data/helios_scaling"
LOGDIR="${OUTDIR}/logs"

ACCOUNT="${ACCOUNT:-plgccbmc15-cpu}"
PARTITION="${PARTITION:-plgrid}"
MEM="${MEM:-8G}"                      # vs ~2.1 GiB measured peak RSS
DOSE_RATES="${DOSE_RATES:-50 10}"
THREAD_COUNTS="${THREAD_COUNTS:-1 2 4 8 16 32 64 128}"

mkdir -p "$LOGDIR"

# Walltime per job, ~2x the measured run time. Only the single-core jobs need
# more than the partition's 15-minute default, and they need it because one core
# cannot be made faster by rearranging the work.
#
#   threads   50 Gy/s   10 Gy/s        (measured where available, else scaled
#      1       ~13 min   ~11 min        from the 10 Gy/s curve in HELIOS.md)
#      2        ~7        ~6
#      4        ~4.5      ~4
#      8        ~5        ~4.5
#     16        ~3.7      ~3
#     32        ~2.5      ~2.3
#     64         1.9       1.5
#    128         1.5       1.0
walltime_for() {
  case "$1" in
    1) echo "00:30:00" ;;
    2) echo "00:20:00" ;;
    *) echo "00:15:00" ;;
  esac
}

echo "repo      : ${REPO}"
echo "results   : ${OUTDIR}"
echo "account   : ${ACCOUNT}"
echo "partition : ${PARTITION}"
echo "memory    : ${MEM} per job (peak RSS measured at ~2.1 GiB)"
echo "threads   : ${THREAD_COUNTS}"
echo "rates     : ${DOSE_RATES} Gy/s to water"
echo

submitted=0
for n in $THREAD_COUNTS; do
  walltime="$(walltime_for "$n")"
  for rate in $DOSE_RATES; do
    jobid=$(sbatch --parsable \
      --job-name="ks-n${n}-d${rate}" \
      --account="${ACCOUNT}" \
      --partition="${PARTITION}" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="${n}" \
      --mem="${MEM}" \
      --time="${walltime}" \
      --output="${LOGDIR}/ks-n${n}-d${rate}-%j.out" \
      --export=ALL,THREADS="${n}",DOSE_RATES="${rate}",OUTDIR="${OUTDIR}",REPO="${REPO}" \
      "${REPO}/profiling/helios_scaling/scaling_job.sbatch")
    echo "submitted ${jobid}  threads=${n}  dose=${rate} Gy/s  cpus=${n}  mem=${MEM}  walltime=${walltime}"
    submitted=$((submitted + 1))
  done
done

echo
echo "${submitted} jobs submitted. If the queue is empty they run concurrently,"
echo "so expect results in ~15 minutes rather than ${submitted} x 15."
echo
echo "watch:    squeue -u \$USER"
echo "collect:  python ${REPO}/profiling/helios_scaling/collect.py"
