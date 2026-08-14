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

# Everything cluster-specific -- modules, account, partition, the thread ladder
# and whether exclusive is the default -- comes from sites.sh, so this file is
# the same on every machine.
# shellcheck source=sites.sh
source "${REPO}/profiling/cluster_scaling/sites.sh"
SITE="${SITE:-$(site_detect)}"
if ! site_configure "$SITE"; then
  echo "unknown site '${SITE}'. Set SITE=helios or SITE=ares, or add a branch" >&2
  echo "to profiling/cluster_scaling/sites.sh for this machine." >&2
  exit 2
fi

# Results per site, so two clusters do not overwrite each other's numbers --
# the file name carries the thread count and dose rate but not the machine, and
# a mixed directory is a table nobody can interpret.
OUTDIR="${OUTDIR:-${REPO}/profiling/data/${SITE}_scaling}"
LOGDIR="${OUTDIR}/logs"

ACCOUNT="$SITE_ACCOUNT"
PARTITION="$SITE_PARTITION"
MEM="${MEM:-8G}"                      # vs ~2.1 GiB measured peak RSS
# Whole node per job; the default is per-site (sites.sh). Off on Helios because
# it charges 192 cores to run a one-core job, on everywhere smaller. This
# benchmark is memory-bandwidth-bound, so a co-tenant competes for exactly the
# resource being measured: leaving it off on Helios inflated the mid-range of
# the study by up to 195 %.
EXCLUSIVE="$SITE_EXCLUSIVE"
DOSE_RATES="${DOSE_RATES:-50 10}"
THREAD_COUNTS="$SITE_THREADS"

mkdir -p "$LOGDIR"

# Walltimes come from sites.sh: ~2x the measured run time, with only the
# single- and two-core jobs needing more than the partition default. One core
# cannot be made faster by rearranging the work.

echo "site      : ${SITE_NAME} (${SITE}), ${SITE_CORES_PER_NODE} cores/node"
echo "modules   : ${SITE_MODULES:-none}"
echo "repo      : ${REPO}"
echo "results   : ${OUTDIR}"
echo "account   : ${ACCOUNT}"
echo "partition : ${PARTITION}"
echo "memory    : ${MEM} per job (peak RSS measured at ~2.1 GiB)"
echo "exclusive : $([ "$EXCLUSIVE" = "1" ] && echo "yes -- whole node per job" || echo "no -- nodes may be shared, see README")"
echo "threads   : ${THREAD_COUNTS}"
echo "rates     : ${DOSE_RATES} Gy/s to water"
echo

submitted=0
for n in $THREAD_COUNTS; do
  walltime="$(site_walltime "$n")"
  for rate in $DOSE_RATES; do
    exclusive_args=()
    [ "$EXCLUSIVE" = "1" ] && exclusive_args=(--exclusive)
    jobid=$(sbatch --parsable \
      "${exclusive_args[@]}" \
      --job-name="ks-n${n}-d${rate}" \
      --account="${ACCOUNT}" \
      --partition="${PARTITION}" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="${n}" \
      --mem="${MEM}" \
      --time="${walltime}" \
      --output="${LOGDIR}/ks-n${n}-d${rate}-%j.out" \
      --export=ALL,THREADS="${n}",DOSE_RATES="${rate}",OUTDIR="${OUTDIR}",REPO="${REPO}",SITE="${SITE}" \
      "${REPO}/profiling/cluster_scaling/scaling_job.sbatch")
    echo "submitted ${jobid}  threads=${n}  dose=${rate} Gy/s  cpus=${n}  mem=${MEM}  walltime=${walltime}$([ "$EXCLUSIVE" = "1" ] && echo "  exclusive")"
    submitted=$((submitted + 1))
  done
done

echo
echo "${submitted} jobs submitted. If the queue is empty they run concurrently,"
echo "so expect results in ~15 minutes rather than ${submitted} x 15."
echo
echo "watch:    squeue -u \$USER"
echo "collect:  python ${REPO}/profiling/cluster_scaling/collect.py"
