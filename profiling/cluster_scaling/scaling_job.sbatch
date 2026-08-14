#!/usr/bin/env bash
# One job of the Helios thread-scaling study: run the full-electrode grid at a
# fixed thread count, at each of the requested dose rates.
#
# Not submitted directly -- `submit_all.sh` fills in the core count, the
# walltime and the run list, because those three have to agree with each other
# and picking them by hand is how a study ends up with a job that dies at 97 %.
#
# Passed in via --export:
#   THREADS    thread count for this job (also the job's --cpus-per-task)
#   DOSE_RATES space-separated dose rates to water, Gy/s (normally one)
#   OUTDIR     where the per-run JSON goes
#   REPO       repository root
#   SITE       which cluster (selects the module line -- see sites.sh)
#
# Account, partition, cores, memory, walltime and log path all come from the
# sbatch command line in submit_all.sh, so that the resource request and the
# estimates it is derived from sit next to each other in one file. The
# directives below are defaults, in case this is ever submitted by hand.
#
#SBATCH --job-name=ks-scaling
#SBATCH --account=plgccbmc15-cpu
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -uo pipefail
cd "${REPO:?REPO not set}"

# The module line differs per cluster; sites.sh holds it. Failing here is much
# better than falling through to a system python that has no numba, which shows
# up several minutes later as an ImportError in every job at once.
# shellcheck source=sites.sh
source "${REPO}/profiling/cluster_scaling/sites.sh"
site_configure "${SITE:?SITE not set}" || { echo "FATAL: unknown site ${SITE}" >&2; exit 1; }
if [ -n "${SITE_MODULES}" ]; then
  # shellcheck disable=SC2086
  module load ${SITE_MODULES} || {
    echo "FATAL: 'module load ${SITE_MODULES}' failed on ${SITE}." >&2
    echo "Find the right module with 'module spider python' and fix" >&2
    echo "SITE_MODULES in profiling/cluster_scaling/sites.sh." >&2
    exit 1
  }
fi
source venv/bin/activate
python -c "import numba, numpy" || {
  echo "FATAL: numba/numpy not importable after loading ${SITE_MODULES}." >&2
  echo "Is venv/ built against this cluster's python?" >&2
  exit 1
}

mkdir -p "$OUTDIR"

# The affinity check is the point of this block, not decoration. A batch job's
# script inherits the job cpuset, so `--cpus-per-task=N` really does give N
# CPUs here -- unlike an interactive allocation, where the shell is pinned to
# one CPU and every thread count silently collapses to 1 (docs/HELIOS.md §3).
# Verifying it costs nothing and turns "the speed-up curve is flat" from a
# mystery into an assertion failure.
AFFINITY=$(python -c "import os; print(len(os.sched_getaffinity(0)))")
echo "site=${SITE}  host=$(hostname)  job=${SLURM_JOB_ID}  cpus-per-task=${SLURM_CPUS_PER_TASK}  affinity=${AFFINITY}  threads=${THREADS}  mem=${SLURM_MEM_PER_NODE:-?}M"
# What the OS thinks the NUMA layout of this job is. On a machine with
# Sub-NUMA Clustering the CPU ids a job receives may span several domains in a
# non-obvious order, and that decides how the first-touch placement lands.
command -v numactl >/dev/null && numactl --show 2>/dev/null | sed "s/^/  numactl: /"
if [ "$AFFINITY" -lt "$THREADS" ]; then
  echo "FATAL: only ${AFFINITY} CPUs visible but ${THREADS} threads requested." >&2
  exit 1
fi

for rate in $DOSE_RATES; do
  tag="threads${THREADS}_dose${rate}"
  echo "=== ${tag} : $(date +%T) ==="
  # --backend batched at every thread count including 1: comparing thread counts
  # means holding the backend fixed, and the unbatched one would take hours here.
  /usr/bin/time -v python -u examples/ifj_aic144/run_markus_2mm.py full_electrode \
      --threads "$THREADS" \
      --backend batched \
      --dose-rate-water-Gy-s "$rate" \
      --json "${OUTDIR}/${tag}.json" \
    2>&1 | grep -Ev "^  step " | tail -25
  echo "=== ${tag} done : $(date +%T) ==="
done
