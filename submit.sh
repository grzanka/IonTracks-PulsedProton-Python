#!/usr/bin/env bash
# Submit the cluster thread-scaling study. Run this on a Helios or Ares access
# node -- the machine is detected and its settings come from
# profiling/cluster_scaling/sites.sh:
#
#     ./submit.sh
#
# That is the whole interface. It checks the environment, submits one job per
# (thread count, dose rate) at 10 and 50 Gy/s to water on the full-electrode
# grid, and prints the two commands you need afterwards: one to watch the
# queue, one to read the results. The thread ladder follows the machine:
# 1..128 on Helios, 1..48 on Ares, chosen at its NUMA and socket boundaries.
#
# Nothing computes here; the jobs do. They are independent, so on an empty queue
# they run concurrently and the study is done in ~15 minutes -- the length of
# its longest single run, which is the 1-thread 50 Gy/s case at ~13 min.
#
# Options, all optional:
#   ./submit.sh --threads "1 8 64"    only these thread counts
#   ./submit.sh --rates "50"          only this dose rate (Gy/s to water)
#   ./submit.sh --account plgXXX-cpu  charge a different grant (default plgccbmc15-cpu)
#   ./submit.sh --mem 16G             memory per job (default 8G; ~2.1 GiB is used)
#   ./submit.sh --exclusive           a whole node per job. Slower to schedule and it
#                                     charges 192 cores for a 1-core run, but this
#                                     benchmark is bandwidth-bound and a co-tenant
#                                     competes for the very thing being measured.
#   ./submit.sh --dry-run             print what would be submitted, submit nothing
#
# Results land in profiling/data/cluster_scaling/ as one JSON per run, with
# job logs under logs/. See docs/HELIOS.md for what the numbers mean.

set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"

THREAD_COUNTS="${THREAD_COUNTS:-}"  # empty = take the site's ladder
DOSE_RATES="50 10"
SITE="${SITE:-}"
ACCOUNT="${ACCOUNT:-}"
# Per job, not per core. The plgrid partition hands out DefMemPerCPU=1536 MB, so
# a 1-core job would otherwise get 1.5 GB against a measured peak of ~2.1 GiB
# and be OOM-killed -- the grid does not shrink when the thread count does.
MEM="${MEM:-8G}"
EXCLUSIVE="${EXCLUSIVE:-}"          # empty = take the site default
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --threads) THREAD_COUNTS="$2"; shift 2 ;;
    --rates)   DOSE_RATES="$2";    shift 2 ;;
    --account) ACCOUNT="$2";       shift 2 ;;
    --mem)     MEM="$2";           shift 2 ;;
    --exclusive) EXCLUSIVE=1;      shift ;;
    --shared)    EXCLUSIVE=0;      shift ;;
    --site)      SITE="$2";        shift 2 ;;
    --dry-run) DRY_RUN=1;          shift ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

# --- preflight --------------------------------------------------------------
# Each of these is a failure that would otherwise surface as eight jobs dying
# one after another, minutes apart, with the reason buried in a log file.

fail() { echo "ERROR: $*" >&2; exit 1; }

command -v sbatch >/dev/null || fail "sbatch not found -- is this a Slurm login node?"

[ -d "${REPO}/venv" ] || fail "no venv at ${REPO}/venv -- create it first:
    module load GCCcore/13.3.0 Python/3.12.3
    python -m venv venv && source venv/bin/activate && pip install -e '.[dev]'"

[ -f "${REPO}/examples/ifj_aic144/run_markus_2mm.py" ] || fail "run from the repository root"

if [ "$DRY_RUN" = "1" ]; then
  # Resolve the site config rather than printing "site default": the point of a
  # dry run is to see the account, partition and ladder that would actually be
  # used, on a machine where getting one of them wrong wastes a queue slot.
  # shellcheck source=profiling/cluster_scaling/sites.sh
  source "${REPO}/profiling/cluster_scaling/sites.sh"
  resolved_site="${SITE:-$(site_detect)}"
  if ! site_configure "$resolved_site"; then
    echo "dry run -- unknown site '${resolved_site}'." >&2
    echo "Pass --site helios|ares, or add a branch to" >&2
    echo "profiling/cluster_scaling/sites.sh." >&2
    exit 2
  fi
  echo "dry run -- nothing will be submitted"
  echo "site      : ${SITE_NAME} (${resolved_site}), ${SITE_CORES_PER_NODE} cores/node"
  echo "modules   : ${SITE_MODULES:-none}"
  echo "account   : ${ACCOUNT:-$SITE_ACCOUNT}"
  echo "partition : ${SITE_PARTITION}"
  echo "threads   : ${THREAD_COUNTS:-$SITE_THREADS}"
  echo "rates     : ${DOSE_RATES} Gy/s to water"
  echo "memory    : ${MEM} per job"
  echo "exclusive : $([ "${EXCLUSIVE:-$SITE_EXCLUSIVE}" = "1" ] && echo yes || echo no)"
  echo "results   : profiling/data/${resolved_site}_scaling/"
  n_jobs=0
  for _ in ${THREAD_COUNTS:-$SITE_THREADS}; do for _ in $DOSE_RATES; do n_jobs=$((n_jobs+1)); done; done
  echo "would submit ${n_jobs} jobs"
  exit 0
fi

# Compute nodes on Helios cannot submit; the error Slurm gives is a bare
# "Access/permission denied", which is not much of a hint on its own.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  fail "this looks like a compute node (inside job ${SLURM_JOB_ID}). Submit from an access node."
fi

# --- submit -----------------------------------------------------------------
# Only export the overrides that were actually given; anything empty falls
# through to the site defaults in sites.sh.
env \
  ${SITE:+SITE="${SITE}"} \
  ${THREAD_COUNTS:+THREAD_COUNTS="${THREAD_COUNTS}"} \
  ${ACCOUNT:+ACCOUNT="${ACCOUNT}"} \
  ${EXCLUSIVE:+EXCLUSIVE="${EXCLUSIVE}"} \
  DOSE_RATES="${DOSE_RATES}" MEM="${MEM}" \
  bash "${REPO}/profiling/cluster_scaling/submit_all.sh"
