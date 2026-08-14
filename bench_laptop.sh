#!/usr/bin/env bash
# Benchmark this code on a laptop, the counterpart to ./submit.sh on Helios.
#
#     ./bench_laptop.sh                 # everything, ~80 min
#     ./bench_laptop.sh --stage topology   # what cores this machine has (instant)
#     ./bench_laptop.sh --stage cores      # P vs E cores, DRAM-resident grid (~10 min)
#     ./bench_laptop.sh --stage scaling    # the 1/2/4/8 ladder, full electrode (~70 min)
#
# It runs the same simulation as the Helios study -- the full-electrode grid at
# 1, 2, 4 and 8 threads, at 10 and 50 Gy/s -- so the two machines can be put in
# one table. What it adds is everything a laptop needs and a compute node does
# not:
#
#   * threads pinned to *known* core types. On a hybrid CPU "4 threads" is not
#     a description of a run: two P-cores plus two E-cores and four E-cores
#     differ by about a factor of two, and which you get is the scheduler's
#     choice unless you pin. Every result records which cores it used.
#   * one logical CPU per physical core, so the curve is cores and not a mix of
#     cores and SMT siblings.
#   * power state, governor, package temperature and the mean clock actually
#     sustained during each run, all folded into the result. On a laptop these
#     are not footnotes; a run on battery or on a warm machine is a different
#     measurement.
#   * a cool-down between runs, so run N+1 does not inherit run N's heat.
#
# Before starting, for numbers worth keeping:
#   - plug the laptop in;
#   - set a performance power profile;
#   - close everything else, especially browsers;
#   - do not use the machine while it runs.
#
# Results: profiling/data/laptop_scaling/<ladder>/threads{N}_dose{R}.json
# Read them with:  python profiling/cluster_scaling/collect.py <dir>

set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

STAGE="all"
TIER="full_electrode"
# The core-type study needs a grid on the *DRAM* side of the last-level cache,
# or it measures clock and IPC rather than the memory behaviour this whole
# investigation is about. Every named tier below full_electrode fits in a
# laptop's ~12 MiB L3 (`wide` is 11.3 MiB), so the radius is set directly:
# r = 0.09 cm is a 186^2 x 210 grid, 222 MiB of carrier arrays -- comfortably
# DRAM-resident -- at 12 % of the full electrode's cost.
CORE_STUDY_RADIUS_CM="0.09"
DOSE_RATES="50 10"
THREAD_COUNTS="1 2 4 8"
COOLDOWN=60
YES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --stage)    STAGE="$2";         shift 2 ;;
    --tier)     TIER="$2";          shift 2 ;;
    --rates)    DOSE_RATES="$2";    shift 2 ;;
    --threads)  THREAD_COUNTS="$2"; shift 2 ;;
    --cooldown) COOLDOWN="$2";      shift 2 ;;
    --yes|-y)   YES=1;              shift ;;
    -h|--help)  sed -n '2,33p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

fail() { echo "ERROR: $*" >&2; exit 1; }

# --- preflight --------------------------------------------------------------
[ "$(uname -s)" = "Linux" ] || fail "this needs Linux: it pins with taskset and reads
core types, clocks and temperatures from sysfs. On macOS or Windows, run the
simulation directly (see docs/BENCHMARKS-LAPTOP.md) -- the numbers will not be
pinned and are not comparable to the ones in that file."

command -v taskset >/dev/null || fail "taskset not found (install util-linux)"

if [ -d "${REPO}/venv" ]; then
  # shellcheck disable=SC1091
  source "${REPO}/venv/bin/activate"
fi
PYTHON="$(command -v python || command -v python3)" || fail "no python found"
export PYTHON
"$PYTHON" -c "import numba, numpy" 2>/dev/null || fail "numba/numpy missing. Install with:
    python -m venv venv && source venv/bin/activate && pip install -e '.[dev]'"

# --- stage: topology --------------------------------------------------------
echo "=== CPU topology ==="
"$PYTHON" -m profiling.laptop_scaling.topology
echo

if [ "$STAGE" = "topology" ]; then exit 0; fi

# A laptop on battery will downclock, and the resulting curve says more about
# the power profile than about the code. Warn loudly rather than refuse: someone
# may genuinely want the battery numbers.
for supply in /sys/class/power_supply/A*/online /sys/class/power_supply/AC*/online; do
  if [ -r "$supply" ] && [ "$(cat "$supply")" != "1" ]; then
    echo "WARNING: running on battery. Clocks will be capped and the results will"
    echo "         describe the power profile more than the code. Plug in first."
    echo
  fi
done

if [ "$YES" != "1" ]; then
  case "$STAGE" in
    all)     estimate="~80 minutes" ;;
    scaling) estimate="~70 minutes" ;;
    cores)   estimate="~10 minutes" ;;
    *)       estimate="unknown" ;;
  esac
  echo "Stage '${STAGE}' will take ${estimate} and should have the machine to itself."
  printf "Start? [y/N] "
  read -r reply
  case "$reply" in [yY]*) ;; *) echo "aborted"; exit 0 ;; esac
fi

export REPO COOLDOWN

# --- stage: cores -----------------------------------------------------------
# P-cores against E-cores at matched thread counts, on a grid small enough that
# this costs minutes. The question is how much of the P-core advantage survives
# on a memory-bound kernel: clock helps, but both core types queue behind the
# same memory controller.
if [ "$STAGE" = "all" ] || [ "$STAGE" = "cores" ]; then
  echo "=== core-type study: P vs E, 186^2 x 210 grid (222 MiB, DRAM-resident) ==="
  RADIUS="$CORE_STUDY_RADIUS_CM" \
  LADDERS="perf econ" \
  THREAD_COUNTS="1 2 4" \
  DOSE_RATES="50" \
  COOLDOWN=20 \
  OUTROOT="${REPO}/profiling/data/laptop_core_types" \
    bash profiling/laptop_scaling/bench.sh
fi

# --- stage: scaling ---------------------------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "scaling" ]; then
  echo
  echo "=== scaling ladder: '${TIER}' grid, the Helios comparison ==="
  TIER="$TIER" \
  LADDERS="perf" \
  THREAD_COUNTS="$THREAD_COUNTS" \
  DOSE_RATES="$DOSE_RATES" \
  COOLDOWN="$COOLDOWN" \
  OUTROOT="${REPO}/profiling/data/laptop_scaling" \
    bash profiling/laptop_scaling/bench.sh
fi

echo
echo "=== all done ==="
echo "collect:  python profiling/cluster_scaling/collect.py profiling/data/laptop_scaling/perf"
echo "compare:  docs/BENCHMARKS-LAPTOP.md  and  docs/HELIOS.md"
