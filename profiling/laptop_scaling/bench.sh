#!/usr/bin/env bash
# Laptop scaling benchmark: the same runs as the Helios study, at 1/2/4/8
# threads, pinned to known core types.
#
# Called by ../../bench_laptop.sh; that is the interface. See README.md here for
# why each stage exists and what it is meant to show.
#
# Passed in as environment or defaults below:
#   TIER, DOSE_RATES, THREAD_COUNTS, LADDERS, COOLDOWN, OUTROOT, REPO

set -uo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

TIER="${TIER:-full_electrode}"
RADIUS="${RADIUS:-}"                  # empty = use the tier's own radius
DOSE_RATES="${DOSE_RATES:-50 10}"
THREAD_COUNTS="${THREAD_COUNTS:-1 2 4 8}"
LADDERS="${LADDERS:-perf}"
COOLDOWN="${COOLDOWN:-60}"
OUTROOT="${OUTROOT:-${REPO}/profiling/data/laptop_scaling}"

PY="${PYTHON:-python}"

# --- environment reporting --------------------------------------------------
# All of this goes into the run record. A laptop benchmark without it is not
# interpretable: the same script on the same machine gives different answers on
# battery, in a power-saving profile, or after it has warmed up.

power_state() {
  local ac="unknown"
  for supply in /sys/class/power_supply/A*/online /sys/class/power_supply/AC*/online; do
    [ -r "$supply" ] && ac=$([ "$(cat "$supply")" = "1" ] && echo "mains" || echo "BATTERY")
  done
  echo "$ac"
}

governor() {
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown"
}

energy_pref() {
  cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo "n/a"
}

# Package temperature, in degrees C. Used only to report thermal state before a
# run; if the machine is already hot the number that follows is a throttled one.
package_temp_c() {
  local hottest=0 t
  for zone in /sys/class/thermal/thermal_zone*/temp; do
    [ -r "$zone" ] || continue
    t=$(( $(cat "$zone") / 1000 ))
    [ "$t" -gt "$hottest" ] && hottest=$t
  done
  echo "$hottest"
}

# Mean current clock of a CPU set, MHz. Sampled during a run by the watcher
# below; the drop between the first and last run of a stage is the thermal
# throttling that a laptop scaling curve otherwise silently absorbs.
mean_mhz() {
  local cpus="$1" total=0 count=0 khz
  for cpu in ${cpus//,/ }; do
    khz=$(cat "/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_cur_freq" 2>/dev/null) || continue
    total=$((total + khz)); count=$((count + 1))
  done
  [ "$count" -gt 0 ] && echo $((total / count / 1000)) || echo 0
}

# --- one run ----------------------------------------------------------------

run_one() {
  local ladder="$1" threads="$2" rate="$3"
  local outdir="${OUTROOT}/${ladder}"
  mkdir -p "$outdir"

  local cpus composition
  cpus=$($PY -m profiling.laptop_scaling.topology --cpulist "$ladder" "$threads") || return 1
  composition=$($PY -m profiling.laptop_scaling.topology --describe "$cpus")

  local temp_before mhz_before
  temp_before=$(package_temp_c)
  mhz_before=$(mean_mhz "$cpus")

  echo
  echo "--- ${ladder}: ${threads} thread(s) on CPUs ${cpus} (${composition}), ${rate} Gy/s"
  echo "    before: ${temp_before} C, ${mhz_before} MHz"

  # Sample the pinned CPUs' clocks every 5 s for as long as the run lasts, so a
  # thermally throttled run can be told apart from a slow one.
  local mhz_log; mhz_log=$(mktemp)
  ( while :; do mean_mhz "$cpus" >> "$mhz_log"; sleep 5; done ) &
  local watcher=$!

  local radius_args=()
  [ -n "$RADIUS" ] && radius_args=(--sampled-radius-cm "$RADIUS")

  local start; start=$(date +%s)
  taskset -c "$cpus" $PY -u examples/ifj_aic144/run_markus_2mm.py "$TIER" \
      --threads "$threads" \
      --backend batched \
      --dose-rate-water-Gy-s "$rate" \
      "${radius_args[@]}" \
      --json "${outdir}/threads${threads}_dose${rate}.json" \
    2>&1 | grep -Ev "^  step " | tail -6
  # Read PIPESTATUS on its own line: any command substitution on the same line
  # would run first and there is no need to reason about whether it clobbers it.
  local status=${PIPESTATUS[0]}
  local elapsed=$(( $(date +%s) - start ))

  kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null

  local mhz_mean=0
  [ -s "$mhz_log" ] && mhz_mean=$(awk '{s+=$1; n++} END {if (n) printf "%d", s/n}' "$mhz_log")
  rm -f "$mhz_log"

  echo "    after:  $(package_temp_c) C, mean ${mhz_mean} MHz during the run, ${elapsed} s wall"

  # Fold the laptop-specific context into the JSON the runner wrote, so a result
  # carries the conditions it was measured under and cannot be quoted without
  # them.
  if [ "$status" -eq 0 ]; then
    $PY - "${outdir}/threads${threads}_dose${rate}.json" <<EOF
import json, sys
path = sys.argv[1]
with open(path) as handle:
    record = json.load(handle)
record.update({
    "ladder": "${ladder}",
    "pinned_cpus": "${cpus}",
    "core_composition": "${composition}",
    "temp_before_c": ${temp_before},
    "mean_mhz_during_run": ${mhz_mean:-0},
    "power_source": "$(power_state)",
    "governor": "$(governor)",
    "energy_performance_preference": "$(energy_pref)",
})
with open(path, "w") as handle:
    json.dump(record, handle, indent=2)
EOF
  else
    echo "    FAILED (exit ${status})" >&2
  fi

  # Let the package cool before the next point, so run N+1 does not start from
  # the thermal state run N left behind. Skipped after the last run by the
  # caller's loop structure only incidentally -- 60 s is cheap either way.
  if [ "${COOLDOWN}" -gt 0 ]; then
    echo "    cooling down ${COOLDOWN}s"
    sleep "${COOLDOWN}"
  fi
}

# --- go ---------------------------------------------------------------------

echo "tier        : ${TIER}${RADIUS:+ (radius overridden to ${RADIUS} cm)}"
echo "ladders     : ${LADDERS}"
echo "threads     : ${THREAD_COUNTS}"
echo "dose rates  : ${DOSE_RATES} Gy/s to water"
echo "power       : $(power_state), governor $(governor), EPP $(energy_pref)"
echo "temperature : $(package_temp_c) C"
echo "output      : ${OUTROOT}"

for ladder in $LADDERS; do
  for rate in $DOSE_RATES; do
    for n in $THREAD_COUNTS; do
      run_one "$ladder" "$n" "$rate"
    done
  done
done

echo
echo "=== done. Collect with:"
for ladder in $LADDERS; do
  echo "    python profiling/helios_scaling/collect.py ${OUTROOT}/${ladder}"
done
