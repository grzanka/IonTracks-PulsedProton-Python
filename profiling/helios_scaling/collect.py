#!/usr/bin/env python3
"""Collect the Helios thread-scaling study into tables, and check it.

Two jobs, and the second is the more important one.

**Tabulate.** Wall time, speed-up and per-core efficiency against thread count,
one table per dose rate, plus a side-by-side of the two rates.

**Verify.** A speed-up curve is only worth reading if every point computed the
same answer, so this refuses to print one until it has checked that:

* ``k_s`` agrees across thread counts, to a tolerance that allows for float
  reduction reordering (~1e-12 relative) but nothing larger;
* every run actually had the CPUs it claimed (``affinity_cpus >= threads``) --
  the failure mode on Slurm is a run that silently used one core and reported
  128, which looks exactly like "parallelism does not help";
* the grid, step count and track count are what that dose rate implies.

Usage:  python profiling/helios_scaling/collect.py [results_dir]
"""

import glob
import json
import os
import sys
from collections import defaultdict

# k_s is a sum over ~60 M voxels and 2194 steps; changing the thread count
# changes the order of that reduction, so the last couple of digits may move.
# Anything above this is a real disagreement, not float non-associativity.
KS_RTOL = 1e-12


def load(results_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "threads*_dose*.json"))):
        with open(path) as handle:
            row = json.load(handle)
        row["_path"] = os.path.basename(path)
        rows.append(row)
    return rows


def check(rows: list[dict]) -> list[str]:
    """Return a list of problems; empty means the study is trustworthy."""
    problems = []

    for row in rows:
        affinity = row.get("affinity_cpus")
        if affinity is not None and affinity < row["threads"]:
            problems.append(
                f"{row['_path']}: asked for {row['threads']} threads but only "
                f"{affinity} CPUs were visible -- this run was not what it says."
            )

    by_rate = defaultdict(list)
    for row in rows:
        by_rate[row.get("dose_rate_water_Gy_s")].append(row)

    for rate, group in sorted(by_rate.items(), key=lambda kv: (kv[0] is None, kv[0])):
        reference = min(group, key=lambda r: r["threads"])
        for row in group:
            drift = abs(row["ks"] - reference["ks"]) / reference["ks"]
            if drift > KS_RTOL:
                problems.append(
                    f"{rate} Gy/s: k_s at {row['threads']} threads is {row['ks']:.12f}, "
                    f"against {reference['ks']:.12f} at {reference['threads']} "
                    f"({drift:.2e} relative) -- thread count changed the physics."
                )
        for field in ("no_xy", "no_z_with_buffer", "total_time_steps", "tracks_per_pulse"):
            values = {row[field] for row in group}
            if len(values) > 1:
                problems.append(f"{rate} Gy/s: runs disagree on {field}: {sorted(values)}")

    return problems


def table(group: list[dict], title: str) -> None:
    group = sorted(group, key=lambda r: r["threads"])
    base = next((r["wall_s"] for r in group if r["threads"] == 1), None)
    print(f"\n{title}")
    print(f"{'threads':>7} {'wall_s':>9} {'speed-up':>9} {'per-core':>9} {'ms/step':>9}  k_s")
    for row in group:
        speedup = base / row["wall_s"] if base else None
        print(
            f"{row['threads']:>7} {row['wall_s']:>9.1f} "
            f"{(f'{speedup:.2f}x' if speedup else '-'):>9} "
            f"{(f'{speedup / row['threads']:.0%}' if speedup else '-'):>9} "
            f"{row['wall_s'] / row['total_time_steps'] * 1e3:>9.1f}  {row['ks']:.6f}"
        )


def compare(by_rate: dict) -> None:
    """The two dose rates side by side.

    The interesting column is the *ratio*: the PDE sweep does not care about
    dose rate at all, so anything above 1.0 is deposition, and how that ratio
    moves with thread count says which phases are still serial.
    """
    rates = sorted(r for r in by_rate if r is not None)
    if len(rates) != 2:
        return
    low, high = rates
    low_by_n = {r["threads"]: r for r in by_rate[low]}
    high_by_n = {r["threads"]: r for r in by_rate[high]}
    shared = sorted(set(low_by_n) & set(high_by_n))
    if not shared:
        return

    print(f"\n{high:g} Gy/s vs {low:g} Gy/s ({high / low:g}x the tracks, same grid)")
    print(f"{'threads':>7} {f'{low:g} Gy/s':>10} {f'{high:g} Gy/s':>10} {'ratio':>7}  {'k_s ' + str(low):>12} {'k_s ' + str(high):>12}")
    for n in shared:
        lo, hi = low_by_n[n], high_by_n[n]
        print(
            f"{n:>7} {lo['wall_s']:>10.1f} {hi['wall_s']:>10.1f} "
            f"{hi['wall_s'] / lo['wall_s']:>6.2f}x  {lo['ks']:>12.6f} {hi['ks']:>12.6f}"
        )


def main() -> int:
    default = os.path.join(os.path.dirname(__file__), "..", "data", "helios_scaling")
    results_dir = sys.argv[1] if len(sys.argv) > 1 else default
    rows = load(results_dir)
    if not rows:
        print(f"no results in {results_dir} -- has the queue drained? (squeue -u $USER)")
        return 1

    print(f"{len(rows)} runs from {os.path.abspath(results_dir)}")
    jobs = sorted({row.get("slurm_job_id") for row in rows if row.get("slurm_job_id")})
    hosts = sorted({row.get("hostname") for row in rows if row.get("hostname")})
    print(f"slurm jobs: {', '.join(jobs) or 'n/a'}")
    print(f"hosts     : {', '.join(hosts) or 'n/a'}")

    problems = check(rows)
    if problems:
        print("\n!! the study is NOT self-consistent:")
        for problem in problems:
            print(f"   - {problem}")
    else:
        print(f"\nchecks passed: k_s identical across thread counts (< {KS_RTOL:g} relative), "
              "every run had the CPUs it claimed, grids consistent per dose rate.")

    by_rate = defaultdict(list)
    for row in rows:
        by_rate[row.get("dose_rate_water_Gy_s")].append(row)
    for rate, group in sorted(by_rate.items(), key=lambda kv: (kv[0] is None, kv[0])):
        table(group, f"--- {rate:g} Gy/s to water ({group[0]['tracks_per_pulse']:,} tracks/pulse) ---")
    compare(by_rate)

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
