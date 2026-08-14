#!/usr/bin/env python3
"""Per-phase kernel benchmark: which phase of a time step costs what, and how
each one scales with thread count on a grid too large to cache.

Why this exists
---------------
`profiling/sweep.py` times whole runs on the small "converged" grid, where the
answer came out "threads do not help". That grid is 40 MiB of carrier arrays --
smaller than this node's aggregate L3 (768 MiB) -- so it never touches DRAM in
anger, and each parallel region is short enough that fork/join is comparable to
the work. Neither is true for the full-electrode grid, which is the one we
actually want to run.

This script isolates the four things a time step does and times each one
separately on the *real* grid, so a scaling curve can be attributed to a phase
rather than to a run:

    sweep       the Lax-Wendroff kernel                (numba, prange)
    copyback    `array[1:-1,1:-1,1:-1] = next[...]`    (numpy, SERIAL)
    broadcast   the batched deposition's phase 2       (numba, prange)
    boundary    the reflecting-wall plane mirror       (numpy, serial, 4 planes)

Every phase is also reported as GB/s of DRAM traffic, counting the minimum
number of bytes it must move (see `_TRAFFIC`), so the numbers can be compared
against a STREAM-like ceiling instead of only against each other.

First touch matters
-------------------
`np.zeros` gets zero-filled pages lazily: a page is physically placed on the
NUMA node of whichever thread first *writes* it. With `--init serial` the main
thread touches everything, so all 1.9 GiB lands on one of this node's 8 NUMA
domains and every worker thread reads across the fabric from a single memory
controller. `--init parallel` first-touches through the same flattened-(i, j)
prange decomposition the kernels use, so each thread's pages are local to it.
The gap between the two is the NUMA penalty, and on this hardware it is large.

Usage (must be its own srun step -- an interactive shell is pinned to 1 CPU):

    srun --overlap --ntasks=1 --cpus-per-task=190 --cpu-bind=none \
      python -m profiling.bench_kernels --threads 1,8,24,48,96,190 \
      --init parallel --json profiling/data/bench_kernels.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Numba reads NUMBA_NUM_THREADS at import time and caps set_num_threads() at it,
# so it has to be raised before `import numba` anywhere in the process.
os.environ.setdefault("NUMBA_NUM_THREADS", str(max(1, len(os.sched_getaffinity(0)))))

import numba  # noqa: E402
from numba import prange  # noqa: E402

from pulsed_ion_chamber.config import SimulationConfig  # noqa: E402
from pulsed_ion_chamber.state import apply_lateral_boundary  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "ifj_aic144"))


def build_config(tier: str, grid_size_um: float | None = None) -> SimulationConfig:
    """The AIC-144 Markus 2 mm config at one grid tier, memory check disabled.

    Imported from the example rather than duplicated, so a benchmark cannot
    drift away from the scenario it claims to measure.
    """
    from run_markus_2mm import BEAM_KWARGS, CHAMBER_PHYSICS_KWARGS, GRID_TIERS

    sampled_radius_cm, buffer_radius = GRID_TIERS[tier]
    return SimulationConfig(
        **BEAM_KWARGS,
        **CHAMBER_PHYSICS_KWARGS,
        grid_size_um=grid_size_um if grid_size_um else 10.0,
        sampled_radius_cm=sampled_radius_cm,
        buffer_radius=buffer_radius,
        no_z_electrode=5,
        max_voxels=400_000_000,
    )


@numba.njit(parallel=True, cache=True)
def _first_touch_parallel(array, value):
    """Write `value` over the whole array through the kernels' own (i, j)
    decomposition, so each page is first touched by the thread that will own it.

    Deliberately *not* `array[:] = value`: the point is which thread does the
    writing, not that the writing happens.
    """
    no_i, no_j, no_k = array.shape
    for idx in prange(no_i * no_j):
        i = idx // no_j
        j = idx % no_j
        for k in range(no_k):
            array[i, j, k] = value


@numba.njit(parallel=True, cache=True)
def _copyback_parallel(dst, src):
    """The copy-back, threaded over the same flattened (i, j) as the sweep.

    Exists only so the benchmark can separate "the copy is serial" from "the
    copy is unavoidable": a threaded copy is the cheap fix, deleting the copy
    is the real one.
    """
    no_i, no_j, no_k = dst.shape
    for idx in prange((no_i - 2) * (no_j - 2)):
        i = 1 + idx // (no_j - 2)
        j = 1 + idx % (no_j - 2)
        for k in range(1, no_k - 1):
            dst[i, j, k] = src[i, j, k]


# Bytes of DRAM traffic each phase must move, per time step, as a function of
# one carrier array's size B. Reads and writes are counted alike; a write is
# charged once even though a write-allocate costs a read too, so the reported
# GB/s is a lower bound on real bus traffic.
_TRAFFIC = {
    # read p, n; write p_next, n_next (neighbour reads assumed cache hits)
    "sweep": lambda B: 4 * B,
    # read next, write current, both species
    "copyback": lambda B: 4 * B,
    "copyback_parallel": lambda B: 4 * B,
    # read+write two species over the columns a track reached
    "broadcast": lambda B: 4 * B,
    # 4 lateral planes, read + write, both species: 8 * (B / no_xy) * 2
    "boundary": lambda B, no_xy: 16 * B / no_xy,
}


def _time(fn, repeats: int) -> float:
    """Best-of-`repeats` seconds. Best, not mean: we want the phase's cost, not
    the cost of whatever else the node was doing during one sample."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench(config: SimulationConfig, threads: int, init: str, repeats: int) -> dict:
    from pulsed_ion_chamber.solver_numba_parallel import (
        _broadcast_density_numba_parallel,
        _lax_wendroff_step_numba_parallel,
    )

    numba.set_num_threads(threads)
    shape = (config.no_xy, config.no_xy, config.no_z_with_buffer)

    arrays = [np.empty(shape) for _ in range(4)]
    if init == "parallel":
        for a in arrays:
            _first_touch_parallel(a, 1e6)
    else:
        for a in arrays:
            a[:] = 1e6  # numpy: main thread touches every page
    positive, negative, positive_next, negative_next = arrays

    total_density = np.full((config.no_xy, config.no_xy), 1.0)
    (p_lat, p_zm, p_zp, p_cen), (n_lat, n_zm, n_zp, n_cen) = config.scheme_coefficients()
    alpha_dt = 1.6e-6 * config.dt

    def sweep():
        _lax_wendroff_step_numba_parallel(
            positive, negative, positive_next, negative_next,
            config.no_xy, config.no_z_with_buffer, config.no_z_electrode, config.no_z,
            config.mid_xy, config.scoring_radius_sq,
            p_lat, p_zm, p_zp, p_cen, n_lat, n_zm, n_zp, n_cen, alpha_dt,
        )

    def copyback():
        positive[1:-1, 1:-1, 1:-1] = positive_next[1:-1, 1:-1, 1:-1]
        negative[1:-1, 1:-1, 1:-1] = negative_next[1:-1, 1:-1, 1:-1]

    def copyback_parallel():
        _copyback_parallel(positive, positive_next)
        _copyback_parallel(negative, negative_next)

    def broadcast():
        _broadcast_density_numba_parallel(
            positive, negative, total_density, config.no_xy, config.no_z,
            config.no_z_electrode, config.Gaussian_factor, config.mid_xy,
            config.scoring_radius_sq,
        )

    def boundary():
        apply_lateral_boundary(positive, config.lateral_boundary)
        apply_lateral_boundary(negative, config.lateral_boundary)

    phases = {
        "sweep": sweep,
        "copyback": copyback,
        "copyback_parallel": copyback_parallel,
        "broadcast": broadcast,
        "boundary": boundary,
    }
    for fn in phases.values():  # compile / warm the pages before timing
        fn()

    array_bytes = float(np.prod(shape) * 8)
    row = {
        "threads": threads,
        "init": init,
        "no_xy": config.no_xy,
        "no_z_with_buffer": config.no_z_with_buffer,
        "array_MiB": array_bytes / 2**20,
        "total_time_steps": config.total_time_steps,
    }
    for name, fn in phases.items():
        seconds = _time(fn, repeats)
        traffic = (
            _TRAFFIC[name](array_bytes, config.no_xy)
            if name == "boundary"
            else _TRAFFIC[name](array_bytes)
        )
        row[f"{name}_ms"] = seconds * 1e3
        row[f"{name}_GBps"] = traffic / seconds / 1e9
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="full_electrode")
    parser.add_argument("--grid-size-um", type=float, default=None)
    parser.add_argument("--threads", default="1", help="comma-separated thread counts")
    parser.add_argument("--init", default="parallel", choices=("parallel", "serial", "both"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    thread_counts = [int(t) for t in args.threads.split(",")]
    inits = ("serial", "parallel") if args.init == "both" else (args.init,)
    config = build_config(args.tier, args.grid_size_um)

    print(f"tier={args.tier}  grid={config.no_xy}^2 x {config.no_z_with_buffer}  "
          f"steps={config.total_time_steps}  affinity={len(os.sched_getaffinity(0))}  "
          f"NUMBA_NUM_THREADS={numba.config.NUMBA_NUM_THREADS}  "
          f"layer={numba.config.THREADING_LAYER}")

    rows = []
    header = f"{'thr':>4} {'init':>8} " + " ".join(
        f"{p:>12}" for p in ("sweep", "copyback", "cpyb_par", "broadcast", "boundary")
    ) + f" {'step_ms':>9} {'est_run_s':>9}"
    print(header)
    for init in inits:
        for threads in thread_counts:
            row = bench(config, threads, init, args.repeats)
            # A step = sweep + copyback + boundary, plus broadcast while the
            # pulse is still injecting (81 % of steps in this scenario).
            row["step_ms_current"] = row["sweep_ms"] + row["copyback_ms"] + row["boundary_ms"]
            row["est_run_s_current"] = row["step_ms_current"] * config.total_time_steps / 1e3
            rows.append(row)
            print(
                f"{threads:>4} {init:>8} "
                + " ".join(
                    f"{row[k]:>8.1f}ms" for k in
                    ("sweep_ms", "copyback_ms", "copyback_parallel_ms", "broadcast_ms", "boundary_ms")
                )
                + f" {row['step_ms_current']:>9.1f} {row['est_run_s_current']:>9.1f}"
            )
            print(
                "     " + " " * 8 + " "
                + " ".join(
                    f"{row[k]:>8.1f}GB/s" for k in
                    ("sweep_GBps", "copyback_GBps", "copyback_parallel_GBps",
                     "broadcast_GBps", "boundary_GBps")
                )
            )

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump({"rows": rows, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, handle, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
