#!/usr/bin/env python3
"""GPU vs CPU wall-clock benchmark for the AIC-144 Markus 2 mm scenario.

Runs the same physical case (`examples/ifj_aic144/run_markus_2mm.py`) across a
ladder of grid sizes on the CUDA backend and, for comparison, the batched CPU
backend at a chosen thread count, and reports wall time, `k_s` agreement and
the GPU carrier-array footprint for each.

The point it demonstrates is the crossover of docs/PERFORMANCE.md sec. 6: both
hot loops are memory-bandwidth-bound, so the GPU only wins once the grid is too
big to live in a CPU's cache, and the win then grows with size. The headline is
the full Markus electrode at 5 um voxels -- 466 M voxels, 14.9 GiB of carrier
arrays -- which does not fit in host cache at all but sits comfortably in the
A100's 40 GiB of HBM.

Every backend is checked to produce the same `k_s` before its time is reported;
a run whose `k_s` disagrees beyond a relative tolerance is flagged, because a
wall time is meaningless if the physics moved.

Run (one command per line, each independently):

    python profiling/bench_gpu.py --ladder crossover --threads 32
    python profiling/bench_gpu.py --ladder full --threads 32 --json profiling/data/bench_gpu.json
    python profiling/bench_gpu.py --sizes 0.265@5 --backends gpu --threads 32

A `--sizes` entry is `sampled_radius_cm@grid_um` (grid_um optional, default 10),
e.g. `0.265@5` is the full electrode at 5 um. `--backends` picks any of
`gpu`, `cpu` (batched at `--threads`); default runs both.
"""

import argparse
import json
import os
import platform
import time

import numpy as np

# The scenario's beam/chamber/gas kwargs live in the example; import them so the
# physics benchmarked here is byte-for-byte the published case, not a copy that
# can drift.
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "ifj_aic144"))
from run_markus_2mm import BEAM_KWARGS, CHAMBER_PHYSICS_KWARGS, WATER_TO_AIR  # noqa: E402

from pulsed_ion_chamber.config import SimulationConfig  # noqa: E402

KS_RTOL = 1e-6  # generous: the backends agree to ~1e-15, this only catches real divergence

# Named ladders. Each entry is (sampled_radius_cm, grid_size_um).
LADDERS = {
    # Spans the crossover on a single A100: small grids the CPU wins, large ones
    # the GPU wins by a growing margin. All at 10 um.
    "crossover": [
        (0.014, 10.0),
        (0.05, 10.0),
        (0.08, 10.0),
        (0.12, 10.0),
        (0.18, 10.0),
    ],
    # Adds the full electrode at 10 um and 5 um -- the memory-bound headline.
    "full": [
        (0.05, 10.0),
        (0.12, 10.0),
        (0.18, 10.0),
        (0.265, 10.0),
        (0.265, 5.0),
    ],
}


def build_config(radius_cm: float, grid_um: float, dose_rate_water_Gy_s: float) -> SimulationConfig:
    """Markus 2 mm config at an arbitrary column radius and voxel size.

    Same beam/chamber/gas as `run_markus_2mm.build_config`, but with the voxel
    size and radius free and the voxel/RAM guards relaxed so a 5 um full
    electrode (which the default 1e8-voxel cap would reject) can be built. GPU
    memory is checked separately, on the device, by the CUDA backend itself.
    """
    return SimulationConfig(
        **BEAM_KWARGS,
        **CHAMBER_PHYSICS_KWARGS,
        dose_rate_Gy_s=dose_rate_water_Gy_s * WATER_TO_AIR,
        grid_size_um=grid_um,
        sampled_radius_cm=radius_cm,
        buffer_radius=3,
        no_z_electrode=5,
        max_voxels=2e9,
        memory_budget_fraction=0.95,
    )


def run_one(cfg: SimulationConfig, backend: str, threads: int) -> dict:
    """Run one config on one backend, timed, returning wall time and k_s."""
    if backend == "gpu":
        from pulsed_ion_chamber.solver_cuda import run_simulation_cuda

        t0 = time.perf_counter()
        result = run_simulation_cuda(cfg, progress=False)
        wall = time.perf_counter() - t0
    elif backend == "cpu":
        from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel

        t0 = time.perf_counter()
        result = run_simulation_numba_parallel(cfg, progress=False, num_threads=threads)
        wall = time.perf_counter() - t0
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return {"wall_s": wall, "ks": float(result.ks)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ladder", choices=sorted(LADDERS), default=None, help="a named size ladder")
    parser.add_argument(
        "--sizes",
        default=None,
        help="comma-separated sampled_radius_cm@grid_um entries (grid_um optional), e.g. 0.12@10,0.265@5",
    )
    parser.add_argument(
        "--backends",
        default="cpu,gpu",
        help="comma-separated: gpu, cpu (batched at --threads). Default both.",
    )
    parser.add_argument("--threads", type=int, default=32, help="CPU thread count for the batched backend")
    parser.add_argument("--dose-rate-water-Gy-s", type=float, default=10.0)
    parser.add_argument("--json", default=None, help="also write results as JSON")
    args = parser.parse_args()

    if args.sizes:
        sizes = []
        for entry in args.sizes.split(","):
            if "@" in entry:
                r, g = entry.split("@")
                sizes.append((float(r), float(g)))
            else:
                sizes.append((float(entry), 10.0))
    elif args.ladder:
        sizes = LADDERS[args.ladder]
    else:
        parser.error("give --ladder or --sizes")

    backends = args.backends.split(",")

    # Warm up every backend once, outside the timed region -- JIT compile for the
    # CPU kernels, kernel compile for the CUDA ones.
    if "cpu" in backends:
        from pulsed_ion_chamber.solver_numba_parallel import warmup_parallel

        warmup_parallel()
    if "gpu" in backends:
        from pulsed_ion_chamber.solver_cuda import warmup_cuda

        warmup_cuda()

    print(f"host={platform.node()}  threads(cpu)={args.threads}  dose={args.dose_rate_water_Gy_s} Gy/s(water)")
    header = f"{'grid':>16} {'voxels':>9} {'tracks/pulse':>13} {'steps':>6} {'GPU GiB':>8}"
    for b in backends:
        header += f" {b + ' wall_s':>13} {b + ' ks':>12}"
    if "cpu" in backends and "gpu" in backends:
        header += f" {'speedup':>8}"
    print(header)

    rows = []
    for radius_cm, grid_um in sizes:
        cfg = build_config(radius_cm, grid_um, args.dose_rate_water_Gy_s)
        dev_gib = 4 * cfg.no_xy * cfg.no_xy * cfg.no_z_with_buffer * 8 / 2**30
        row = {
            "sampled_radius_cm": radius_cm,
            "grid_size_um": grid_um,
            "no_xy": cfg.no_xy,
            "no_z_with_buffer": cfg.no_z_with_buffer,
            "voxels": cfg.no_xy**2 * cfg.no_z_with_buffer,
            "tracks_per_pulse": cfg.number_of_tracks_per_pulse,
            "total_time_steps": cfg.total_time_steps,
            "gpu_carrier_gib": dev_gib,
        }
        line = (
            f"{cfg.no_xy}x{cfg.no_xy}x{cfg.no_z_with_buffer:<4}".rjust(16)
            + f" {row['voxels'] / 1e6:>7.0f}M"
            + f" {cfg.number_of_tracks_per_pulse / 1e6:>11.1f}M"
            + f" {cfg.total_time_steps:>6}"
            + f" {dev_gib:>8.1f}"
        )
        for b in backends:
            res = run_one(cfg, b, args.threads)
            row[f"{b}_wall_s"] = res["wall_s"]
            row[f"{b}_ks"] = res["ks"]
            line += f" {res['wall_s']:>13.2f} {res['ks']:>12.6f}"
        if "cpu" in backends and "gpu" in backends:
            speedup = row["cpu_wall_s"] / row["gpu_wall_s"]
            row["speedup"] = speedup
            dks = abs(row["gpu_ks"] - row["cpu_ks"]) / row["cpu_ks"]
            line += f" {speedup:>7.2f}x" + ("  KS DIVERGENCE!" if dks > KS_RTOL else "")
        print(line)
        rows.append(row)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "host": platform.node(),
                    "threads_cpu": args.threads,
                    "dose_rate_water_Gy_s": args.dose_rate_water_Gy_s,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "rows": rows,
                },
                handle,
                indent=2,
            )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
