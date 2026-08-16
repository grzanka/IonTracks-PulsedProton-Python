#!/usr/bin/env python3
"""Grace Hopper (GH200) benchmark: how fine a grid this code can actually run.

`profiling/bench_gpu.py` answers "when does the GPU beat the CPU". On a GH200
that question is settled -- it does, by more than an A100 did -- and the
interesting one becomes **how far the resolution can be pushed**, because the
superchip offers three places to put the carrier arrays instead of one:

* **device** -- 96 GiB of HBM3 at ~4 TB/s. The fast case, and the ceiling.
* **managed** -- `cudaMallocManaged`. Pages migrate between HBM and host
  LPDDR5X on demand, so the grid may exceed HBM.
* **host** -- `cudaHostAlloc`. The arrays sit in Grace's LPDDR5X for the whole
  run and the GPU reads them across NVLink-C2C (~450 GB/s each way,
  cache-coherent). No migration machinery, so no page-fault storms; the cost
  is flat instead of cliff-shaped.

The physics is the published AIC-144 Markus 2 mm case, imported from the
example so it cannot drift, and every mode is checked to produce the same
collection efficiency before its time is reported.

Four ladders, each a separate command:

    python profiling/bench_gh200.py --ladder resolution --max-steps 200

    python profiling/bench_gh200.py --ladder memory --max-steps 200

    python profiling/bench_gh200.py --ladder oversubscribe --max-steps 100

    python profiling/bench_gh200.py --ladder blocks --max-steps 200

`--max-steps` is what makes the fine grids measurable: a 1 um column needs
~26,000 time steps, hours of wall clock, and nothing about the per-step cost
needs all of them. A fixed step count is also reproducible across machines
where a wall-clock cut is not. Give `--max-steps 0` to run each case to
completion and get a real `k_s`.

Two derived numbers per row:

* **Gvox/s** -- interior voxels swept per second, the resolution-independent
  measure of how fast the machine is going.
* **GB/s** -- effective bandwidth, counting the sweep's compulsory traffic only
  (2 carrier reads + 2 writes = 32 B per voxel). Neighbour loads are assumed
  to hit cache; they mostly do. Compare against ~4000 for HBM3 and ~450 for
  C2C to see which memory a row is actually running out of.
"""

import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "ifj_aic144"))
from run_markus_2mm import BEAM_KWARGS, CHAMBER_PHYSICS_KWARGS, WATER_TO_AIR  # noqa: E402

from pulsed_ion_chamber.config import SimulationConfig  # noqa: E402
from pulsed_ion_chamber.resources import format_bytes  # noqa: E402

# Bytes of carrier array per interior voxel that the sweep must move: two
# species read, two written, float64.
BYTES_PER_VOXEL_SWEPT = 32

# Ladders. Each case is (sampled_radius_cm, grid_size_um, memory, advise).
# radius 0.05 cm is the crossover column of docs/GPU.md sec. 4 -- small enough
# that 1 um voxels fit in HBM, large enough to fill the GPU.
LADDERS = {
    # Refine the voxel size at fixed radius, 10 um -> 1 um. The last row is
    # 2.0 G voxels and 60 GiB, which no other machine in this repo's docs can
    # hold in fast memory at all.
    "resolution": [
        (0.05, 10.0, "device", "device"),
        (0.05, 5.0, "device", "device"),
        (0.05, 2.0, "device", "device"),
        (0.05, 1.0, "device", "device"),
    ],
    # The same grid through all three allocators, so the cost of leaving HBM is
    # measured rather than assumed. Sized to fit in HBM, which is the only way
    # to compare them on equal terms.
    "memory": [
        (0.05, 2.0, "device", "device"),
        (0.05, 2.0, "managed", "device"),
        (0.05, 2.0, "managed", "host"),
        (0.05, 2.0, "managed", "none"),
        (0.05, 2.0, "host", "none"),
    ],
    # "oversubscribe" is built at run time by _oversubscribe_cases(), because
    # the radius that overflows HBM depends on how much host memory the job was
    # given -- see that function.
}

OVERSUBSCRIBE_GRID_UM = 1.0


def _oversubscribe_cases(grid_um: float):
    """Cases whose carrier arrays do NOT fit in HBM, sized to this job.

    A fixed radius cannot work here: the grid has to be larger than the GPU's
    96 GiB and still within the *job's* host headroom, and on Helios that
    headroom is whatever `--mem` asked for -- 12 GiB in a default 8-core
    allocation, 478 GiB with the whole node. So the radius is solved for
    instead: aim 30 % past HBM, back off to 85 % of the combined ceiling if
    that would not fit, and say so plainly if nothing overflows.
    """
    import cupy as cp

    from pulsed_ion_chamber.resources import available_memory_bytes

    free_device, _ = cp.cuda.runtime.memGetInfo()
    host_free = available_memory_bytes() or 0
    ceiling = 0.85 * (0.9 * free_device + 0.8 * host_free)
    target = min(1.3 * free_device, ceiling)
    if target <= 0.9 * free_device:
        print(
            f"note: this job's host headroom ({format_bytes(host_free)}) cannot hold a grid past "
            f"the GPU's {format_bytes(free_device)}. Ask sbatch for more memory "
            "(submit_gh200.sh takes a whole node for exactly this reason)."
        )
        return []

    # Solve target = 4 * 8 * no_xy^2 * nzb for the radius, with a trial config
    # supplying nzb (which depends only on the voxel size, not the radius).
    probe = build_config(0.05, grid_um, 10.0)
    no_xy = (target / (32.0 * probe.no_z_with_buffer)) ** 0.5
    radius_cm = round((no_xy - 2 * probe.buffer_radius) / 2.0 * grid_um * 1e-4, 5)
    return [
        (radius_cm, grid_um, "device", "device"),
        (radius_cm, grid_um, "managed", "device"),
        (radius_cm, grid_um, "managed", "host"),
        (radius_cm, grid_um, "host", "none"),
    ]


BLOCK_SIZES = (128, 256, 512, 1024)
BLOCKS_CASE = (0.05, 2.0)


def build_config(radius_cm: float, grid_um: float, dose_rate_water_Gy_s: float) -> SimulationConfig:
    """Markus 2 mm config at an arbitrary column radius and voxel size.

    Same beam/chamber/gas as `run_markus_2mm.build_config`, with both host-side
    guards lifted: `max_voxels` because a 1 um column is 2 G voxels, and the
    RAM budget because these arrays never touch host RAM under `memory="device"`
    and the CUDA backend does its own device-side check either way. On a Helios
    GH200 the job's host memory (`--mem`) is routinely a tenth of the GPU's, so
    the host guard would otherwise reject every run this file exists to make.
    """
    return SimulationConfig(
        **BEAM_KWARGS,
        **CHAMBER_PHYSICS_KWARGS,
        dose_rate_Gy_s=dose_rate_water_Gy_s * WATER_TO_AIR,
        grid_size_um=grid_um,
        sampled_radius_cm=radius_cm,
        buffer_radius=3,
        no_z_electrode=5,
        max_voxels=1e11,
        memory_budget_fraction=None,
    )


def run_case(cfg: SimulationConfig, memory: str, advise: str, max_steps, dry_run: bool) -> dict:
    """One timed run. Returns wall time, per-step cost and the efficiency
    reached, or the exception's message if the grid did not fit."""
    from pulsed_ion_chamber.solver_cuda import run_simulation_cuda

    if dry_run:
        return {"skipped": "dry run"}
    try:
        t0 = time.perf_counter()
        result = run_simulation_cuda(
            cfg,
            progress=False,
            max_steps=max_steps,
            memory=memory,
            advise=advise,
            return_fields=False,
        )
        wall = time.perf_counter() - t0
    except MemoryError as exc:
        return {"error": f"MemoryError: {str(exc).split('.')[0]}"}
    except Exception as exc:
        # A block size the sweep cannot be launched at (register pressure) or a
        # driver refusal is a result of this benchmark, not a crash of it: the
        # remaining rows still have to run.
        return {"error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"}

    steps = getattr(result, "steps_completed", None) or cfg.total_time_steps
    # f_t at the last completed step: the physics check that works for a
    # truncated run, where k_s (defined after full clearance) is NaN.
    f_end = float(result.f_t[steps - 1])
    interior = (cfg.no_xy - 2) ** 2 * (cfg.no_z_with_buffer - 2)
    per_step = wall / steps
    return {
        "wall_s": wall,
        "steps": steps,
        "ms_per_step": per_step * 1e3,
        "gvox_per_s": interior / per_step / 1e9,
        "gb_per_s": interior * BYTES_PER_VOXEL_SWEPT / per_step / 1e9,
        "f_end": f_end,
        "ks": float(result.ks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ladder", choices=sorted(LADDERS) + ["blocks", "oversubscribe"], default=None
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help="comma-separated sampled_radius_cm@grid_um entries instead of a ladder, e.g. 0.265@2",
    )
    parser.add_argument("--memory", default="auto", help='allocator for --sizes runs: auto, device, managed, host')
    parser.add_argument("--advise", default="device", help='managed placement policy: device, host, none')
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="stop each run after this many steps (0 = run to completion and report a real k_s)",
    )
    parser.add_argument("--dose-rate-water-Gy-s", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="print the grid sizes without running anything")
    parser.add_argument("--json", default=None, help="also write results as JSON")
    args = parser.parse_args()
    max_steps = args.max_steps or None

    from pulsed_ion_chamber import solver_cuda

    if not args.dry_run:
        solver_cuda.warmup_cuda()

    if args.sizes:
        cases = []
        for entry in args.sizes.split(","):
            radius, _, grid_um = entry.partition("@")
            cases.append((float(radius), float(grid_um or 10.0), args.memory, args.advise, None))
        label = args.sizes
    elif args.ladder == "blocks":
        radius, grid_um = BLOCKS_CASE
        cases = [(radius, grid_um, "device", "device", tpb) for tpb in BLOCK_SIZES]
        label = "blocks"
    elif args.ladder == "oversubscribe":
        label = "oversubscribe"
        cases = [(r, g, m, a, None) for (r, g, m, a) in _oversubscribe_cases(OVERSUBSCRIBE_GRID_UM)]
    else:
        label = args.ladder or "resolution"
        cases = [(r, g, m, a, None) for (r, g, m, a) in LADDERS[label]]

    print(f"host={platform.node()}  ladder={label}  max_steps={max_steps or 'all'}")
    print(
        f"{'grid':>20} {'Mvoxel':>8} {'arrays':>10} {'memory':>8} {'advise':>7} {'tpb':>5}"
        f" {'steps':>6} {'ms/step':>9} {'Gvox/s':>8} {'GB/s':>7} {'f_end':>10} {'k_s':>10}"
    )

    rows = []
    for radius_cm, grid_um, memory, advise, tpb in cases:
        launch_error = None
        if tpb is not None:
            solver_cuda.set_threads_per_block(tpb)
            try:
                solver_cuda.warmup_cuda()
            except Exception as exc:
                # 1024 threads/block does not launch: the sweep's registers plus
                # 3 x 1024 x 8 B of shared arrays exceed what an SM can give one
                # block. That is the measurement for that row.
                launch_error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"
        cfg = build_config(radius_cm, grid_um, args.dose_rate_water_Gy_s)
        voxels = cfg.no_xy**2 * cfg.no_z_with_buffer
        arrays = 4 * voxels * 8
        row = {
            "sampled_radius_cm": radius_cm,
            "grid_size_um": grid_um,
            "memory": memory,
            "advise": advise,
            "threads_per_block": tpb or solver_cuda.THREADS_PER_BLOCK,
            "no_xy": cfg.no_xy,
            "no_z_with_buffer": cfg.no_z_with_buffer,
            "voxels": voxels,
            "carrier_bytes": arrays,
            "total_time_steps": cfg.total_time_steps,
        }
        line = (
            f"{cfg.no_xy}x{cfg.no_xy}x{cfg.no_z_with_buffer}".rjust(20)
            + f" {voxels / 1e6:>8.0f}"
            + f" {format_bytes(arrays):>10}"
            + f" {memory:>8} {advise:>7} {row['threads_per_block']:>5}"
        )
        if launch_error:
            row["error"] = launch_error
        else:
            row.update(run_case(cfg, memory, advise, max_steps, args.dry_run))
        if "error" in row:
            print(line + f"  -> {row['error']}")
        elif "skipped" in row:
            print(line + f"  -> {row['skipped']} ({cfg.total_time_steps} steps for a full run)")
        else:
            print(
                line + f" {row['steps']:>6} {row['ms_per_step']:>9.2f}"
                f" {row['gvox_per_s']:>8.2f} {row['gb_per_s']:>7.0f} {row['f_end']:>10.6f}"
                # k_s is defined after full clearance, so a truncated row has none.
                + (f" {row['ks']:>10.6f}" if row["ks"] == row["ks"] else f" {'--':>10}")
            )
        rows.append(row)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "host": platform.node(),
                    "ladder": label,
                    "max_steps": max_steps,
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
