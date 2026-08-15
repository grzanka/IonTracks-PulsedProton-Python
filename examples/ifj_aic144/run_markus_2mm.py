#!/usr/bin/env python3
"""IFJ PAN AIC-144: PTW Markus 23343 (2 mm gap), macropulse, 10 Gy/s by default.

Two Kanai carrier species resolved separately, dry air at 20 degC, a
reflecting (zero-flux) chamber wall, and a collection tail sized on the
slowest carrier.

See examples/ifj_aic144/README.md for the scenario table and results,
docs/PHYSICS.md for what every assumption means and why, and
docs/PERFORMANCE.md for timings and scaling.

Run:  python examples/ifj_aic144/run_markus_2mm.py [tier] [--threads N]
            [--backend auto|serial|batched] [--dose-rate-water-Gy-s R] [--json FILE]
            [--sampled-radius-mm R] [--dry-run] [--estimate-runtime-seconds N]

The default is one thread and the unbatched backend, which is what the tier
table below was measured with. `--threads N` switches to the batched backend
(`solver_numba_parallel`) and is how the `full_electrode` tier becomes
affordable on a cluster -- roughly 10x on ~96 cores. There it needs its own
srun step, and the thread count that pays is not the largest one available:
see docs/HELIOS.md.

`--dry-run` builds the config and prints its memory sizing (estimated peak
allocation vs. what this machine actually has free), then exits without
allocating the grid or running anything -- a way to catch "this grid is too
big for this machine" before committing to it. See docs/BENCHMARKS-LAPTOP.md
sec. 3 for how the estimate has tracked measured peak RSS on the
full_electrode tier.

`--estimate-runtime-seconds N` runs the *real* backend on the *real* grid --
real allocation, real JIT warmup, same thread count -- for N seconds
(default 5), then extrapolates and exits without doing the full run. Unlike a
runtime estimate built from isolated single-track/single-step samples, this
one exercises the exact code path (including the batched deposition
`--threads > 1` selects) a real run of this config would use, so the number
is a genuine estimate rather than a proxy for one. See docs/PERFORMANCE.md
sec. 7 and `pulsed_ion_chamber.benchmark.estimate_full_runtime_empirical`.
"""

import argparse
import json
import os
import platform
import resource
import time

from pulsed_ion_chamber.benchmark import estimate_full_runtime_empirical
from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import (
    AIR_DENSITY_20C_KG_M3,
    ION_DIFFUSION_NEGATIVE_CM2_S,
    ION_DIFFUSION_POSITIVE_CM2_S,
    ION_MOBILITY_NEGATIVE_CM2_VS,
    ION_MOBILITY_POSITIVE_CM2_VS,
)
from pulsed_ion_chamber.resources import format_bytes, memory_report
from pulsed_ion_chamber.solver_numba import run_simulation_numba, warmup
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel, warmup_parallel

# --- beam and chamber, straight from the archived source_config.yaml ---------
BEAM_KWARGS = dict(
    E_MeV_u=56.2,  # 60 MeV nominal, degraded to 56.2 at the measurement plane
    voltage_V=300.0,
    electrode_gap_cm=0.2,  # PTW Markus 23343 -> E = 1500 V/cm = 150 kV/m
    pulse_duration_s=540e-6,  # macropulse
    repetition_rate_hz=50.0,  # every 20 ms -> duty cycle 1/37
    n_pulses=1,
    rf_frequency_hz=26.26e6,  # diagnostic only; see config.py on RF averaging
    seed=20260527,  # archive seed (different RNG, so not track-for-track equal)
)

# Water-to-air conversion for this beam quality, calibrated for the campaign.
# The scenario is quoted as a dose rate *to water*; the solver wants air.
WATER_TO_AIR = 0.891
# 10 Gy/s to water -> 8.91 Gy/s to air -> 0.1782 Gy to air per pulse. This is
# the *nominal* dose: tracks fill the scored column uniformly. The archive
# instead counted tracks over the full 0.12 mm column while confining them to
# 0.7 of it, i.e. 2.04x this areal density -- set chamber_fill_fraction=0.7 to
# reproduce that (see README).
DEFAULT_DOSE_RATE_WATER_GY_S = 10.0

# --- gas, carriers and boundary treatment (docs/PHYSICS.md sections 8, 10, 12) ---
CHAMBER_PHYSICS_KWARGS = dict(
    # Two Kanai species instead of one averaged pair: dt is set by the negative
    # ion (faster and more diffusive), the collection tail by the positive.
    # Costs ~35% more time steps; changes k_s by -0.4%.
    mu_positive_cm2_Vs=ION_MOBILITY_POSITIVE_CM2_VS,
    mu_negative_cm2_Vs=ION_MOBILITY_NEGATIVE_CM2_VS,
    D_positive_cm2_s=ION_DIFFUSION_POSITIVE_CM2_S,
    D_negative_cm2_s=ION_DIFFUSION_NEGATIVE_CM2_S,
    W_eV=33.0,  # proton value used for this campaign (library default 34.2)
    air_density_kg_m3=AIR_DENSITY_20C_KG_M3,  # dry air at 20 degC
    # Zero-flux wall: the right model for a column sampled from the interior of
    # a large, uniformly irradiated chamber. Converges at buffer_radius=3 where
    # an absorbing wall needs 4-6.
    lateral_boundary="reflecting",
    # separation_time_steps is the half-gap transit of the slowest carrier, so
    # 2.6 is one full transit plus a 30% margin.
    n_clearance_separation_times=2.6,
    # 10 sigma discards exp(-50) = 1.9e-22 of each track -- below float64
    # epsilon, so bit-identical to depositing over the whole grid.
    track_cutoff_sigmas=10.0,
)

# --- grid tiers, all at 10 um voxels ----------------------------------------
# Wall times measured single-threaded on one development machine (solver_numba,
# except full_electrode which used the batched backend); k_s from those runs.
#
# EVERY tier is biased low by the finite-column edge deficit, which falls only
# as 1/radius: k_s(r) = 1.1119 - 1.512 um / r. Correct for it rather than
# picking a bigger radius -- see docs/PHYSICS.md section 14.
GRID_TIERS = {
    # name:            (sampled_radius_cm, buffer_radius)
    "dev": (0.003, 3),  #  12^2 x 210,      3 157 tracks,    0.2 s, k_s = 1.0580 (4.9% low)
    "archive": (0.008, 3),  #  22^2 x 210,     22 447 tracks,    1.8 s, k_s = 1.0929 (1.7% low)
    "standard": (0.014, 3),  #  34^2 x 210,     68 744 tracks,    8.8 s, k_s = 1.1011 (1.0% low)
    "wide": (0.018, 3),  #  42^2 x 210,    113 638 tracks,   14.5 s, k_s = 1.1035 (0.8% low)
    "full_electrode": (0.265, 3),  # 536^2 x 210, 24 630 400 tracks, 562 s, k_s = 1.1111 (0.1% low)
}
DEFAULT_TIER = "archive"

# Finite-column edge deficit: a track near the rim of the sampled disc has
# neighbours on one side only, so the local density -- and hence alpha*n+*n- --
# is lower there. The shortfall is a perimeter/area effect and so falls as 1/r.
# Fitted over 80 um <= r <= 2650 um to within 3e-4 in k_s; see
# docs/PHYSICS.md section 14.
EDGE_DEFICIT_UM = 1.512
INFINITE_COLUMN_KS = 1.1119


def build_config(
    tier: str = DEFAULT_TIER,
    dose_rate_water_Gy_s: float = DEFAULT_DOSE_RATE_WATER_GY_S,
    sampled_radius_cm: float | None = None,
) -> SimulationConfig:
    """SimulationConfig for one grid tier of the Markus 2 mm macropulse case.

    ``dose_rate_water_Gy_s`` is the time-averaged dose rate *to water*, the way
    the scenario is quoted; it is converted to air here. Raising it is the one
    knob that changes the physics without changing the grid: track count scales
    with it linearly, recombination with roughly its square, and the PDE sweep
    not at all -- which makes it the cleanest way to move this problem from
    PDE-bound to deposition-bound. 50 Gy/s is the FLASH-adjacent case.

    ``sampled_radius_cm`` overrides the tier's column radius, in cm -- the
    convention every other length in this codebase uses (docs/PHYSICS.md,
    GRID_TIERS above). The CLI's ``--sampled-radius-mm`` is in mm instead (a
    Markus electrode is quoted in mm) and converts to this at the argument
    boundary in ``main()``, below.

    The tiers are a ladder of *scientific* interest, and the sizes that fall
    between them can matter for a different reason: what a benchmark needs is
    a grid on the correct side of the machine's last-level cache, and on a
    laptop every named tier below `full_electrode` fits in L3. Setting the
    radius directly is how `bench_laptop.sh` gets a DRAM-resident grid that
    still costs minutes.
    """
    if tier not in GRID_TIERS:
        raise ValueError(f"Unknown tier {tier!r}; expected one of {sorted(GRID_TIERS)}.")
    tier_radius_cm, buffer_radius = GRID_TIERS[tier]
    sampled_radius_cm = tier_radius_cm if sampled_radius_cm is None else sampled_radius_cm
    return SimulationConfig(
        **BEAM_KWARGS,
        **CHAMBER_PHYSICS_KWARGS,
        dose_rate_Gy_s=dose_rate_water_Gy_s * WATER_TO_AIR,
        grid_size_um=10.0,
        sampled_radius_cm=sampled_radius_cm,
        buffer_radius=buffer_radius,
        no_z_electrode=5,
    )


def main(
    tier: str = DEFAULT_TIER,
    threads: int = 1,
    json_path: str | None = None,
    backend: str = "auto",
    dose_rate_water_Gy_s: float = DEFAULT_DOSE_RATE_WATER_GY_S,
    sampled_radius_mm: float | None = None,
    dry_run: bool = False,
    estimate_runtime_seconds: float | None = None,
) -> None:
    # The CLI takes mm (a chamber's collecting electrode is quoted in mm --
    # this one is 5.3 mm across -- and "2.65" reads better than "0.265").
    # build_config()/SimulationConfig keep the whole codebase's cm convention
    # (docs/PHYSICS.md, GRID_TIERS above, every other length here), so the
    # conversion happens once, right at this boundary.
    sampled_radius_cm = None if sampled_radius_mm is None else sampled_radius_mm / 10.0
    config = build_config(tier, dose_rate_water_Gy_s, sampled_radius_cm)
    # "auto": one thread keeps the unbatched backend, so a plain run reproduces
    # the tier table; more than one needs the batched backend, the only one
    # where a thread count means anything.
    #
    # The override matters for one real case: a single-core *baseline* for a
    # large tier. `full_electrode` on the unbatched backend would deposit
    # m * w^2 * no_z per step and take hours, so the 572 s reference figure is
    # the batched backend at `--threads 1`, not the unbatched one. Comparing
    # thread counts means holding the backend fixed.
    batched = threads != 1 if backend == "auto" else backend == "batched"
    print(
        f"=== IFJ AIC-144, Markus 2 mm, macropulse, {dose_rate_water_Gy_s:g} Gy/s to water"
        f" -- '{tier}' grid ==="
    )
    print(config.summary())
    backend = "solver_numba_parallel" if batched else "solver_numba"
    print(f"Backend               : {backend}, {threads} thread(s)")
    print()

    if dry_run:
        # Sizing only: the memory guard in SimulationConfig.__post_init__ has
        # already run (it would have raised MemoryError above if this config
        # did not fit), so getting here at all means "yes, within budget" --
        # this just makes the margin visible instead of silent. Deliberately
        # no runtime estimate here: the only accurate one requires actually
        # allocating the grid and running it for a while (see
        # --estimate-runtime-seconds below), which isn't "dry" any more.
        print("--- dry run: memory ---")
        print(memory_report(config.estimated_memory_bytes, config.memory_budget_fraction))
        print("\nNo simulation was run (--dry-run).")
        return

    if estimate_runtime_seconds is not None:
        # Not dry: this allocates the real grid and runs the real backend for
        # real, just for a short, wall-clock-bounded sample instead of to
        # completion. See benchmark.estimate_full_runtime_empirical for why
        # that is worth the extra cost over --dry-run's instant estimate.
        print(
            f"--- empirical runtime estimate: real {backend} backend, {threads} thread(s), "
            f"~{estimate_runtime_seconds:g}s sample ---"
        )
        # Pass the already-resolved `batched` decision (from --backend/--threads
        # above, line 199) explicitly rather than letting the estimator
        # re-derive it from num_threads alone -- otherwise `--backend batched
        # --threads 1` or `--backend serial --threads 8` would sample a
        # different backend than the header just printed and a real run would
        # actually use.
        est = estimate_full_runtime_empirical(
            config, num_threads=threads, max_wall_s=estimate_runtime_seconds, batched=batched
        )
        print(f"steps measured            : {est['steps_measured']:,} / {est['total_time_steps']:,}")
        print(
            f"measured time for those   : {est['elapsed_measured_s']:.2f} s "
            f"({est['ms_per_step_measured']:.1f} ms/step)"
        )
        if est["exact"]:
            print(
                f"\nThe whole run finished inside the {estimate_runtime_seconds:g}s budget -- "
                "the figure below is the real wall time, not an extrapolation."
            )
        print(f"estimated total wall time : {est['estimated_seconds']:.0f} s ({est['estimated_hours']:.2g} h)")
        print("\nNo full run was performed (--estimate-runtime-seconds).")
        return

    if batched:
        warmup_parallel()
        t0 = time.perf_counter()
        result = run_simulation_numba_parallel(config, progress=True, num_threads=threads)
    else:
        warmup()  # one-off JIT compilation, excluded from the timing below
        t0 = time.perf_counter()
        result = run_simulation_numba(config, progress=True)
    elapsed_s = time.perf_counter() - t0
    # ru_maxrss is already the process's peak so far, not an incremental
    # figure -- see resource.getrusage(2) -- and by this point the carrier
    # arrays and (for the unbatched backend) the arrival-time draw have both
    # been through their peak, so this is the run's peak RSS.
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    radius_um = config.sampled_radius_cm * 1e4
    corrected = result.ks + EDGE_DEFICIT_UM / radius_um
    print(f"\nWall time ({backend}, {threads} thread(s)): {elapsed_s:.1f} s")
    # ru_maxrss is KiB on Linux, bytes on macOS -- this repo's benchmarking is
    # Linux-only (bench_laptop.sh, submit.sh both require it), so KiB is what
    # this number will be in practice; noted rather than special-cased.
    print(f"Peak RSS                                : {format_bytes(peak_rss_kb * 1024)}")
    print(f"  vs. estimated peak allocation         : {format_bytes(config.estimated_memory_bytes)}")
    print(f"Collection efficiency f = {result.f_t[-1]:.4f}")
    print(f"Recombination correction k_s = 1/f = {result.ks:.4f}")
    print(
        f"Corrected for the finite-column edge deficit (+{EDGE_DEFICIT_UM / radius_um:.4f}): "
        f"k_s -> {corrected:.4f}"
    )
    print(
        "\nPublished IonTracks v2 (FEniCSx) result for this case: k_s = 1.1629 (exact, 1/f)"
        "\n-- but at 2.04x this areal track density, so the two are not directly"
        "\ncomparable until the density convention is reconciled; see README."
    )

    if json_path:
        # Machine-readable, so a Slurm job array or a scaling sweep can collect
        # runs without re-parsing the text above.
        with open(json_path, "w") as handle:
            json.dump(
                {
                    "tier": tier,
                    "sampled_radius_cm": config.sampled_radius_cm,
                    "threads": threads,
                    "dose_rate_water_Gy_s": dose_rate_water_Gy_s,
                    "dose_rate_air_Gy_s": config.dose_rate_Gy_s,
                    # What the process was actually allowed to run on. Recorded
                    # because a thread count is meaningless without it -- under
                    # Slurm the two disagree more often than not (docs/HELIOS.md).
                    "affinity_cpus": len(os.sched_getaffinity(0)),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "hostname": platform.node(),
                    "backend": backend,
                    "wall_s": elapsed_s,
                    "peak_rss_kb": peak_rss_kb,
                    "estimated_memory_bytes": config.estimated_memory_bytes,
                    "f": float(result.f_t[-1]),
                    "ks": float(result.ks),
                    "ks_corrected": float(corrected),
                    "no_xy": config.no_xy,
                    "no_z_with_buffer": config.no_z_with_buffer,
                    "total_time_steps": config.total_time_steps,
                    "tracks_per_pulse": config.number_of_tracks_per_pulse,
                },
                handle,
                indent=2,
            )
        print(f"\nwrote {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", nargs="?", default=DEFAULT_TIER, choices=sorted(GRID_TIERS))
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="thread count for the batched backend; 1 (default) uses the unbatched one",
    )
    parser.add_argument("--json", default=None, help="also write the result as JSON")
    parser.add_argument(
        "--sampled-radius-mm",
        type=float,
        default=None,
        help="override the tier's column radius, in mm (for sizing a grid against a cache)",
    )
    parser.add_argument(
        "--dose-rate-water-Gy-s",
        type=float,
        default=DEFAULT_DOSE_RATE_WATER_GY_S,
        help=f"time-averaged dose rate to water (default {DEFAULT_DOSE_RATE_WATER_GY_S})",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "serial", "batched"),
        help="auto (default) picks by thread count; force 'batched' for a single-core baseline",
    )
    sizing = parser.add_mutually_exclusive_group()
    sizing.add_argument(
        "--dry-run",
        action="store_true",
        help="print the memory sizing, then exit without allocating the grid or running anything",
    )
    sizing.add_argument(
        "--estimate-runtime-seconds",
        type=float,
        default=None,
        metavar="N",
        help=(
            "run the real backend on the real grid for ~N seconds (allocating memory, "
            "warming up Numba) and extrapolate, then exit without doing the full run"
        ),
    )
    args = parser.parse_args()
    main(
        args.tier,
        args.threads,
        args.json,
        args.backend,
        args.dose_rate_water_Gy_s,
        args.sampled_radius_mm,
        args.dry_run,
        args.estimate_runtime_seconds,
    )
