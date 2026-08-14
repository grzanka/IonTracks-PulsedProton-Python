#!/usr/bin/env python3
"""IFJ PAN AIC-144: PTW Markus 23343 (2 mm gap), macropulse, 10 Gy/s.

Two Kanai carrier species resolved separately, dry air at 20 degC, a
reflecting (zero-flux) chamber wall, and a collection tail sized on the
slowest carrier.

See examples/ifj_aic144/README.md for the scenario table and results,
docs/PHYSICS.md for what every assumption means and why, and
docs/PERFORMANCE.md for timings and scaling.

Run:  python examples/ifj_aic144/run_markus_2mm.py [tier] [--threads N] [--json FILE]

The default is one thread and the unbatched backend, which is what the tier
table below was measured with. `--threads N` switches to the batched backend
(`solver_numba_parallel`) and is how the `full_electrode` tier becomes
affordable on a cluster -- roughly 10x on ~96 cores. There it needs its own
srun step, and the thread count that pays is not the largest one available:
see docs/HELIOS.md.
"""

import argparse
import json
import time

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import (
    AIR_DENSITY_20C_KG_M3,
    ION_DIFFUSION_NEGATIVE_CM2_S,
    ION_DIFFUSION_POSITIVE_CM2_S,
    ION_MOBILITY_NEGATIVE_CM2_VS,
    ION_MOBILITY_POSITIVE_CM2_VS,
)
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
    # 10 Gy/s to water x 0.891 (water->air, calibrated for this beam) gives
    # 0.1782 Gy to air per pulse. This is the *nominal* dose: tracks fill the
    # scored column uniformly. The archive instead counted tracks over the full
    # 0.12 mm column while confining them to 0.7 of it, i.e. 2.04x this areal
    # density -- set chamber_fill_fraction=0.7 to reproduce that (see README).
    dose_rate_Gy_s=8.91,
    seed=20260527,  # archive seed (different RNG, so not track-for-track equal)
)

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
    "full_electrode": (0.265, 3),  # 536^2 x 210, 24 630 400 tracks, 12.8 min, k_s = 1.1111 (0.1% low)
}
DEFAULT_TIER = "archive"

# Finite-column edge deficit: a track near the rim of the sampled disc has
# neighbours on one side only, so the local density -- and hence alpha*n+*n- --
# is lower there. The shortfall is a perimeter/area effect and so falls as 1/r.
# Fitted over 80 um <= r <= 2650 um to within 3e-4 in k_s; see
# docs/PHYSICS.md section 14.
EDGE_DEFICIT_UM = 1.512
INFINITE_COLUMN_KS = 1.1119


def build_config(tier: str = DEFAULT_TIER) -> SimulationConfig:
    """SimulationConfig for one grid tier of the Markus 2 mm macropulse case."""
    if tier not in GRID_TIERS:
        raise ValueError(f"Unknown tier {tier!r}; expected one of {sorted(GRID_TIERS)}.")
    sampled_radius_cm, buffer_radius = GRID_TIERS[tier]
    return SimulationConfig(
        **BEAM_KWARGS,
        **CHAMBER_PHYSICS_KWARGS,
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
) -> None:
    config = build_config(tier)
    # "auto": one thread keeps the unbatched backend, so a plain run reproduces
    # the tier table; more than one needs the batched backend, the only one
    # where a thread count means anything.
    #
    # The override matters for one real case: a single-core *baseline* for a
    # large tier. `full_electrode` on the unbatched backend would deposit
    # m * w^2 * no_z per step and take hours, so the 680 s reference figure is
    # the batched backend at `--threads 1`, not the unbatched one. Comparing
    # thread counts means holding the backend fixed.
    batched = threads != 1 if backend == "auto" else backend == "batched"
    print(f"=== IFJ AIC-144, Markus 2 mm, macropulse, 10 Gy/s -- '{tier}' grid ===")
    print(config.summary())
    backend = "solver_numba_parallel" if batched else "solver_numba"
    print(f"Backend               : {backend}, {threads} thread(s)")
    print()

    if batched:
        warmup_parallel()
        t0 = time.perf_counter()
        result = run_simulation_numba_parallel(config, progress=True, num_threads=threads)
    else:
        warmup()  # one-off JIT compilation, excluded from the timing below
        t0 = time.perf_counter()
        result = run_simulation_numba(config, progress=True)
    elapsed_s = time.perf_counter() - t0

    radius_um = config.sampled_radius_cm * 1e4
    corrected = result.ks + EDGE_DEFICIT_UM / radius_um
    print(f"\nWall time ({backend}, {threads} thread(s)): {elapsed_s:.1f} s")
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
                    "threads": threads,
                    "backend": backend,
                    "wall_s": elapsed_s,
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
        "--backend",
        default="auto",
        choices=("auto", "serial", "batched"),
        help="auto (default) picks by thread count; force 'batched' for a single-core baseline",
    )
    args = parser.parse_args()
    main(args.tier, args.threads, args.json, args.backend)
