#!/usr/bin/env python3
"""90 MeV/u iron in air: one ion on the axis, Cucinotta RDD, 2 mm gap at 200 V.

A cross-check against an IonTracks-FEniCSx run on an unstructured mesh
(R = 120 um cylinder, 5 um on-axis spacing). See examples/fe90_air/README.md
for the physics, the sizing study and what the numbers mean.

With one track there is no general recombination -- the result is pure
initial/columnar recombination, and the only thing that sets it is how densely
the RDD's core is resolved. That is why the interesting mode is the ladder
rather than any single run.

Run:  python examples/fe90_air/run_fe90.py [--h-um H] [--radius-um R]
            [--threads N] [--ladder] [--dry-run] [--json FILE]

`--ladder` runs the h = 10, 5, 2.5 um rungs at R = 120 um and prints k_s
against spacing; `--ladder --fine` adds the 1.25 um rung, which is 2.4 GiB and
takes about 17 minutes on two cores.

Two k_s values are always reported and they answer different questions:

  k_s (in-domain)  loss as a fraction of the charge *inside the grid*. This is
                   what a solver reports if it simply divides its own recombined
                   charge by its own injected charge, and it is what the FEniCS
                   run reports too.
  k_s (chamber)    loss as a fraction of the charge the ion actually *created*,
                   which includes the delta-ray halo out to ~10 cm that no
                   affordable grid contains. See rdd.chamber_ks and README sec 4.2.

The second is the physically meaningful one. The gap between them is not small:
a 280 um column holds only 72 % of the track's energy.
"""

import argparse
import json
import time
from pathlib import Path

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import (
    ION_DIFFUSION_NEGATIVE_CM2_S,
    ION_DIFFUSION_POSITIVE_CM2_S,
    ION_MOBILITY_NEGATIVE_CM2_VS,
    ION_MOBILITY_POSITIVE_CM2_VS,
)
from pulsed_ion_chamber.rdd import chamber_ks
from pulsed_ion_chamber.resources import format_bytes, memory_report
from pulsed_ion_chamber.solver_numba import run_simulation_numba
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel

RDD_CSV = Path(__file__).parent / "data" / "rdd_cucinotta_fe90_air.csv"

# Kanai (1998) per-species transport. Resolved separately rather than averaged
# because the collection tail is set by the slower positive ion and dt by the
# faster, more diffusive negative one -- see docs/PHYSICS.md sec. 8.
TWO_SPECIES = dict(
    mu_positive_cm2_Vs=ION_MOBILITY_POSITIVE_CM2_VS,
    mu_negative_cm2_Vs=ION_MOBILITY_NEGATIVE_CM2_VS,
    D_positive_cm2_s=ION_DIFFUSION_POSITIVE_CM2_S,
    D_negative_cm2_s=ION_DIFFUSION_NEGATIVE_CM2_S,
)

# R = 120 um matches the FEniCS cylinder. The ladder holds it fixed and varies
# only the spacing, so every rung contains the same share of the track and the
# only thing moving is how well the core is resolved.
LADDER_UM = (10.0, 5.0, 2.5)
FINE_RUNG_UM = 1.25


def build_config(h_um: float, radius_um: float, n_tracks: int = 1) -> SimulationConfig:
    """One ion on the axis, deposited from the tabulated RDD.

    `lateral_boundary="absorbing"` with `scoring_region="full_grid"` is the
    right pair for a single track: the gas outside the column is empty, so
    nothing diffuses back in (which is what "reflecting" would assert), and the
    books have to close over the whole array rather than over a disc the RDD
    spills out of anyway.
    """
    # Enough lateral margin for ~20 um of buffer at any spacing: the diffusion
    # length over the 147 us collection is ~36 um, so the scored disc needs to
    # sit some way inside the wall.
    buffer_radius = max(2, int(round(20.0 / h_um)))
    return SimulationConfig(
        E_MeV_u=90.0,
        particle="iron",
        voltage_V=200.0,
        electrode_gap_cm=0.2,
        # One ion, injected in a single step; the run is then all collection.
        pulse_duration_s=1e-7,
        repetition_rate_hz=50.0,
        dose_rate_Gy_s=1e-12,
        n_pulses=1,
        n_tracks=n_tracks,
        track_placement="axis",
        rdd_csv=str(RDD_CSV),
        grid_size_um=h_um,
        sampled_radius_cm=radius_um * 1e-4,
        buffer_radius=buffer_radius,
        no_z_electrode=3,
        lateral_boundary="absorbing",
        scoring_region="full_grid",
        max_voxels=1e10,
        seed=1,
        **TWO_SPECIES,
    )


def run_one(config: SimulationConfig, threads: int, progress: bool = False) -> dict:
    started = time.perf_counter()
    if threads > 1:
        result = run_simulation_numba_parallel(config, progress=progress, num_threads=threads)
    else:
        result = run_simulation_numba(config, progress=progress)
    elapsed = time.perf_counter() - started

    ks_chamber = chamber_ks(result.ks, config.in_domain_let_fraction)
    return {
        "h_um": config.grid_size_um,
        "grid": f"{config.no_xy}x{config.no_xy}x{config.no_z_with_buffer}",
        "width_um": config.no_xy * config.grid_size_um,
        "peak_density_cm3": float(config.track_stencil.density_cm3.max()),
        "in_domain_fraction": config.in_domain_let_fraction,
        "dt_ns": config.dt * 1e9,
        "steps": config.total_time_steps,
        "ks_in_domain": result.ks,
        "ks_chamber": ks_chamber,
        "loss_percent": 100.0 * (1.0 - 1.0 / ks_chamber),
        "memory": format_bytes(config.estimated_memory_bytes),
        "wall_s": elapsed,
    }


def print_row_header() -> None:
    print(
        f"{'h[um]':>6} {'grid':>16} {'n0[cm^-3]':>11} {'in-dom':>7} "
        f"{'dt[ns]':>7} {'steps':>6} {'ks(domain)':>11} {'ks(chamber)':>12} "
        f"{'loss':>7} {'RAM':>10} {'wall':>9}"
    )


def print_row(row: dict) -> None:
    print(
        f"{row['h_um']:>6g} {row['grid']:>16} {row['peak_density_cm3']:>11.3e} "
        f"{100 * row['in_domain_fraction']:>6.2f}% {row['dt_ns']:>7.1f} {row['steps']:>6} "
        f"{row['ks_in_domain']:>11.6f} {row['ks_chamber']:>12.6f} "
        f"{row['loss_percent']:>6.2f}% {row['memory']:>10} {row['wall_s']:>8.1f}s"
    )


def main(args) -> list:
    spacings = list(LADDER_UM) if args.ladder else [args.h_um]
    if args.ladder and args.fine:
        spacings.append(FINE_RUNG_UM)

    if args.dry_run:
        for h_um in spacings:
            config = build_config(h_um, args.radius_um)
            print(f"--- h = {h_um} um, R = {args.radius_um} um ---")
            print(config.summary())
            print(memory_report(config.estimated_memory_bytes, config.memory_budget_fraction))
            print()
        return []

    rows = []
    print_row_header()
    for h_um in spacings:
        config = build_config(h_um, args.radius_um)
        rows.append(run_one(config, args.threads, progress=args.progress))
        print_row(rows[-1])

    if len(rows) > 1:
        print()
        print(
            "k_s is expected to keep rising as h falls: the RDD core is denser than "
            "any affordable voxel resolves, and recombination goes as <n^2>. "
            "It stops rising once h is below the ~1 um diffusion length reached in "
            "the first 0.1 us -- see examples/fe90_air/README.md sec. 4.3."
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h-um", type=float, default=5.0, help="voxel size, all three axes (default 5)")
    parser.add_argument(
        "--radius-um", type=float, default=120.0, help="scored disc radius (default 120, the FEniCS mesh)"
    )
    parser.add_argument("--threads", type=int, default=1, help="1 uses the serial backend, >1 the batched one")
    parser.add_argument("--ladder", action="store_true", help="run the h = 10, 5, 2.5 um resolution ladder")
    parser.add_argument("--fine", action="store_true", help="add the 1.25 um rung (2.4 GiB, ~17 min on 2 cores)")
    parser.add_argument("--dry-run", action="store_true", help="print sizing and exit without allocating")
    parser.add_argument("--progress", action="store_true", help="print per-step progress")
    parser.add_argument("--json", default=None, help="also write the rows as JSON")
    parsed = parser.parse_args()

    results = main(parsed)
    if parsed.json and results:
        Path(parsed.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {parsed.json}")
