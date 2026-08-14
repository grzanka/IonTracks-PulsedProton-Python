#!/usr/bin/env python3
"""Run one tier of the AIC-144 Markus scenario and write its full record.

Produces, in the output directory:

    collected_charge.csv          per time step: time, n_positive, n_negative,
                                  injected_positive, injected_negative, recombination
    injection_rate.png            beam arrival vs time
    carrier_evolution.png         carriers present vs time
    recombination_rate.png        pairs lost per time step
    track_density_cross_section.png   where the tracks actually landed

Usage:
    python examples/ifj_aic144/report.py [tier] [output_dir]
"""

import sys
import time
from pathlib import Path

from pulsed_ion_chamber.output import write_collected_charge_csv
from pulsed_ion_chamber.plots import save_diagnostic_plots
from pulsed_ion_chamber.resources import format_bytes
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel, warmup_parallel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_markus_2mm import EDGE_DEFICIT_UM, GRID_TIERS, build_config  # noqa: E402


def main(tier: str = "archive", output_dir: str = "") -> None:
    config = build_config(tier)
    directory = Path(output_dir or f"out/ifj_aic144/markus_2mm_{tier}")

    print(f"=== IFJ AIC-144, Markus 2 mm, macropulse, 10 Gy/s -- '{tier}' ===")
    print(config.summary())
    print(f"\nEstimated peak memory: {format_bytes(config.estimated_memory_bytes)}")
    print(f"Writing to {directory}/\n")

    warmup_parallel()
    started = time.perf_counter()
    result = run_simulation_numba_parallel(config, progress=True, num_threads=1)
    elapsed = time.perf_counter() - started

    csv_path = write_collected_charge_csv(result, directory / "collected_charge.csv")
    plot_paths = save_diagnostic_plots(result, directory, title=f"Markus 2 mm, {tier}")

    radius_um = config.sampled_radius_cm * 1e4
    corrected = result.ks + EDGE_DEFICIT_UM / radius_um
    injected = result.injected_positive.sum() * config.unit_length_cm**3
    recombined = result.recombination.sum() * config.unit_length_cm**3

    print(f"\nWall time            : {elapsed:.1f} s ({elapsed / 60:.2f} min), single thread")
    print(f"Tracks               : {config.number_of_tracks_per_pulse:,}")
    print(f"Injected (each sign) : {injected:.4e} ion pairs")
    print(f"Recombined           : {recombined:.4e} ion pairs ({100 * recombined / injected:.2f} %)")
    print(f"Collection efficiency: f = {result.f_t[-1]:.6f}")
    print(f"k_s = 1/f            : {result.ks:.6f}")
    print(f"k_s, edge-corrected  : {corrected:.6f}  (+{EDGE_DEFICIT_UM / radius_um:.4f})")
    print(f"\nWrote {csv_path}")
    for path in plot_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    tier = sys.argv[1] if len(sys.argv) > 1 else "archive"
    if tier not in GRID_TIERS:
        sys.exit(f"Unknown tier {tier!r}; expected one of {sorted(GRID_TIERS)}")
    main(tier, sys.argv[2] if len(sys.argv) > 2 else "")
