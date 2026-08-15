#!/usr/bin/env python3
"""Draw the four diagnostic figures from a run someone else already performed.

Reads exactly what `run_markus_2mm.py ... --save-run DIR` wrote --
`collected_charge.csv`, `track_density_xy.npy`, `run_meta.json` -- and writes:

    injection_rate.png            beam arrival vs time
    carrier_evolution.png         carriers present vs time
    recombination_rate.png        pairs lost per time step
    track_density_cross_section.png   where the tracks actually landed

Deliberately does not import Numba, Numba's solver backends, or anything with
a thread count: this script only reads files and calls matplotlib, so it has
nothing to parallelise and no `--threads` flag. If it's slow, the run that
produced RUN_DIR is what to speed up (`run_markus_2mm.py --threads N
--save-run DIR`), not this step.

Usage:
    python examples/ifj_aic144/plot.py RUN_DIR [output_dir]

RUN_DIR is where `--save-run` wrote its files; output_dir defaults to RUN_DIR
itself, so the figures land next to the data they were drawn from.
"""

import sys
from pathlib import Path

from pulsed_ion_chamber.output import load_run_record
from pulsed_ion_chamber.plots import save_diagnostic_plots_from_table


def main(run_dir: str, output_dir: str = "") -> None:
    run_dir = Path(run_dir)
    directory = Path(output_dir) if output_dir else run_dir

    record = load_run_record(run_dir)
    meta = record.meta
    print(f"=== Plotting {run_dir}/ ===")
    print(f"Tracks               : {meta['tracks_per_pulse']:,}")
    print(f"Injected (each sign) : {meta['injected_ion_pairs']:.4e} ion pairs")
    print(f"Recombined           : {meta['recombined_ion_pairs']:.4e} ion pairs")
    print(f"Collection efficiency: f = {meta['f']:.6f}")
    print(f"k_s = 1/f            : {meta['ks']:.6f}")
    print(f"\nWriting to {directory}/\n")

    plot_paths = save_diagnostic_plots_from_table(
        record.config, record.table, record.track_density_xy / record.config.unit_length_cm**2, directory
    )
    for path in plot_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
