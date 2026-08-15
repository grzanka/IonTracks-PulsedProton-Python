"""Writing a run's per-time-step record to disk, and reading it back.

The solver accumulates *densities* summed over voxels, because the collection
efficiency is a ratio in which the voxel volume cancels. Everything written
here is converted to absolute ion-pair counts by multiplying through by the
voxel volume, so the numbers are physical and directly comparable with other
codes' output.

:func:`save_run_record` / :func:`load_run_record` are the pair that lets
plotting happen without a solver: a run is performed once
(`examples/ifj_aic144/run_markus_2mm.py --save-run DIR`) and the plots are
drawn later, possibly in a different process, from files alone
(`examples/ifj_aic144/plot.py DIR`). `run_meta.json` carries just the config
scalars the figures need (see `pulsed_ion_chamber.plots`) plus a few
already-computed summary numbers -- not the whole `SimulationConfig`, which
would tie the file format to config.py's field set.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import numpy as np

__all__ = [
    "COLLECTED_CHARGE_COLUMNS",
    "collected_charge_table",
    "write_collected_charge_csv",
    "track_density_per_cm2",
    "save_run_record",
    "load_run_record",
    "RunRecord",
]

# The config scalars examples/ifj_aic144/plot.py's figures actually read
# (see pulsed_ion_chamber.plots.save_diagnostic_plots_from_table) -- not the
# whole SimulationConfig, so the saved-run format does not have to track
# every field config.py happens to grow.
_PLOT_CONFIG_FIELDS = (
    "dt",
    "pulse_duration_s",
    "no_xy",
    "unit_length_cm",
    "inner_radius",
    "sampling_radius",
    "chamber_fill_fraction",
)

COLLECTED_CHARGE_COLUMNS = (
    "time",
    "n_positive",
    "n_negative",
    "injected_positive",
    "injected_negative",
    "recombination",
)


def collected_charge_table(result) -> dict:
    """Per-time-step record as a dict of equal-length arrays, in ion pairs.

    ``time`` is in seconds (end of the step). ``n_positive``/``n_negative`` are
    the carriers present in the scored region at the end of the step;
    ``injected_*`` are those created during it; ``recombination`` is the pairs
    lost during it.
    """
    voxel_volume_cm3 = result.config.unit_length_cm**3
    return {
        "time": result.time_s,
        "n_positive": result.n_positive * voxel_volume_cm3,
        "n_negative": result.n_negative * voxel_volume_cm3,
        "injected_positive": result.injected_positive * voxel_volume_cm3,
        "injected_negative": result.injected_negative * voxel_volume_cm3,
        "recombination": result.recombination * voxel_volume_cm3,
    }


def write_collected_charge_csv(result, path: Union[str, Path]) -> Path:
    """Write the per-time-step record, one row per step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = collected_charge_table(result)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLLECTED_CHARGE_COLUMNS)
        for row in zip(*(table[name] for name in COLLECTED_CHARGE_COLUMNS)):
            writer.writerow(f"{value:.12e}" for value in row)
    return path


def track_density_per_cm2(result) -> np.ndarray:
    """Track centres per cm^2, as a (no_xy, no_xy) map.

    The solver counts track centres per voxel column; dividing by the column's
    footprint turns that into the areal density the beam actually delivered,
    which is the quantity worth checking against the nominal fluence.
    """
    voxel_area_cm2 = result.config.unit_length_cm**2
    return result.track_density_xy / voxel_area_cm2


@dataclass
class RunRecord:
    """A saved run, loaded back for plotting -- no solver, no config object.

    ``config`` exposes only the scalars `pulsed_ion_chamber.plots` reads (see
    `_PLOT_CONFIG_FIELDS`), as a plain `SimpleNamespace`, not a
    `SimulationConfig`. ``table`` is the same shape `collected_charge_table`
    returns, already in ion pairs. ``track_density_xy`` is the raw
    track-centre count per voxel column, as the solver recorded it (not yet
    divided by voxel area).
    """

    config: SimpleNamespace
    table: dict
    track_density_xy: np.ndarray
    meta: dict  # everything in run_meta.json, including fields config doesn't carry


def save_run_record(result, directory: Union[str, Path]) -> dict:
    """Write everything `load_run_record` + the plots need to reproduce this
    run's figures, without saving the run itself.

    Three files, all in ``directory``:

    * ``collected_charge.csv`` -- the per-time-step record (see
      :func:`write_collected_charge_csv`).
    * ``track_density_xy.npy`` -- raw track-centre counts per voxel column.
    * ``run_meta.json`` -- the config scalars the plots read
      (`_PLOT_CONFIG_FIELDS`) plus a summary (`f`, `ks`, track and charge
      totals) for a plotting script to report without recomputing them.

    Returns the three paths, keyed ``"csv"``, ``"track_density"``, ``"meta"``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = result.config

    csv_path = write_collected_charge_csv(result, directory / "collected_charge.csv")

    density_path = directory / "track_density_xy.npy"
    np.save(density_path, result.track_density_xy)

    voxel_volume_cm3 = config.unit_length_cm**3
    meta = {field: getattr(config, field) for field in _PLOT_CONFIG_FIELDS}
    meta.update(
        f=float(result.f_t[-1]),
        ks=float(result.ks),
        tracks_per_pulse=int(config.number_of_tracks_per_pulse),
        injected_ion_pairs=float(result.injected_positive.sum() * voxel_volume_cm3),
        recombined_ion_pairs=float(result.recombination.sum() * voxel_volume_cm3),
    )
    meta_path = directory / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    return {"csv": csv_path, "track_density": density_path, "meta": meta_path}


def load_run_record(directory: Union[str, Path]) -> RunRecord:
    """Read back what :func:`save_run_record` wrote -- the inverse.

    Raises `FileNotFoundError` naming whichever of the three files is
    missing, rather than letting a bare `csv`/`json`/`np.load` traceback
    stand in for "this directory wasn't written by save_run_record".
    """
    directory = Path(directory)
    csv_path = directory / "collected_charge.csv"
    density_path = directory / "track_density_xy.npy"
    meta_path = directory / "run_meta.json"
    for path in (csv_path, density_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- is {directory} a directory written by save_run_record() "
                "(e.g. `run_markus_2mm.py ... --save-run DIR`)?"
            )

    meta = json.loads(meta_path.read_text())
    config = SimpleNamespace(**{field: meta[field] for field in _PLOT_CONFIG_FIELDS})

    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    header, rows = rows[0], rows[1:]
    columns = {name: np.array([float(row[i]) for row in rows]) for i, name in enumerate(header)}
    # collected_charge.csv is already in ion pairs (write_collected_charge_csv
    # applies the voxel-volume conversion), so this table is directly what
    # save_diagnostic_plots_from_table expects -- no reconversion here.
    table = {name: columns[name] for name in COLLECTED_CHARGE_COLUMNS}

    track_density_xy = np.load(density_path)

    return RunRecord(config=config, table=table, track_density_xy=track_density_xy, meta=meta)
