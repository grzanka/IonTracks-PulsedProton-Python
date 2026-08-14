"""Writing a run's per-time-step record to disk.

The solver accumulates *densities* summed over voxels, because the collection
efficiency is a ratio in which the voxel volume cancels. Everything written
here is converted to absolute ion-pair counts by multiplying through by the
voxel volume, so the numbers are physical and directly comparable with other
codes' output.
"""

import csv
from pathlib import Path
from typing import Union

import numpy as np

__all__ = ["COLLECTED_CHARGE_COLUMNS", "collected_charge_table", "write_collected_charge_csv"]

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
