"""Explicit-loop Lax-Wendroff finite-difference solver for the coupled
drift-diffusion-recombination equations of positive/negative ion densities
in a parallel-plate ionization chamber, exposed to a pulsed proton beam.

    d n_+-/dt = D * lapl(n_+-) -+ mu * E * d(n_+-)/dz - alpha * n+ * n-

This is deliberately written as plain nested Python for-loops (matching
hadrons/python/continuous_beam.py in the original IonTracks-Cython repo)
rather than vectorized NumPy slicing: it is meant as a clear, obviously
correct but slow starting point for a later multi-threading/GPU port.
The two loops below -- track insertion (_insert_track) and the PDE update
(_lax_wendroff_step) -- are where essentially all the run time goes and
are the natural targets for parallelization.
"""

from dataclasses import dataclass
from math import ceil, exp, floor
from typing import Optional

import numpy as np

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import RECOMBINATION_ALPHA_CM3_S
from pulsed_ion_chamber.pulses import build_track_schedule, sample_xy_inside_cylinder


def apply_lateral_boundary(array: np.ndarray, mode: str) -> None:
    """Enforce the outer-ring boundary condition in place, after a step.

    "absorbing" is a no-op: the Lax-Wendroff sweep never writes the outer
    ring, so whatever is there stays put and charge that reaches it leaves the
    simulation -- the original IonTracks-Cython behaviour.

    "reflecting" mirrors each interior neighbour outwards, giving a
    zero-gradient (zero-flux) wall. That is the natural boundary condition
    IonTracks-FEniCSx leaves on the chamber wall, and the physically right one
    for a column sampled from the interior of a large uniformly irradiated
    volume: the neighbouring gas that is not simulated returns as much charge
    as it removes, so there is no net flux across the wall.

    The z ends are deliberately untouched in both modes -- charge that drifts
    past the electrode buffer has been collected and should leave.
    """
    if mode != "reflecting":
        return
    array[0, :, :] = array[1, :, :]
    array[-1, :, :] = array[-2, :, :]
    array[:, 0, :] = array[:, 1, :]
    array[:, -1, :] = array[:, -2, :]


def _stencil_bounds(centre: float, cutoff_voxels: float, no_xy: int) -> tuple[int, int]:
    """Half-open grid-index range [lo, hi) covered by a track's cutoff radius,
    clipped to the grid. Shared by all three backends so they truncate
    identically."""
    lo = int(ceil(centre - cutoff_voxels))
    hi = int(floor(centre + cutoff_voxels)) + 1
    return max(lo, 0), min(hi, no_xy)


@dataclass
class Result:
    config: SimulationConfig
    time_s: np.ndarray  # shape (total_time_steps,)
    f_t: np.ndarray  # collection efficiency vs. time, same shape
    ks: float  # recombination correction factor 1 / f_t[-1], after full clearance
    positive_array: np.ndarray  # final charge-carrier density snapshot [cm^-3]
    negative_array: np.ndarray


def _insert_track(positive_array, negative_array, x, y, config) -> float:
    """Add one Gaussian ion track's charge density to both carrier arrays.

    The deposit is confined to the square bounding box of the track's cutoff
    radius (config.track_cutoff_voxels); outside it the Gaussian is below
    exp(-cutoff_sigmas^2/2) of the track's total charge and is dropped. Cost is
    O(stencil^2 * no_z) rather than O(no_xy^2 * no_z).
    """
    inserted = 0.0
    b2 = config.track_radius_cm**2
    h2 = config.unit_length_cm**2
    mid = config.mid_xy
    scoring_radius_sq = config.scoring_radius_sq
    gaussian_factor = config.Gaussian_factor
    cutoff = config.track_cutoff_voxels
    i_lo, i_hi = _stencil_bounds(x, cutoff, config.no_xy)
    j_lo, j_hi = _stencil_bounds(y, cutoff, config.no_xy)
    for k in range(config.no_z_electrode, config.no_z + config.no_z_electrode):
        for i in range(i_lo, i_hi):
            for j in range(j_lo, j_hi):
                r2 = ((i - x) ** 2 + (j - y) ** 2) * h2
                ion_density = gaussian_factor * exp(-r2 / b2)
                positive_array[i, j, k] += ion_density
                negative_array[i, j, k] += ion_density
                if (i - mid) ** 2 + (j - mid) ** 2 < scoring_radius_sq:
                    inserted += ion_density
    return inserted


def _lax_wendroff_step(
    positive_array, negative_array, positive_next, negative_next, config, pos_weights, neg_weights
) -> float:
    """Advance both carrier densities by one time step and accumulate the
    recombination that occurred inside the scored gap region.

    pos_weights/neg_weights are the per-species (lateral, z_minus, z_plus,
    centre) stencil weights from config.scheme_coefficients(); they are equal
    unless the two Kanai carrier species are being resolved separately.

    O(no_xy^2 * no_z_with_buffer) per call, executed once per time step --
    the main target for parallelization (independent per-voxel updates).
    """
    p_lat, p_zm, p_zp, p_cen = pos_weights
    n_lat, n_zm, n_zp, n_cen = neg_weights
    recombined = 0.0
    alpha_dt = RECOMBINATION_ALPHA_CM3_S * config.dt
    mid = config.mid_xy
    scoring_radius_sq = config.scoring_radius_sq
    no_z_electrode = config.no_z_electrode
    no_z = config.no_z

    for i in range(1, config.no_xy - 1):
        for j in range(1, config.no_xy - 1):
            for k in range(1, config.no_z_with_buffer - 1):
                p = positive_array[i, j, k]
                n = negative_array[i, j, k]

                p_new = (
                    p_zm * positive_array[i, j, k - 1]
                    + p_zp * positive_array[i, j, k + 1]
                    + p_lat * positive_array[i, j - 1, k]
                    + p_lat * positive_array[i, j + 1, k]
                    + p_lat * positive_array[i - 1, j, k]
                    + p_lat * positive_array[i + 1, j, k]
                    + p_cen * p
                )
                # negative ions drift opposite to positive ions, so the
                # +z/-z neighbour coefficients are swapped relative to p_new
                n_new = (
                    n_zm * negative_array[i, j, k + 1]
                    + n_zp * negative_array[i, j, k - 1]
                    + n_lat * negative_array[i, j - 1, k]
                    + n_lat * negative_array[i, j + 1, k]
                    + n_lat * negative_array[i - 1, j, k]
                    + n_lat * negative_array[i + 1, j, k]
                    + n_cen * n
                )

                recomb = alpha_dt * p * n
                positive_next[i, j, k] = p_new - recomb
                negative_next[i, j, k] = n_new - recomb

                if no_z_electrode < k < (no_z + no_z_electrode) and (i - mid) ** 2 + (
                    j - mid
                ) ** 2 < scoring_radius_sq:
                    recombined += recomb

    return recombined


def run_simulation(config: SimulationConfig, rng: Optional[np.random.Generator] = None, progress: bool = True) -> Result:
    """Simulate a pulse train and return the collection-efficiency time
    series f(t) and the final recombination correction factor k_s = 1/f.
    """
    rng = rng if rng is not None else np.random.default_rng(config.seed)
    schedule = build_track_schedule(config, rng)

    shape = (config.no_xy, config.no_xy, config.no_z_with_buffer)
    positive_array = np.zeros(shape)
    negative_array = np.zeros(shape)
    positive_next = np.zeros(shape)
    negative_next = np.zeros(shape)

    pos_weights, neg_weights = config.scheme_coefficients()

    no_initialised = 0.0
    no_recombined = 0.0
    f_t = np.ones(config.total_time_steps)
    report_every = max(1, config.total_time_steps // 20)

    for step in range(config.total_time_steps):
        for _ in range(schedule[step]):
            x, y = sample_xy_inside_cylinder(rng, config.mid_xy, config.sampling_radius, config.no_xy)
            no_initialised += _insert_track(positive_array, negative_array, x, y, config)

        no_recombined += _lax_wendroff_step(
            positive_array, negative_array, positive_next, negative_next, config, pos_weights, neg_weights
        )

        positive_array[1:-1, 1:-1, 1:-1] = positive_next[1:-1, 1:-1, 1:-1]
        negative_array[1:-1, 1:-1, 1:-1] = negative_next[1:-1, 1:-1, 1:-1]
        apply_lateral_boundary(positive_array, config.lateral_boundary)
        apply_lateral_boundary(negative_array, config.lateral_boundary)

        f_t[step] = 1.0 if no_initialised == 0.0 else (no_initialised - no_recombined) / no_initialised

        if progress and step % report_every == 0:
            print(f"  step {step + 1}/{config.total_time_steps}  f = {f_t[step]:.4f}")

    time_s = (np.arange(config.total_time_steps) + 1) * config.dt
    ks = 1.0 / f_t[-1]
    return Result(config=config, time_s=time_s, f_t=f_t, ks=ks, positive_array=positive_array, negative_array=negative_array)
