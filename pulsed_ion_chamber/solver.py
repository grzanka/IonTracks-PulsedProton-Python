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
from math import exp, sqrt
from typing import Optional

import numpy as np

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import ION_DIFFUSION_CM2_S, ION_MOBILITY_CM2_VS, RECOMBINATION_ALPHA_CM3_S
from pulsed_ion_chamber.pulses import build_track_schedule, sample_xy_inside_cylinder


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

    O(no_xy^2 * no_z) per call -- the cost scales with the number of tracks
    per pulse, which in turn scales with dose rate and sampled area.
    """
    inserted = 0.0
    b2 = config.track_radius_cm**2
    h2 = config.unit_length_cm**2
    mid = config.mid_xy
    inner_radius = config.inner_radius
    gaussian_factor = config.Gaussian_factor
    for k in range(config.no_z_electrode, config.no_z + config.no_z_electrode):
        for i in range(config.no_xy):
            for j in range(config.no_xy):
                r2 = ((i - x) ** 2 + (j - y) ** 2) * h2
                ion_density = gaussian_factor * exp(-r2 / b2)
                positive_array[i, j, k] += ion_density
                negative_array[i, j, k] += ion_density
                if sqrt((i - mid) ** 2 + (j - mid) ** 2) < inner_radius:
                    inserted += ion_density
    return inserted


def _lax_wendroff_step(
    positive_array, negative_array, positive_next, negative_next, config, sx, sc_pos_z, sc_neg_z, sc_center
) -> float:
    """Advance both carrier densities by one time step and accumulate the
    recombination that occurred inside the scored gap region.

    O(no_xy^2 * no_z_with_buffer) per call, executed once per time step --
    the main target for parallelization (independent per-voxel updates).
    """
    recombined = 0.0
    alpha_dt = RECOMBINATION_ALPHA_CM3_S * config.dt
    mid = config.mid_xy
    inner_radius = config.inner_radius
    no_z_electrode = config.no_z_electrode
    no_z = config.no_z

    for i in range(1, config.no_xy - 1):
        for j in range(1, config.no_xy - 1):
            for k in range(1, config.no_z_with_buffer - 1):
                p = positive_array[i, j, k]
                n = negative_array[i, j, k]

                p_new = (
                    sc_pos_z * positive_array[i, j, k - 1]
                    + sc_neg_z * positive_array[i, j, k + 1]
                    + sx * positive_array[i, j - 1, k]
                    + sx * positive_array[i, j + 1, k]
                    + sx * positive_array[i - 1, j, k]
                    + sx * positive_array[i + 1, j, k]
                    + sc_center * p
                )
                # negative ions drift opposite to positive ions, so the
                # +z/-z neighbour coefficients are swapped relative to p_new
                n_new = (
                    sc_pos_z * negative_array[i, j, k + 1]
                    + sc_neg_z * negative_array[i, j, k - 1]
                    + sx * negative_array[i, j - 1, k]
                    + sx * negative_array[i, j + 1, k]
                    + sx * negative_array[i - 1, j, k]
                    + sx * negative_array[i + 1, j, k]
                    + sc_center * n
                )

                recomb = alpha_dt * p * n
                positive_next[i, j, k] = p_new - recomb
                negative_next[i, j, k] = n_new - recomb

                if no_z_electrode < k < (no_z + no_z_electrode) and sqrt((i - mid) ** 2 + (j - mid) ** 2) < inner_radius:
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

    sx = ION_DIFFUSION_CM2_S * config.dt / config.unit_length_cm**2
    cz = ION_MOBILITY_CM2_VS * config.Efield_V_cm * config.dt / config.unit_length_cm
    sc_pos_z = sx + cz * (cz + 1.0) / 2.0
    sc_neg_z = sx + cz * (cz - 1.0) / 2.0
    sc_center = 1.0 - cz * cz - 6.0 * sx

    no_initialised = 0.0
    no_recombined = 0.0
    f_t = np.ones(config.total_time_steps)
    report_every = max(1, config.total_time_steps // 20)

    for step in range(config.total_time_steps):
        for _ in range(schedule[step]):
            x, y = sample_xy_inside_cylinder(rng, config.mid_xy, config.inner_radius, config.no_xy)
            no_initialised += _insert_track(positive_array, negative_array, x, y, config)

        no_recombined += _lax_wendroff_step(
            positive_array, negative_array, positive_next, negative_next, config, sx, sc_pos_z, sc_neg_z, sc_center
        )

        positive_array[1:-1, 1:-1, 1:-1] = positive_next[1:-1, 1:-1, 1:-1]
        negative_array[1:-1, 1:-1, 1:-1] = negative_next[1:-1, 1:-1, 1:-1]

        f_t[step] = 1.0 if no_initialised == 0.0 else (no_initialised - no_recombined) / no_initialised

        if progress and step % report_every == 0:
            print(f"  step {step + 1}/{config.total_time_steps}  f = {f_t[step]:.4f}")

    time_s = (np.arange(config.total_time_steps) + 1) * config.dt
    ks = 1.0 / f_t[-1]
    return Result(config=config, time_s=time_s, f_t=f_t, ks=ks, positive_array=positive_array, negative_array=negative_array)
