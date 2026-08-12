"""A single-threaded Numba-JIT version of the two hot loops in solver.py.

This is the *first, smallest possible* step of the parallelization
exercise the rest of this repository is a starting point for: same
algorithm, same explicit loops, no vectorization and no multi-threading
(no `parallel=True`, no `prange`) -- just compiling the existing Python
loops to machine code with `@numba.njit`. It exists to give a concrete,
easy "before/after" number before reaching for anything more involved
(multiprocessing, `prange`, or a GPU backend).

Numba's `nopython` mode cannot take the `SimulationConfig` dataclass
directly (it isn't a numba-compatible type), so the jitted functions take
plain scalars/arrays instead; run_simulation_numba() unpacks the config
once per call, mirroring what run_simulation() does in solver.py.
"""

from math import exp, sqrt
from typing import Optional

import numba
import numpy as np

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import ION_DIFFUSION_CM2_S, ION_MOBILITY_CM2_VS, RECOMBINATION_ALPHA_CM3_S
from pulsed_ion_chamber.pulses import build_track_schedule, sample_xy_inside_cylinder
from pulsed_ion_chamber.solver import Result


@numba.njit(cache=True)
def _insert_track_numba(
    positive_array, negative_array, x, y, no_xy, no_z, no_z_electrode, h2, b2, gaussian_factor, mid_xy, inner_radius
):
    inserted = 0.0
    for k in range(no_z_electrode, no_z + no_z_electrode):
        for i in range(no_xy):
            for j in range(no_xy):
                r2 = ((i - x) ** 2 + (j - y) ** 2) * h2
                ion_density = gaussian_factor * exp(-r2 / b2)
                positive_array[i, j, k] += ion_density
                negative_array[i, j, k] += ion_density
                if sqrt((i - mid_xy) ** 2 + (j - mid_xy) ** 2) < inner_radius:
                    inserted += ion_density
    return inserted


@numba.njit(cache=True)
def _lax_wendroff_step_numba(
    positive_array,
    negative_array,
    positive_next,
    negative_next,
    no_xy,
    no_z_with_buffer,
    no_z_electrode,
    no_z,
    mid_xy,
    inner_radius,
    sx,
    sc_pos_z,
    sc_neg_z,
    sc_center,
    alpha_dt,
):
    recombined = 0.0
    for i in range(1, no_xy - 1):
        for j in range(1, no_xy - 1):
            for k in range(1, no_z_with_buffer - 1):
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

                if no_z_electrode < k < (no_z + no_z_electrode) and sqrt((i - mid_xy) ** 2 + (j - mid_xy) ** 2) < inner_radius:
                    recombined += recomb

    return recombined


def warmup() -> None:
    """Trigger Numba JIT compilation of both hot loops on minimal dummy
    arrays, so a subsequent timed call to run_simulation_numba() measures
    only execution time, not one-time compilation.

    Deliberately does NOT go through run_simulation_numba()/SimulationConfig
    -- it calls the jitted kernels directly with trivial, always-valid
    scalar arguments, so it can't hang the way a degenerate config could.
    """
    shape = (4, 4, 4)
    p, n, p_next, n_next = (np.zeros(shape) for _ in range(4))
    _insert_track_numba(p, n, 2.0, 2.0, 4, 1, 1, 1.0, 1.0, 1.0, 2, 1.0)
    _lax_wendroff_step_numba(p, n, p_next, n_next, 4, 4, 1, 1, 2, 1.0, 0.01, 0.01, 0.01, 0.97, 1e-9)


def run_simulation_numba(config: SimulationConfig, rng: Optional[np.random.Generator] = None, progress: bool = True) -> Result:
    """Same algorithm as solver.run_simulation(), with the two hot loops
    compiled by Numba (single-threaded: no parallel=True, no prange)."""
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
    alpha_dt = RECOMBINATION_ALPHA_CM3_S * config.dt

    h2 = config.unit_length_cm**2
    b2 = config.track_radius_cm**2

    no_initialised = 0.0
    no_recombined = 0.0
    f_t = np.ones(config.total_time_steps)
    report_every = max(1, config.total_time_steps // 20)

    for step in range(config.total_time_steps):
        for _ in range(schedule[step]):
            x, y = sample_xy_inside_cylinder(rng, config.mid_xy, config.inner_radius, config.no_xy)
            no_initialised += _insert_track_numba(
                positive_array,
                negative_array,
                x,
                y,
                config.no_xy,
                config.no_z,
                config.no_z_electrode,
                h2,
                b2,
                config.Gaussian_factor,
                config.mid_xy,
                config.inner_radius,
            )

        no_recombined += _lax_wendroff_step_numba(
            positive_array,
            negative_array,
            positive_next,
            negative_next,
            config.no_xy,
            config.no_z_with_buffer,
            config.no_z_electrode,
            config.no_z,
            config.mid_xy,
            config.inner_radius,
            sx,
            sc_pos_z,
            sc_neg_z,
            sc_center,
            alpha_dt,
        )

        positive_array[1:-1, 1:-1, 1:-1] = positive_next[1:-1, 1:-1, 1:-1]
        negative_array[1:-1, 1:-1, 1:-1] = negative_next[1:-1, 1:-1, 1:-1]

        f_t[step] = 1.0 if no_initialised == 0.0 else (no_initialised - no_recombined) / no_initialised

        if progress and step % report_every == 0:
            print(f"  step {step + 1}/{config.total_time_steps}  f = {f_t[step]:.4f}")

    time_s = (np.arange(config.total_time_steps) + 1) * config.dt
    ks = 1.0 / f_t[-1]
    return Result(config=config, time_s=time_s, f_t=f_t, ks=ks, positive_array=positive_array, negative_array=negative_array)
