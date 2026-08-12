"""The baseline single-threaded backend: same physics as solver.py, but
restructured and Numba-JIT-compiled for speed on one thread, still with
no `parallel=True`/`prange` -- that's the next step, not this one.

solver.py deliberately mirrors the original IonTracks-Cython loop
structure one-to-one (k outermost, i, j innermost) as a readable
reference. This module instead exploits three things that structure
leaves on the table, on top of JIT compilation:

1. **Loop order matches memory layout.** Arrays are C-contiguous with `k`
   (the z-axis) as the fastest-varying index. Looping `k` innermost (here:
   i, j outer, k inner) walks memory sequentially instead of striding
   across it -- solver.py's `_insert_track` loops (k outer, i, j inner) do
   the opposite. `_lax_wendroff_step`'s loop order was already k-innermost,
   so it doesn't need reordering, only the two optimizations below.
2. **Track insertion doesn't depend on k.** A track's 2D Gaussian density
   at grid point (i, j) is deposited identically into every z-layer of the
   gap (see `_insert_track_numba`'s docstring) -- solver.py's k-outermost
   loop recomputes `exp(...)` for every (i, j) once per k layer (`no_z`
   times); hoisting that computation to before the k-loop removes that
   redundancy entirely.
3. **The 2D Gaussian is separable.** `exp(-((i-x)^2+(j-y)^2)*h2/b2)` factors
   into `exp(-(i-x)^2*h2/b2) * exp(-(j-y)^2*h2/b2)` (`e^(a+b) = e^a * e^b`).
   Precomputing each 1D factor once (`O(no_xy)` calls to `exp`) and taking
   their product per (i, j) (`O(no_xy^2)` cheap multiplications) avoids
   `O(no_xy^2)` calls to `exp` -- exact, not an approximation.

Both hot loops also compare *squared* distances against a precomputed
`inner_radius_sq` instead of calling `sqrt` per voxel/track (see
SimulationConfig.inner_radius_sq in config.py).

Numba's `nopython` mode cannot take the `SimulationConfig` dataclass
directly (it isn't a numba-compatible type), so the jitted functions take
plain scalars/arrays instead; run_simulation_numba() unpacks the config
once per call, mirroring what run_simulation() does in solver.py.
"""

from math import exp
from typing import Optional

import numba
import numpy as np
import numpy.typing as npt

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import ION_DIFFUSION_CM2_S, ION_MOBILITY_CM2_VS, RECOMBINATION_ALPHA_CM3_S
from pulsed_ion_chamber.pulses import build_track_schedule, sample_xy_inside_cylinder
from pulsed_ion_chamber.solver import Result

FloatArray3D = npt.NDArray[np.float64]
FloatArray1D = npt.NDArray[np.float64]


@numba.njit(cache=True)
def _insert_track_numba(
    positive_array: FloatArray3D,
    negative_array: FloatArray3D,
    x: float,
    y: float,
    no_xy: int,
    no_z: int,
    no_z_electrode: int,
    h2: float,
    b2: float,
    gaussian_factor: float,
    mid_xy: int,
    inner_radius_sq: float,
) -> float:
    """Add one Gaussian ion track's charge density to both carrier arrays.

    The track runs the length of the gap parallel to the field, so its 2D
    Gaussian cross-section is deposited identically into every z-layer
    from no_z_electrode to no_z_electrode + no_z -- there is no k-dependence
    in the density itself, only in how many layers it gets added to.
    """
    # Separable Gaussian: exp(-((i-x)^2+(j-y)^2)*h2/b2) = gauss_i[i] * gauss_j[j].
    # O(no_xy) calls to exp instead of O(no_xy^2).
    gauss_i = np.empty(no_xy)
    for i in range(no_xy):
        gauss_i[i] = exp(-((i - x) ** 2) * h2 / b2)
    gauss_j = np.empty(no_xy)
    for j in range(no_xy):
        gauss_j[j] = exp(-((j - y) ** 2) * h2 / b2)

    k_lo = no_z_electrode
    k_hi = no_z_electrode + no_z

    inserted = 0.0
    for i in range(no_xy):
        di_sq = (i - mid_xy) ** 2
        gi = gauss_i[i]
        for j in range(no_xy):
            ion_density = gaussian_factor * gi * gauss_j[j]
            for k in range(k_lo, k_hi):
                positive_array[i, j, k] += ion_density
                negative_array[i, j, k] += ion_density
            if di_sq + (j - mid_xy) ** 2 < inner_radius_sq:
                inserted += ion_density * no_z

    return inserted


@numba.njit(cache=True)
def _lax_wendroff_step_numba(
    positive_array: FloatArray3D,
    negative_array: FloatArray3D,
    positive_next: FloatArray3D,
    negative_next: FloatArray3D,
    no_xy: int,
    no_z_with_buffer: int,
    no_z_electrode: int,
    no_z: int,
    mid_xy: int,
    inner_radius_sq: float,
    sx: float,
    sc_pos_z: float,
    sc_neg_z: float,
    sc_center: float,
    alpha_dt: float,
) -> float:
    """Advance both carrier densities by one time step and accumulate the
    recombination that occurred inside the scored gap region.

    Whether (i, j) falls inside the scored cylinder doesn't depend on k,
    so it's evaluated once per (i, j) here rather than once per (i, j, k).
    """
    recombined = 0.0
    for i in range(1, no_xy - 1):
        di_sq = (i - mid_xy) ** 2
        for j in range(1, no_xy - 1):
            inside = di_sq + (j - mid_xy) ** 2 < inner_radius_sq

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

                if inside and no_z_electrode < k < (no_z + no_z_electrode):
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


def run_simulation_numba(
    config: SimulationConfig, rng: Optional[np.random.Generator] = None, progress: bool = True
) -> Result:
    """Same algorithm as solver.run_simulation(), with the two hot loops
    restructured and compiled by Numba (single-threaded: no parallel=True,
    no prange)."""
    rng = rng if rng is not None else np.random.default_rng(config.seed)
    schedule = build_track_schedule(config, rng)

    shape = (config.no_xy, config.no_xy, config.no_z_with_buffer)
    positive_array: FloatArray3D = np.zeros(shape)
    negative_array: FloatArray3D = np.zeros(shape)
    positive_next: FloatArray3D = np.zeros(shape)
    negative_next: FloatArray3D = np.zeros(shape)

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
    f_t: FloatArray1D = np.ones(config.total_time_steps)
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
                config.inner_radius_sq,
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
            config.inner_radius_sq,
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

    time_s: FloatArray1D = (np.arange(config.total_time_steps) + 1) * config.dt
    ks = 1.0 / f_t[-1]
    return Result(config=config, time_s=time_s, f_t=f_t, ks=ks, positive_array=positive_array, negative_array=negative_array)
