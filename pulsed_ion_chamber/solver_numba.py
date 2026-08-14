"""Baseline backend: one track at a time, single grid pass per track.

The reference implementation of the physics in docs/PHYSICS.md. Simpler than
solver_numba_parallel.py -- no batching, no row index, no scratch array -- and
the right choice when few tracks arrive per time step, where batching's
once-per-step broadcast would be pure overhead. For dense pulses (hundreds or
thousands of tracks per step) use the batched backend instead; the two agree to
1e-9 and share the same RNG stream.

Three exact identities make deposition cheap. All three are algebra, not
approximation:

1. **No k dependence.** A track is a line through the full gap at constant LET,
   so its 2D cross-section is written unchanged into every one of `no_z`
   layers. Computing it once outside the k loop, rather than recomputing
   `exp(...)` inside it, removes a factor of `no_z` from the transcendental
   count.
2. **Separability.** `exp(-((i-x)^2 + (j-y)^2) h^2/b^2)` equals
   `exp(-(i-x)^2 h^2/b^2) * exp(-(j-y)^2 h^2/b^2)`, so a w x w stencil needs
   `2w` calls to `exp` and `w^2` multiplications instead of `w^2` calls to
   `exp`. On the reference grid that is 56 exponentials per track rather than
   784.
3. **Truncation.** Beyond `track_cutoff_sigmas` the Gaussian is below
   `exp(-50)` of its peak at the default 10 sigma -- far below what a float64
   sum can represent -- so it is simply not evaluated. This is what makes the
   per-track cost independent of how wide the grid is.

Loop order is `k` innermost throughout, matching the C-contiguous array layout
(`k` is the fastest-varying axis), so memory is walked sequentially rather than
strided.

Numba's nopython mode cannot take the `SimulationConfig` dataclass, so the
jitted functions below take plain scalars and arrays; `run_simulation_numba`
unpacks the config once per call.
"""

from math import ceil, exp, floor
from typing import Optional

import numba
import numpy as np
import numpy.typing as npt

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import RECOMBINATION_ALPHA_CM3_S
from pulsed_ion_chamber.pulses import build_track_schedule, sample_xy_inside_cylinder
from pulsed_ion_chamber.state import Diagnostics, Result, apply_lateral_boundary

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
    scoring_radius_sq: float,
    cutoff_voxels: float,
) -> float:
    """Deposit one track's Gaussian into both carrier arrays.

    Returns the charge that landed inside the scored region, so the caller can
    accumulate the injected total without a second pass.

    `(x, y)` is in fractional voxel units. `h2` is the voxel area and `b2` the
    squared Gaussian track radius, both in cm^2, so `r^2 * h2 / b2` is the
    dimensionless exponent. `gaussian_factor` is `N0 / (pi b^2)`, the peak
    density of one track.

    See the module docstring for the three identities used here; concretely,
    per track this costs `2w` exponentials, `w^2` multiplications and
    `w^2 * no_z` array additions per species.
    """
    # Stencil bounding box, clipped to the grid. ceil/floor rather than round
    # so the box contains the cutoff circle rather than cutting inside it.
    i_lo = max(int(ceil(x - cutoff_voxels)), 0)
    i_hi = min(int(floor(x + cutoff_voxels)) + 1, no_xy)
    j_lo = max(int(ceil(y - cutoff_voxels)), 0)
    j_hi = min(int(floor(y + cutoff_voxels)) + 1, no_xy)

    # The two 1D factors of the separable Gaussian, over the stencil only.
    # Their outer product reconstructs the 2D profile exactly.
    gauss_i = np.empty(i_hi - i_lo)
    for i in range(i_lo, i_hi):
        gauss_i[i - i_lo] = exp(-((i - x) ** 2) * h2 / b2)
    gauss_j = np.empty(j_hi - j_lo)
    for j in range(j_lo, j_hi):
        gauss_j[j - j_lo] = exp(-((j - y) ** 2) * h2 / b2)

    k_lo = no_z_electrode
    k_hi = no_z_electrode + no_z

    inserted = 0.0
    for i in range(i_lo, i_hi):
        di_sq = (i - mid_xy) ** 2  # hoisted: the radius test does not depend on j
        gi = gauss_i[i - i_lo]
        for j in range(j_lo, j_hi):
            # One multiplication reconstructs the 2D Gaussian at (i, j).
            ion_density = gaussian_factor * gi * gauss_j[j - j_lo]
            # The same value goes into every gap layer -- the track is a line.
            for k in range(k_lo, k_hi):
                positive_array[i, j, k] += ion_density
                negative_array[i, j, k] += ion_density
            # Scored columns contribute no_z identical layers, hence the factor
            # rather than a sum.
            if di_sq + (j - mid_xy) ** 2 < scoring_radius_sq:
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
    scoring_radius_sq: float,
    p_lat: float,
    p_zm: float,
    p_zp: float,
    p_cen: float,
    n_lat: float,
    n_zm: float,
    n_zp: float,
    n_cen: float,
    alpha_dt: float,
) -> float:
    """Advance both carrier densities by one time step.

    Serial twin of
    :func:`~pulsed_ion_chamber.solver_numba_parallel._lax_wendroff_step_numba_parallel`
    -- see that function's docstring for the stencil weights, the sign
    convention that makes the two species drift in opposite directions, and
    worked coefficient values. The only difference here is the loop structure:
    plain nested `for` instead of a flattened `prange`.

    Returns ``(recombined, total_positive, total_negative)`` summed over the
    scored region; the carrier totals come free because the updated values are
    already in registers.
    """
    recombined = 0.0
    total_positive = 0.0
    total_negative = 0.0
    for i in range(1, no_xy - 1):
        di_sq = (i - mid_xy) ** 2
        for j in range(1, no_xy - 1):
            inside = di_sq + (j - mid_xy) ** 2 < scoring_radius_sq

            for k in range(1, no_z_with_buffer - 1):
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
                p_out = p_new - recomb
                n_out = n_new - recomb
                positive_next[i, j, k] = p_out
                negative_next[i, j, k] = n_out

                if inside and no_z_electrode < k < (no_z + no_z_electrode):
                    recombined += recomb
                    total_positive += p_out
                    total_negative += n_out

    return recombined, total_positive, total_negative


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
    _insert_track_numba(p, n, 2.0, 2.0, 4, 1, 1, 1.0, 1.0, 1.0, 2, 1.0, 4.0)
    _lax_wendroff_step_numba(
        p, n, p_next, n_next, 4, 4, 1, 1, 2, 1.0, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97, 1e-9
    )


def run_simulation_numba(
    config: SimulationConfig, rng: Optional[np.random.Generator] = None, progress: bool = True
) -> Result:
    """Simulate a pulse train and return the full run record.

    Deposits tracks one at a time (see module docstring). For dense pulses --
    hundreds or thousands of tracks per time step -- prefer
    :func:`~pulsed_ion_chamber.solver_numba_parallel.run_simulation_numba_parallel`,
    which batches the deposition; the two agree to 1e-9.
    """
    rng = rng if rng is not None else np.random.default_rng(config.seed)
    schedule = build_track_schedule(config, rng)

    # Two arrays per species: the sweep reads `*_array` and writes `*_next`,
    # so no voxel update can observe a half-updated neighbour.
    shape = (config.no_xy, config.no_xy, config.no_z_with_buffer)
    positive_array: FloatArray3D = np.zeros(shape)
    negative_array: FloatArray3D = np.zeros(shape)
    positive_next: FloatArray3D = np.zeros(shape)
    negative_next: FloatArray3D = np.zeros(shape)

    (p_lat, p_zm, p_zp, p_cen), (n_lat, n_zm, n_zp, n_cen) = config.scheme_coefficients()
    alpha_dt = RECOMBINATION_ALPHA_CM3_S * config.dt

    h2 = config.unit_length_cm**2
    b2 = config.track_radius_cm**2

    no_initialised = 0.0
    no_recombined = 0.0
    f_t: FloatArray1D = np.ones(config.total_time_steps)
    diagnostics = Diagnostics(config)
    report_every = max(1, config.total_time_steps // 20)

    for step in range(config.total_time_steps):
        injected_this_step = 0.0
        for _ in range(schedule[step]):
            x, y = sample_xy_inside_cylinder(rng, config.mid_xy, config.sampling_radius, config.no_xy)
            diagnostics.count_track(x, y)
            injected_this_step += _insert_track_numba(
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
                config.scoring_radius_sq,
                config.track_cutoff_voxels,
            )

        no_initialised += injected_this_step
        recombined, total_p, total_n = _lax_wendroff_step_numba(
            positive_array,
            negative_array,
            positive_next,
            negative_next,
            config.no_xy,
            config.no_z_with_buffer,
            config.no_z_electrode,
            config.no_z,
            config.mid_xy,
            config.scoring_radius_sq,
            p_lat,
            p_zm,
            p_zp,
            p_cen,
            n_lat,
            n_zm,
            n_zp,
            n_cen,
            alpha_dt,
        )
        no_recombined += recombined
        diagnostics.record(step, injected_this_step, recombined, total_p, total_n)

        # Interior only: the sweep never writes the outer shell, so `*_next`
        # has no valid boundary to carry over. apply_lateral_boundary then
        # sets that shell according to the configured wall condition.
        positive_array[1:-1, 1:-1, 1:-1] = positive_next[1:-1, 1:-1, 1:-1]
        negative_array[1:-1, 1:-1, 1:-1] = negative_next[1:-1, 1:-1, 1:-1]
        apply_lateral_boundary(positive_array, config.lateral_boundary)
        apply_lateral_boundary(negative_array, config.lateral_boundary)

        f_t[step] = 1.0 if no_initialised == 0.0 else (no_initialised - no_recombined) / no_initialised

        if progress and step % report_every == 0:
            print(f"  step {step + 1}/{config.total_time_steps}  f = {f_t[step]:.4f}")

    time_s: FloatArray1D = (np.arange(config.total_time_steps) + 1) * config.dt
    ks = 1.0 / f_t[-1]
    return diagnostics.build_result(config, time_s, f_t, ks, positive_array, negative_array)
