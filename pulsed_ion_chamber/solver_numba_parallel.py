"""Shared-memory multi-core backend: the same physics as solver_numba.py,
restructured for ~190-core scaling on one Helios node. Two changes on top
of solver_numba.py, in order of how much they actually matter:

1. **Track insertion is batched per time step, not called per track.**
   `_insert_track_numba` deposits one track by looping over the *entire*
   grid and, for every (i, j), writing the same density into every one of
   `no_z` z-layers (see its docstring: the track runs the length of the
   gap). That inner k-loop is O(no_z) work repeated once per track --
   but on a given time step, all of that step's tracks (hundreds, on the
   "converged" grid from the README) end up broadcasting into the *same*
   z-layers. Summing all of a step's per-(i,j) Gaussian contributions
   first, and only then doing the O(no_z) broadcast *once* for the
   combined density, turns O(n_tracks_this_step * no_xy^2 * no_z) into
   O(n_tracks_this_step * no_xy^2 + no_xy^2 * no_z). Measured on the
   converged-grid config (see the "Running on many cores" section of
   README.md), this alone is roughly two orders of magnitude fewer
   floating-point ops and, just as importantly, cuts the number of
   Numba parallel-region launches for track insertion from ~719,000
   (one per track) to ~2,000 (one per time step that has any tracks) --
   see point 2.
2. **Both hot loops parallelize over a *flattened* (i, j) index**, not
   just the outer `i` loop. `i` alone only ranges over `no_xy` values (56
   for the converged grid); handing `prange` just that loop caps the
   parallel granularity at 56 work items -- fine for a few dozen threads,
   but measured directly on this repository's 190-core target, using
   *more* threads than that made both kernels *slower*, not faster (most
   threads got zero `i` values and idled, while Numba's fork-join still
   paid full cross-NUMA synchronization cost for the region). Flattening
   `i`*`j` into one loop of length `no_xy**2` (3136 for that grid) gives
   every thread real work at any thread count up to a few thousand.

Even after both fixes, empirically the sweet spot on this machine (dual
AMD EPYC 9654, 8 NUMA domains, `omp` threading layer) was **not** 190
threads -- see README.md for the measured curve. Past roughly 64-96
threads, per-launch fork-join/barrier synchronization cost across NUMA
domains grows faster than the shrinking amount of work each thread gets,
and wall time gets *worse*. `run_simulation_numba_parallel`'s
`num_threads` argument exists precisely so callers can pick the measured
sweet spot instead of assuming "more cores = faster".

Why the parallel writes below need no locking: `_lax_wendroff_step_numba_parallel`
only *reads* the previous-step arrays and *writes* the next-step arrays;
different (i, j) iterations write disjoint `[i, j, :]` slices of the
*next* arrays only. `_insert_tracks_step_numba_parallel` writes
`positive_array`/`negative_array` in place, but again each (i, j)
iteration owns a disjoint `[i, j, :]` slice, and all of a step's tracks
are folded into one `total_density` per voxel *before* that slice is
touched, so there is no cross-track or cross-thread write race. The
scalar accumulators (`inserted`, `recombined`) are ordinary `prange`
reductions, which Numba detects automatically for `+=` (the result can
differ from the single-threaded value in the last few ULPs, since
float addition is not associative -- see the tolerance in
test_solver_numba_parallel.py).

What does NOT parallelize here: the time-step loop itself, because step
N's arrays depend on step N-1's. All the parallelism is *within* each
step, across voxels.
"""

from math import exp
from typing import Optional

import numba
import numpy as np
import numpy.typing as npt
from numba import prange

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import RECOMBINATION_ALPHA_CM3_S
from pulsed_ion_chamber.pulses import build_track_schedule, sample_xy_inside_cylinder
from pulsed_ion_chamber.solver import Result, apply_lateral_boundary

FloatArray3D = npt.NDArray[np.float64]
FloatArray1D = npt.NDArray[np.float64]
FloatArray2D = npt.NDArray[np.float64]


@numba.njit(parallel=True, cache=True)
def _insert_tracks_step_numba_parallel(
    positive_array: FloatArray3D,
    negative_array: FloatArray3D,
    gauss_i: FloatArray2D,  # shape (n_tracks, no_xy): gauss_i[t, i] for track t
    gauss_j: FloatArray2D,  # shape (n_tracks, no_xy): gauss_j[t, j] for track t
    no_xy: int,
    no_z: int,
    no_z_electrode: int,
    gaussian_factor: float,
    mid_xy: int,
    scoring_radius_sq: float,
) -> float:
    """Deposit ALL of one time step's tracks in a single pass over the grid.

    Same separable-Gaussian identity as _insert_track_numba (exp(a+b) =
    exp(a)*exp(b)), but summed over tracks *before* the O(no_z) broadcast
    into every z-layer, instead of once per track -- see module docstring.
    """
    n_tracks = gauss_i.shape[0]
    k_lo = no_z_electrode
    k_hi = no_z_electrode + no_z

    inserted = 0.0
    for idx in prange(no_xy * no_xy):
        i = idx // no_xy
        j = idx % no_xy
        di_sq = (i - mid_xy) ** 2

        total_density = 0.0
        for t in range(n_tracks):
            total_density += gauss_i[t, i] * gauss_j[t, j]
        total_density *= gaussian_factor

        for k in range(k_lo, k_hi):
            positive_array[i, j, k] += total_density
            negative_array[i, j, k] += total_density
        if di_sq + (j - mid_xy) ** 2 < scoring_radius_sq:
            inserted += total_density * no_z

    return inserted


@numba.njit(parallel=True, cache=True)
def _lax_wendroff_step_numba_parallel(
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
    """Same as _lax_wendroff_step_numba, parallel over the flattened
    (i, j) index (see module docstring for why not `i` alone). The p_*/n_*
    scalars are the per-species stencil weights from
    config.scheme_coefficients()."""
    recombined = 0.0
    n_inner = no_xy - 2
    for idx in prange(n_inner * n_inner):
        i = 1 + idx // n_inner
        j = 1 + idx % n_inner
        di_sq = (i - mid_xy) ** 2
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

            if inside and no_z_electrode < k < (no_z + no_z_electrode):
                recombined += recomb

    return recombined


def _precompute_track_gaussians(xs: FloatArray1D, ys: FloatArray1D, no_xy: int, h2: float, b2: float):
    """Plain-NumPy, vectorized: gauss_i[t, i] = exp(-((i-xs[t])^2)*h2/b2),
    same for gauss_j with ys -- one exp() per (track, grid-line) pair,
    matching the per-track separable-Gaussian cost of solver_numba.py."""
    coords = np.arange(no_xy)
    gauss_i = np.exp(-((coords[None, :] - xs[:, None]) ** 2) * (h2 / b2))
    gauss_j = np.exp(-((coords[None, :] - ys[:, None]) ** 2) * (h2 / b2))
    return gauss_i, gauss_j


def warmup_parallel() -> None:
    """Trigger Numba JIT compilation of both parallel kernels on minimal
    dummy arrays -- see solver_numba.warmup() for why."""
    shape = (4, 4, 4)
    p, n, p_next, n_next = (np.zeros(shape) for _ in range(4))
    gauss_i, gauss_j = _precompute_track_gaussians(np.array([2.0]), np.array([2.0]), 4, 1.0, 1.0)
    _insert_tracks_step_numba_parallel(p, n, gauss_i, gauss_j, 4, 1, 1, 1.0, 2, 1.0)
    _lax_wendroff_step_numba_parallel(
        p, n, p_next, n_next, 4, 4, 1, 1, 2, 1.0, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97, 1e-9
    )


def run_simulation_numba_parallel(
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
    progress: bool = True,
    num_threads: Optional[int] = None,
) -> Result:
    """Same algorithm as solver_numba.run_simulation_numba(), with both hot
    loops parallelized across CPU cores via Numba prange, and all of a time
    step's track insertions batched into one grid pass (see module
    docstring for why that -- not just adding more threads -- is what
    actually makes this scale).

    num_threads: if given, calls numba.set_num_threads(num_threads) before
    running. Make sure the process's CPU affinity mask actually contains
    that many cores -- see README.md's "Running on many cores" section --
    and see the same section for why the fastest choice on this repository's
    target machine was well under 190.
    """
    if num_threads is not None:
        numba.set_num_threads(num_threads)

    rng = rng if rng is not None else np.random.default_rng(config.seed)
    schedule = build_track_schedule(config, rng)

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
    report_every = max(1, config.total_time_steps // 20)

    for step in range(config.total_time_steps):
        n_tracks_this_step = schedule[step]
        if n_tracks_this_step > 0:
            xs = np.empty(n_tracks_this_step)
            ys = np.empty(n_tracks_this_step)
            for t in range(n_tracks_this_step):
                xs[t], ys[t] = sample_xy_inside_cylinder(rng, config.mid_xy, config.sampling_radius, config.no_xy)
            gauss_i, gauss_j = _precompute_track_gaussians(xs, ys, config.no_xy, h2, b2)
            no_initialised += _insert_tracks_step_numba_parallel(
                positive_array,
                negative_array,
                gauss_i,
                gauss_j,
                config.no_xy,
                config.no_z,
                config.no_z_electrode,
                config.Gaussian_factor,
                config.mid_xy,
                config.scoring_radius_sq,
            )

        no_recombined += _lax_wendroff_step_numba_parallel(
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

        positive_array[1:-1, 1:-1, 1:-1] = positive_next[1:-1, 1:-1, 1:-1]
        negative_array[1:-1, 1:-1, 1:-1] = negative_next[1:-1, 1:-1, 1:-1]
        apply_lateral_boundary(positive_array, config.lateral_boundary)
        apply_lateral_boundary(negative_array, config.lateral_boundary)

        f_t[step] = 1.0 if no_initialised == 0.0 else (no_initialised - no_recombined) / no_initialised

        if progress and step % report_every == 0:
            print(f"  step {step + 1}/{config.total_time_steps}  f = {f_t[step]:.4f}")

    time_s: FloatArray1D = (np.arange(config.total_time_steps) + 1) * config.dt
    ks = 1.0 / f_t[-1]
    return Result(config=config, time_s=time_s, f_t=f_t, ks=ks, positive_array=positive_array, negative_array=negative_array)
