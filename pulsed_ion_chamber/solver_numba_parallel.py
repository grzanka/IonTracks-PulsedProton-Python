"""Batched, multi-core backend: same physics as solver_numba.py, restructured
so that both hot loops have real parallel work and the per-track cost no longer
carries the length of the gap.

Read docs/ALGORITHM.md for the full picture; this docstring covers what is
specific to *this* file.

Two structural differences from solver_numba.py
-----------------------------------------------

**1. Deposition is batched per time step.** solver_numba.py deposits one track
by writing its 2D Gaussian into every one of `no_z` z-layers -- the track runs
the length of the gap, so the profile is z-independent and simply repeats. That
inner k-loop is repeated once per track. But every track arriving in the *same*
time step writes into the *same* z-layers, so the repetition is pure waste: with
13 900 tracks in a step (the full-electrode scenario), each column of 200 voxels
is traversed 13 900 times, adding one number each visit.

Here a step's tracks are first summed into one 2D array, and only then
broadcast down z, once:

    per step, m tracks, stencil w, gap no_z
    solver_numba.py :  m * w^2 * no_z
    this file       :  m * w^2  +  no_xy^2 * no_z

Exact, not approximate: deposition is pure accumulation and addition is
associative, so sum-then-broadcast adds the same numbers as
broadcast-each-then-sum. Only the summation *order* changes, which is why the
cross-backend tests use a relative tolerance rather than demanding bit equality.

**2. Both kernels parallelise over a flattened (i, j) index**, not the outer `i`
loop alone. `i` alone offers only `no_xy` work items -- 22 on the reference
grid -- which caps useful parallelism at a couple of dozen threads no matter
how many are available. Flattening gives `no_xy**2` items.

Why none of the parallel writes need locking
--------------------------------------------

Every `prange` iteration owns a disjoint slice of whatever it writes:

* `_accumulate_track_density_numba_parallel` iterates over rows `i` and writes
  only `total_density[i, :]`.
* `_broadcast_density_numba_parallel` and `_lax_wendroff_step_numba_parallel`
  iterate over `(i, j)` and write only the `[i, j, :]` column.

The Lax-Wendroff sweep additionally reads only the *current* arrays and writes
only the *next* ones, so a thread can never observe a neighbour that another
thread is midway through updating. The scalar accumulators are ordinary `+=`
reductions, which Numba privatises per thread; their result can differ from the
serial order in the last few ULPs because float addition is not associative.

What does not parallelise: the time-step loop itself. Step N reads what step
N-1 wrote.

Threads are usually not the answer
----------------------------------

Both hot loops are memory-bandwidth-bound -- they stream the whole grid. On a
laptop-class machine a single core already reaches ~21 GB/s of a ~29 GB/s
practical ceiling, so the sweep saturates at about 1.7x on two threads and gets
*worse* beyond that. Measure before assuming; `num_threads` exists so callers
can pick the measured sweet spot rather than "all of them". See
docs/PERFORMANCE.md for the curve and for why independent replicas beat threads
for parameter sweeps.
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
from pulsed_ion_chamber.resources import clamp_thread_count
from pulsed_ion_chamber.state import Diagnostics, Result, apply_lateral_boundary

FloatArray3D = npt.NDArray[np.float64]
FloatArray1D = npt.NDArray[np.float64]
FloatArray2D = npt.NDArray[np.float64]
IntArray1D = npt.NDArray[np.int64]


@numba.njit(cache=True)
def _build_row_index(lo_i: IntArray1D, hi_i: IntArray1D, no_xy: int):
    """Bucket this step's tracks by the grid rows their stencil covers.

    Returns a CSR-style pair: ``track_ids[offsets[i] : offsets[i+1]]`` lists
    exactly the tracks whose deposition footprint touches row ``i``.

    Why this exists: once deposition is truncated to a stencil, a track touches
    only ~28 of the grid's rows. The accumulation kernel below parallelises
    over rows, and without an index each row would have to test every track --
    an ``O(no_xy * m)`` scan. On the full-electrode grid that is
    536 x 13 900 = 7.4 million tests per step, comparable to the real work and
    growing with the grid, which would undo the point of the stencil. Building
    the index is ``O(m * w)``, the same order as the work it feeds.

    Built with a counting sort in two passes -- count per row, prefix-sum,
    then scatter -- rather than an argsort, because the keys are small dense
    integers and this is O(n) with no comparisons.
    """
    n_tracks = lo_i.shape[0]

    # Pass 1: how many tracks touch each row. offsets[i+1] is used as the
    # counter for row i so that the prefix sum below lands in the right place.
    offsets = np.zeros(no_xy + 1, dtype=np.int64)
    for t in range(n_tracks):
        for i in range(lo_i[t], hi_i[t]):
            offsets[i + 1] += 1

    # Prefix sum turns counts into start positions.
    for i in range(no_xy):
        offsets[i + 1] += offsets[i]

    # Pass 2: scatter track ids into their rows' slots. `cursor` tracks the
    # next free slot per row and is consumed as we go.
    cursor = offsets[:no_xy].copy()
    track_ids = np.empty(offsets[no_xy], dtype=np.int64)
    for t in range(n_tracks):
        for i in range(lo_i[t], hi_i[t]):
            track_ids[cursor[i]] = t
            cursor[i] += 1
    return offsets, track_ids


@numba.njit(parallel=True, cache=True)
def _accumulate_track_density_numba_parallel(
    total_density: FloatArray2D,
    gauss_i: FloatArray2D,  # (n_tracks, width): gauss_i[t, m] applies at grid row lo_i[t] + m
    gauss_j: FloatArray2D,  # same, for columns
    lo_i: IntArray1D,
    lo_j: IntArray1D,
    hi_j: IntArray1D,
    offsets: IntArray1D,
    track_ids: IntArray1D,
    no_xy: int,
) -> None:
    """Phase 1 of batched deposition: sum every track's 2D Gaussian into one
    ``(no_xy, no_xy)`` array.

    This is where the separable-Gaussian identity pays off. The 2D profile

        exp(-((i-x)^2 + (j-y)^2) h^2/b^2)

    factors exactly into ``gauss_i[t, i] * gauss_j[t, j]`` because
    ``e^(a+b) = e^a e^b``. The two 1D factors were precomputed once per track
    (2w exponentials); reconstructing the 2D profile here costs w^2 plain
    multiplications and no further calls to ``exp``. On a 28-voxel stencil that
    is 56 exponentials instead of 784.

    Parallel over rows: iteration ``i`` writes only ``total_density[i, :]``, so
    even though many tracks contribute to the same row, no two iterations touch
    the same element. ``offsets``/``track_ids`` supply exactly the tracks that
    reach row ``i``, so no track is examined for a row it cannot affect.
    """
    for i in prange(no_xy):
        for pos in range(offsets[i], offsets[i + 1]):
            t = track_ids[pos]
            gi = gauss_i[t, i - lo_i[t]]  # this track's row factor, once per row
            base_j = lo_j[t]
            for j in range(base_j, hi_j[t]):
                total_density[i, j] += gi * gauss_j[t, j - base_j]


@numba.njit(parallel=True, cache=True)
def _broadcast_density_numba_parallel(
    positive_array: FloatArray3D,
    negative_array: FloatArray3D,
    total_density: FloatArray2D,
    no_xy: int,
    no_z: int,
    no_z_electrode: int,
    gaussian_factor: float,
    mid_xy: int,
    scoring_radius_sq: float,
) -> float:
    """Phase 2 of batched deposition: push the step's summed 2D density into
    every z-layer of the gap, and return the charge that landed in the scored
    region.

    This is the ``O(no_z)`` part, and doing it once per *step* rather than once
    per *track* is the whole point of batching. A track is a line through the
    full gap at constant LET, so its cross-section is identical in every layer
    -- there is nothing to recompute, only to copy.

    ``gaussian_factor`` is ``N0 / (pi b^2)``, the peak density of one track; it
    is applied here rather than inside the Gaussian factors so that phase 1
    stays a pure sum of dimensionless shapes.

    Columns no track reached are skipped outright. That costs nothing when the
    grid is saturated and saves the whole broadcast on a sparse one.
    """
    k_lo = no_z_electrode
    k_hi = no_z_electrode + no_z
    inserted = 0.0
    for idx in prange(no_xy * no_xy):
        i = idx // no_xy
        j = idx % no_xy
        density = total_density[i, j]
        if density == 0.0:
            continue
        density *= gaussian_factor
        for k in range(k_lo, k_hi):
            positive_array[i, j, k] += density
            negative_array[i, j, k] += density
        # Injected charge is scored on the same voxel set as recombination, so
        # the two are directly comparable and the voxel volume cancels in f.
        # The column contributes no_z identical layers, hence the factor.
        if (i - mid_xy) ** 2 + (j - mid_xy) ** 2 < scoring_radius_sq:
            inserted += density * no_z
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
) -> tuple:
    """Advance both carrier densities by one time step.

    Solves, for each species,

        dn/dt = D grad^2 n  -+  mu E dn/dz  -  alpha n+ n-

    with an explicit Lax-Wendroff scheme on the 7-point stencil (the voxel and
    its six face neighbours). Drift is along z only, so the four transverse
    neighbours carry diffusion alone.

    Returns ``(recombined, total_positive, total_negative)``, each summed over
    the scored region. The carrier totals are accumulated here rather than
    reduced afterwards because the updated values are already in registers; a
    separate pass would re-read the entire grid every step for nothing.

    Stencil weights
    ---------------
    Precomputed by ``SimulationConfig.scheme_coefficients()`` as
    ``(lateral, z_minus, z_plus, centre)`` per species, from that species\' own
    diffusion number ``s = D dt/h^2`` and Courant number ``c = mu E dt/h``::

        lateral = s
        z_minus = s + c(c+1)/2        weight on the upwind neighbour
        z_plus  = s + c(c-1)/2        weight on the downwind neighbour
        centre  = 1 - c^2 - 6s

    The ``c^2/2`` part of the z weights is the Lax-Wendroff correction: it is
    what makes the scheme second-order in time and, at c = 1, exact for pure
    advection. ``dt`` is chosen so ``6s + c^2 <= 1``, which is what keeps
    ``centre`` non-negative and the scheme stable.

    The two species get *different* weights whenever the Kanai carriers are
    resolved separately (mu+ = 1.36 vs mu- = 2.10 cm^2/Vs). They also drift in
    opposite directions, expressed by applying the negative species\' weights to
    the *swapped* neighbours -- see below.

    Concrete numbers, reference grid (10 um voxels, 1500 V/cm, two species)::

        positive: s = 0.00858, c = 0.621  ->  z_minus 0.512, z_plus -0.109, centre 0.562
        negative: s = 0.01324, c = 0.959  ->  z_minus 0.952, z_plus -0.007, centre 0.001

    Note ``z_plus`` is negative: the downwind neighbour is *subtracted*. That is
    normal for Lax-Wendroff at high Courant number, and is why the scheme is not
    positivity-preserving -- densities can dip slightly below zero in the last
    digits. Harmless here, but worth knowing before trusting a raw density.
    """
    recombined = 0.0
    total_positive = 0.0
    total_negative = 0.0

    # Parallelise over a flattened (i, j): `i` alone would offer only no_xy
    # work items. Each iteration writes exclusively to the [i, j, :] column of
    # the *next* arrays, so iterations are independent (see module docstring).
    n_inner = no_xy - 2
    for idx in prange(n_inner * n_inner):
        i = 1 + idx // n_inner
        j = 1 + idx % n_inner

        # Whether this column is scored depends on (i, j) only, never on k, so
        # the radius test is hoisted out of the k loop -- one comparison per
        # column instead of one per voxel. Squared distance, so no sqrt.
        di_sq = (i - mid_xy) ** 2
        inside = di_sq + (j - mid_xy) ** 2 < scoring_radius_sq

        # k innermost: arrays are C-contiguous with k fastest-varying, so this
        # walks memory sequentially. The [i+-1, j, k] and [i, j+-1, k]
        # neighbours are separate streams, but each is itself sequential in k,
        # which hardware prefetchers handle well.
        for k in range(1, no_z_with_buffer - 1):
            p = positive_array[i, j, k]
            n = negative_array[i, j, k]

            p_new = (
                p_zm * positive_array[i, j, k - 1]  # upwind: positive ions drift towards +z
                + p_zp * positive_array[i, j, k + 1]
                + p_lat * positive_array[i, j - 1, k]  # transverse: diffusion only,
                + p_lat * positive_array[i, j + 1, k]  # so all four share one weight
                + p_lat * positive_array[i - 1, j, k]
                + p_lat * positive_array[i + 1, j, k]
                + p_cen * p
            )
            # Negative ions drift the other way, so their upwind neighbour is
            # the one at k+1. Same weight formulas, swapped neighbours -- that
            # single swap is the entire difference in transport direction.
            n_new = (
                n_zm * negative_array[i, j, k + 1]
                + n_zp * negative_array[i, j, k - 1]
                + n_lat * negative_array[i, j - 1, k]
                + n_lat * negative_array[i, j + 1, k]
                + n_lat * negative_array[i - 1, j, k]
                + n_lat * negative_array[i + 1, j, k]
                + n_cen * n
            )

            # Recombination is operator-split from transport and evaluated from
            # the *old* densities, so it is a lagged explicit sink. Both species
            # lose the same number of pairs -- one positive annihilates with one
            # negative. Per-step loss is ~1e-3 of the local population, and the
            # resulting splitting error measures below 0.05 % on k_s.
            recomb = alpha_dt * p * n
            p_out = p_new - recomb
            n_out = n_new - recomb
            positive_next[i, j, k] = p_out
            negative_next[i, j, k] = n_out

            # Score only inside the disc *and* between the electrodes: charge
            # that has drifted into the electrode buffer has been collected and
            # must not keep contributing.
            if inside and no_z_electrode < k < (no_z + no_z_electrode):
                recombined += recomb
                total_positive += p_out
                total_negative += n_out

    return recombined, total_positive, total_negative


def _precompute_track_gaussians(
    xs: FloatArray1D, ys: FloatArray1D, no_xy: int, h2: float, b2: float, cutoff_voxels: float
):
    """1D Gaussian factors for a step's tracks, over their stencils only.

    Returns ``(gauss_i, gauss_j, lo_i, hi_i, lo_j, hi_j)``. Row ``t`` of
    ``gauss_i`` holds ``exp(-((lo_i[t] + m - xs[t])^2) h^2/b^2)`` for
    ``m in [0, width)``; the kernels read only up to ``hi_i[t] - lo_i[t]``.

    Two things are going on:

    * **Separability.** The 2D Gaussian is the outer product of these two 1D
      arrays (see ``_accumulate_track_density_numba_parallel``), so this costs
      ``O(m * width)`` calls to ``exp`` instead of ``O(m * width^2)``.
    * **Truncation.** ``lo``/``hi`` are the stencil bounds, clipped to the
      grid. Beyond the cutoff the Gaussian is below ``exp(-50)`` of the peak at
      the default 10 sigma -- unrepresentable relative to the running sum, so
      dropping it is lossless.

    A rectangular array is used even though the per-track ranges are ragged
    (clipping at the grid edge shortens some by one). The unused trailing
    column is never read, and a rectangular layout keeps the jitted kernels
    free of ragged indexing.
    """
    # ceil/floor rather than round, so the box always *contains* the cutoff
    # circle rather than cutting inside it.
    lo_i = np.maximum(np.ceil(xs - cutoff_voxels), 0.0).astype(np.int64)
    hi_i = np.minimum(np.floor(xs + cutoff_voxels) + 1.0, float(no_xy)).astype(np.int64)
    lo_j = np.maximum(np.ceil(ys - cutoff_voxels), 0.0).astype(np.int64)
    hi_j = np.minimum(np.floor(ys + cutoff_voxels) + 1.0, float(no_xy)).astype(np.int64)
    width = max(1, int(max((hi_i - lo_i).max(), (hi_j - lo_j).max())))
    offsets = np.arange(width)
    # Broadcast to (n_tracks, width): each row is one track's 1D profile,
    # evaluated at grid coordinates lo + offset.
    gauss_i = np.exp(-((lo_i[:, None] + offsets[None, :] - xs[:, None]) ** 2) * (h2 / b2))
    gauss_j = np.exp(-((lo_j[:, None] + offsets[None, :] - ys[:, None]) ** 2) * (h2 / b2))
    return gauss_i, gauss_j, lo_i, hi_i, lo_j, hi_j


def warmup_parallel() -> None:
    """Compile every jitted kernel on trivial arrays, so a later timed call
    measures execution and not compilation.

    Deliberately does not go through ``run_simulation_numba_parallel``: it
    calls the kernels directly with hand-made, always-valid arguments, so it
    cannot hang the way a degenerate config could.
    """
    shape = (4, 4, 4)
    p, n, p_next, n_next = (np.zeros(shape) for _ in range(4))
    total = np.zeros((4, 4))
    gi, gj, li, hi, lj, hj = _precompute_track_gaussians(
        np.array([2.0]), np.array([2.0]), 4, 1.0, 1.0, 4.0
    )
    offsets, track_ids = _build_row_index(li, hi, 4)
    _accumulate_track_density_numba_parallel(total, gi, gj, li, lj, hj, offsets, track_ids, 4)
    _broadcast_density_numba_parallel(p, n, total, 4, 1, 1, 1.0, 2, 1.0)
    _lax_wendroff_step_numba_parallel(
        p, n, p_next, n_next, 4, 4, 1, 1, 2, 1.0, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97, 1e-9
    )


def run_simulation_numba_parallel(
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
    progress: bool = True,
    num_threads: Optional[int] = None,
) -> Result:
    """Simulate a pulse train and return the full run record.

    Same physics and same RNG stream as
    :func:`~pulsed_ion_chamber.solver_numba.run_simulation_numba`; the
    difference is that a whole time step's tracks are deposited in one pass
    (see the module docstring) and both hot loops run under ``prange``.

    ``num_threads`` is clamped to what the process may actually use. More is
    frequently *not* better: both kernels are memory-bandwidth-bound, so the
    sweep saturates within a couple of threads on a laptop. Measure before
    choosing -- docs/PERFORMANCE.md has the curve.
    """
    if num_threads is not None:
        # Clamped to the process CPU affinity mask and Numba's own maximum, so
        # an over-ambitious request warns and degrades instead of raising.
        numba.set_num_threads(clamp_thread_count(num_threads))

    rng = rng if rng is not None else np.random.default_rng(config.seed)
    schedule = build_track_schedule(config, rng)

    # Two arrays per species: the sweep reads `*_array` and writes `*_next`,
    # so every voxel update is independent of every other. That double buffer
    # is what makes the loop trivially parallel.
    shape = (config.no_xy, config.no_xy, config.no_z_with_buffer)
    positive_array: FloatArray3D = np.zeros(shape)
    negative_array: FloatArray3D = np.zeros(shape)
    positive_next: FloatArray3D = np.zeros(shape)
    negative_next: FloatArray3D = np.zeros(shape)

    (p_lat, p_zm, p_zp, p_cen), (n_lat, n_zm, n_zp, n_cen) = config.scheme_coefficients()
    alpha_dt = RECOMBINATION_ALPHA_CM3_S * config.dt

    h2 = config.unit_length_cm**2  # voxel area, converts index distance to cm^2
    b2 = config.track_radius_cm**2  # Gaussian track radius squared
    cutoff_voxels = config.track_cutoff_voxels
    # Scratch for the batched deposition, allocated once and zeroed per step.
    total_density: FloatArray2D = np.zeros((config.no_xy, config.no_xy))

    no_initialised = 0.0
    no_recombined = 0.0
    f_t: FloatArray1D = np.ones(config.total_time_steps)
    diagnostics = Diagnostics(config)
    report_every = max(1, config.total_time_steps // 20)

    for step in range(config.total_time_steps):
        injected_this_step = 0.0
        n_tracks_this_step = schedule[step]
        if n_tracks_this_step > 0:
            xs = np.empty(n_tracks_this_step)
            ys = np.empty(n_tracks_this_step)
            for t in range(n_tracks_this_step):
                xs[t], ys[t] = sample_xy_inside_cylinder(rng, config.mid_xy, config.sampling_radius, config.no_xy)
            # Deposition in three stages: 1D Gaussian factors per track, a
            # row index so the accumulation kernel can parallelise safely, then
            # sum into 2D and broadcast down z once. See the module docstring.
            gauss_i, gauss_j, lo_i, hi_i, lo_j, hi_j = _precompute_track_gaussians(
                xs, ys, config.no_xy, h2, b2, cutoff_voxels
            )
            total_density[:] = 0.0
            offsets, track_ids = _build_row_index(lo_i, hi_i, config.no_xy)
            _accumulate_track_density_numba_parallel(
                total_density, gauss_i, gauss_j, lo_i, lo_j, hi_j, offsets, track_ids, config.no_xy
            )
            diagnostics.count_tracks(xs, ys)
            injected_this_step = _broadcast_density_numba_parallel(
                positive_array,
                negative_array,
                total_density,
                config.no_xy,
                config.no_z,
                config.no_z_electrode,
                config.Gaussian_factor,
                config.mid_xy,
                config.scoring_radius_sq,
            )

        no_initialised += injected_this_step
        recombined, total_p, total_n = _lax_wendroff_step_numba_parallel(
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

        # Copy the interior back rather than swapping references: the sweep
        # never writes the outer shell, so `*_next` has no valid boundary to
        # swap in. This is ~30 % of a large step and is the obvious target if
        # this ever needs to be faster (it would mean carrying the boundary
        # planes explicitly, which is O(X^2) rather than O(X^2 Z)).
        positive_array[1:-1, 1:-1, 1:-1] = positive_next[1:-1, 1:-1, 1:-1]
        negative_array[1:-1, 1:-1, 1:-1] = negative_next[1:-1, 1:-1, 1:-1]
        apply_lateral_boundary(positive_array, config.lateral_boundary)
        apply_lateral_boundary(negative_array, config.lateral_boundary)

        # Running collection efficiency. Both totals are density sums over the
        # same voxel set, so the voxel volume cancels and never appears.
        f_t[step] = 1.0 if no_initialised == 0.0 else (no_initialised - no_recombined) / no_initialised

        if progress and step % report_every == 0:
            print(f"  step {step + 1}/{config.total_time_steps}  f = {f_t[step]:.4f}")

    time_s: FloatArray1D = (np.arange(config.total_time_steps) + 1) * config.dt
    ks = 1.0 / f_t[-1]
    return diagnostics.build_result(config, time_s, f_t, ks, positive_array, negative_array)
