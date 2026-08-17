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

Threads help exactly when the grid is big
-----------------------------------------

Both hot loops are memory-bandwidth-bound -- they stream the whole grid -- so
what threads buy is memory controllers, not arithmetic. That makes the answer
depend entirely on the size of the grid relative to the machine:

* **Laptop-class, or a grid that fits in cache.** A single core already reaches
  most of the available bandwidth (~21 of ~29 GB/s on a laptop), so the sweep
  saturates around 1.7x on two threads and gets *worse* beyond that.
* **A NUMA server and a grid larger than its aggregate L3.** One core gets a
  single core's share of bandwidth -- 9 GB/s of a node's ~900 GB/s on a dual
  EPYC 9654 -- and more threads buy more memory controllers. Measured 12.2x on
  the full-electrode grid at 32 threads (79 % of ideal still at 8), against
  1.7x on a laptop.

Two things had to be fixed before that second case worked, and both are the
kind of thing that shows up only above a few hundred MiB (docs/HELIOS.md has
the measurements):

**NUMA first touch.** ``np.zeros`` hands back untouched pages; Linux places
each one on the NUMA domain of the thread that first *writes* it. If the main
thread does that, all 1.9 GiB of carrier arrays lands behind one of eight
memory controllers and every worker reads across the fabric from it -- the
sweep then plateaus at 48 GB/s no matter how many threads are added.
``_first_touch_parallel`` writes the arrays through the same flattened-(i, j)
decomposition the kernels use, so each thread's pages are placed local to it,
and the same sweep reaches 230 GB/s.

**No serial phase.** Amdahl is unforgiving at 190 threads: the interior
copy-back this file used to do was ~144 ms of a 354 ms step, plain NumPy, and
would have capped the whole run at 2.5x however fast the kernels got. It is
gone -- see the swap in ``run_simulation_numba_parallel``.
"""

from math import exp
from time import perf_counter
from typing import Optional

import numba
import numpy as np
import numpy.typing as npt
from numba import prange

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import RECOMBINATION_ALPHA_CM3_S
from pulsed_ion_chamber.pulses import CylinderSampler, build_track_schedule
from pulsed_ion_chamber.resources import clamp_thread_count
from pulsed_ion_chamber.state import Diagnostics, Result, apply_lateral_boundary, carry_lateral_ring

FloatArray3D = npt.NDArray[np.float64]
FloatArray1D = npt.NDArray[np.float64]
FloatArray2D = npt.NDArray[np.float64]
IntArray1D = npt.NDArray[np.int64]


@numba.njit(parallel=True, cache=True)
def _first_touch_parallel(array: FloatArray3D) -> None:
    """Zero the array from the threads that will later own its pages.

    Not an optimisation of the zeroing -- ``np.zeros`` is already free, because
    the kernel hands back pages that are mapped but not physically placed.
    Placement happens on first *write*, on the NUMA domain of whichever thread
    does it, and it is permanent for the life of the allocation. So the thread
    that writes a page first decides which memory controller every later access
    to it goes through.

    Iterating the same flattened ``(i, j)`` index as the sweep and the
    broadcast means thread ``t`` touches the columns thread ``t`` will later
    read and write. On the full-electrode grid (1.9 GiB, 8 NUMA domains) this is
    worth 4.6x on the sweep -- 41 ms/step versus 8.9 ms/step. On a
    single-domain machine, or a grid that fits in cache, it does nothing.

    Only correct while Numba's ``prange`` uses a static, contiguous schedule,
    which it does by default (``NUMBA_PARALLEL_CHUNKSIZE`` unset). A dynamic
    schedule would still be correct, just no longer NUMA-local.
    """
    no_i, no_j, no_k = array.shape
    for idx in prange(no_i * no_j):
        i = idx // no_j
        j = idx % no_j
        for k in range(no_k):
            array[i, j, k] = 0.0


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
        # Restricted to the interior the Lax-Wendroff sweep actually visits
        # (1 <= i,j <= no_xy-2): under "full_grid" scoring, scoring_radius_sq
        # alone always passes, and without this bound the never-swept outer
        # ring would be counted as injected charge that can structurally
        # never recombine (see issue #19 P2).
        if (
            1 <= i <= no_xy - 2
            and 1 <= j <= no_xy - 2
            and (i - mid_xy) ** 2 + (j - mid_xy) ** 2 < scoring_radius_sq
        ):
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
            # must not keep contributing. k == no_z_electrode is the *first*
            # gap layer -- deposition writes it (k_lo == no_z_electrode) and it
            # must be scored too, or injected and recombined charge integrate
            # different voxel sets (see issue #19 P1).
            if inside and no_z_electrode <= k < (no_z + no_z_electrode):
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
    _first_touch_parallel(p)
    gi, gj, li, hi, lj, hj = _precompute_track_gaussians(
        np.array([2.0]), np.array([2.0]), 4, 1.0, 1.0, 4.0
    )
    offsets, track_ids = _build_row_index(li, hi, 4)
    _accumulate_track_density_numba_parallel(total, gi, gj, li, lj, hj, offsets, track_ids, 4)
    # mid_xy (the 2.0 below) is a float in production (config.py anchors it
    # to outer_radius = no_xy / 2.0, see issue #19 P6) -- matching that type
    # here is what makes this a real warmup rather than pre-compiling a
    # specialization run_simulation_numba_parallel never actually calls.
    _broadcast_density_numba_parallel(p, n, total, 4, 1, 1, 1.0, 2.0, 1.0)
    _lax_wendroff_step_numba_parallel(
        p, n, p_next, n_next, 4, 4, 1, 1, 2.0, 1.0, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97, 1e-9
    )


class _PhaseTimer:
    """Optional per-phase wall-clock accounting for one run.

    Off by default and free when off. On, it costs two ``perf_counter`` calls
    per phase per step -- ~50 ns against phases measured in milliseconds.

    It exists because on a large grid the interesting question stopped being
    "how long did it take" and became "which phase is still serial": once the
    threaded kernels drop to a few ms, anything left in NumPy or in a Python
    loop is what sets the wall time, and only a breakdown shows which.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.totals: dict[str, float] = {}
        self._t0 = 0.0

    def start(self) -> None:
        if self.enabled:
            self._t0 = perf_counter()

    def stop(self, name: str) -> None:
        if self.enabled:
            self.totals[name] = self.totals.get(name, 0.0) + (perf_counter() - self._t0)

    def report(self, total_time_steps: int) -> str:
        if not self.enabled or not self.totals:
            return ""
        total = sum(self.totals.values())
        lines = [f"{'phase':>22} {'total_s':>9} {'ms/step':>9} {'share':>7}"]
        for name, seconds in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            lines.append(
                f"{name:>22} {seconds:>9.2f} {seconds / total_time_steps * 1e3:>9.2f} "
                f"{seconds / total:>6.1%}"
            )
        lines.append(f"{'accounted':>22} {total:>9.2f} {total / total_time_steps * 1e3:>9.2f}")
        return "\n".join(lines)


def run_simulation_numba_parallel(
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
    progress: bool = True,
    num_threads: Optional[int] = None,
    phase_timing: bool = False,
    max_wall_s: Optional[float] = None,
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

    ``max_wall_s``, if given, stops the loop once that many seconds have
    elapsed instead of running all ``config.total_time_steps`` -- see
    :func:`~pulsed_ion_chamber.solver_numba.run_simulation_numba` for what
    that does to the returned ``Result`` and the ``steps_completed`` /
    ``loop_elapsed_s`` attributes it adds.
    """
    if num_threads is not None:
        # Clamped to the process CPU affinity mask and Numba's own maximum, so
        # an over-ambitious request warns and degrades instead of raising.
        numba.set_num_threads(clamp_thread_count(num_threads))

    rng = rng if rng is not None else np.random.default_rng(config.seed)
    schedule = build_track_schedule(config, rng)
    # Draws the identical stream of doubles the per-track sampler would, in
    # blocks -- 166x faster, and 24.6 M tracks of a full-electrode run would
    # otherwise spend 110 s in a Python loop. See pulses.CylinderSampler.
    # Axis placement draws nothing, so the sampler is not built at all rather
    # than built and bypassed -- that keeps a "random" run's RNG stream exactly
    # what it was before this branch existed.
    on_axis = config.track_placement == "axis"
    sampler = (
        None if on_axis else CylinderSampler(rng, config.mid_xy, config.sampling_radius, config.no_xy)
    )
    stencil = None if config.track_stencil is None else config.track_stencil.density_cm3

    # Two arrays per species: the sweep reads `*_array` and writes `*_next`,
    # so every voxel update is independent of every other. That double buffer
    # is what makes the loop trivially parallel.
    #
    # `np.empty` + a threaded zero-fill, not `np.zeros`: the fill is what places
    # the pages on NUMA domains, and it must be done by the worker threads
    # rather than by this one. See `_first_touch_parallel`.
    shape = (config.no_xy, config.no_xy, config.no_z_with_buffer)
    positive_array: FloatArray3D = np.empty(shape)
    negative_array: FloatArray3D = np.empty(shape)
    positive_next: FloatArray3D = np.empty(shape)
    negative_next: FloatArray3D = np.empty(shape)
    for array in (positive_array, negative_array, positive_next, negative_next):
        _first_touch_parallel(array)

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
    # Which of the two swap strategies below applies; resolved once, not per
    # step, so the hot loop carries no string comparison.
    reflecting_wall = config.lateral_boundary == "reflecting"
    timer = _PhaseTimer(phase_timing)
    loop_t0 = perf_counter()
    steps_completed = 0

    for step in range(config.total_time_steps):
        injected_this_step = 0.0
        n_tracks_this_step = schedule[step]
        if n_tracks_this_step > 0:
            timer.start()
            if on_axis:
                xs = np.full(n_tracks_this_step, config.mid_xy)
                ys = xs
            else:
                xs, ys = sampler.sample(n_tracks_this_step)
            timer.stop("sample_xy")
            if stencil is not None:
                # The tabulated-RDD path needs none of the Gaussian machinery:
                # this step's 2D density is the prebuilt stencil times the track
                # count, because every track sits on the same centre line. Phase
                # 2 below is then shared verbatim with the Gaussian path, which
                # is what keeps the two backends and the two track models from
                # drifting apart.
                timer.start()
                np.multiply(stencil, float(n_tracks_this_step), out=total_density)
                timer.stop("accumulate")
            else:
                # Deposition in three stages: 1D Gaussian factors per track, a
                # row index so the accumulation kernel can parallelise safely, then
                # sum into 2D and broadcast down z once. See the module docstring.
                timer.start()
                gauss_i, gauss_j, lo_i, hi_i, lo_j, hi_j = _precompute_track_gaussians(
                    xs, ys, config.no_xy, h2, b2, cutoff_voxels
                )
                timer.stop("track_gaussians")
                timer.start()
                total_density[:] = 0.0
                offsets, track_ids = _build_row_index(lo_i, hi_i, config.no_xy)
                timer.stop("row_index")
                timer.start()
                _accumulate_track_density_numba_parallel(
                    total_density, gauss_i, gauss_j, lo_i, lo_j, hi_j, offsets, track_ids, config.no_xy
                )
                timer.stop("accumulate")
            timer.start()
            diagnostics.count_tracks(xs, ys)
            timer.stop("count_tracks")
            timer.start()
            injected_this_step = _broadcast_density_numba_parallel(
                positive_array,
                negative_array,
                total_density,
                config.no_xy,
                config.no_z,
                config.no_z_electrode,
                # The RDD stencil already carries absolute density; only the
                # Gaussian path factors its peak out of phase 1 and back in here.
                1.0 if stencil is not None else config.Gaussian_factor,
                config.mid_xy,
                config.scoring_radius_sq,
            )
            timer.stop("broadcast")

        no_initialised += injected_this_step
        timer.start()
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
        timer.stop("lax_wendroff")
        no_recombined += recombined
        diagnostics.record(step, injected_this_step, recombined, total_p, total_n)
        timer.start()

        # Swap the buffers instead of copying the interior back. The copy was
        # 144 ms of a 354 ms full-electrode step, in serial NumPy, and computed
        # nothing; on 190 threads it alone would have capped the run at 2.5x.
        #
        # What the copy was really for is the outer shell: the sweep writes only
        # the interior, so after a swap the new current buffer's ring is stale.
        # Both wall conditions can be served without touching the interior:
        #
        #   reflecting -- the ring is rewritten from the interior every step
        #                 anyway, so the stale values are never read.
        #   otherwise  -- the ring holds accumulated deposition tails that must
        #                 survive, so carry those four planes across: O(X*Z),
        #                 not the O(X^2*Z) the full copy cost.
        #
        # The z end planes need no care in either case: they start at zero and
        # nothing ever writes them (deposition starts at k = no_z_electrode),
        # which is what the sweep's k = 0 and k = -1 reads assume.
        if reflecting_wall:
            positive_array, positive_next = positive_next, positive_array
            negative_array, negative_next = negative_next, negative_array
            apply_lateral_boundary(positive_array, config.lateral_boundary)
            apply_lateral_boundary(negative_array, config.lateral_boundary)
        else:
            carry_lateral_ring(positive_next, positive_array)
            carry_lateral_ring(negative_next, negative_array)
            positive_array, positive_next = positive_next, positive_array
            negative_array, negative_next = negative_next, negative_array
        timer.stop("swap_and_boundary")

        # Running collection efficiency. Both totals are density sums over the
        # same voxel set, so the voxel volume cancels and never appears.
        f_t[step] = 1.0 if no_initialised == 0.0 else (no_initialised - no_recombined) / no_initialised

        if progress and step % report_every == 0:
            print(f"  step {step + 1}/{config.total_time_steps}  f = {f_t[step]:.4f}")

        steps_completed = step + 1
        if max_wall_s is not None and (perf_counter() - loop_t0) >= max_wall_s:
            break

    if phase_timing:
        print(timer.report(config.total_time_steps))

    time_s: FloatArray1D = (np.arange(config.total_time_steps) + 1) * config.dt
    # f_t is initialised to all-ones, so a run stopped early by max_wall_s
    # would otherwise report ks = 1.0 -- a physically meaningful-looking "no
    # recombination at all" for a run that never got far enough to say that.
    # NaN makes the truncation impossible to miss (see issue #19 P7).
    truncated = max_wall_s is not None and steps_completed < config.total_time_steps
    ks = float("nan") if truncated else 1.0 / f_t[-1]
    result = diagnostics.build_result(config, time_s, f_t, ks, positive_array, negative_array)
    if max_wall_s is not None:
        result.steps_completed = steps_completed
        result.loop_elapsed_s = perf_counter() - loop_t0
    return result
