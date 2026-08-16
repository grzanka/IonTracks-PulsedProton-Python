"""GPU backend: the same physics as solver_numba_parallel.py, run on a CUDA
device with the carrier arrays resident on the GPU for the whole simulation.

Read docs/GPU.md for the full picture and the measured results; this docstring
covers what is specific to *this* file and why it is shaped the way it is.

When the GPU wins
-----------------
Both hot loops are memory-bandwidth-bound (docs/PERFORMANCE.md sec. 6). The
Lax-Wendroff sweep streams the whole grid once per step and does ~20 double
loads per voxel; on a large grid that is the entire cost. An A100 has ~1.5 TB/s
of HBM2e against ~9 GB/s per CPU core and ~900 GB/s aggregate on a full NUMA
server, so the sweep is exactly the kind of kernel a GPU is built for -- but
*only once the grid is large enough to fill it*. A grid that fits in a CPU's L3
(every AIC-144 tier below ``full_electrode``) is faster on one CPU core than on
the GPU, because there is not enough parallel work to hide the launch latency
and the host<->device round trip. The crossover, measured on this A100, is
around the ``standard``/``wide`` tier; the win grows without bound above it and
the headline case is the 5 um full-electrode grid (15.5 GiB of carrier arrays,
which does not fit in host L3 at all but sits comfortably in 40 GiB of HBM).

Three design decisions
----------------------

**1. The arrays never leave the device.** ``cp.zeros`` allocates the four
carrier arrays (and the 2D deposition scratch) once, in GPU memory, and every
kernel operates on them in place. The only per-step host<->device traffic is
three float64 scalars coming back (the scored reduction) -- 24 bytes a step
against the ~15 GiB the sweep touches on device. Copying a carrier array back
each step, as a naive port would, would make the PCIe bus, not the GPU, the
bottleneck and lose the entire point.

**2. The sweep's reduction is fused into the sweep kernel.** The step needs
three sums over the scored region -- recombined pairs and the two carrier
totals. Computing them with a second pass (CuPy ``.sum`` over the updated
arrays) would re-read the whole grid, roughly doubling the step's bandwidth and
so halving the throughput on the bandwidth-bound case that matters. Instead
each thread already has its voxel's contribution in registers, so the kernel
does a shared-memory tree reduction per block and one ``atomicAdd`` per block
into a three-element device buffer. That is ``n_blocks`` atomics per scalar per
step rather than one per scored voxel, and no extra grid traversal.

**3. Deposition uses atomics, and its scored sum does not.** The 2D density
accumulation runs one thread per track, each adding its truncated Gaussian
stencil into the shared ``total_density`` array with ``atomicAdd`` -- tracks
overlap, so the writes genuinely collide, but float64 atomics are native on
sm_80 and deposition is a minority of a large run's cost (docs/PERFORMANCE.md
sec. 1). The injected-charge scalar the caller needs is then a plain CuPy
masked reduction over the small ``(no_xy, no_xy)`` density, with a precomputed
mask that reproduces the kernels' scored-region test exactly (issue #19 P1/P2)
-- no atomics, and cheap because it is 2D, not 3D.

Agreement with the CPU backends
-------------------------------
The per-voxel update is the *same* arithmetic as solver_numba_parallel.py,
written in the same order, so the density field matches the serial backend to
near machine epsilon -- the only differences are last-bit (CUDA contracts
``a*b + c`` to an FMA by default, and the two reductions sum in a different
order). The scalar summaries (f_t, k_s) therefore agree to a relative
tolerance rather than bit-for-bit, exactly as the two CPU backends already do
with each other (tests/test_backends_agree.py). Track positions are drawn from
the identical RNG stream (the same CylinderSampler), so track_density_xy is
bit-identical.

This backend is optional: it imports CuPy and Numba's CUDA target lazily, so a
CPU-only install of the package is unaffected and only pays the import cost when
run_simulation_cuda is actually called.
"""

import warnings
from math import ceil, exp, floor
from time import perf_counter
from typing import Optional

import numpy as np
import numpy.typing as npt

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import RECOMBINATION_ALPHA_CM3_S
from pulsed_ion_chamber.pulses import CylinderSampler, build_track_schedule
from pulsed_ion_chamber.state import Diagnostics, Result, apply_lateral_boundary, carry_lateral_ring

FloatArray1D = npt.NDArray[np.float64]

# Threads per block for the sweep kernel. Must be a compile-time constant (the
# shared-memory reduction arrays are sized by it) and a power of two (the tree
# reduction halves the stride each round). 256 is a good default on the A100:
# enough threads to fill an SM, small enough that the shared arrays (3 x 256 x
# 8 = 6 KiB) leave room for high occupancy.
THREADS_PER_BLOCK = 256


# Kernels are defined inside _load_cuda() so that importing this module does not
# require CuPy or a CUDA toolchain -- see the module docstring. The compiled
# kernels are cached module-level after the first call.
_CUDA = None


def _load_cuda():
    """Import CuPy + Numba's CUDA target and JIT-compile the kernels once.

    Returns a small namespace object holding the compiled kernels and the CuPy
    module. Raises a clear error, not an ImportError deep in a hot loop, if the
    GPU stack is missing.
    """
    global _CUDA
    if _CUDA is not None:
        return _CUDA

    try:
        import cupy as cp
        from numba import cuda, float64
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "run_simulation_cuda needs CuPy and Numba's CUDA target. Install them "
            "into the environment (e.g. `pip install cupy-cuda12x numba`) on a node "
            "with an NVIDIA GPU and a matching CUDA toolkit. The CPU backends "
            "(run_simulation_numba / run_simulation_numba_parallel) need neither."
        ) from exc

    try:
        from numba.core.errors import NumbaPerformanceWarning
    except ImportError:  # pragma: no cover - defensive across numba versions
        NumbaPerformanceWarning = Warning

    if not cuda.is_available():  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "No CUDA device is available to Numba. Check that this process can see "
            "a GPU (nvidia-smi) and that CUDA_HOME / the driver are set up."
        )

    tpb = THREADS_PER_BLOCK

    @cuda.jit(cache=True)
    def _sweep_kernel(
        pos,
        neg,
        pos_next,
        neg_next,
        no_xy,
        nzb,
        no_z_electrode,
        no_z,
        mid_xy,
        scoring_radius_sq,
        p_lat,
        p_zm,
        p_zp,
        p_cen,
        n_lat,
        n_zm,
        n_zp,
        n_cen,
        alpha_dt,
        result,
    ):
        """Advance both carrier densities by one time step, and reduce the three
        scored scalars into ``result`` (recombined, total_positive,
        total_negative).

        One thread per interior voxel. The flattened index is decoded with ``k``
        (the C-contiguous fastest axis) fastest-varying too, so consecutive
        threads read consecutive memory and every neighbour load coalesces. The
        arithmetic is a line-for-line copy of
        solver_numba_parallel._lax_wendroff_step_numba_parallel -- same terms,
        same order -- so the written field matches the CPU backends bit-close.
        """
        s_rec = cuda.shared.array(tpb, float64)
        s_p = cuda.shared.array(tpb, float64)
        s_n = cuda.shared.array(tpb, float64)
        tid = cuda.threadIdx.x
        idx = cuda.grid(1)

        n_i = no_xy - 2
        n_k = nzb - 2
        total_interior = n_i * n_i * n_k

        rec_loc = 0.0
        p_loc = 0.0
        n_loc = 0.0

        if idx < total_interior:
            # k innermost: consecutive idx -> consecutive k -> unit stride.
            k = 1 + idx % n_k
            rem = idx // n_k
            j = 1 + rem % n_i
            i = 1 + rem // n_i

            p = pos[i, j, k]
            n = neg[i, j, k]
            p_new = (
                p_zm * pos[i, j, k - 1]  # upwind: positive ions drift towards +z
                + p_zp * pos[i, j, k + 1]
                + p_lat * pos[i, j - 1, k]
                + p_lat * pos[i, j + 1, k]
                + p_lat * pos[i - 1, j, k]
                + p_lat * pos[i + 1, j, k]
                + p_cen * p
            )
            # Negative ions drift the other way: swapped z-neighbours, same weights.
            n_new = (
                n_zm * neg[i, j, k + 1]
                + n_zp * neg[i, j, k - 1]
                + n_lat * neg[i, j - 1, k]
                + n_lat * neg[i, j + 1, k]
                + n_lat * neg[i - 1, j, k]
                + n_lat * neg[i + 1, j, k]
                + n_cen * n
            )
            recomb = alpha_dt * p * n
            p_out = p_new - recomb
            n_out = n_new - recomb
            pos_next[i, j, k] = p_out
            neg_next[i, j, k] = n_out

            # Scored region: inside the disc and between the electrodes. Same set
            # deposition scores over, so injected and recombined charge integrate
            # the same voxels (issue #19 P1).
            if (i - mid_xy) ** 2 + (j - mid_xy) ** 2 < scoring_radius_sq and (
                no_z_electrode <= k < no_z + no_z_electrode
            ):
                rec_loc = recomb
                p_loc = p_out
                n_loc = n_out

        # Block reduction: every thread writes its contribution (0 if out of
        # range or unscored), tree-reduce in shared memory, then thread 0 folds
        # the block's partial into the global result with three atomics. This is
        # n_blocks atomics per scalar per step, not one per scored voxel.
        s_rec[tid] = rec_loc
        s_p[tid] = p_loc
        s_n[tid] = n_loc
        cuda.syncthreads()

        stride = tpb // 2
        while stride > 0:
            if tid < stride:
                s_rec[tid] += s_rec[tid + stride]
                s_p[tid] += s_p[tid + stride]
                s_n[tid] += s_n[tid + stride]
            cuda.syncthreads()
            stride //= 2

        if tid == 0:
            cuda.atomic.add(result, 0, s_rec[0])
            cuda.atomic.add(result, 1, s_p[0])
            cuda.atomic.add(result, 2, s_n[0])

    @cuda.jit(cache=True)
    def _accumulate_kernel(total_density, xs, ys, no_xy, h2_over_b2, cutoff_voxels):
        """Sum this step's tracks' 2D Gaussians into ``total_density``.

        One thread per track. Each writes its truncated ``w x w`` stencil with
        ``atomicAdd`` -- overlapping tracks collide, which is what the atomics
        are for. ``gaussian_factor`` is applied later, in the broadcast, so this
        stays a pure sum of dimensionless shapes (matching the CPU backend's
        two-phase structure). ceil/floor, not round, so the box contains the
        cutoff circle rather than cutting inside it.
        """
        t = cuda.grid(1)
        if t >= xs.shape[0]:
            return
        x = xs[t]
        y = ys[t]
        i_lo = int(ceil(x - cutoff_voxels))
        if i_lo < 0:
            i_lo = 0
        i_hi = int(floor(x + cutoff_voxels)) + 1
        if i_hi > no_xy:
            i_hi = no_xy
        j_lo = int(ceil(y - cutoff_voxels))
        if j_lo < 0:
            j_lo = 0
        j_hi = int(floor(y + cutoff_voxels)) + 1
        if j_hi > no_xy:
            j_hi = no_xy
        for i in range(i_lo, i_hi):
            gi = exp(-((i - x) ** 2) * h2_over_b2)
            for j in range(j_lo, j_hi):
                gj = exp(-((j - y) ** 2) * h2_over_b2)
                cuda.atomic.add(total_density, (i, j), gi * gj)

    @cuda.jit(cache=True)
    def _broadcast_kernel(pos, neg, total_density, no_xy, no_z, no_z_electrode, gaussian_factor):
        """Push the step's summed 2D density into every gap z-layer of both
        carrier arrays. One thread per gap *voxel*, ``k`` fastest-varying.

        A track is a line through the full gap at constant LET, so its
        cross-section is identical in every layer -- this is a copy, not a
        recompute, and doing it once per step rather than once per track is the
        point of batching (docs/ALGORITHM.md).

        The index is decoded with ``k`` fastest -- the same layout the sweep
        uses -- so consecutive threads write consecutive, unit-stride memory and
        the writes coalesce. A thread-per-column version instead has each warp
        write 32 columns that are ``no_z_with_buffer`` apart in memory, a 32-way
        scatter that measured ~10x slower and made deposition, not the sweep,
        the dominant cost of a large run (docs/GPU.md).
        """
        idx = cuda.grid(1)
        if idx >= no_xy * no_xy * no_z:
            return
        k = no_z_electrode + idx % no_z
        rem = idx // no_z
        j = rem % no_xy
        i = rem // no_xy
        density = total_density[i, j]
        if density == 0.0:
            return
        density *= gaussian_factor
        pos[i, j, k] += density
        neg[i, j, k] += density

    class _CudaKernels:
        pass

    ns = _CudaKernels()
    ns.cp = cp
    ns.cuda = cuda
    ns.sweep = _sweep_kernel
    ns.accumulate = _accumulate_kernel
    ns.broadcast = _broadcast_kernel
    # Sparse pulse steps and the tiny warmup arrays launch single-block grids,
    # which Numba flags as under-utilised. That is expected here (deposition
    # scales with the step's track count, which is often small), so the backend
    # silences that specific, unactionable warning around its own launches.
    ns.perf_warning = NumbaPerformanceWarning
    _CUDA = ns
    return ns


def _build_scored_mask(config: SimulationConfig, cp) -> "object":
    """The ``(no_xy, no_xy)`` 0/1 mask of columns the injected-charge sum runs
    over -- exactly the kernels' scored-region test, minus the z condition.

    ``1 <= i,j <= no_xy-2`` (the interior the sweep actually visits) *and*
    inside the scored disc. Under "full_grid" scoring the disc test always
    passes, so the interior bound is what keeps the never-swept outer ring out
    of the injected total (issue #19 P2).
    """
    no_xy = config.no_xy
    coords = np.arange(no_xy)
    di = coords - config.mid_xy
    dist_sq = di[:, None] ** 2 + di[None, :] ** 2
    interior = np.zeros((no_xy, no_xy), dtype=bool)
    interior[1 : no_xy - 1, 1 : no_xy - 1] = True
    mask = interior & (dist_sq < config.scoring_radius_sq)
    return cp.asarray(mask.astype(np.float64))


def warmup_cuda() -> None:
    """Compile every CUDA kernel on trivial arrays, so a later timed call
    measures execution and not JIT compilation.

    Mirrors warmup()/warmup_parallel(): calls the kernels directly with
    hand-made, always-valid arguments rather than going through
    run_simulation_cuda, so it cannot hang on a degenerate config.
    """
    ns = _load_cuda()
    cp = ns.cp
    shape = (4, 4, 4)
    p, n, p_next, n_next = (cp.zeros(shape) for _ in range(4))
    total = cp.zeros((4, 4))
    result = cp.zeros(3)
    xs = cp.asarray(np.array([2.0]))
    ys = cp.asarray(np.array([2.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ns.perf_warning)
        ns.accumulate[(1,), (THREADS_PER_BLOCK,)](total, xs, ys, 4, 1.0, 4.0)
        ns.broadcast[(1,), (THREADS_PER_BLOCK,)](p, n, total, 4, 1, 1, 1.0)
        ns.sweep[(1,), (THREADS_PER_BLOCK,)](
            p, n, p_next, n_next, 4, 4, 1, 1, 2.0, 1.0, 0.01, 0.01, 0.01, 0.97, 0.01, 0.01, 0.01, 0.97, 1e-9, result
        )
    cp.cuda.Stream.null.synchronize()


def _check_gpu_memory(config: SimulationConfig, cp) -> None:
    """Refuse a grid that will not fit in GPU memory, before allocating it.

    The host-side ``SimulationConfig`` memory guard checks system RAM, which on
    a fat node is far larger than the GPU's. Four carrier arrays plus the 2D
    scratch must fit in device memory with headroom for the CUDA context and the
    final host copy's staging.
    """
    no_xy = config.no_xy
    nzb = config.no_z_with_buffer
    carrier_bytes = 4 * no_xy * no_xy * nzb * 8
    scratch_bytes = no_xy * no_xy * 8 * 3  # total_density + mask + slack
    needed = carrier_bytes + scratch_bytes
    free, total = cp.cuda.runtime.memGetInfo()
    if needed > 0.9 * free:
        from pulsed_ion_chamber.resources import format_bytes

        raise MemoryError(
            f"This {no_xy}x{no_xy}x{nzb} grid needs {format_bytes(needed)} of GPU memory "
            f"(4 carrier arrays), but only {format_bytes(free)} of {format_bytes(total)} is free "
            "on the device. Coarsen grid_size_um or reduce sampled_radius_cm, or use a GPU with "
            "more memory."
        )


def run_simulation_cuda(
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
    progress: bool = True,
    max_wall_s: Optional[float] = None,
) -> Result:
    """Simulate a pulse train on the GPU and return the full run record.

    Same physics and same RNG stream as
    :func:`~pulsed_ion_chamber.solver_numba_parallel.run_simulation_numba_parallel`
    -- the carrier arrays live on the GPU for the whole run and only three
    scalars a step come back to the host. See the module docstring for the
    design and docs/GPU.md for measured results and when this backend is worth
    using (large, DRAM-sized grids; a small grid is faster on one CPU core).

    ``max_wall_s``, if given, stops the loop once that many seconds have
    elapsed, exactly as the CPU backends do, and sets ``steps_completed`` /
    ``loop_elapsed_s`` on the returned Result for the runtime estimator.
    """
    ns = _load_cuda()
    cp = ns.cp

    _check_gpu_memory(config, cp)

    rng = rng if rng is not None else np.random.default_rng(config.seed)
    schedule = build_track_schedule(config, rng)
    sampler = CylinderSampler(rng, config.mid_xy, config.sampling_radius, config.no_xy)

    # Four carrier arrays resident on the device for the whole run. The sweep
    # reads *_array and writes *_next; the buffers are then swapped, never
    # copied (docs/ALGORITHM.md). C-contiguous, k fastest -- what the kernel's
    # coalesced access assumes.
    shape = (config.no_xy, config.no_xy, config.no_z_with_buffer)
    positive_array = cp.zeros(shape)
    negative_array = cp.zeros(shape)
    positive_next = cp.zeros(shape)
    negative_next = cp.zeros(shape)

    (p_lat, p_zm, p_zp, p_cen), (n_lat, n_zm, n_zp, n_cen) = config.scheme_coefficients()
    alpha_dt = RECOMBINATION_ALPHA_CM3_S * config.dt

    h2 = config.unit_length_cm**2
    b2 = config.track_radius_cm**2
    h2_over_b2 = h2 / b2
    cutoff_voxels = config.track_cutoff_voxels

    total_density = cp.zeros((config.no_xy, config.no_xy))
    scored_mask = _build_scored_mask(config, cp)
    result_buf = cp.zeros(3)  # (recombined, total_positive, total_negative), reused per step

    no_xy = config.no_xy
    nzb = config.no_z_with_buffer
    n_interior = (no_xy - 2) * (no_xy - 2) * (nzb - 2)
    sweep_blocks = (n_interior + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    n_gap_voxels = no_xy * no_xy * config.no_z
    broadcast_blocks = (n_gap_voxels + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    no_initialised = 0.0
    no_recombined = 0.0
    f_t: FloatArray1D = np.ones(config.total_time_steps)
    diagnostics = Diagnostics(config)
    report_every = max(1, config.total_time_steps // 20)
    reflecting_wall = config.lateral_boundary == "reflecting"
    loop_t0 = perf_counter()
    steps_completed = 0

    for step in range(config.total_time_steps):
        injected_this_step = 0.0
        n_tracks_this_step = schedule[step]
        if n_tracks_this_step > 0:
            xs, ys = sampler.sample(n_tracks_this_step)
            xs_d = cp.asarray(xs)
            ys_d = cp.asarray(ys)
            total_density.fill(0.0)
            dep_blocks = (xs.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            # A sparse step launches a single-block grid; that under-utilisation
            # is inherent to depositing few tracks, not a bug, so silence the
            # (unactionable) warning around this one launch. The sweep and
            # broadcast grids are always large on any grid worth running here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ns.perf_warning)
                ns.accumulate[(dep_blocks,), (THREADS_PER_BLOCK,)](
                    total_density, xs_d, ys_d, no_xy, h2_over_b2, cutoff_voxels
                )
            # Injected charge over the scored disc: a cheap 2D masked reduction,
            # not an atomic in the kernel. gaussian_factor and the no_z layers
            # are folded in here, matching the CPU backend's broadcast sum.
            injected_this_step = float(cp.sum(total_density * scored_mask)) * config.Gaussian_factor * config.no_z
            ns.broadcast[(broadcast_blocks,), (THREADS_PER_BLOCK,)](
                positive_array, negative_array, total_density, no_xy, config.no_z, config.no_z_electrode, config.Gaussian_factor
            )
            diagnostics.count_tracks(xs, ys)

        no_initialised += injected_this_step

        result_buf.fill(0.0)
        ns.sweep[(sweep_blocks,), (THREADS_PER_BLOCK,)](
            positive_array,
            negative_array,
            positive_next,
            negative_next,
            no_xy,
            nzb,
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
            result_buf,
        )
        # Three scalars back to the host. This copy also synchronises the stream,
        # so the sweep is complete before the buffer swap below reads its output.
        recombined, total_p, total_n = (float(v) for v in cp.asnumpy(result_buf))
        no_recombined += recombined
        diagnostics.record(step, injected_this_step, recombined, total_p, total_n)

        # Swap, never copy the interior back. Under a reflecting wall the ring is
        # rewritten from the interior every step, so the stale ring left by the
        # swap is never read; otherwise the four boundary planes (Gaussian tails
        # deposition put there) must be carried across. CuPy arrays take the
        # same slicing as NumPy, so state.py's helpers work unchanged on device.
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

        f_t[step] = 1.0 if no_initialised == 0.0 else (no_initialised - no_recombined) / no_initialised

        if progress and step % report_every == 0:
            print(f"  step {step + 1}/{config.total_time_steps}  f = {f_t[step]:.4f}")

        steps_completed = step + 1
        if max_wall_s is not None and (perf_counter() - loop_t0) >= max_wall_s:
            break

    time_s: FloatArray1D = (np.arange(config.total_time_steps) + 1) * config.dt
    truncated = max_wall_s is not None and steps_completed < config.total_time_steps
    ks = float("nan") if truncated else 1.0 / f_t[-1]
    # The field snapshot the Result carries lives on the host; copy the two final
    # arrays back once, at the end (not per step).
    positive_host = cp.asnumpy(positive_array)
    negative_host = cp.asnumpy(negative_array)
    result = diagnostics.build_result(config, time_s, f_t, ks, positive_host, negative_host)
    if max_wall_s is not None:
        result.steps_completed = steps_completed
        result.loop_elapsed_s = perf_counter() - loop_t0
    return result
