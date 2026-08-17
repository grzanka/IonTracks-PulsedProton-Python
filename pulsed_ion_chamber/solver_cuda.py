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
the headline case is the 5 um full-electrode grid (13.9 GiB of carrier arrays,
which does not fit in host L3 at all but sits comfortably in 40 GiB of HBM).

Three design decisions
----------------------

**1. The arrays never leave the device.** ``cp.zeros`` allocates the four
carrier arrays (and the 2D deposition scratch) once, in GPU memory, and every
kernel operates on them in place. The only per-step host<->device traffic is
three float64 scalars coming back (the scored reduction) -- 24 bytes a step
against the ~14 GiB the sweep touches on device. Copying a carrier array back
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

Grids larger than the GPU: unified memory
-----------------------------------------
``memory="managed"`` allocates the carrier arrays with ``cudaMallocManaged``
instead of ``cudaMalloc``, which lets a grid exceed the device's own memory and
spill into host RAM. On a **Grace Hopper (GH200) superchip** that is not the
consolation prize it is on a PCIe machine: the Grace CPU's LPDDR5X and the
Hopper GPU's HBM3 are one hardware-coherent address space joined by NVLink-C2C
at ~450 GB/s each way, about 7x a PCIe 5.0 x16 link, and the GPU walks the
host's own page tables (``PageableMemoryAccessUsesHostPageTables=1``) so the
overflow is *streamed on demand* rather than migrated page by page. The
practical consequence for this code is a resolution limit set by the node's
480 GiB of LPDDR5X rather than by the GPU's 96 GiB of HBM -- see
docs/BENCHMARKS-HELIOS-GH200.md for what that buys and what it costs per step.

``memory="auto"`` (the default) picks device memory when the grid fits in it
and managed memory when it does not, so a run that used to raise MemoryError
now runs, slower, instead of not running at all.

This backend is optional: it imports CuPy and Numba's CUDA target lazily, so a
CPU-only install of the package is unaffected and only pays the import cost when
run_simulation_cuda is actually called.
"""

import os
import warnings
from contextlib import contextmanager
from math import ceil, exp, floor
from time import perf_counter
from typing import Optional

import numpy as np
import numpy.typing as npt

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import RECOMBINATION_ALPHA_CM3_S
from pulsed_ion_chamber.pulses import CylinderSampler, build_track_schedule
from pulsed_ion_chamber.resources import available_memory_bytes, format_bytes
from pulsed_ion_chamber.state import Diagnostics, Result, apply_lateral_boundary, carry_lateral_ring

FloatArray1D = npt.NDArray[np.float64]

# Threads per block for the sweep kernel. Must be a compile-time constant (the
# shared-memory reduction arrays are sized by it) and a power of two (the tree
# reduction halves the stride each round). 256 is a good default on the A100:
# enough threads to fill an SM, small enough that the shared arrays (3 x 256 x
# 8 = 6 KiB) leave room for high occupancy. It is also within a factor of a few
# per cent of optimal on the GH200's Hopper die (docs/BENCHMARKS-HELIOS-GH200.md
# sec. 6 measures 128/256/512/1024), so the default is unchanged there.
#
# Override for an experiment with PULSED_ION_CHAMBER_CUDA_TPB=512, or by calling
# set_threads_per_block() before the first run. Kernels are recompiled when it
# changes; Numba's on-disk cache keys on closure values, so the two block sizes
# do not collide in the cache.
THREADS_PER_BLOCK = int(os.environ.get("PULSED_ION_CHAMBER_CUDA_TPB", 256))


def set_threads_per_block(threads: int) -> None:
    """Set the CUDA block size used by every kernel, discarding compiled ones.

    Must be a power of two in [32, 1024]: the sweep's shared-memory tree
    reduction halves its stride each round, and 1024 is the hardware maximum.
    """
    global THREADS_PER_BLOCK, _CUDA
    if threads < 32 or threads > 1024 or threads & (threads - 1):
        raise ValueError(f"threads_per_block must be a power of two in [32, 1024], got {threads!r}.")
    if threads != THREADS_PER_BLOCK:
        THREADS_PER_BLOCK = threads
        _CUDA = None  # force recompile against the new shared-array size


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
    if _CUDA is not None and _CUDA.tpb == THREADS_PER_BLOCK:
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
    ns.tpb = tpb
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


def carrier_bytes(config: SimulationConfig) -> int:
    """Device bytes the four float64 carrier arrays occupy. The run's footprint,
    to within the (2D, negligible) scratch."""
    return 4 * config.no_xy * config.no_xy * config.no_z_with_buffer * 8


def _managed_pool(cp):
    """A CuPy memory pool backed by ``cudaMallocManaged``, created once.

    A pool rather than raw ``malloc_managed`` because the run allocates and
    frees the same four shapes repeatedly across a benchmark ladder, and
    managed allocations are expensive to create (the driver has to build page
    tables for them).
    """
    global _MANAGED_POOL
    if _MANAGED_POOL is None:
        _MANAGED_POOL = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    return _MANAGED_POOL


_MANAGED_POOL = None


@contextmanager
def _allocating_managed(cp, enabled: bool):
    """Route CuPy allocations made inside the block through the managed pool.

    Scoped and restored, so importing this module or running a device-memory
    simulation never changes the process-wide allocator that other code (or a
    later run in the same process) relies on.
    """
    if not enabled:
        yield
        return
    previous = cp.cuda.get_allocator()
    cp.cuda.set_allocator(_managed_pool(cp).malloc)
    try:
        yield
    finally:
        cp.cuda.set_allocator(previous)


def _host_page_tables(cp) -> bool:
    """True where the GPU walks the host's own page tables, so a plain malloc'd
    pointer is directly dereferenceable from a kernel.

    This is the defining property of an address-translation-services machine
    like GH200 (``cudaDevAttrPageableMemoryAccessUsesHostPageTables``); on a
    PCIe host it is 0 and ``memory="host"`` cannot work.
    """
    try:
        return bool(cp.cuda.Device().attributes.get("PageableMemoryAccessUsesHostPageTables", 0))
    except Exception:  # pragma: no cover - driver dependent
        return False


def _host_shared_zeros(cp, shape) -> "object":
    """A float64 array in *host* memory that CUDA kernels can address directly.

    The GH200 alternative to ``cudaMallocManaged``: allocate with the host
    allocator, first-touch it on the CPU so the pages are placed in Grace's
    LPDDR5X, and hand the GPU the same virtual address. There is no migration
    machinery at all -- the Hopper die reads and writes those pages across
    NVLink-C2C, cache-coherently, at ~450 GB/s. Compared with managed memory
    this trades peak bandwidth (HBM3 is ~8x faster) for a flat, predictable
    cost with no page-fault storms and no eviction, which is the better bargain
    once the grid is several times the size of HBM.

    Allocated with ``cudaHostAlloc`` rather than plain ``malloc``: under
    unified addressing the pinned host pointer is a valid device pointer, which
    a ``malloc``'d one is not -- ``cuPointerGetAttribute`` rejects it and Numba
    cannot take the array as a kernel argument, ATS or no ATS. The pages are
    still ordinary host DRAM, still first-touched by the CPU, and still reached
    over C2C; pinning only means the kernel may not page them out, which for a
    simulation that touches every page every step is what we want anyway.

    Wrapped as an unowned CuPy allocation whose ``owner`` is the pinned
    allocation, so the pages stay alive exactly as long as the CuPy view does.
    """
    count = int(np.prod(shape))
    pinned = cp.cuda.alloc_pinned_memory(count * 8)
    host = np.frombuffer(pinned, dtype=np.float64, count=count).reshape(shape)
    host[...] = 0.0  # CPU first touch: the pages are placed in Grace's LPDDR5X
    memory = cp.cuda.UnownedMemory(pinned.ptr, count * 8, owner=pinned)
    return cp.ndarray(shape, dtype=np.float64, memptr=cp.cuda.MemoryPointer(memory, 0))


# Driver-API constants for the host-preferred hint. CUDA 13's *runtime*
# cudaMemAdvise takes a cudaMemLocation struct, and CuPy's binding only exposes
# the int-device form, which rejects the CPU's -1 -- so the host policy goes
# through cuMemAdvise_v2 in libcuda directly.
_CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3
_CU_MEM_ADVISE_SET_ACCESSED_BY = 5
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_LOCATION_TYPE_HOST = 2


def _driver_mem_advise(ptr: int, nbytes: int, advice: int, location_type: int, location_id: int) -> None:
    """``cuMemAdvise_v2`` through ctypes. Raises on any failure; the caller
    treats hints as optional."""
    import ctypes

    class _CUmemLocation(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

    lib = ctypes.CDLL("libcuda.so.1")
    fn = lib.cuMemAdvise_v2
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, _CUmemLocation]
    status = fn(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), advice, _CUmemLocation(location_type, location_id))
    if status != 0:
        raise RuntimeError(f"cuMemAdvise_v2 returned {status}")


def _advise(cp, arrays, policy: str, device_budget_bytes: int) -> None:
    """Tell the driver where the managed carrier arrays should live.

    Only meaningful for managed memory; a no-op on device allocations. Measured
    on a 7.7 GiB grid that fits in HBM, at 200 steps
    (docs/BENCHMARKS-HELIOS-GH200.md sec. 5):

    ``"device"``  **15.6 ms/step.** Prefer HBM and prefetch, in argument order,
    until ``device_budget_bytes`` is spent; the remainder is left to fault in.
    Within 2 % of a plain ``cudaMalloc`` run (15.3 ms), which is the result that
    matters: managed memory costs nothing when the grid fits, so it is safe as
    the default and the spill is available for free when the grid does not.

    ``"none"``  **98.8 ms/step,** 6.4x worse. No hints, so every page is
    migrated on its first fault and the sweep pays for it. The reason
    ``"device"`` exists.

    ``"host"``  **273.7 ms/step,** 18x worse. Pins every page in host LPDDR5X
    and maps it into the GPU, so the sweep streams over NVLink-C2C. This is a
    *penalty* on a grid that fits -- it declines the HBM it was offered -- and
    is here for the oversubscribed case it was written for, where there is no
    resident working set to protect and migration is pure overhead. That case
    is not yet measured; do not assume it wins there either.

    Note it loses even to ``memory="host"`` (96.7 ms/step) doing ostensibly the
    same thing, so the difference is migration machinery, not the link.
    """
    if policy == "none":
        return
    runtime = cp.cuda.runtime
    device_id = cp.cuda.Device().id
    remaining = device_budget_bytes
    for array in arrays:
        ptr = array.data.ptr
        nbytes = array.nbytes
        try:
            if policy == "host":
                _driver_mem_advise(
                    ptr, nbytes, _CU_MEM_ADVISE_SET_PREFERRED_LOCATION, _CU_MEM_LOCATION_TYPE_HOST, 0
                )
                _driver_mem_advise(
                    ptr, nbytes, _CU_MEM_ADVISE_SET_ACCESSED_BY, _CU_MEM_LOCATION_TYPE_DEVICE, device_id
                )
            elif policy == "device":
                runtime.memAdvise(ptr, nbytes, runtime.cudaMemAdviseSetPreferredLocation, device_id)
                if remaining >= nbytes:
                    runtime.memPrefetchAsync(ptr, nbytes, device_id, cp.cuda.Stream.null.ptr)
                    remaining -= nbytes
        except Exception as exc:  # pragma: no cover - driver/toolkit dependent
            # Hints are an optimisation, never a correctness requirement: a
            # driver that refuses them still runs the simulation.
            warnings.warn(f"cudaMemAdvise/Prefetch failed ({exc}); continuing without placement hints.")
            return


def _check_gpu_memory(config: SimulationConfig, cp, memory: str) -> None:
    """Refuse a grid that will not fit, before allocating it.

    Each of the three allocators has its own ceiling: ``"device"`` is bounded by
    the GPU alone, ``"host"`` by the host alone, ``"managed"`` by their sum.
    The host-side ``SimulationConfig`` guard cannot stand in for any of them --
    it checks system RAM, which on a fat node is far larger than the GPU's and
    on a scheduled one is far *smaller* than it looks. "What the host has" is
    the job's cgroup
    headroom, not the node's free RAM (``resources.available_memory_bytes``).
    Getting that wrong on Helios means the OOM killer at step 300 instead of a
    MemoryError before allocating.
    """
    no_xy = config.no_xy
    nzb = config.no_z_with_buffer
    scratch_bytes = no_xy * no_xy * 8 * 3  # total_density + mask + slack
    needed = carrier_bytes(config) + scratch_bytes
    free, total = cp.cuda.runtime.memGetInfo()
    host_free = available_memory_bytes()

    if memory == "host":
        # Entirely in host RAM: only the cgroup's headroom matters.
        if host_free is not None and needed > 0.8 * host_free:
            raise MemoryError(
                f"This {no_xy}x{no_xy}x{nzb} grid needs {format_bytes(needed)} of host memory, but "
                f"only {format_bytes(host_free)} is available to this process (the guard reads the "
                "job's cgroup limit, not the node's RAM -- on Helios ask sbatch for more with "
                "--mem=). Coarsen grid_size_um or reduce sampled_radius_cm."
            )
        return

    if memory == "device":
        if needed > 0.9 * free:
            raise MemoryError(
                f"This {no_xy}x{no_xy}x{nzb} grid needs {format_bytes(needed)} of GPU memory "
                f"(4 carrier arrays), but only {format_bytes(free)} of {format_bytes(total)} is free "
                "on the device. Coarsen grid_size_um or reduce sampled_radius_cm, use a GPU with "
                'more memory, or pass memory="managed" to spill into host RAM (fast on a '
                "GH200's NVLink-C2C, slow over PCIe -- see docs/GPU.md)."
            )
        return

    # 0.9 of the device (CUDA context, kernel scratch) and 0.8 of the host
    # headroom (interpreter, NumPy temporaries, the final snapshot copy).
    ceiling = 0.9 * free + (0.8 * host_free if host_free else 0.0)
    if needed > ceiling:
        host_note = f" plus {format_bytes(host_free)} of host headroom" if host_free else ""
        raise MemoryError(
            f"This {no_xy}x{no_xy}x{nzb} grid needs {format_bytes(needed)} of unified memory, "
            f"but only {format_bytes(free)} of device memory{host_note} is available "
            f"({format_bytes(ceiling)} after headroom). Coarsen grid_size_um, reduce "
            "sampled_radius_cm, or ask the scheduler for more host memory (on Helios: "
            "sbatch --mem=...; the guard reads the job's cgroup limit, not the node's RAM)."
        )


def run_simulation_cuda(
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
    progress: bool = True,
    max_wall_s: Optional[float] = None,
    max_steps: Optional[int] = None,
    memory: str = "auto",
    advise: str = "device",
    return_fields: bool = True,
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
    ``max_steps`` stops after a fixed number of steps instead, which is what a
    benchmark of a grid too large to run to completion wants: a step count is
    reproducible where a wall-clock cut is not, so ms/step from two machines
    can be compared.

    ``return_fields=False`` drops the final density snapshot from the Result
    (the time series and ``k_s`` are unaffected). The snapshot is two host
    arrays the size of the grid, and on a scheduled node the job's host memory
    is routinely a fraction of the GPU's: a 60 GiB grid on a Helios GH200 with
    ``--mem=12G`` runs happily and then dies copying the answer home. Anything
    that only needs the scalars should turn it off.

    ``memory`` selects the allocator for the carrier arrays: ``"device"``
    (``cudaMalloc``, fails if the grid does not fit), ``"managed"``
    (``cudaMallocManaged``, may spill into host RAM), or ``"auto"`` (device
    when it fits, managed when it does not). ``advise`` is the placement policy
    for managed memory -- ``"device"``, ``"host"`` or ``"none"``, see
    :func:`_advise`; it is ignored under ``"device"`` memory.
    """
    ns = _load_cuda()
    cp = ns.cp
    tpb = ns.tpb

    if memory not in ("auto", "device", "managed", "host"):
        raise ValueError(f'memory must be "auto", "device", "managed" or "host", got {memory!r}.')
    if advise not in ("device", "host", "none"):
        raise ValueError(f'advise must be "device", "host" or "none", got {advise!r}.')
    if memory == "auto":
        free_device, _ = cp.cuda.runtime.memGetInfo()
        if carrier_bytes(config) <= 0.9 * free_device:
            memory = "device"
        elif _host_page_tables(cp):
            # Oversubscribed, on an ATS machine: stream from host memory, do NOT
            # migrate. This is a measured choice, and the measurement was brutal
            # -- the full electrode at 2 um (7.1 G voxels, 212 GiB against 95 GiB
            # of HBM) ran at 32.1 *seconds* per step on managed memory with the
            # prefetch-to-device policy, 84x slower than the same GPU's HBM rate
            # on a grid that fits. Migration can only pay when there is a
            # resident working set to reuse, and this sweep touches every one of
            # those 212 GiB every single step, so prefetching into HBM just
            # evicts what the next block of threads was about to need.
            # docs/BENCHMARKS-HELIOS-GH200.md sec. 7.
            memory = "host"
        else:
            memory = "managed"
    elif memory == "managed":
        free_device, _ = cp.cuda.runtime.memGetInfo()
        if carrier_bytes(config) > 1.05 * free_device and _host_page_tables(cp):
            warnings.warn(
                f"This grid ({format_bytes(carrier_bytes(config))}) is larger than device memory "
                f"({format_bytes(free_device)} free), and managed memory migrates pages the sweep "
                "re-reads every step: the 212 GiB full electrode measured 32 s/step this way. "
                'Prefer memory="host" on this machine -- it streams over NVLink-C2C without the '
                "migration machinery. See docs/BENCHMARKS-HELIOS-GH200.md sec. 7.",
                stacklevel=2,
            )
    if memory == "host" and not _host_page_tables(cp):
        raise RuntimeError(
            'memory="host" needs a GPU that walks the host page tables (an ATS machine such as a '
            "Grace Hopper GH200); this device reports "
            'cudaDevAttrPageableMemoryAccessUsesHostPageTables=0. Use memory="managed" instead, '
            "which works everywhere."
        )
    managed = memory == "managed"

    _check_gpu_memory(config, cp, memory)

    rng = rng if rng is not None else np.random.default_rng(config.seed)
    schedule = build_track_schedule(config, rng)
    sampler = CylinderSampler(rng, config.mid_xy, config.sampling_radius, config.no_xy)

    # Four carrier arrays resident on the device for the whole run. The sweep
    # reads *_array and writes *_next; the buffers are then swapped, never
    # copied (docs/ALGORITHM.md). C-contiguous, k fastest -- what the kernel's
    # coalesced access assumes.
    shape = (config.no_xy, config.no_xy, config.no_z_with_buffer)
    if memory == "host":
        # Host-allocated, CPU-first-touched, GPU-addressable over NVLink-C2C.
        # Already zeroed by np.zeros, and there is nothing to advise.
        fields = [_host_shared_zeros(cp, shape) for _ in range(4)]
    else:
        with _allocating_managed(cp, managed):
            # empty -> advise -> fill, not zeros: placement hints must land
            # before the pages are first touched, or the zeroing migrates the
            # whole grid to the device and the "host" policy immediately
            # migrates it back.
            fields = [cp.empty(shape) for _ in range(4)]
        if managed:
            free_device, _ = cp.cuda.runtime.memGetInfo()
            _advise(cp, fields, advise, device_budget_bytes=int(0.9 * free_device))
        for field in fields:
            field.fill(0.0)
    positive_array, negative_array, positive_next, negative_next = fields
    del fields

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
    sweep_blocks = (n_interior + tpb - 1) // tpb
    n_gap_voxels = no_xy * no_xy * config.no_z
    broadcast_blocks = (n_gap_voxels + tpb - 1) // tpb

    no_initialised = 0.0
    no_recombined = 0.0
    f_t: FloatArray1D = np.ones(config.total_time_steps)
    diagnostics = Diagnostics(config)
    report_every = max(1, config.total_time_steps // 20)
    reflecting_wall = config.lateral_boundary == "reflecting"
    loop_t0 = perf_counter()
    steps_completed = 0
    last_step = config.total_time_steps if max_steps is None else min(config.total_time_steps, max_steps)

    for step in range(last_step):
        injected_this_step = 0.0
        n_tracks_this_step = schedule[step]
        if n_tracks_this_step > 0:
            xs, ys = sampler.sample(n_tracks_this_step)
            xs_d = cp.asarray(xs)
            ys_d = cp.asarray(ys)
            total_density.fill(0.0)
            dep_blocks = (xs.shape[0] + tpb - 1) // tpb
            # A sparse step launches a single-block grid; that under-utilisation
            # is inherent to depositing few tracks, not a bug, so silence the
            # (unactionable) warning around this one launch. The sweep and
            # broadcast grids are always large on any grid worth running here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ns.perf_warning)
                ns.accumulate[(dep_blocks,), (tpb,)](
                    total_density, xs_d, ys_d, no_xy, h2_over_b2, cutoff_voxels
                )
            # Injected charge over the scored disc: a cheap 2D masked reduction,
            # not an atomic in the kernel. gaussian_factor and the no_z layers
            # are folded in here, matching the CPU backend's broadcast sum.
            injected_this_step = float(cp.sum(total_density * scored_mask)) * config.Gaussian_factor * config.no_z
            ns.broadcast[(broadcast_blocks,), (tpb,)](
                positive_array, negative_array, total_density, no_xy, config.no_z, config.no_z_electrode, config.Gaussian_factor
            )
            diagnostics.count_tracks(xs, ys)

        no_initialised += injected_this_step

        result_buf.fill(0.0)
        ns.sweep[(sweep_blocks,), (tpb,)](
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

    loop_elapsed_s = perf_counter() - loop_t0
    time_s: FloatArray1D = (np.arange(config.total_time_steps) + 1) * config.dt
    truncated = steps_completed < config.total_time_steps
    # k_s is the efficiency after *full* clearance, so a truncated run has none.
    ks = float("nan") if truncated else 1.0 / f_t[-1]
    # The field snapshot the Result carries lives on the host; copy the two final
    # arrays back once, at the end (not per step) -- or not at all, if the
    # caller said it does not want them (see return_fields).
    if return_fields:
        positive_host = cp.asnumpy(positive_array)
        negative_host = cp.asnumpy(negative_array)
    else:
        positive_host = np.zeros((0, 0, 0))
        negative_host = np.zeros((0, 0, 0))
    result = diagnostics.build_result(config, time_s, f_t, ks, positive_host, negative_host)
    if max_wall_s is not None or max_steps is not None:
        result.steps_completed = steps_completed
        result.loop_elapsed_s = loop_elapsed_s
    if managed:
        # A managed pool holds its blocks for reuse; on an oversubscribed grid
        # those are hundreds of GiB of host pages that the next run in the same
        # process (a benchmark ladder) does not want still charged to it. Drop
        # the references first -- free_all_blocks only reclaims unused blocks.
        del positive_array, negative_array, positive_next, negative_next
        _managed_pool(cp).free_all_blocks()
    return result
