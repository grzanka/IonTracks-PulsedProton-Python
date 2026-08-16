# Running on the GPU (CUDA backend)

A third solver backend, `pulsed_ion_chamber.solver_cuda.run_simulation_cuda`,
runs the identical physics of the two CPU backends on an NVIDIA GPU, with the
carrier arrays resident on the device for the whole simulation. For the physics
see [PHYSICS.md](PHYSICS.md), for the algorithm [ALGORITHM.md](ALGORITHM.md),
for the CPU cost model [PERFORMANCE.md](PERFORMANCE.md).

**The one-line answer:** the GPU wins once the grid is too large to live in a
CPU's cache, and the win then grows with size. On one A100-40GB the full Markus
electrode at 10 µm runs **15× faster than 32 CPU cores** (219 s → 14.5 s), and
the 5 µm full electrode — 466 M voxels, 13.9 GiB of carrier arrays, which no CPU
cache can hold — runs in **156 s** on the GPU against **50 min on 32 cores
(19×)**. `k_s` is identical to the CPU reference to six digits in every case. A
*small* grid is faster on one CPU core: below ~r = 0.05 cm the GPU loses to
launch latency, so this backend is for large grids only.

---

## 1. The machine

Measured on a Cyfronet **Athena** worker node:

| | |
|---|---|
| GPU | 1 × **NVIDIA A100-SXM4-40GB** (sm_80), 40 GiB HBM2e, ~1.5 TB/s |
| host | 32 CPU cores (of the node) allocated, ~1 TiB RAM |
| driver / CUDA | driver 595.71, CUDA 13.2 capable; toolkit `CUDA/12.8.0` module |
| stack | Python 3.10.4, CuPy 14.1.1 (`cupy-cuda12x`), Numba 0.67 |

One number drives everything below: **~1.5 TB/s of HBM against ~9 GB/s per CPU
core** (and ~900 GB/s aggregate on a full NUMA server). Both hot loops are
memory-bandwidth-bound (PERFORMANCE.md §6), so the GPU is the right tool — but
only once there is enough parallel work to fill it and hide the host↔device
round trip.

## 2. Setup

The four `module load` lines must be repeated in every new shell (the venv does
not remember them). `module load` must not be piped — `module load ... | tail`
runs in a subshell and silently loses the environment.

```bash
module load Python/3.10.4
```

```bash
module load CUDA/12.8.0
```

```bash
module load GCC/13.3.0
```

```bash
python -m venv .venv-gpu
```

```bash
source .venv-gpu/bin/activate
```

```bash
pip install -e ".[dev]"
```

```bash
pip install cupy-cuda12x
```

```bash
pytest tests/test_cuda_backend.py -q  # 9 GPU tests; skip cleanly with no GPU
```

`cupy-cuda12x` is a pre-built wheel that bundles its own CUDA runtime, so it
needs only the driver; the `CUDA/12.8.0` module is what Numba's CUDA target uses
to compile its kernels (it finds `libnvvm` through `CUDA_HOME`). The GPU stack is
imported lazily inside `run_simulation_cuda`, so a CPU-only install of the
package (`pip install -e .`, no CuPy) is unaffected and the other two backends
need neither dependency.

## 3. How to run

From the example CLI (`docs/PERFORMANCE.md` §2 has the scenario):

```bash
python examples/ifj_aic144/run_markus_2mm.py wide --backend cuda
```

From the benchmark harness, one size (`sampled_radius_cm@grid_um`):

```bash
python profiling/bench_gpu.py --sizes 0.265@5 --backends gpu
```

The whole crossover-to-full-electrode ladder, GPU against 32 CPU cores:

```bash
python profiling/bench_gpu.py --ladder full --backends cpu,gpu --threads 32
```

From Python:

```python
from pulsed_ion_chamber import SimulationConfig, run_simulation_cuda
result = run_simulation_cuda(SimulationConfig(), progress=False)
```

## 4. The crossover — where the GPU starts to win

The whole point is size. Both loops stream the grid, so the GPU's bandwidth only
pays once the grid stops fitting in a CPU's cache. Measured on the AIC-144
Markus 2 mm scenario, 10 µm voxels, two carrier species, GPU vs the batched CPU
backend on 32 cores of the same node:

| grid | voxels | tracks/pulse | CPU 32-core | GPU (A100) | speed-up |
|---|---|---|---|---|---|
| 34² × 210 (`standard`) | 0.2 M | 0.07 M | 3.0 s | 5.1 s | 0.59× |
| 106² × 210 | 2 M | 0.9 M | 6.0 s | 5.5 s | 1.08× |
| 166² × 210 | 6 M | 2.2 M | 20.4 s | 6.1 s | 3.35× |
| 246² × 210 | 13 M | 5.1 M | 61.3 s | 7.2 s | 8.54× |
| 366² × 210 | 28 M | 11.4 M | 92.1 s | 9.0 s | 10.2× |
| **536² × 210** (`full_electrode`) | **60 M** | **24.6 M** | **219.1 s** | **14.5 s** | **15.1×** |

`k_s` is identical to six digits in every row (1.101078 → 1.111081 down the
ladder). The crossover is around r = 0.05 cm (106²): below it the GPU is
launch-latency- and round-trip-bound and one CPU core is faster; above it the
GPU wall time barely moves (5.1 → 14.5 s across a 300× range in voxels) while the
CPU grows with the grid, so the ratio keeps climbing.

## 5. The full electrode at 5 µm — the memory-bound headline

The reason to reach for a 40 GiB GPU: refine the full electrode to 5 µm voxels
and it becomes a **1066² × 410 grid, 466 M voxels, 13.9 GiB** of carrier arrays
(four float64 grids), with the time step halved by the von Neumann limit so
4580 steps instead of 2194. That does not fit in any CPU cache, and on a CPU it
is `h⁻⁴`-expensive (PERFORMANCE.md §3); on the A100 it sits in HBM with room to
spare.

| | CPU 32-core | GPU (A100) | speed-up |
|---|---|---|---|
| **5 µm full electrode** (466 M vox, 4580 steps) | 3000 s (50 min)¹ | **156 s** | **19.2×** |
| `k_s` | 1.1104725895 | 1.1104725895 | identical (8e-16)² |
| device memory | — | 13.9 GiB / 40 GiB | — |

¹ Batched CPU backend, 32 threads, same node — measured, not extrapolated
(`profiling/data/gpu_athena/cpu_5um_full_electrode.json`).
² Relative difference in `k_s`; the two agree to the last two bits even on this
466 M-voxel, 4580-step run — the reduction order differs, the physics does not.

This is the case the GPU was brought in for: a converged-resolution full-electrode
run that is a coffee break on the GPU instead of an hour on a whole CPU node, and
that would not fit in a laptop's or a Helios node's cache at all.

## 6. How it works

Three CUDA kernels (Numba `@cuda.jit`, operating on CuPy arrays through the CUDA
array interface), everything else CuPy:

1. **Sweep** — one thread per interior voxel advances both carrier densities by
   one Lax-Wendroff step. The flattened index is decoded with `k` (the
   C-contiguous fastest axis) fastest-varying, so consecutive threads touch
   consecutive memory and every 7-point-stencil neighbour load coalesces. The
   arithmetic is a line-for-line copy of the CPU kernel.
2. **Accumulate** — one thread per track adds its truncated Gaussian stencil
   into a 2D density array with `atomicAdd`. Tracks overlap, so the writes
   collide, but float64 atomics are native on sm_80 and deposition is a minority
   of a large run's cost.
3. **Broadcast** — one thread per gap voxel copies the 2D density down every
   z-layer of both carrier arrays.

Three decisions make it fast:

- **The arrays never leave the device.** They are allocated once in HBM; the
  only per-step host↔device traffic is three float64 scalars coming back (the
  scored reduction), 24 bytes against the ~14 GiB the sweep touches. Copying a
  carrier array back each step would make PCIe, not the GPU, the bottleneck.
- **The reduction is fused into the sweep.** Each step needs three sums over the
  scored region; a second CuPy pass would re-read the whole grid and roughly
  double the step's bandwidth. Instead each thread already has its contribution
  in registers, so the kernel does a shared-memory tree reduction per block and
  one `atomicAdd` per block into a three-element buffer — no extra grid
  traversal.
- **The injected-charge sum uses no atomics.** It is a cheap CuPy masked
  reduction over the small 2D density, with a precomputed mask that reproduces
  the kernels' scored-region test exactly (issue #19 P1/P2).

## 7. Agreement with the CPU backends

The per-voxel update is the *same* arithmetic as `solver_numba`, in the same
order, so the density **field matches the serial backend to near machine
epsilon** — measured max relative difference 5e-16 (single species) to 1.5e-15
(two species, over 2194 steps). The only differences are last-bit: CUDA
contracts `a*b + c` to an FMA by default, and the two reductions sum in a
different order. The scalar summaries (`f_t`, `k_s`) therefore agree to a
relative tolerance rather than bit-for-bit — exactly as the two CPU backends
already do with each other. Track positions come from the identical RNG stream
(the same `CylinderSampler`), so `track_density_xy` is bit-identical.

`tests/test_cuda_backend.py` checks all of this against the serial reference —
`f_t`, `k_s`, the full field, the track map, both wall conditions, both scoring
regions, and the two-species stencil — and skips cleanly where there is no GPU.

## 8. The optimisation that mattered

The first working port ran the 5 µm full electrode in **581 s**, and profiling
found the cost was not the sweep (101 s) but the **broadcast (470 s)**. The
broadcast kernel had one thread per `(i, j)` column looping over `k`; a warp's 32
threads then wrote to 32 columns `no_z_with_buffer` apart in memory — a 32-way
scatter, fully uncoalesced. Rewriting it as one thread per gap voxel with `k`
fastest-varying (the same layout the sweep uses) made the writes coalesce and cut
the kernel **~9.5×** (125 → 13 ms/step at 5 µm), taking the whole run from 581 s
to **156 s**. This is the standard GPU lesson — on a bandwidth-bound kernel the
memory access pattern is the algorithm — and it is why the sweep and the
broadcast both decode their flat index with `k` innermost.

Block size is 256 threads (a good default for a bandwidth-bound stencil on the
A100; it also sizes the reduction's shared arrays at 3 × 256 × 8 = 6 KiB). The
sweep reaches roughly 45 % of the A100's peak bandwidth, helped by L2 reuse of
the transverse neighbours; the deposition atomics are cheap here because tracks
spread across a wide grid rarely collide.

## 9. Caveats and future work

- **One GPU, one run.** This backend uses a single device and does not shard a
  grid across GPUs. For parameter studies, independent single-process replicas
  (one per GPU, or the CPU replica pattern of PERFORMANCE.md §6) beat trying to
  parallelise one run further.
- **Small grids belong on the CPU.** Below the crossover the CPU backends win;
  the example CLI still defaults to them, and `--backend cuda` is opt-in.
- **`--estimate-runtime-seconds` is CPU-only.** The empirical estimator knows
  only the two CPU backends; the GPU backend is fast enough on the example tiers
  that there is nothing to estimate — just run it.
- **Memory guard.** `run_simulation_cuda` checks the grid against free *device*
  memory before allocating (the host-side `SimulationConfig` guard checks system
  RAM, which on a fat node is far larger), and raises a clear `MemoryError`
  rather than failing inside a kernel.
- Further speed-ups left on the table: shared-memory tiling of the sweep's
  stencil, mixed-precision (the field is float64 throughout to match the CPU
  reference bit-close), and CUDA streams to overlap deposition sampling with the
  sweep.
