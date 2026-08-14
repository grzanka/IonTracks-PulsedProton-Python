# Running on Helios (Cyfronet)

How to run this code on a Cyfronet Helios node, how many cores to ask for, and
what wall time to expect. For the physics see [PHYSICS.md](PHYSICS.md), for the
algorithm [ALGORITHM.md](ALGORITHM.md), for the general cost model
[PERFORMANCE.md](PERFORMANCE.md).

**The one-line answer:** ask for **8–24 cores in one NUMA domain**, use
`--threads` equal to what you asked for, and spend anything beyond that on
independent replicas rather than on more threads for one run.

---

## 1. The machine

| | |
|---|---|
| node | 2 × AMD EPYC 9654, 96 cores each, **192 cores**, SMT off |
| NUMA | **8 domains of 24 cores** (NPS4 × 2 sockets) |
| L3 | 32 MiB per CCD (8 cores), **768 MiB per node** |
| RAM | 377 GiB, ~367 GiB usable |
| bandwidth | ~9 GB/s from one core; ~200 GB/s from one NUMA domain; ~900 GB/s node peak |

Two of those numbers drive everything below. **One core gets 9 GB/s**, which is
about 1 % of the node — a single-threaded run on Helios is *slower* than the
same run on a laptop, because a laptop core has a much larger share of its
machine's (smaller) bandwidth. And **L3 is 768 MiB**, so a grid below roughly
200 MiB per carrier array behaves completely differently from one above it.

## 2. Setup

```bash
module load GCCcore/13.3.0 Python/3.12.3   # repeat in EVERY new shell
source venv/bin/activate                    # the venv does not remember the module
pip install -e ".[dev]"
pytest                                      # ~10 s, 65 tests
```

`module load` must not be piped (`module load ... | tail` runs it in a subshell
and silently loses the environment).

## 3. The cpuset trap — read this before believing any timing

An interactive allocation hands back a shell whose **own process is pinned to
one CPU**, however many the job holds. `numba.set_num_threads(96)` in that shell
is clamped to 1, with a warning that is easy to miss in a long log.

```bash
python -c "import os; print(len(os.sched_getaffinity(0)))"   # prints 1
```

Every multi-core run needs its own step:

```bash
srun --overlap --ntasks=1 --cpus-per-task=24 --cpu-bind=none \
  python examples/ifj_aic144/run_markus_2mm.py full_electrode --threads 24
```

`--overlap` lets the step share the allocation's resources with the shell;
`--cpu-bind=none` stops Slurm from pinning the whole step onto one core.

**Do not set `OMP_PROC_BIND` / `OMP_PLACES`.** Measured on the full-electrode
grid: default placement 70 s, `OMP_PROC_BIND=spread OMP_PLACES=cores` **593 s**
— 8.5× worse. Under `--cpu-bind=none` the OpenMP place list does not describe
the CPUs the step actually holds, and the runtime stacks threads onto a handful
of cores. Leave placement to Slurm and the kernel.

## 4. How many cores to ask for

Measured, AIC-144 Markus 2 mm `full_electrode` tier (536² × 210 voxels,
1.9 GiB of carrier arrays, 24.6 M tracks, 2194 steps), one whole run each:

| threads | wall | speed-up | ms/step | k_s |
|---|---|---|---|---|
| 1 | see §5 | 1× | | 1.111065 |
| 8 | | | | 1.111065 |
| 16 | | | | 1.111065 |
| 24 | | | | 1.111065 |
| 48 | 113 s | | 51 | 1.111065 |
| 96 | **70 s** | | 32 | 1.111065 |
| 190 | 124 s | | 56 | 1.111065 |

`k_s` is identical to six digits at every thread count — the thread count
changes the order of a float reduction and nothing else.

Two things are worth reading off that table. The **best single-run point is
around one NUMA domain**, and past ~96 threads it gets worse, not better:
threads spread across both sockets, the kernels' static `prange` chunks stop
matching where the pages live, and run-to-run variance grows to tens of percent.
And the **curve is not monotonic** — 48 threads measured slower than 96 more
than once. Treat any single measurement above ~24 threads as ±30 %.

So: **8–24 threads per run.** Beyond that, use the cores for replicas — a Slurm
job array of single-domain jobs over seeds, dose rates or voltages gets close to
linear aggregate throughput, which more threads on one run does not.

## 5. What one core costs

*(filled in below by the single-core reference run)*

## 6. Why threads help here but not on a laptop

`docs/PERFORMANCE.md` §6 used to say threads make things worse, measured on the
5 µm "converged" grid: 40 MiB of carrier arrays, comfortably inside this node's
768 MiB of L3. That conclusion was correct for that grid and wrong as a general
rule, and the full-electrode grid is the opposite case: 1.9 GiB, DRAM-resident,
where one core is bandwidth-starved and threads buy memory controllers.

Three things had to be fixed before the big grid actually scaled. All three were
invisible below a few hundred MiB, which is why they survived the earlier study.

**1. NUMA first touch — 4.6× on the sweep.** `np.zeros` returns pages that are
mapped but not physically placed; Linux places each one on the NUMA domain of
the thread that first *writes* it. The old code's first writer was the main
thread, so all 1.9 GiB landed behind one of eight memory controllers:

| threads | sweep, main-thread first touch | sweep, per-thread first touch |
|---|---|---|
| 1 | 209 ms (9 GB/s) | 214 ms (9 GB/s) |
| 8 | 40 ms (48 GB/s) | 27 ms (72 GB/s) |
| 24 | 41 ms (47 GB/s) | **8.9 ms (216 GB/s)** |
| 96 | 44 ms (44 GB/s) | 11 ms (174 GB/s) |
| 190 | 56 ms (34 GB/s) | 8.4 ms (230 GB/s) |

The left column is flat at ~48 GB/s from 8 threads on — that is one memory
controller, and no number of threads changes it. `_first_touch_parallel` in
`solver_numba_parallel.py` now zeroes the arrays through the same flattened
`(i, j)` decomposition the kernels use.

**2. The serial copy-back — Amdahl.** Each step ended with
`positive_array[1:-1,1:-1,1:-1] = positive_next[...]`, single-threaded NumPy
over 1.9 GiB: **144 ms, unchanged at every thread count**, against a sweep that
now takes 8 ms. It alone capped the whole run at 2.5×. It is gone — the buffers
are swapped instead, and the outer ring that the copy was really there for is
either rewritten from the interior (reflecting wall) or carried across as four
planes, `O(X·Z)` instead of `O(X²·Z)`.

**3. Per-track Python — 110 s of sampling.** `sample_xy_inside_cylinder` cost
4.6 µs per track in the interpreter, so 24.6 M tracks spent ~110 s there before
any physics happened. `pulses.CylinderSampler` draws the same numbers in blocks,
166× faster, and *bit-identically*: it consumes the identical stream of
`rng.random()` doubles in the identical order, carrying a block's leftover into
the next call, so published `k_s` values stay tied to their seeds.

### What did not work

- **Threading the boundary update.** Four planes out of a 536² × 210 grid, and a
  `prange` version measured **5× slower** than the NumPy one (5.3 ms/step vs
  1.0 ms). NumPy issues them as four memcpys; the hand-rolled kernel writes
  element by element into pages that live on one or two NUMA domains. Fusing the
  eight parallel regions into two changed nothing, which is what ruled out
  launch overhead as the explanation.
- **Pinning threads** — see §3, 8.5× worse.
- **More than ~96 threads** — see §4.

## 7. Where the remaining time goes

Full-electrode run at 96 threads, `phase_timing=True`:

| phase | ms/step | share | threaded? |
|---|---|---|---|
| Lax-Wendroff sweep | 14.8 | 48 % | yes |
| broadcast of the batched density | 4.6 | 15 % | yes |
| swap + boundary planes | 1.0 | 3 % | no (memcpy) |
| per-track Gaussian factors | 2.9 | 9 % | **no** — serial `np.exp` |
| accumulate tracks into 2D | 2.0 | 7 % | yes |
| row index | 0.9 | 3 % | **no** — serial `njit` |
| track histogram, xy sampling | 0.8 | 3 % | no |

The sweep is the run, as it should be. The obvious next target is
`_precompute_track_gaussians` (~2.9 ms/step, 9 %): it is a serial NumPy `exp`
over `n_tracks × stencil`, and threading it is straightforward. It was left
alone deliberately — Numba's `exp` and NumPy's can differ in the last ULP, and
that would break the bit-for-bit reproduction of the archived `k_s` values that
everything else here preserves.

## 8. Reproducing the numbers on this page

```bash
# per-phase kernel scaling, including the NUMA first-touch comparison (~3 min)
srun --overlap --ntasks=1 --cpus-per-task=190 --cpu-bind=none \
  python -m profiling.bench_kernels --tier full_electrode \
  --threads 1,8,24,48,96,190 --init both \
  --json profiling/data/bench_kernels_full_electrode.json

# whole-run thread scaling on the full electrode (~20 min)
bash profiling/run_full_electrode_sweep.sh 8 16 24 48 96

# one run with the per-phase breakdown
srun --overlap --ntasks=1 --cpus-per-task=24 --cpu-bind=none python -c "
import sys; sys.path.insert(0, 'examples/ifj_aic144')
from run_markus_2mm import build_config
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel
run_simulation_numba_parallel(build_config('full_electrode'),
                              num_threads=24, progress=False, phase_timing=True)"
```
