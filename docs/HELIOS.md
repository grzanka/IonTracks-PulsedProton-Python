# Running on Helios (Cyfronet)

How to run this code on a Cyfronet Helios node, how many cores to ask for, and
what wall time to expect. For the physics see [PHYSICS.md](PHYSICS.md), for the
algorithm [ALGORITHM.md](ALGORITHM.md), for the general cost model
[PERFORMANCE.md](PERFORMANCE.md).

**The one-line answer:** for the shortest wall time on one run, ask for
**32 cores** (full electrode: 572 s → 47 s, 12.2×); 64 if the dose rate is high. For the most science per
core-hour, ask for **8** and spend the rest of the node on independent replicas:
24 concurrent 8-thread jobs do 12x the work per node-hour that one 128-thread
job does. Never ask for the whole node for one run, and **always pass
`--exclusive` for numbers you will quote** — a co-tenant on the node inflated
this study's mid-range by up to 195 %.

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
grid at 96 threads: default placement 70 s, `OMP_PROC_BIND=spread
OMP_PLACES=cores` **593 s** — 8.5× worse. (Both measured on a shared node, so
read the pair as a ratio, not as absolute times.) Under `--cpu-bind=none` the OpenMP place list does not describe
the CPUs the step actually holds, and the runtime stacks threads onto a handful
of cores. Leave placement to Slurm and the kernel.

## 4. How many cores to ask for

Measured, AIC-144 Markus 2 mm `full_electrode` tier (536² × 210 voxels, 1.9 GiB
of carrier arrays, 2194 steps), at two dose rates. Sixteen runs, **one job per
point with a whole node to itself** — `./submit.sh --exclusive`:

| threads | 10 Gy/s | speed-up | eff. | 50 Gy/s | speed-up | eff. |
|---|---|---|---|---|---|---|
| 1 | 572 s | 1.00× | 100 % | 713 s | 1.00× | 100 % |
| 2 | 299 s | 1.91× | 96 % | 398 s | 1.79× | 90 % |
| 4 | 162 s | 3.53× | 88 % | 237 s | 3.01× | 75 % |
| 8 | 91 s | 6.32× | 79 % | 149 s | 4.77× | 60 % |
| 16 | 57 s | 9.99× | 62 % | 147 s | 4.85× | 30 % |
| 32 | **47 s** | **12.23×** | 38 % | 102 s | 6.99× | 22 % |
| 64 | 47 s | 12.18× | 19 % | **82 s** | **8.68×** | 14 % |
| 128 | 49 s | 11.59× | 9 % | 83 s | 8.55× | 7 % |

`k_s` is `1.111065` in all eight 10 Gy/s runs and `1.446434` in all eight at
50 Gy/s — identical to six digits at every thread count. `collect.py` will not
print this table until it has checked that.

**Scaling is good to about 8 threads and then decays smoothly.** 79 % of ideal
at 8 threads, 62 % at 16. This is a memory-bandwidth-bound stencil over 1.9 GiB;
that it parallelises this well is the whole point of the NUMA first-touch and
copy-back work in §7.

**The optimum is 32 threads at 10 Gy/s and 64 at 50 Gy/s — and nothing beyond.**
Both curves are flat from there and *rise* at 128. There is no configuration in
which asking for the whole node helps a single run.

**Two different things stop the two curves**, and the arithmetic says which:

| | plateau | implied serial fraction | Amdahl ceiling |
|---|---|---|---|
| 10 Gy/s | 12.2× at 32 threads | ~5 % | ~19× |
| 50 Gy/s | 8.7× at 64 threads | ~10 % | ~10× |

At 50 Gy/s the measured 8.7× is already at its Amdahl ceiling: the run is capped
by the deposition work that was never threaded (§8), and no amount of bandwidth
would help. At 10 Gy/s the ceiling is ~19× and only 12.2× is reached, so *that*
curve is stopped by memory controllers rather than by serial code. Two different
problems that happen to produce similarly-shaped curves.

**Per-core efficiency argues for narrow jobs.** A whole node's aggregate
throughput, jobs × speed-up:

| shape | 10 Gy/s | 50 Gy/s |
|---|---|---|
| 24 × 8-thread | **152×** | **114×** |
| 12 × 16-thread | 120× | 58× |
| 6 × 32-thread | 73× | 42× |
| 1 × 128-thread | 12× | 9× |

So a parameter study — seeds, dose rates, voltages — should be a Slurm array of
**8-thread jobs**, which gets 12× the work per node-hour that one wide job does.
Reach for 32–64 threads only when one specific answer is wanted quickly.

### Ask for exclusive nodes, or measure something else

`--cpus-per-task=N` asks Slurm for cores, not for a node, so other work lands
beside yours — including other jobs from the same study. The first run of this
study was not exclusive, and eleven of its sixteen jobs shared a node with
another of our own. For a memory-bandwidth-bound benchmark a co-tenant competes
for precisely the quantity being measured:

| threads | shared node | exclusive | inflation |
|---|---|---|---|
| 4 | 207 s | 162 s | +28 % |
| 8 | 195 s | 91 s | **+115 %** |
| 16 | 169 s | 57 s | **+195 %** |
| 32 | 131 s | 47 s | **+179 %** |
| 128 | 69 s | 49 s | +39 % |

The distortion peaks in the middle of the range, which is the worst possible
shape: it does not merely add noise, it *moves the optimum*. Read off the shared
numbers, 128 threads looked best and the mid-range looked like a plateau. Both
were artefacts. **Use `./submit.sh --exclusive` for any number you intend to
quote**; the cost is that a 1-core job books 192 cores for twelve minutes.

## 5. Dose rate: 10 vs 50 Gy/s

The same grid at five times the track count. The pair separates the two halves
of the code, because the PDE sweep does not depend on dose rate *at all* while
the deposition phases scale linearly with it:

| threads | 10 Gy/s | 50 Gy/s | ratio |
|---|---|---|---|
| 1 | 572 s | 713 s | 1.25× |
| 2 | 299 s | 398 s | 1.33× |
| 4 | 162 s | 237 s | 1.46× |
| 8 | 91 s | 149 s | 1.65× |
| 16 | 57 s | 147 s | 2.57× |
| 32 | 47 s | 102 s | 2.18× |
| 64 | 47 s | 82 s | 1.75× |
| 128 | 49 s | 83 s | 1.69× |

**Five times the dose costs 1.25× the time on one core.** That is the batched
deposition working as designed: a step's tracks are summed into one 2D array and
broadcast down the gap once, so the `O(no_z)` part is paid per *step*, not per
*track*. The PDE sweep, which is most of a single-core run, does not notice the
dose rate at all.

**But the ratio doubles as threads are added**, peaking at 2.6× around 16
threads. That drift is the diagnosis: the sweep parallelises and the leftover
per-track work does not. Phase timing at 96 threads makes it explicit —
`_precompute_track_gaussians`, a serial NumPy `exp` over `n_tracks × stencil`,
is 9 % of the 10 Gy/s run and **33 %** of the 50 Gy/s one, level with the sweep
itself (§8).

**So the higher the dose rate, the less a wide job buys.** At 50 Gy/s the best
speed-up available is 8.7× against 10 Gy/s's 12.2×, and it needs twice the cores
to get there. A FLASH-regime campaign should be an array of 8-thread jobs, where
the ratio is still only 1.65×.

The physics moves much further than the cost: `k_s` goes from 1.1111 to 1.4464 —
10.0 % of the charge recombining at 10 Gy/s against 30.9 % at 50 Gy/s. Five times
the dose rate, three times the loss, which is the superlinearity expected of a
recombination term in `n₊n₋`. See PHYSICS.md.

### How this compares to Ares

The same study on Cyfronet Ares (2 × Xeon 8268, 48 cores, ~200 GB/s) came out
slower on both axes — 968 s on one core against 572 s, and 105 s at its best
against 47 s. The per-core gap is a flat 1.7–1.9× from one thread upward, so it
is the core and not the scaling. [BENCHMARKS-ARES.md](BENCHMARKS-ARES.md) §6 has
the numbers and a testable hypothesis (AVX-512 frequency licensing on Cascade
Lake) for why an ostensibly faster-clocked core loses.

Worth stating because the prediction went the other way: Helios is not just
wider here, it is also quicker per core on this kernel.

## 6. What one core costs

572 s for the full electrode at 10 Gy/s, 713 s at 50 Gy/s, and **2.0 s for the
`archive` tier**. The same
`archive` run takes 1.8 s on a 2024 laptop (Intel Core Ultra 5 225U), so a
Helios core is about **10 % slower than a laptop core** on this workload. That is
the expected shape: an EPYC 9654 core runs at a lower clock and gets ~9 GB/s of
memory bandwidth, roughly 1 % of its node's ~900 GB/s, where a laptop core gets
most of its machine's ~29 GB/s.

The consequence worth internalising: **Helios is not a faster computer, it is a
wider one.** Nothing here gets quicker by being moved to Helios unless it is
either given many cores on a grid large enough to use them, or replicated.

## 7. Why threads help here but not on a laptop

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

## 8. Where the remaining time goes

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

## 9. Reproducing the numbers on this page

The §4 and §5 tables, from a Helios **access** node (compute nodes cannot
submit; Slurm's error is a bare "Access/permission denied"):

```bash
./submit.sh --exclusive                        # 16 jobs, ~15 min if the queue is free
squeue -u $USER
python profiling/cluster_scaling/collect.py     # tables, and the consistency checks
```

Drop `--exclusive` only for a rough look — see §4 for what it costs in accuracy.
`profiling/cluster_scaling/README.md` explains the job layout.

The supporting measurements, from inside an interactive allocation:

```bash
# per-phase kernel scaling, including the NUMA first-touch comparison (~3 min)
srun --overlap --ntasks=1 --cpus-per-task=190 --cpu-bind=none \
  python -m profiling.bench_kernels --tier full_electrode \
  --threads 1,8,24,48,96,190 --init both \
  --json profiling/data/bench_kernels_full_electrode.json

# one run with the per-phase breakdown of §8
srun --overlap --ntasks=1 --cpus-per-task=32 --cpu-bind=none python -c "
import sys; sys.path.insert(0, 'examples/ifj_aic144')
from run_markus_2mm import build_config
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel
run_simulation_numba_parallel(build_config('full_electrode'),
                              num_threads=32, progress=False, phase_timing=True)"
```

The laptop counterpart is `./bench_laptop.sh` — see
[BENCHMARKS-LAPTOP.md](BENCHMARKS-LAPTOP.md) and
`profiling/laptop_scaling/README.md`.
