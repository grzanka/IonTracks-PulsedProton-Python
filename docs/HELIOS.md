# Running on Helios (Cyfronet)

How to run this code on a Cyfronet Helios node, how many cores to ask for, and
what wall time to expect. For the physics see [PHYSICS.md](PHYSICS.md), for the
algorithm [ALGORITHM.md](ALGORITHM.md), for the general cost model
[PERFORMANCE.md](PERFORMANCE.md).

**The one-line answer:** for the shortest wall time on one run, ask for
**~128 cores** (full electrode: 593 s → 69 s, 8.6×). For the most science per
core-hour, ask for **2–8** and spend the rest of the node on independent
replicas: sixteen 8-thread jobs do 49x the work per hour that one 128-thread job
does. Never ask for all 190 — it is consistently slower than 128.

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
OMP_PLACES=cores` **593 s** — 8.5× worse. Under `--cpu-bind=none` the OpenMP place list does not describe
the CPUs the step actually holds, and the runtime stacks threads onto a handful
of cores. Leave placement to Slurm and the kernel.

## 4. How many cores to ask for

Measured, AIC-144 Markus 2 mm `full_electrode` tier (536² × 210 voxels, 1.9 GiB
of carrier arrays, 2194 steps), at two dose rates. Sixteen runs, each its own
Slurm job on its own node, reproducible with `./submit.sh`:

| threads | 10 Gy/s | speed-up | eff. | 50 Gy/s | speed-up | eff. |
|---|---|---|---|---|---|---|
| 1 | 593 s | 1.00× | 100 % | 755 s | 1.00× | 100 % |
| 2 | 351 s | 1.69× | 85 % | 399 s | 1.89× | 95 % |
| 4 | 207 s | 2.87× | 72 % | 325 s | 2.32× | 58 % |
| 8 | 195 s | 3.05× | 38 % | 370 s | 2.04× | 25 % |
| 16 | 169 s | 3.51× | 22 % | 256 s | 2.95× | 18 % |
| 32 | 131 s | 4.54× | 14 % | 177 s | 4.27× | 13 % |
| 64 | 78 s | 7.56× | 12 % | 139 s | 5.41× | 8 % |
| 128 | **69 s** | **8.63×** | 7 % | **97 s** | **7.80×** | 6 % |

`k_s` is `1.111065` in all eight 10 Gy/s runs and `1.446434` in all eight at
50 Gy/s — identical to six digits, at every thread count. The thread count
changes the order of one float reduction and nothing else, and `collect.py`
will not print this table until it has checked that. It is what makes the rest
of it worth reading.

Four things to read off it.

**Scaling is real but far from linear, and it is bandwidth that is missing, not
parallelism.** Amdahl is not the binding constraint at 10 Gy/s: the phases that
are still serial come to ~10 s of a 593 s run, which would permit ~59×. The
measured 8.6× is what the memory controllers will give.

**The curve has plateaus, and they are the memory hierarchy.** 4 → 8 threads
buys almost nothing (207 s → 195 s): eight cores are one CCD sharing one link,
and four already saturate it. Progress resumes only when a job spans more of
them — 32 → 64 threads is the biggest single step in the table (131 s → 78 s).
Cores are not what is being bought; memory controllers are.

**Per-core efficiency runs the other way, and that is the actionable half.**
2 threads returns 85–95 % of ideal, 128 threads returns 6–7 %. A node carved
into **sixteen 8-thread jobs** does 16 × 3.05 = 49× the work per hour that one
128-thread job's 8.6× does. For a parameter study — seeds, dose rates,
voltages — that is the configuration to use, as a Slurm job array. Reach for
128 threads only when one specific answer is wanted quickly.

**Do not ask for the whole node.** 190 threads measured 124 s against 128
threads' 69 s: past one socket the threads span both, the kernels' static
`prange` chunks stop matching where the pages live, and it gets worse. There is
no configuration in which asking for all 190 cores is right.

### A caveat on this table: the jobs were not exclusive

`./submit.sh` asks for `--cpus-per-task=N`, not for a node, so Slurm is free to
put other work on the same node — including other jobs from the same study.
Eleven of these sixteen runs shared a node with another of our own, and for a
benchmark that is *memory-bandwidth-bound* a co-tenant competes for precisely
the quantity being measured.

Re-measured on an uncontended node, at 50 Gy/s:

| threads | in the table (shared) | uncontended | inflation |
|---|---|---|---|
| 4 | 325 s | 290 s | +12 % |
| 8 | 370 s | 313 s | +18 % |

So read the table as ±20 %, and do not tune against differences smaller than
that. `./submit.sh --exclusive` gives a whole node per job and removes the
effect, at the price of charging 192 cores to run a 1-core job — worth it for a
number that is going to be quoted, wasteful otherwise.

**What the re-measurement did *not* overturn is the 4 → 8 inversion at 50 Gy/s.**
Uncontended, 8 threads (313 s) is still slower than 4 (290 s). That is real: at
this dose rate the deposition phases are a third of the run, they parallelise
over grid rows with an uneven number of tracks each, and adding threads inside a
single CCD adds imbalance and synchronisation without adding bandwidth. The
10 Gy/s column, where deposition is a tenth of the run, shows the same step as a
plateau rather than a reversal (207 s → 195 s).

## 5. Dose rate: 10 vs 50 Gy/s

The same grid at five times the track count. This pair separates the two halves
of the code, because the PDE sweep does not depend on dose rate *at all* and the
deposition phases scale linearly with it:

| threads | 10 Gy/s | 50 Gy/s | ratio |
|---|---|---|---|
| 1 | 593 s | 755 s | 1.27× |
| 2 | 351 s | 399 s | 1.14× |
| 4 | 207 s | 325 s | 1.57× |
| 8 | 195 s | 370 s | 1.90× |
| 16 | 169 s | 256 s | 1.52× |
| 32 | 131 s | 177 s | 1.35× |
| 64 | 78 s | 139 s | 1.78× |
| 128 | 69 s | 97 s | 1.41× |

**Five times the dose costs 1.3–1.9× the time, not 5×.** That is the batched
deposition working as designed: a step's tracks are summed into one 2D array and
broadcast down the gap once, so the `O(no_z)` part is paid per *step* and not
per *track* (see `solver_numba_parallel`). The PDE sweep, which is most of the
run, does not notice the dose rate at all.

**But the ratio drifts upward with thread count**, from ~1.2× at low counts to
~1.4–1.9× at high ones, and that drift is the diagnosis: the sweep parallelises
and the leftover per-track work does not. At 128 threads the sweep has shrunk to
a few ms/step while `_precompute_track_gaussians` — a serial NumPy `exp` over
`n_tracks × stencil` — has not moved. Phase timing at 96 threads makes it
explicit: that one phase is 9 % of the 10 Gy/s run and **33 %** of the 50 Gy/s
run, level with the sweep itself.

**So the higher the dose rate, the lower the thread count worth asking for.**
At 50 Gy/s, 128 threads returns 7.8× against 10 Gy/s's 8.6×, and the gap widens
with every further core. A FLASH-regime study — high dose rate, many parameter
points — should be a job array of 8-thread jobs, not a queue of wide ones.

The physics moves too, and much further than the cost: `k_s` goes from 1.1111 to
1.4464, i.e. 10.0 % of the charge recombines at 10 Gy/s against 30.9 % at
50 Gy/s. Five times the dose rate, three times the loss — the superlinearity
expected from a recombination term that goes as `n₊n₋`. See PHYSICS.md.

## 6. What one core costs

593 s for the full electrode at 10 Gy/s, 755 s at 50 Gy/s, and **2.0 s for the
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
