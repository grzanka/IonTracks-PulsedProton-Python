# Execution time

Cost model, measured timings and scaling for `pulsed_ion_chamber`. The physics
these numbers correspond to is described in [PHYSICS.md](PHYSICS.md), and the
implementation they measure in [ALGORITHM.md](ALGORITHM.md).

All timings are single-threaded on one development machine (`solver_numba` or
`solver_numba_parallel` with `num_threads=1`), excluding one-off JIT
compilation. Treat them as ratios and orders of magnitude, not as a benchmark.

---

## 1. Where the time goes

A run is a loop over time steps. Each step does two things:

| phase | cost | scales with |
|---|---|---|
| **track insertion** | `O(n_tracks · stencil²)` | tracks per pulse; independent of grid width |
| **PDE sweep + broadcast** | `O(n_steps · no_xy² · no_z)` | grid volume × step count; independent of dose |

`n_tracks ∝ dose_rate × sampled_radius²` and `no_xy ∝ sampled_radius`, so which
phase dominates depends on the scenario:

- **Small column, high dose rate** → insertion-bound.
- **Wide column, or low dose rate** → PDE-bound.

The crossover is roughly `n_tracks · stencil² ≈ n_steps · no_xy² · no_z`. On the
AIC-144 `archive` tier the two are within a factor of a few; by
`sampled_radius_cm = 0.05` the PDE sweep is ~90 % of the run.

The truncation stencil (PHYSICS §4) is what makes the insertion cost
grid-independent. Without it, insertion is `O(n_tracks · no_xy² · no_z)` and
grows as the fourth power of the column radius.

---

## 2. Reference timings

AIC-144 Markus 2 mm scenario (`examples/ifj_aic144/run_markus_2mm.py`), 10 µm
voxels, `buffer_radius=3`, two carrier species, 10 σ stencil, 2194 time steps:

| tier | `sampled_radius_cm` | grid | tracks/pulse | wall time (1 thread) | k_s | edge bias |
|---|---|---|---|---|---|---|
| `dev` | 0.003 (30 µm) | 12²×210 | 3 157 | 0.2 s | 1.0580 | −4.9 % |
| `archive` | 0.008 (80 µm) | 22²×210 | 22 447 | 1.8 s | 1.0929 | −1.7 % |
| `standard` | 0.014 (140 µm) | 34²×210 | 68 744 | 8.8 s | 1.1011 | −1.0 % |
| `wide` | 0.018 (180 µm) | 42²×210 | 113 638 | 14.5 s | 1.1035 | −0.8 % |
| `full_electrode` | 0.265 (2.65 mm) | 536²×210 | 24 630 400 | 12.8 min | 1.1111 | −0.1 % |

Cost grows as `r²` but the finite-column bias only falls as `1/r`, so no tier is
converged and enlarging the column is a bad way to chase the last percent. Apply
the `1/r` correction of PHYSICS §14 instead: `archive` plus the correction beats
`full_electrode` outright, in 1.8 s rather than 12.8 min.

**Backends.** `solver_numba.py` deposits one track at a time and is the
simpler baseline. `solver_numba_parallel.py` batches a whole step's deposition
and is the one to use for dense pulses and large grids; at `num_threads=1` it
is otherwise equivalent, and it is what the timings above use.

---

## 3. Scaling laws

**Column radius** is the steep one. With the stencil:

```
insertion  ∝ r²        (track count)
PDE sweep  ∝ r²        (grid area)
```

so the whole run is `∝ r²` — but only because the stencil decouples insertion
from grid width. Without truncation insertion alone is `∝ r⁴`, and that term
takes over completely on a wide column.

Measured, `solver_numba_parallel`, 10 σ stencil vs. no truncation:

| `sampled_radius_cm` | grid | no truncation | 10 σ stencil | speed-up |
|---|---|---|---|---|
| 0.008 | 22² | 0.90 s | 0.90 s | 1.00× |
| 0.030 | 66² | 11.9 s | 10.7 s | 1.11× |
| 0.050 | 106² | 41.4 s | 29.5 s | 1.40× |
| 0.080 | 166² | 166.8 s | 105.8 s | 1.58× |

The stencil is 28 voxels across, so it is *larger* than the grid on the
`archive` tier and does nothing there; the gain appears once the grid exceeds
the stencil and grows without bound thereafter. `k_s` is identical to all
printed digits in every row.

**Grid spacing** is the expensive one. Halving `grid_size_um` quadruples the
transverse grid, doubles the axial one, *and* halves `dt` through the von
Neumann condition, so the PDE term goes as `h⁻⁴`. Measured on the `archive`
tier: 0.93 s at 10 µm against 29.5 s at 5 µm, **32×** for one refinement level.

**Dose rate** scales the insertion term linearly and leaves the PDE term
untouched.

**Carrier model.** Resolving two species instead of one averaged pair costs
+35 % time steps: `dt` drops 20 % (set by the faster, more diffusive negative
ion) and the collection tail lengthens (set by the slower positive ion). That
surcharge lands entirely on the PDE term, so it is ~10 % of wall time in
insertion-bound scenarios and the full +35 % in PDE-bound ones — worst in the
low-dose-rate cases, where there is least other work to hide it.

---

## 4. Choosing the cutoff

`track_cutoff_sigmas` trades deposition cost against discarded charge. Measured
on the `archive` tier, where the stencil is comparable to the grid:

| cutoff | stencil width | charge discarded | k_s | wall time |
|---|---|---|---|---|
| none | (whole grid) | 0 | 1.09292081 | 1.93 s |
| 12 σ | 33.9 vox | 5.4e-32 | 1.09292081 | 1.92 s |
| **10 σ** | **28.3 vox** | **1.9e-22** | **1.09292081** | **1.80 s** |
| 8 σ | 22.6 vox | 1.3e-14 | 1.09292081 | 1.60 s |
| 6 σ | 17.0 vox | 1.5e-08 | 1.09292081 | 1.30 s |
| 4 σ | 11.3 vox | 3.4e-04 | 1.09291370 | 1.01 s |
| 3 σ | 8.5 vox | 1.1e-02 | 1.09256161 | 0.89 s |

The default 10 σ is bit-identical to no truncation. Anything below ~6 σ starts
to move `k_s`, in the direction the discarded-charge column predicts. On a grid
much wider than the stencil these differences vanish, because the cost is then
dominated by the PDE sweep — so tightening the cutoff buys the most exactly
where it is least justified.

---

## 5. Simulating the full electrode

The Classic Markus PTW 23343 collecting electrode is 5.3 mm across
(r = 2.65 mm, 0.2206 cm², 0.0441 cm³ of gas across the 2 mm gap). This has been
run, not just estimated — `run_markus_2mm.py full_electrode`:

```
grid          536 × 536 × 210 = 60.3 M voxels
tracks/pulse  24 630 400
time steps    2 194
wall time     768 s = 12.8 min   (batched backend, one thread; 843 s with the
                                 per-step record enabled, 646 s at 2 threads)
peak RSS      2.02 GiB           (1.80 GiB of carrier arrays + interpreter and temporaries)
result        f = 0.900037,  k_s = 1.111065
```

### Where those 768 seconds go

Each phase timed separately on the same grid, `CPU/wall = 1.00` throughout —
this is one core, not fourteen:

| phase | per step | steps | total | share |
|---|---|---|---|---|
| Lax-Wendroff sweep | 158 ms | 2 194 | 347 s | 45 % |
| copy-back of `_next` → current | 97 ms | 2 194 | 213 s | 28 % |
| broadcast of the batched density | 63 ms | 1 775 | 112 s | 15 % |
| xy rejection sampling | — | — | 60 s | 8 % |
| track deposition (phase 1) | 16 ms | 1 775 | 28 s | 4 % |
| **total** | | | **760 s** | vs 768 s measured |

The Lax-Wendroff sweep sustains **12 GB/s** on a single core (Intel Core Ultra 5
225U), which is where a memory-bound stencil over 1.8 GiB of arrays should land.
Nothing here is surprising once the arithmetic is done: the run streams the
carrier arrays several times per step, and that is the whole story.

Two of those rows are avoidable overhead rather than physics:

- **The copy-back is 28 % of the run and computes nothing.** It exists because
  the sweep writes only the interior of the `_next` arrays, so the boundary ring
  has to be carried over. Double-buffering with a pointer swap would remove it,
  but it is not a drop-in: under `lateral_boundary="absorbing"` the ring holds
  accumulated state that the `_next` arrays do not have, so the swap would have
  to copy the boundary planes explicitly — `O(X² + X·Z)` instead of `O(X²·Z)`.
- **xy sampling is 8 %**, spent in a Python-level loop calling
  `sample_xy_inside_cylinder` once per track — 2.4 µs each, 24.6 M times.
  Vectorising the rejection sampling over a whole batch would recover most of it.

Together they are ~35 % of this run, with no effect on the physics.

### Accuracy of the estimate

The cost model of §1 predicted ~18 min (2 min insertion + 16 min PDE), so it was
**40 % conservative**. The insertion term was about right; the per-step term was
not. Extrapolating per-step cost as `no_xy²` from a 166² grid assumes a constant
cost per voxel, but the measured figure *improved* from 7.4 ns/voxel at 166² to
4.9 ns/voxel at 536² — larger contiguous sweeps stream better, and the
zero-column skip in the broadcast helps once most of the grid is quiet. Treat
`no_xy²` extrapolation of the per-step term as an upper bound.

Peak RSS came in 12 % above `config.estimated_memory_bytes`, which counts the
four carrier arrays, the arrival-time draw and the 2D scratch but not the
interpreter, Numba's runtime or NumPy's transient temporaries. The default
`memory_budget_fraction = 0.8` absorbs that margin comfortably.

Without the deposition stencil the insertion term alone would be ~4 h on the
batched backend and ~16 days on the unbatched one.

**What it is good for, and what it is not.** At this size the run is PDE-bound
and memory-bandwidth-bound — the arrays are far larger than any cache and each
voxel-step touches ~20 doubles. This is the one regime where adding threads
should genuinely help, unlike the small grids of §6.

But as a way to get an accurate `k_s` it is a poor trade. The full electrode is
still 0.1 % below the infinite-column limit, while an 80 µm column corrected for
the `1/r` edge deficit lands within 0.06 % of it in **1.8 seconds** — 400× faster
and closer (PHYSICS §14). The full-electrode run's real value is that it
*verifies* that extrapolation across a 15× range in radius. Running the true
electrode geometry only becomes necessary alongside edge physics the model does
not currently have — guard-ring field distortion, non-uniform fluence — at which
point the answer would no longer be a scaled copy of the interior.

## 6. Using many cores

**Both hot loops are memory-bandwidth-bound, not compute-bound.** They stream
the whole grid every step, and on a machine whose memory controller a single
core can nearly saturate, extra threads have nothing left to win. Measure
before assuming.

### The ceiling is the memory subsystem

A pure STREAM triad — the simplest possible memory-bound kernel — on the
development machine (Intel Core Ultra 5 225U, 12 cores / 14 threads, LPDDR5,
one NUMA node):

| threads | triad | interior copy |
|---|---|---|
| 1 | 21.5 GB/s | 20.8 GB/s |
| 2 | 25.5 | 23.8 |
| 4 | 29.2 | 24.1 |
| 14 | 26.3 | 26.7 |

One core already reaches ~21 GB/s of a ~29 GB/s practical ceiling, so **nothing
memory-bound on this machine can gain more than ~1.35x**, whatever the thread
count.

The Lax-Wendroff sweep behaves exactly as that predicts — peaking at two
threads and degrading beyond:

| threads | sweep | broadcast | deposition |
|---|---|---|---|
| 1 | 195 ms (1.00x) | 66 ms (1.00x) | 9.2 ms (1.00x) |
| **2** | 117 ms (**1.67x**) | 55 ms (1.20x) | 4.7 ms (1.99x) |
| 4 | 124 ms (1.58x) | 52 ms (1.26x) | 3.0 ms (3.11x) |
| 12 | 133 ms (1.46x) | 57 ms (1.17x) | 2.0 ms (4.56x) |

Only deposition scales well — it is compute-bound — but it is 4 % of the run.
This is not threading overhead: the `omp` and `workqueue` layers give identical
curves (1.83x / 1.80x at two threads, 1.47x / 1.43x at twelve).

### Amdahl does the rest

Only about 64 % of a large run is parallel at all. The copy-back is plain NumPy
(~30 % of a step) and the xy rejection sampling is a Python loop (~8 % of the
run); neither is threaded. Combined with the bandwidth ceiling above:

| threads | full electrode (536^2 x 210) | speed-up | paired ratio, 2.4 M-voxel grid |
|---|---|---|---|
| 1 | 843.2 s (14.05 min) | 1.00x | 1.00x |
| **2** | **646.1 s (10.77 min)** | **1.30x** | **1.30x** |
| 12 | 685.3 s (11.42 min) | 1.23x | 1.22x |

**Two threads beats twelve**, and the two independent measurements agree to
0.01x on a grid 25x apart in size. The 2- and 12-thread full-electrode runs
were themselves paired back-to-back, so their 6 % difference is drift-free.
`k_s = 1.111065` at every thread count -- threading changes nothing physically.

### Measuring this on a laptop

Absolute timings drift by ~30 % with thermal state — enough to swamp the effect
and even to invert it. A naive sequential sweep on the full electrode produced
*994 s at 4 threads and 1033 s at 2*, both apparently slower than one thread,
purely because they ran after twelve minutes of sustained full load.

Use a **paired design**: alternate the two conditions back-to-back so any drift
slower than one pair cancels, and compare ratios rather than absolute times.
Once settled, repeat timings are stable to under 1 %, and the paired ratios
above are reproducible to ±0.02.

### The right way to spend many cores: replicas

Since one run gains little from threads, the way to use a large machine is
**many independent single-threaded runs** — different seeds, or a sweep over
dose rate, energy and voltage — via `multiprocessing.Pool` or a Slurm job
array. Measured on a 192-core node: 64 concurrent single-threaded replicas
finished in 69 s against ~1600 s sequentially, a ~23x aggregate speed-up for
64x the statistics. The shortfall from a theoretical 64x is memory-bandwidth
and L3 contention plus concurrent JIT compilation; `--exclusive` single-core
Slurm jobs avoid most of it.

Note that all the numbers above are from one laptop-class chip with a single
memory controller. A server part with eight controllers has far more bandwidth
headroom, and the same run should scale considerably better there — the
*shape* of the argument transfers, the numbers do not.

### Slurm / cpuset caveat

An allocation requested as `--ntasks=N --cpus-per-task=1` hands back a shell
whose own process is restricted to a single core, even when the whole node is
reserved. `numba.set_num_threads()` cannot help if the OS exposes one CPU.
Launch multi-core or multi-process work as its own step:

```bash
srun --overlap --ntasks=1 --cpus-per-task=190 --cpu-bind=none python your_script.py
```

Verify before trusting any thread-count benchmark:

```bash
python -c "import os; print(len(os.sched_getaffinity(0)))"
```

---

## 7. Estimating before running

`pulsed_ion_chamber.benchmark.estimate_full_runtime(config)` times a handful of
track insertions and PDE steps on the config's actual grid and extrapolates,
without running the simulation. Use it to size a job before submitting it. It
reports the per-track and per-step costs separately, which also tells you which
of the two phases in §1 a given scenario is bound by — and therefore which knob
is worth turning.
