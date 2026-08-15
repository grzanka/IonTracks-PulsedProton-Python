# Execution time

**Cost model and scaling laws** for `pulsed_ion_chamber` -- what a run costs as
a function of the knobs, independent of the machine it runs on. The physics
these numbers correspond to is described in [PHYSICS.md](PHYSICS.md), and the
implementation they measure in [ALGORITHM.md](ALGORITHM.md).

**Measured wall times live on the machine pages**, because they are properties
of a machine and not of this code:

| | |
|---|---|
| [BENCHMARKS-LAPTOP.md](BENCHMARKS-LAPTOP.md) | Intel Core Ultra 5 225U. Tier timings, the full-electrode run, why one thread is the right number there. |
| [HELIOS.md](HELIOS.md) | Cyfronet Helios, dual EPYC 9654. How to run it, how many cores to ask for, thread scaling and dose-rate scaling. |

Ratios quoted below were measured on the laptop unless stated otherwise; treat
them as ratios and orders of magnitude, not as a benchmark.

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

## 2. Reference scenario

Everything below is measured on the AIC-144 Markus 2 mm scenario
(`examples/ifj_aic144/run_markus_2mm.py`): 10 µm voxels, `buffer_radius=3`, two
carrier species, 10 σ stencil, 2194 time steps, five grid tiers from a 30 µm
column to the full 2.65 mm electrode.

The tier table -- grid, track count, wall time, `k_s` and edge bias per tier --
is in [BENCHMARKS-LAPTOP.md](BENCHMARKS-LAPTOP.md) §2, with the Helios
equivalent in [HELIOS.md](HELIOS.md) §4. `k_s` is identical on both; only the
wall times differ.

Cost grows as `r²` while the finite-column bias falls only as `1/r`, so no tier
is converged and enlarging the column is a bad way to chase the last percent.
Apply the `1/r` correction of PHYSICS §14 instead: `archive` plus the correction
beats `full_electrode` outright, in seconds rather than minutes.

**Backends.** `solver_numba.py` deposits one track at a time and is the simpler
baseline. `solver_numba_parallel.py` batches a whole step's deposition and is
the one to use for dense pulses, large grids, and anything with a thread count;
at `num_threads=1` it is otherwise equivalent.

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

The Classic Markus PTW 23343 collecting electrode is 5.3 mm across (r = 2.65 mm,
0.2206 cm², 0.0441 cm³ of gas across the 2 mm gap), which is a 536 × 536 × 210
grid: 60.3 M voxels, 1.9 GiB of carrier arrays, 24.6 M tracks per pulse. It has
been run on both machines, not estimated:

| | laptop, 1 core | Ares, 1 core | Helios, 1 core | Ares, 24 cores | Helios, 32 cores |
|---|---|---|---|---|---|
| wall time | **562 s**¹ | 968 s | 572 s | 105 s | **47 s** |
| `k_s` | 1.111065 | 1.111065 | 1.111065 | 1.111065 | 1.111065 |

¹ after the optimisations of HELIOS.md §6 (copy-back → buffer swap, batched xy
sampling); before them it was 768 s. The laptop core is the fastest single
core of the three, edging out Helios by 10 s.

At this size the run is PDE-bound and memory-bandwidth-bound — the arrays are
far larger than any cache and each voxel-step touches ~20 doubles. That is the
regime where threads genuinely help, and the only one.

`k_s` is identical on every machine at every thread count, which is the check
that makes the wall-time row worth comparing at all.

Details: [BENCHMARKS-LAPTOP.md](BENCHMARKS-LAPTOP.md) §3 for the single-core
phase breakdown and the memory accounting, [HELIOS.md](HELIOS.md) for the thread
scaling and what had to change to get it, [BENCHMARKS-ARES.md](BENCHMARKS-ARES.md)
for why the older Xeon loses by 1.7× per core despite the higher clock.

**Is it worth running?** As a way to get an accurate `k_s`, no — an 80 µm column
plus the `1/r` edge correction lands closer to the infinite-column limit, far
faster (PHYSICS §14). Its value is that it *verifies* that extrapolation across
a 15× range in radius. The true electrode geometry only becomes necessary
alongside edge physics the model does not have — guard-ring field distortion,
non-uniform fluence.

---

## 6. Using many cores

**It depends entirely on whether the grid fits in cache.** Both hot loops are
memory-bandwidth-bound, so what threads buy is memory controllers — and a grid
that lives in L3 has already got all the bandwidth it is going to get.

**Small grids: use one thread.** Measured on a dual-EPYC 192-core node with the
5 µm "converged" grid (40 MiB of carrier arrays, against 768 MiB of node L3),
wall time got *worse* with more threads: 22.9 s at 1, 37–44 s at 8–190 with the
`omp` layer. Each parallel region does a few hundred microseconds of work and
there are thousands of them, so fork/join and cross-NUMA barrier cost dominates.

**Large grids: threads are the whole point.** The full-electrode grid is 1.9 GiB
— DRAM-resident, and one core can pull only ~9 GB/s of a ~900 GB/s node. The
same run is 572 s on one core and 47 s on 32, with `k_s` identical to six
digits — 12.2×, at 38 % of ideal. Getting there needed three fixes that only matter above a few hundred
MiB — NUMA first touch, deleting the serial copy-back, and taking per-track
sampling out of the Python interpreter.

The crossover is roughly where the carrier arrays exceed the machine's total
L3. Below it, one thread; above it, as many as the memory controllers can feed.

**It is also a property of the machine, not of this code.** The same grid that
wants one thread on a laptop (where one core already reaches ~70 % of the
machine's bandwidth) wants ~96 on a Helios node (where one core reaches ~1 %).
Both pages state their own answer:
[BENCHMARKS-LAPTOP.md](BENCHMARKS-LAPTOP.md) §4, [HELIOS.md](HELIOS.md) §4 —
the latter also lists the optimisations that turned out to make things *worse*,
which is the more transferable half.

**For parameter studies, independent replicas beat threads either way**:
different seeds, or a sweep over dose rate, energy and voltage, run as separate
single-threaded processes. Measured: 64 concurrent single-threaded replicas via
`multiprocessing.Pool(64)` finished in 69 s against ~1600 s for the same 64 runs
sequentially — a ~23× aggregate speed-up for 64× the statistics. The shortfall
from a theoretical 64× is memory-bandwidth and L3 contention plus concurrent JIT
compilation; a Slurm job array of `--exclusive` single-core jobs avoids most of
it.

`run_simulation_numba_parallel(num_threads=...)` clamps the request to the
process's CPU affinity mask and to Numba's configured maximum, warning when it
does, so a benchmark cannot report a thread count that never existed.

### Slurm / cpuset caveat

An interactive allocation hands back a shell whose own process is restricted to
a single core, even when the whole node is reserved, so every thread count in it
silently collapses to 1. Verify before trusting any thread-count benchmark:

```bash
python -c "import os; print(len(os.sched_getaffinity(0)))"
```

[HELIOS.md](HELIOS.md) §3 has the full treatment and the launch incantation.

---

## 7. Estimating before running

`pulsed_ion_chamber.benchmark.estimate_full_runtime(config)` times a handful of
track insertions and PDE steps on the config's actual grid and extrapolates,
without running the simulation. Use it to size a job before submitting it. It
reports the per-track and per-step costs separately, which also tells you which
of the two phases in §1 a given scenario is bound by — and therefore which knob
is worth turning. Its per-track sample uses the unbatched kernel, so on a
large, batched (`--threads > 1`) run treat the resulting number as an upper
bound, not a wall-time prediction — see the caution `--dry-run` prints below.

Memory gets the same treatment, and for the same reason: a full-electrode-sized
grid is gigabytes, and finding that out from the OOM killer twenty minutes into
a run is worse than finding out before it starts. `SimulationConfig` computes
`estimated_memory_bytes` for every config (four carrier arrays, the
arrival-time draw, and the batched backend's 2D scratch — whichever phase's
allocation peaks) and refuses the config in its constructor if it exceeds
`memory_budget_fraction` (default 0.8) of currently available RAM.
`pulsed_ion_chamber.resources.memory_report(config.estimated_memory_bytes,
config.memory_budget_fraction)` turns that into a human-readable comparison
against this machine's total and available RAM.

`examples/ifj_aic144/run_markus_2mm.py --dry-run` wires both of the above
together: it builds the config (so the memory guard has already run), prints
the memory report and the runtime estimate, and exits without simulating
anything.
[docs/BENCHMARKS-LAPTOP.md](BENCHMARKS-LAPTOP.md) sec. 3 has the measured peak
RSS this estimate has been checked against on the full-electrode grid.
