# Benchmarks: laptop

Measured wall times on the development laptop. The companion page is
[HELIOS.md](HELIOS.md) (Cyfronet Helios node); the machine-independent cost
model and scaling laws are in [PERFORMANCE.md](PERFORMANCE.md).

Read this page to size a run you are about to start on a laptop, and to know
which numbers in the repository came from here.

---

## 1. The machine

| | |
|---|---|
| CPU | Intel Core Ultra 5 225U |
| hardware threads | 14 |
| memory bandwidth | ~29 GB/s practical ceiling; **~21 GB/s reachable from a single core** |
| sweep bandwidth achieved | 12 GB/s on one core over 1.8 GiB of arrays |
| P-core clock sustained | 4783–4793 MHz on one core, 4202 on two, 3412 on eight |

The second row of numbers is the one that matters, and it is what makes this
machine behave so differently from a compute node. **One core here can already
pull ~70 % of the whole machine's bandwidth.** Both hot loops are
memory-bandwidth-bound, so there is very little left for a second thread to
win — see §4.

To confirm the spec on your own machine:

```bash
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|L3"
```

## 2. Reference timings — AIC-144 Markus 2 mm

`examples/ifj_aic144/run_markus_2mm.py`, 10 µm voxels, `buffer_radius=3`, two
carrier species, 10 σ stencil, 2194 time steps, single-threaded, excluding JIT:

| tier | `sampled_radius_cm` | grid | tracks/pulse | wall time | k_s | edge bias |
|---|---|---|---|---|---|---|
| `dev` | 0.003 (30 µm) | 12²×210 | 3 157 | 0.2 s | 1.0580 | −4.9 % |
| `archive` | 0.008 (80 µm) | 22²×210 | 22 447 | 1.8 s | 1.0929 | −1.7 % |
| `standard` | 0.014 (140 µm) | 34²×210 | 68 744 | 8.8 s | 1.1011 | −1.0 % |
| `wide` | 0.018 (180 µm) | 42²×210 | 113 638 | 14.5 s | 1.1035 | −0.8 % |
| `full_electrode` | 0.265 (2.65 mm) | 536²×210 | 24 630 400 | 12.8 min | 1.1111 | −0.1 % |

`k_s` is a property of the physics and is identical on both machines; only the
wall-time column is a property of this one.

**How this compares to a compute node.** The `archive` tier takes 1.8 s here and
2.0 s on one Helios core — a laptop core is ~10 % *faster*. Helios only wins by
being wide: same tier, same code (HELIOS.md §6).

## 3. Simulating the full electrode

The Classic Markus PTW 23343 collecting electrode is 5.3 mm across
(r = 2.65 mm, 0.2206 cm², 0.0441 cm³ of gas across the 2 mm gap). This has been
run here, not just estimated:

```
grid          536 × 536 × 210 = 60.3 M voxels
tracks/pulse  24 630 400
time steps    2 194
wall time     768 s = 12.8 min   (batched backend, one thread)
peak RSS      2.02 GiB           (1.80 GiB of carrier arrays + interpreter and temporaries)
result        f = 0.900037,  k_s = 1.111065
```

### Where those 768 seconds went

Each phase timed separately on the same grid, `CPU/wall = 1.00` throughout —
one core, not fourteen:

| phase | per step | steps | total | share |
|---|---|---|---|---|
| Lax-Wendroff sweep | 158 ms | 2 194 | 347 s | 45 % |
| copy-back of `_next` → current | 97 ms | 2 194 | 213 s | 28 % |
| broadcast of the batched density | 63 ms | 1 775 | 112 s | 15 % |
| xy rejection sampling | — | — | 60 s | 8 % |
| track deposition (phase 1) | 16 ms | 1 775 | 28 s | 4 % |
| **total** | | | **760 s** | vs 768 s measured |

The sweep sustains 12 GB/s on a single core, which is where a memory-bound
stencil over 1.8 GiB of arrays should land. Nothing here is surprising once the
arithmetic is done: the run streams the carrier arrays several times per step,
and that is the whole story.

**This table is the "before" picture.** Two of its rows were avoidable overhead
rather than physics, and both have since been removed (HELIOS.md §7): the
copy-back became a buffer swap, and the per-track xy sampling became a blocked,
bit-identical batch draw.

**Re-measured after those fixes: 562.1 s on one P-core** (`./bench_laptop.sh
--stage scaling`), against 768 s before — a 27 % cut, and close to the ~500 s
that was estimated here while it was still a guess. `k_s = 1.111065`, matching
Helios and Ares exactly.

### The laptop core is the fastest of the three machines

| | single core, full electrode, 10 Gy/s |
|---|---|
| **laptop** (Core Ultra 5 225U, 1 P-core) | **562 s** |
| Helios (EPYC 9654) | 572 s |
| Ares (Xeon Platinum 8268) | 968 s |

A 2024 laptop P-core edges out a Helios core and beats an Ares core by 1.7× on
this kernel. Nothing about a compute node makes a *core* fast; what a node sells
is width, and this code can only use ~12× of Helios's 192. See
[BENCHMARKS-ARES.md](BENCHMARKS-ARES.md) §6 for why the Xeon loses.

### Accuracy of the estimate

The cost model of PERFORMANCE.md §1 predicted ~18 min (2 min insertion +
16 min PDE), so it was **40 % conservative**. The insertion term was about
right; the per-step term was not. Extrapolating per-step cost as `no_xy²` from a
166² grid assumes a constant cost per voxel, but the measured figure *improved*
from 7.4 ns/voxel at 166² to 4.9 ns/voxel at 536² — larger contiguous sweeps
stream better, and the zero-column skip in the broadcast helps once most of the
grid is quiet. Treat `no_xy²` extrapolation of the per-step term as an upper
bound.

Peak RSS came in 12 % above `config.estimated_memory_bytes`, which counts the
four carrier arrays, the arrival-time draw and the 2D scratch but not the
interpreter, Numba's runtime or NumPy's transient temporaries. The default
`memory_budget_fraction = 0.8` absorbs that margin comfortably.

Without the deposition stencil the insertion term alone would be ~4 h on the
batched backend and ~16 days on the unbatched one.

### RAM on two P-cores, both dose rates

§4 finds two P-cores is the optimal thread count for this grid, so that is
also the setting to know the memory footprint of. Measured with the
peak-RSS instrumentation `examples/ifj_aic144/run_markus_2mm.py` now carries
(`resource.getrusage().ru_maxrss`), same command as §4's ladder,
`--threads 2`:

| dose rate | wall time | tracks/pulse | peak RSS | `config.estimated_memory_bytes` | margin |
|---|---|---|---|---|---|
| 10 Gy/s | 381.9 s | 24 630 400 | **2.03 GiB** | 1.80 GiB (carrier arrays dominate) | +12.8 % |
| 50 Gy/s | 496.0 s | 123 152 000 | **2.16 GiB** | 1.84 GiB (arrival-time draw dominates) | +17.7 % |

`k_s` matches §4's single-run figures to 6 decimal places (1.111065 and
1.446434), and both wall times reproduce the ladder in §4 within ~2 %
(381.9 s vs. 373.2 s; 496.0 s vs. 499.2 s) — laptop-to-laptop-run noise from
thermal state, not a different measurement (§4 already flags this ladder as
thermally throttled; this re-run started a few degrees warmer).

**The cross-check.** `estimated_memory_bytes` is `max(track_schedule_bytes,
carrier_array_bytes + scratch_bytes)` (config.py `_estimate_and_check_memory`)
— it takes whichever of the two phases peaks, not their sum, since the
arrival-time draw is freed before the solver arrays are touched. Which term
wins depends on the dose rate: at 10 Gy/s the four carrier arrays (1.80 GiB,
fixed by the grid) dominate; at 50 Gy/s the arrival-time draw for 123 M tracks
(2 float64 arrays, 1.84 GiB) edges narrowly ahead of them. Both peak-RSS
figures land above the estimate by roughly the same margin the single-thread
run in this section found (12 %), a little more at 50 Gy/s — plausibly the
larger transient allocation (1.84 GiB vs. 375 MiB at 10 Gy/s) leaves more
allocator overhead behind it, though this hasn't been isolated further.
Either way `memory_budget_fraction`'s default 0.8 headroom absorbs it with
enormous room to spare on this 30.8 GiB laptop — the interesting case is a
smaller machine, which is exactly what §7's `--dry-run` flag is for.

### Is it worth running?

As a way to get an accurate `k_s`, no. The full electrode is still 0.1 % below
the infinite-column limit, while an 80 µm column corrected for the `1/r` edge
deficit lands within 0.06 % of it in **1.8 seconds** — 400× faster and closer
(PHYSICS §14). The full-electrode run's real value is that it *verifies* that
extrapolation across a 15× range in radius. Running the true electrode geometry
only becomes necessary alongside edge physics the model does not currently
have — guard-ring field distortion, non-uniform fluence — at which point the
answer would no longer be a scaled copy of the interior.

## 4. Threads on a laptop: two P-cores, then a plateau

Measured, `./bench_laptop.sh --stage scaling`, full electrode, both dose rates,
pinned one thread per physical core, checked self-consistent by
[`profiling/cluster_scaling/collect.py`](../profiling/cluster_scaling/collect.py)
(`k_s` identical to 12 decimal places at every thread count, at both rates):

| threads | cores | 10 Gy/s wall | speed-up | eff. | 50 Gy/s wall | speed-up | eff. | clock (10 / 50 Gy/s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1P | 562.1 s | 1.00× | 100 % | 715.3 s | 1.00× | 100 % | 4793 / 4797 MHz |
| 2 | 2P | 373.2 s | 1.51× | 75 % | 499.2 s | 1.43× | 72 % | 4202 / 4282 MHz |
| 4 | 2P + 2E | 382.4 s | 1.47× | 37 % | 492.1 s | 1.45× | 36 % | 3663 / 3761 MHz |
| 8 | 2P + 6E | 391.0 s | 1.44× | 18 % | 489.6 s | 1.46× | 18 % | 3342 / 3412 MHz |

**At 10 Gy/s, two P-cores isn't just where adding more stops paying off — it's
the minimum.** Wall time bottoms out at 373 s and then gets *worse*: 382 s at
4 threads, 391 s at 8. The physics hasn't moved (`k_s = 1.111065` in all four
rows); the clock has — 4793 → 3342 MHz, 30 % down, as E-cores join in. A laptop
shares one power/thermal budget across every core, P or E, so past the second
P-core the extra threads aren't reaching any spare bandwidth — they're
competing for the bandwidth the first two cores already claimed, and dragging
the whole package's clock down while they do it.

**At 50 Gy/s the same four thread counts behave differently.** Wall time keeps
falling past 2 cores, just barely: 499 → 492 → 490 s, a further 2 % from six
added E-cores. It doesn't reverse. The efficiency column looks the same as the
10 Gy/s row (37 %, then 18 %), but past 2 threads that number is mostly
arithmetic — near-flat wall time divided by double the core count — not a sign
that the E-cores are doing nothing.

**The difference is which stage the extra tracks land in.** 50 Gy/s pushes 5×
more tracks through the same grid (123.2M vs 24.6M per pulse). The stages that
scale with track count — rejection sampling and deposition — are embarrassingly
parallel and take cycles gladly from an E-core; the sweep and broadcast that
dominate the 10 Gy/s run are bandwidth-bound and already saturated at 2
P-cores (§3), so there is nothing spare for a third or fourth core to pick up.
Comparing the rates directly makes the split visible — the PDE sweep does not
care about dose rate at all, so this ratio isolates the track-bound work:

| threads | 10 Gy/s | 50 Gy/s | ratio |
|---|---|---|---|
| 1 | 562.1 s | 715.3 s | 1.27× |
| 2 | 373.2 s | 499.2 s | 1.34× |
| 4 | 382.4 s | 492.1 s | 1.29× |
| 8 | 391.0 s | 489.6 s | 1.25× |

If nothing scaled with thread count this ratio would be flat. Instead it rises
from 1.27× to a peak of 1.34× at 2 threads, then eases back to 1.25× at 8. The
peak is the PDE-bound part reaching its floor almost immediately (1 → 2
threads) while the track-bound part is still running at its 1-thread speed;
the ratio falls back over 4 and 8 threads as the track-bound part is finally
given somewhere to go.

**So "two P-cores" is the budget for the bandwidth-bound majority of this
workload, not an absolute ceiling for every workload on it.** A dose rate high
enough to keep the E-cores fed on deposition work can still find a little more
throughput past two threads — a ~2 % effect, next to the 1.43–1.51× the second
P-core buys, bought here with a 30 % clock drop and six extra cores.

This is exactly what trips the self-consistency checker: every ladder here
exceeds its >15 % clock-variance threshold. That's the expected signature of a
laptop measurement, not a broken one — see
[`profiling/cluster_scaling/collect.py`](../profiling/cluster_scaling/collect.py)
for why it treats "the clock moved" as the alarm worth raising here.

Contrast with Helios, where 2 threads returns 1.91× at 96 % efficiency and the
clock does not move. **Whether threads help, and how many are worth adding, is
a property of the machine** — and on this one, past the second P-core, it's
also a property of how many tracks per pulse you asked for.

## 5. P-cores versus E-cores

`./bench_laptop.sh --stage cores`, on a 186² × 210 grid (222 MiB — DRAM-resident
on this machine, unlike every named tier below `full_electrode`), 50 Gy/s:

| threads | perf ladder | | econ ladder | | P advantage |
|---|---|---|---|---|---|
| | cores | wall | cores | wall | |
| 1 | 1P | **79.6 s** | 1E | **122.0 s** | **1.53×** |
| 2 | 2P | 52.3 s | 2E | 73.2 s | 1.40× |
| 4 | 2P + 2E | 53.4 s | 4E | 68.6 s | 1.28× |

`k_s = 1.442382` in all six runs.

**A P-core beats an E-core by more than its clock does.** The clock ratio is
4783/3785 = 1.26×; the measured single-thread ratio is **1.53×**. The surplus is
memory-level parallelism: a bandwidth-bound stencil lives on how many cache
misses a core can keep in flight, and the P-core's deeper out-of-order window
sustains more of them. Clock is the smaller half of the story.

**The advantage shrinks as threads are added** — 1.53× → 1.40× → 1.28×,
converging toward the clock ratio. That is the memory controller becoming the
limit instead of the core: once bandwidth is the binding constraint, it stops
mattering which kind of core is waiting for it.

**Adding E-cores to a saturated pair of P-cores buys nothing.** 2P is 52.3 s and
2P + 2E is 53.4 s — slightly *worse*. Meanwhile 4E (68.6 s) does beat 2E
(73.2 s), because four small cores are still short of the bandwidth two big ones
already claim. The practical rule on this machine: **two P-cores is the whole
budget**, and anything past it is contention.

## 6. Running it yourself

```bash
./bench_laptop.sh --stage topology  # instant, verifies core detection
```

```bash
./bench_laptop.sh --stage cores  # ~10 min
```

```bash
./bench_laptop.sh --stage scaling  # ~70 min
```

```bash
python profiling/cluster_scaling/collect.py profiling/data/laptop_scaling/perf
```

Plug in, use a performance profile, close everything else. See
[`profiling/laptop_scaling/README.md`](../profiling/laptop_scaling/README.md).

For parameter studies, run independent single-threaded processes — but expect
well under linear aggregate throughput, for the same bandwidth and power reasons
as above.

## 7. Sizing a run before starting it: `--dry-run` and `--estimate-runtime-seconds`

The `full_electrode` grid is 2 GiB on this laptop (§3); on a machine with less
RAM to spare, or a grid pushed wider than this one, finding that out from the
OOM killer partway through a 6–12 minute run is worse than finding out before
it starts. Two flags on `run_markus_2mm.py` size a run without running all of
it, for the two different questions "how much RAM?" and "how long?" — kept as
two flags, not one, because a trustworthy answer to each costs something
different.

**`--dry-run`: memory only, instant, no allocation.** Builds the config (so
the constructor's own memory guard has already run — see
[`resources.py`](../pulsed_ion_chamber/resources.py)) and prints the sizing
without touching the solver at all:

```
$ python examples/ifj_aic144/run_markus_2mm.py full_electrode --threads 2 --dry-run
...
--- dry run: memory ---
Estimated peak allocation : 1.80 GiB
Total RAM on this machine : 30.84 GiB
Currently available RAM   : 19.75 GiB
Budget (80% of available)    : 15.80 GiB
Fits within budget        : yes

No simulation was run (--dry-run).
```

This number is trustworthy directly — it is exactly
`config.estimated_memory_bytes`, the figure §3 checked against measured peak
RSS (12–18 % low, comfortably inside the default 80 % budget).
`pulsed_ion_chamber.resources.memory_report()` is the reusable half of this —
it takes any `(required_bytes, budget_fraction)` pair and returns the same
required-vs-available comparison, so the same check can be dropped into
another script without re-deriving it.

**Runtime deliberately is not part of `--dry-run`.** An earlier version of
this flag also printed a runtime estimate built from timing the unbatched,
one-track-at-a-time kernel in isolation — which, on a grid this size, is
**~500–1000× slower** than the batched backend a real `--threads 2` run
actually uses (21 h estimated against a 382 s actual run). That number was
never trustworthy enough to belong next to an instant, no-allocation flag,
so it was removed rather than kept behind a caveat.

**`--estimate-runtime-seconds N`: real backend, real grid, ~N seconds.**
Allocates the full grid, warms up Numba, and runs the *actual* backend
`--threads` would select — for real, just only for `N` seconds (default 5)
instead of to completion — then extrapolates linearly from the measured
per-step cost:

```
$ python examples/ifj_aic144/run_markus_2mm.py full_electrode --threads 2 --estimate-runtime-seconds 5
...
--- empirical runtime estimate: real solver_numba_parallel backend, 2 thread(s), ~5s sample ---
steps measured            : 31 / 2,194
measured time for those   : 5.03 s (162.2 ms/step)
estimated total wall time : 356 s (0.099 h)

No full run was performed (--estimate-runtime-seconds).
```

Measured against the actual 382 s full run (§3, same config): **356 s, a 7 %
under-estimate** — two orders of magnitude closer than the isolated-kernel
estimate above, because it exercises the batched deposition and the real
thread count instead of a proxy for either. The 7 % gap has a specific cause
worth knowing about on *this* class of machine: the 5 s sample is drawn from
the first 31 steps, when the package is at its coolest; §4 already shows this
laptop's sustained clock dropping from ~4790 MHz to ~4175 MHz as a
full-electrode run heats it up, so an early sample runs faster than the run's
own steady-state average. A cluster node that does not throttle (HELIOS.md)
would not have this particular bias. Treat the result as same-order-of-magnitude,
not a bound in either direction — see
`pulsed_ion_chamber.benchmark.estimate_full_runtime_empirical` for the other
contributor (missing the cheaper, deposition-free clearance-phase steps,
which pulls the other way) and PERFORMANCE.md §7.

`--dry-run` and `--estimate-runtime-seconds` are mutually exclusive: the
former promises not to touch the solver, the latter always does.
