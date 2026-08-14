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

### Is it worth running?

As a way to get an accurate `k_s`, no. The full electrode is still 0.1 % below
the infinite-column limit, while an 80 µm column corrected for the `1/r` edge
deficit lands within 0.06 % of it in **1.8 seconds** — 400× faster and closer
(PHYSICS §14). The full-electrode run's real value is that it *verifies* that
extrapolation across a 15× range in radius. Running the true electrode geometry
only becomes necessary alongside edge physics the model does not currently
have — guard-ring field distortion, non-uniform fluence — at which point the
answer would no longer be a scaled copy of the interior.

## 4. Threads on a laptop: two, and only two

Measured, `./bench_laptop.sh --stage scaling`, full electrode, 10 Gy/s, pinned
one thread per physical core:

| threads | cores | wall | speed-up | eff. | sustained clock |
|---|---|---|---|---|---|
| 1 | 1P | 562 s | 1.00× | 100 % | 4793 MHz |
| 2 | 2P | 373 s | 1.51× | 75 % | 4202 MHz |

**And part of even that 1.51× is not scaling at all — it is thermal.** The clock
falls 12 % between the two runs, and at 8 threads (2P + 6E) it is down to
3412 MHz, 29 % below single-core. On a laptop the second thread is competing for
bandwidth *and* for a package power budget the first thread was already using.
That is why every run here records the clock it sustained; a laptop speed-up
figure without one is not interpretable.

Contrast with Helios, where 2 threads returns 1.91× at 96 % efficiency and the
clock does not move. **Whether threads help is a property of the machine**, and
on this one the answer is "barely".

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
./bench_laptop.sh --stage topology   # instant, verifies core detection
./bench_laptop.sh --stage cores      # ~10 min
./bench_laptop.sh --stage scaling    # ~70 min
python profiling/cluster_scaling/collect.py profiling/data/laptop_scaling/perf
```

Plug in, use a performance profile, close everything else. See
[`profiling/laptop_scaling/README.md`](../profiling/laptop_scaling/README.md).

For parameter studies, run independent single-threaded processes — but expect
well under linear aggregate throughput, for the same bandwidth and power reasons
as above.
