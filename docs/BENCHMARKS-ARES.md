# Benchmarks: Ares (Cyfronet)

**Measured.** 14 runs, one exclusive node each, `./submit.sh`. The predictions
in §5 were written before any of it and are left exactly as they were; §6.3
scores them, and **the headline one was wrong** — see §6.

Companion pages: [HELIOS.md](HELIOS.md) (the same study on a much wider node)
and [BENCHMARKS-LAPTOP.md](BENCHMARKS-LAPTOP.md). The machine-independent cost
model is in [PERFORMANCE.md](PERFORMANCE.md).

---

## 1. The machine

| | |
|---|---|
| node | 2 × Intel Xeon Platinum 8268 (Cascade Lake-SP), 24 cores each, **48 cores** |
| SMT | **off** (`siblings 24` = `cpu cores 24`) |
| clock | 2.90 GHz base, **3.9 GHz observed** |
| L3 | 35.75 MiB per socket, **~72 MiB per node** |
| NUMA | **4 domains of 12 cores** — Sub-NUMA Clustering is on |
| RAM | ~192 GB (4 × ~48 GB) |
| memory | 6 × DDR4-2933 per socket → ~140 GB/s per socket, **~200 GB/s per node** |
| ISA | AVX-512 (F, DQ, CD, BW, VL, VNNI) |

### The CPU numbering is interleaved, and it matters

```
node 0 cpus: 0 1 2 3 7 8 12 13 14 18 19 20
node 1 cpus: 4 5 6 9 10 11 15 16 17 21 22 23
node 2 cpus: 24 25 26 27 30 31 35 36 37 42 43 44
node 3 cpus: 28 29 32 33 34 38 39 40 41 45 46 47

distances:  10 local,  11 sibling domain (same socket),  21 across sockets
```

Consecutive CPU ids are **not** in the same NUMA domain. Two consequences:

* A job given "8 CPUs" may get them spread over two or three domains, or packed
  into one, depending on what else is on the node. Those are different machines
  from this code's point of view — different numbers of memory controllers —
  so an unpinned, non-exclusive thread ladder here is not comparable point to
  point. This is the same problem the laptop has with P and E cores, arriving
  by a different route.
* The `_first_touch_parallel` placement (docs/HELIOS.md §7) assumes a thread
  keeps the pages it touched. That still holds, but with four domains rather
  than eight and a much smaller distance penalty (21 vs Helios's cross-socket
  hop), so the effect should be **smaller here than on Helios**. §5.

The thread ladder is therefore **1, 2, 4, 8, 12, 24, 48** — 12 is one NUMA
domain, 24 one socket, 48 the node. Those are where the curve should bend.

## 2. Setup

On an Ares access node:

```bash
git clone <repo>
```

```bash
cd IonTracks-PulsedProton-Python
```

```bash
module load python/3.12.3-gcccore-13.3.0
```

```bash
python -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install -e ".[dev]"
```

```bash
pytest  # ~10 s, 65 tests
```

**The module name differs from Helios's, the software does not.** Ares spells it
`python/3.12.3-gcccore-13.3.0` — lower case, toolchain folded into one name —
where Helios wants `GCCcore/13.3.0 Python/3.12.3`. Both resolve to the same
Python 3.12.3, and the venv then pulls the same numba 0.67.0 and numpy 2.5.2.
That is a piece of luck worth stating: an Ares-vs-Helios comparison is a
hardware comparison, with no compiler or library version confounding it.

`sites.sh` knows both spellings, so `./submit.sh` needs no argument. If a module
is ever renamed, the job script aborts with the failing `module load` rather
than falling through to a system python without numba.

## 3. Running the study

```bash
./submit.sh --dry-run  # confirm site, account, partition, ladder
```

```bash
./submit.sh  # 14 jobs: 7 thread counts x 2 dose rates
```

```bash
squeue -u $USER
```

```bash
python profiling/cluster_scaling/collect.py profiling/data/ares_scaling
```

The site is detected from `$SLURM_CLUSTER_NAME`; `--site ares` forces it.
Account `plgccbmc15-cpu`, partition `plgrid`, and **`--exclusive` is the default
here** — see §4.

## 4. Should whole nodes be reserved? Yes, and more so than on Helios

Three reasons, in order of how much they matter.

**It is measured, not hypothetical.** On Helios, leaving nodes shared inflated
the middle of the scaling curve by up to **195 %** (16 threads: 169 s shared
against 57 s exclusive) and *moved the apparent optimum* from 32 threads to 128.
Both the numbers and the recommendation drawn from them were wrong. See
[HELIOS.md](HELIOS.md) §4.

**A co-tenant costs proportionally more here.** Bandwidth is the resource being
measured, and Ares has ~200 GB/s against Helios's ~900. One neighbouring job
doing ordinary memory-heavy work takes a far larger share of a much smaller
pool. Ares is *more* vulnerable to this, not less.

**Sub-NUMA Clustering makes co-tenancy change the topology, not just the load.**
Without `--exclusive`, which CPUs Slurm hands you depends on what is already
running, so "8 threads" may be 8 cores in one domain in one run and spread over
three in the next — a different number of memory controllers for the same
nominal thread count. That is not noise that averages out; it is a different
experiment each time.

**And it is cheap here.** An exclusive node costs 48 core-hours per hour against
Helios's 192, so the objection that made it non-default on Helios — booking 192
cores to run one — is four times weaker. Hence `SITE_EXCLUSIVE=1` for Ares.

Use `./submit.sh --shared` if you deliberately want the cheap, rough version;
just do not put those numbers in this file.

## 5. Predictions, written before measuring

The claim this whole investigation rests on is that scaling is set by
**bandwidth per core**, not by core count. Ares is the useful test because it
sits between the other two machines in exactly that quantity:

| | cores | node BW | 1-core BW | node ÷ 1-core |
|---|---|---|---|---|
| laptop | 12 | ~29 GB/s | ~21 GB/s | ~1.4 |
| **Ares** | 48 | ~200 GB/s | ~13 GB/s | **~15** |
| Helios | 192 | ~900 GB/s | ~9 GB/s | ~100 |

That last column is roughly the speed-up a bandwidth-bound code can hope for
before the controllers saturate. So:

1. **Single core faster than Helios.** Higher clock (3.9 vs ~3.5 GHz) and more
   bandwidth per core. Expect the 10 Gy/s full-electrode run around
   **450–550 s** against Helios's 572 s.
2. **Best speed-up ~10–14×, reached at 24–48 threads.** Bandwidth allows ~15×;
   the Amdahl ceiling from the un-threaded deposition is ~19× at 10 Gy/s
   (HELIOS.md §4), so bandwidth binds first, just barely.
3. **Therefore roughly Helios's best wall time (47 s) on a quarter of the
   cores**, at much better per-core efficiency. If that holds, the practical
   conclusion is that this workload belongs on Ares, not Helios — Helios's
   width is mostly unusable here.
4. **The curve bends at 12 and 24**, the NUMA-domain and socket boundaries,
   rather than at the powers of two.
5. **NUMA first touch matters less than on Helios** — four domains instead of
   eight, and a 21 vs 11 distance ratio rather than Helios's eight-domain
   fabric. Worth re-running `profiling/bench_kernels.py --init both` to see how
   much less.
6. **AVX-512 changes nothing measurable.** The kernel is bandwidth-bound, and
   Cascade Lake's full-width AVX-512 against Zen 4's double-pumped
   implementation should be invisible behind the memory wall. If it is *not*
   invisible, the "bandwidth-bound" diagnosis needs revisiting.

## 6. Results

Full-electrode grid (536² × 210, 1.9 GiB, 2194 steps), 14 exclusive-node jobs.
`k_s` is `1.111065` at 10 Gy/s and `1.446434` at 50 Gy/s in **every** run, so
the machine and the thread count change nothing about the answer.

### 6.1 Thread scaling

| threads | 10 Gy/s | speed-up | eff. | 50 Gy/s | speed-up | eff. |
|---|---|---|---|---|---|---|
| 1 | 968 s | 1.00× | 100 % | 1269 s | 1.00× | 100 % |
| 2 | 501 s | 1.93× | 97 % | 693 s | 1.83× | 92 % |
| 4 | 286 s | 3.39× | 85 % | 449 s | 2.83× | 71 % |
| 8 | 168 s | 5.76× | 72 % | 287 s | 4.42× | 55 % |
| 12 (1 NUMA domain) | 141 s | 6.89× | 57 % | 251 s | 5.05× | 42 % |
| 24 (1 socket) | **105 s** | **9.19×** | 38 % | 212 s | 5.98× | 25 % |
| 48 (whole node) | 129 s | 7.49× | 16 % | **201 s** | **6.30×** | 13 % |

**Using the whole node is worse than using half of it** at 10 Gy/s: 48 threads
is 129 s against 24 threads' 105 s. The same shape as Helios past its optimum,
arriving 4× earlier in core count. At 50 Gy/s 48 threads is marginally best, and
only because that curve is still climbing out of its serial deposition floor.

Implied serial fractions from the plateaus: ~7 % at 10 Gy/s (ceiling ~14×,
measured 9.2×, so bandwidth binds first) and ~14 % at 50 Gy/s (ceiling ~7×,
measured 6.3×, essentially at it). Same division of blame as Helios.

### 6.2 Three machines

| | laptop¹ | Ares | Helios |
|---|---|---|---|
| cores | 12 (2P + 8E + 2LP-E) | 48 | 192 |
| single core, 10 Gy/s | **562 s** | **968 s** | **572 s** |
| best wall time, 10 Gy/s | 373 s (2 thr)² | **105 s** | **47 s** |
| cores at that point | 2 | 24 | 32 |
| best speed-up | 1.51× | 9.19× | 12.23× |
| efficiency there | 75 % | 38 % | 38 % |
| node throughput, 8-thread jobs | n/a | 6 × 5.76 = **35×** | 24 × 6.32 = **152×** |

² the laptop ladder past 2 threads was still running when this was written; 2
threads is where its curve had already flattened.

**The single-core row is the one to look at.** A 2024 laptop P-core (562 s) is
marginally *faster* than a Helios core (572 s) and 1.7× faster than an Ares core
(968 s). Compute nodes do not sell fast cores; they sell many of them, and this
kernel can use ~12× of Helios's 192.

¹ `./bench_laptop.sh`, see [BENCHMARKS-LAPTOP.md](BENCHMARKS-LAPTOP.md).

**Helios wins on both axes, and the per-core one is the surprise.** At matched
thread counts an Ares core is **1.7–1.9× slower**:

| threads | Ares | Helios | ratio |
|---|---|---|---|
| 1 | 968 s | 572 s | 1.69× |
| 2 | 501 s | 299 s | 1.67× |
| 4 | 286 s | 162 s | 1.76× |
| 8 | 168 s | 91 s | 1.86× |

A flat ~1.7× offset from one thread upward is not a scaling difference — it is
the *core* being slower at this work, before parallelism enters.

### 6.3 Which predictions held

| # | prediction | outcome |
|---|---|---|
| 1 | single core **faster** than Helios, 450–550 s | **wrong, and by a lot**: 968 s, 1.7× *slower* |
| 2 | best speed-up 10–14× at 24–48 threads | **half right**: 9.19× — just below the range — but at 24 threads as expected |
| 3 | ≈ Helios's best wall time on a quarter of the cores | **wrong**: 105 s against 47 s, 2.2× slower at its best |
| 4 | curve bends at 12 and 24 | **untestable as run** — see the affinity note below |
| 5 | first touch matters less than on Helios | **not tested** — needs `bench_kernels.py --init both` on Ares |
| 6 | AVX-512 invisible behind the memory wall | **doubtful** — see the hypothesis below |

The bandwidth-per-core model of §5 predicted the *shape* of the scaling curve
tolerably (9.2× against a predicted 10–14×, saturating where predicted) and the
*absolute speed* not at all. It had no term for single-core throughput, and that
is where the entire 2.2× difference in best wall time comes from.

### Why is an Ares core 1.7× slower? A hypothesis, not a finding

Clock favours Ares on paper: 3.9 GHz observed against Helios's ~3.5. So
something is taking it back, and the size of the gap is suggestive.

**AVX-512 frequency licensing.** Skylake-SP and Cascade Lake drop core frequency
when running sustained 512-bit code — for a Platinum 8268 the all-core AVX-512
turbo is roughly 2.4 GHz against 3.9 for scalar work, a ratio of ~1.6× that
lands close to the measured 1.7×. Zen 4 implements AVX-512 as double-pumped
256-bit and takes no such licence penalty. LLVM will happily vectorise this
7-point stencil to 512 bits on Cascade Lake, so the code may be buying vector
width and paying for it in clock, on a kernel that is bandwidth-bound and cannot
use the width.

That would make prediction 6 wrong in an interesting direction: AVX-512 is not
invisible, it is a *cost*.

**How to test it**, cheaply and on one thread:

```bash
# force 256-bit vectors and re-run the single-core point
NUMBA_CPU_FEATURES="+avx2,-avx512f,-avx512dq,-avx512cd,-avx512bw,-avx512vl" \
  python examples/ifj_aic144/run_markus_2mm.py full_electrode \
    --threads 1 --backend batched --dose-rate-water-Gy-s 10
```

If it comes out materially *faster* than 968 s, the licence penalty is real and
the fix is to pin the vector width rather than to accept the machine as slow.
If it does not move, the cause is elsewhere — DDR4-2933 against DDR5-4800, or
Cascade Lake's mesh latency against Zen 4's prefetchers — and the honest
conclusion is simply that this kernel suits the newer memory system.

### The affinity caveat: `--exclusive` did not confine the cpuset

Every job reported `affinity=48`, whatever its `--cpus-per-task`:

```
site=ares  host=ac0783  cpus-per-task=1   affinity=48  threads=1
site=ares  host=ac0714  cpus-per-task=8   affinity=48  threads=8
site=ares  host=ac0655  cpus-per-task=24  affinity=48  threads=24
```

With `--exclusive` the batch script inherits the whole node's cpuset, so the
thread *count* was enforced by `numba.set_num_threads` but the *placement* was
not. Every run's threads were free to migrate across all 48 CPUs and all four
NUMA domains.

Two consequences, and neither invalidates the table:

* The low-thread points are, if anything, **flattered** — 8 threads had four
  domains' worth of memory controllers available rather than one domain's. So
  the per-core deficit against Helios in §6.2 is a *lower bound*.
* Prediction 4 could not be tested. Bends at 12 and 24 would only appear if
  those thread counts were confined to a domain and a socket, and they were not.
  The curve is smooth, which is what free placement predicts.

Testing prediction 4 needs explicit pinning — `taskset` against CPU lists built
from `numactl -H`, the same approach `profiling/laptop_scaling/topology.py`
takes for the laptop's P and E cores. That is the obvious next run here, and it
is the measurement that would say whether Sub-NUMA Clustering helps this code
or merely complicates it.
