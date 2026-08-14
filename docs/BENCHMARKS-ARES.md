# Benchmarks: Ares (Cyfronet)

> **Placeholder.** The hardware description and the setup instructions below are
> real and verified on the machine; **every results table is empty and marked
> TODO**, because the study has not been run here yet. Run `./submit.sh` on an Ares access node and fill them
> in from `python profiling/cluster_scaling/collect.py profiling/data/ares_scaling`.
>
> The predictions in §5 were written *before* any measurement. Leave them as
> they are when you fill in the numbers, and record whether they held: a
> prediction edited after the fact is worth nothing.

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

```bash
# on an Ares access node
git clone <repo> && cd IonTracks-PulsedProton-Python
module load python/3.12.3-gcccore-13.3.0
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest                                       # ~10 s, 65 tests
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
./submit.sh --dry-run    # confirm site, account, partition, ladder
./submit.sh              # 14 jobs: 7 thread counts x 2 dose rates
squeue -u $USER
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

**TODO — not yet measured.**

### 6.1 Thread scaling

| threads | 10 Gy/s | speed-up | eff. | 50 Gy/s | speed-up | eff. |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 4 | | | | | | |
| 8 | | | | | | |
| 12 (1 NUMA domain) | | | | | | |
| 24 (1 socket) | | | | | | |
| 48 (whole node) | | | | | | |

`k_s` must come out `1.111065` at 10 Gy/s and `1.446434` at 50 Gy/s, identical
to six digits at every thread count, on every machine. `collect.py` checks it.

### 6.2 Three machines side by side

| | laptop | Ares | Helios |
|---|---|---|---|
| cores used for best time | TODO | TODO | 32 |
| best wall time, 10 Gy/s | TODO | TODO | 47 s |
| single core, 10 Gy/s | TODO | TODO | 572 s |
| best speed-up | TODO | TODO | 12.2× |
| efficiency at that point | TODO | TODO | 38 % |

### 6.3 Which predictions held

**TODO** — one line per numbered prediction in §5, kept honest.
