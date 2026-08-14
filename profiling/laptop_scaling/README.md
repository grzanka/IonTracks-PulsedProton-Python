# Laptop scaling benchmark

The counterpart to [`../helios_scaling/`](../helios_scaling/): the same
simulation, 1–8 threads, on a hybrid laptop CPU. Produces the tables in
[`docs/BENCHMARKS-LAPTOP.md`](../../docs/BENCHMARKS-LAPTOP.md).

```bash
./bench_laptop.sh --stage topology   # instant: what cores this machine has
./bench_laptop.sh --stage cores      # ~10 min: P-cores vs E-cores
./bench_laptop.sh --stage scaling    # ~70 min: the 1/2/4/8 ladder
./bench_laptop.sh                    # all three

python profiling/helios_scaling/collect.py profiling/data/laptop_scaling/perf
```

`./bench_laptop.sh` lives at the repository root, beside `./submit.sh`.

## Why this is not just `submit.sh` with smaller numbers

**A thread count does not describe a run on a hybrid CPU.** A Core Ultra 5 225U
has 2 P-cores (with SMT), 8 E-cores and 2 low-power E-cores. "4 threads" could
be 2P+2E or 4E, and those differ by roughly a factor of two. Which one the
kernel gives you depends on the scheduler, the power profile and what else is
running, so an unpinned laptop benchmark is not reproducible against itself, let
alone against anything else. Every run here is pinned with `taskset` to a CPU
list chosen from a detected core group, and records which cores it used.

**One logical CPU per physical core.** The P-cores have two SMT threads each. A
ladder that counts them as separate cores plateaus at the P-core count for
reasons that have nothing to do with the code — on a memory-bound kernel a
second sibling adds contention, not throughput. `topology.py` hands out one CPU
per physical core; the `smt` ladder exists to measure that claim rather than
assume it.

**A laptop's speed is a function of its temperature.** The ladder runs
1 thread first, which is also the longest run, so without care every later point
is measured on a hotter, slower machine and every speed-up comes out inflated.
There is a cool-down between runs (`--cooldown`, default 60 s), each run records
the clock it actually sustained, and `collect.py` refuses to present a ladder
whose sustained clock moved by more than 15 %.

**Battery changes the answer.** Recorded per run, and flagged by `collect.py`.

## The three stages

| stage | grid | what it answers |
|---|---|---|
| `topology` | — | What core types does this machine have, and which CPUs are they? |
| `cores` | 186²×210, r = 0.09 cm | How much of the P-core advantage survives on a memory-bound kernel? Clock favours P, but both core types queue behind the same memory controller — so the gap should be smaller than the clock ratio. |
| `scaling` | `full_electrode` (536²×210) | The Helios comparison: same grid, same dose rates, 1/2/4/8 threads. |

`cores` uses a smaller grid on purpose — it is a *ratio* between core types, and
paying 12 minutes a point to measure a ratio is waste — but **not one of the
named tiers**. Every tier below `full_electrode` fits inside a laptop's ~12 MiB
L3 (`wide` is 11.3 MiB), and on a cache-resident grid this study would compare
clock and IPC rather than the memory behaviour the whole investigation is
about. So it sets the column radius directly: r = 0.09 cm is 222 MiB of carrier
arrays, comfortably DRAM-resident, at 12 % of the full electrode's cost.

`scaling` uses the full electrode because comparability with Helios is the
entire point of it.

## Ladders

| ladder | CPUs it uses | for |
|---|---|---|
| `perf` | fastest first: every P core, then E, then LP-E | "how fast can this laptop run it on N cores" |
| `econ` | E and LP-E cores only | the same thread counts on the small cores |
| `smt` | both SMT threads of as few P cores as possible | a deliberate worst case |

## Files

| | |
|---|---|
| `../../bench_laptop.sh` | The entry point: preflight, topology, then the stages. |
| `topology.py` | Core-type detection and CPU-list generation. Standalone and useful on its own — `python -m profiling.laptop_scaling.topology`. |
| `bench.sh` | The driver: pins, samples clocks and temperature during each run, folds the conditions into the result JSON. |
| `../helios_scaling/collect.py` | Shared with the Helios study; reads either. |

Results: `profiling/data/laptop_scaling/<ladder>/threads{N}_dose{R}.json` and
`profiling/data/laptop_core_types/<ladder>/…`.

## Before running

Plug the laptop in, set a performance power profile, close everything else
(browsers especially), and leave it alone. The full run is ~80 minutes and the
machine should have nothing else to do; otherwise the numbers describe your
desktop session as much as this code.

## Platform

Linux only: it pins with `taskset` and reads core types, clocks and temperatures
from sysfs. On macOS or Windows, run
`examples/ifj_aic144/run_markus_2mm.py` directly — the results will not be
pinned, and are not comparable with
[`docs/BENCHMARKS-LAPTOP.md`](../../docs/BENCHMARKS-LAPTOP.md).
