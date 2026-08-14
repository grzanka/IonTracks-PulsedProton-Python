# Cluster thread-scaling study

Wall time of the full-electrode grid against thread count, at two dose rates,
on an HPC node. Produces the tables in [`docs/HELIOS.md`](../../docs/HELIOS.md)
§4–§5 and [`docs/BENCHMARKS-ARES.md`](../../docs/BENCHMARKS-ARES.md) §6.

```bash
./submit.sh --dry-run                          # confirm the detected machine
./submit.sh                                    # from a Helios or Ares ACCESS node
squeue -u $USER
python profiling/cluster_scaling/collect.py profiling/data/<site>_scaling
```

## One study, several machines

Everything machine-specific lives in [`sites.sh`](sites.sh) — module line,
account, partition, thread ladder, and whether whole nodes are reserved by
default. The job script and the submitter are identical everywhere, so adding a
cluster is a new `case` branch rather than a second copy of the study.

| | Helios | Ares |
|---|---|---|
| node | 2 × EPYC 9654, 192 cores | 2 × Xeon 8268, 48 cores |
| NUMA | 8 domains × 24, contiguous ids | 4 domains × 12, **interleaved ids** |
| ladder | 1 2 4 8 16 32 64 128 | 1 2 4 8 **12 24 48** |
| exclusive by default | no | **yes** |
| results | `profiling/data/helios_scaling/` | `profiling/data/ares_scaling/` |

The ladders differ because the boundaries do: powers of two on Helios, and on
Ares the NUMA domain (12), socket (24) and node (48), which is where its curve
should bend. Exclusive is the Ares default because a co-tenant takes a larger
share of a smaller bandwidth pool *and* changes which NUMA domains the job's
CPUs come from — see `docs/BENCHMARKS-ARES.md` §4.

That is the whole workflow. `./submit.sh` lives at the repository root because
it is the only command anyone needs to remember; everything here is what it
calls.

## What gets run

The `full_electrode` tier (536² × 210, 1.9 GiB of carrier arrays, 2194 steps) at
**1, 2, 4, 8, 16, 32, 64 and 128 threads**, at **10 and 50 Gy/s to water** —
16 runs, submitted as **16 independent jobs**. Always the batched backend,
including at one thread: comparing thread counts means holding the backend
fixed, and the unbatched one would take hours on this grid.

The two dose rates are there because they stress different halves of the code.
The PDE sweep does not depend on dose rate at all; the deposition phases scale
linearly with it. So 10 Gy/s is a PDE-bound run and 50 Gy/s is halfway to a
deposition-bound one, and the difference between their scaling curves is a
direct measurement of how much serial work is left in deposition.

## Files

| | |
|---|---|
| `../../submit.sh` | The entry point. Preflight checks, then calls `submit_all.sh`. |
| `submit_all.sh` | One `sbatch` per (thread count, dose rate), each requesting exactly the cores it uses, with memory and walltime sized from measured runs. |
| `scaling_job.sbatch` | The job itself: loads the module, asserts its CPU affinity matches what it asked for, then runs. |
| `collect.py` | Reads the JSONs into tables — and refuses to print them until it has checked that `k_s` agrees across thread counts and that every run had the CPUs it claimed. |

Results: `profiling/data/cluster_scaling/threads{N}_dose{R}.json`, job logs in
`logs/` beside them.

## Why 16 separate jobs

**One per (thread count, dose rate).** The two dose rates share nothing but the
grid dimensions, so pairing them in one job only serialises them. Split, the
study's turnaround is its longest *single* run — ~13 min, the 1-thread 50 Gy/s
case — instead of its longest *pair*, ~23 min. On an empty queue all 16 run at
once and the whole thing is done in a quarter of an hour.

**Not a job array.** An array shares a single `--cpus-per-task` across every
task, so the 1-thread runs would sit inside a 128-core reservation, wasting
127 cores and queueing far longer than they need to. Separate submissions each
ask for exactly what they use.

## Why `--mem` is explicit

The plgrid partition has `DefMemPerCPU=1536 MB` — memory is rationed per *core*.
This study's peak RSS is ~2.1 GiB at **every** thread count, because it is one
grid rather than one grid per thread. So the default would give the 1-core job
1.5 GB and get it OOM-killed, and the 2-core job a 1.4× margin. Each job asks
for 8 GB, which is free at these sizes and removes the question.

## Why the affinity assertion

The characteristic Slurm failure here is not a crash. It is a run that asked for
128 threads, was given one CPU, took 128 threads' worth of walltime, and
reported a thread count it never had — which looks exactly like "this code does
not parallelise". `scaling_job.sbatch` compares `os.sched_getaffinity` against
the requested thread count and dies immediately if they disagree, and
`collect.py` re-checks the same thing from the recorded JSON. See
[`docs/HELIOS.md`](../../docs/HELIOS.md) §3.

## Adjusting it

```bash
./submit.sh --threads "1 8 64"       # subset of thread counts
./submit.sh --rates "50"             # one dose rate
./submit.sh --account plgXXXX-cpu    # charge a different grant
./submit.sh --mem 16G                # more memory per job
./submit.sh --dry-run                # print what would be submitted
```

The walltimes in `submit_all.sh` are ~2× the measured run times: 30 min for the
single-core jobs, 20 for two cores, and the partition's 15-minute default for
everything else. If you change the tier, the grid spacing or the dose rates,
re-derive them — too short kills a run at 97 %, too long only delays
scheduling.
