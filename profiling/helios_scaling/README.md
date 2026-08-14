# Helios thread-scaling study

Wall time of the full-electrode grid against thread count, at two dose rates.
Produces the tables in [`docs/HELIOS.md`](../../docs/HELIOS.md) §4 and §5.

```bash
./submit.sh                                   # from a Helios ACCESS node
squeue -u $USER
python profiling/helios_scaling/collect.py    # once the queue drains
```

That is the whole workflow. `./submit.sh` lives at the repository root because
it is the only command anyone needs to remember; everything here is what it
calls.

## What gets run

The `full_electrode` tier (536² × 210, 1.9 GiB of carrier arrays, 2194 steps) at
**1, 2, 4, 8, 16, 32, 64 and 128 threads**, at **10 and 50 Gy/s to water** —
16 runs. Always the batched backend, including at one thread: comparing thread
counts means holding the backend fixed, and the unbatched one would take hours
on this grid.

The two dose rates are there because they stress different halves of the code.
The PDE sweep does not depend on dose rate at all; the deposition phases scale
linearly with it. So 10 Gy/s is a PDE-bound run and 50 Gy/s is halfway to a
deposition-bound one, and the difference between their scaling curves is a
direct measurement of how much serial work is left in deposition.

## Files

| | |
|---|---|
| `../../submit.sh` | The entry point. Preflight checks, then calls `submit_all.sh`. |
| `submit_all.sh` | One `sbatch` per thread count, each requesting exactly the cores it uses, each with a walltime sized from measured times. |
| `scaling_job.sbatch` | The job itself: loads the module, asserts its CPU affinity matches what it asked for, then loops over dose rates. |
| `collect.py` | Reads the JSONs into tables — and refuses to print them until it has checked that `k_s` agrees across thread counts and that every run had the CPUs it claimed. |

Results: `profiling/data/helios_scaling/threads{N}_dose{R}.json`, job logs in
`logs/` beside them.

## Why one job per thread count, and not a job array

A job array shares a single `--cpus-per-task` across every task, so a 1-thread
run would sit inside a 128-core reservation, wasting 127 cores for twelve
minutes and queueing far longer than it needs to. Separate submissions let each
job ask for exactly what it uses; they queue independently and generally start
sooner.

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
./submit.sh --threads "1 8 64"   # subset of thread counts
./submit.sh --rates "50"         # one dose rate
./submit.sh --dry-run            # print what would be submitted
```

The walltimes in `submit_all.sh` are ~2× the measured run times. If you change
the tier, the grid spacing or the dose rates, re-derive them — too short kills a
run at 97 %, too long only delays scheduling.
