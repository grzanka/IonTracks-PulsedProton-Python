# Profiling harness and raw data

> **Phase 2 happened.** The thread sweep below was run on a 40 MiB grid, and
> concluded that threads never help. That is true for *that* grid and false in
> general: on a grid larger than the node's 768 MiB of L3 the same code scales
> ~9x, once NUMA first touch, a serial copy-back and a per-track Python loop
> are dealt with. See [`docs/HELIOS.md`](../docs/HELIOS.md), and
> `bench_kernels.py` / `run_full_electrode_sweep.sh` below for the harness that
> showed it. The Phase-1 material is kept as-is; it is the "before".

## The two machine studies

Both are self-contained and have their own README:

| | |
|---|---|
| [`helios_scaling/`](helios_scaling/README.md) | Thread scaling on a Helios node, 1–128 cores, two dose rates. Submit with `./submit.sh` from an access node. |
| [`laptop_scaling/`](laptop_scaling/README.md) | The same runs on a hybrid laptop CPU, 1–8 cores, pinned to known core types. Run with `./bench_laptop.sh`. |

Their results are read by the same collector,
[`helios_scaling/collect.py`](helios_scaling/collect.py), which checks a study
for self-consistency before it will tabulate it.

## Phase 1: raw profiling & scientific-validation data

The exercise this material was built for is specified in
[`../TRAINING_PLAN.md`](../TRAINING_PLAN.md): Phase 1 diagnose, Phase 2
optimise. Phase 2 has since been carried out — see the note above and
[`../docs/HELIOS.md`](../docs/HELIOS.md) — so the plan now reads as a record of
what was asked rather than of what is outstanding.

This directory is training material: a set of scripts and the raw
performance/correctness data they produced on a Cyfronet Helios node
(dual AMD EPYC 9654, 192 physical cores, 8 NUMA domains), for an AI coding
agent (opencode, backed by open-weight models) to analyze independently —
produce plots/diagrams, and reason about *why* the numbers look the way
they do — before attempting to actually improve the code in a later phase.

**The scripts in this directory are the harness, not the analysis.** They
only measure and record. Turning `data/*.csv` into plots, and the plots
into a written diagnosis, is the assigned task, deliberately left undone
here.

## What's already true about this codebase (see the top-level README.md)

`pulsed_ion_chamber.solver_numba_parallel` already implements a
shared-memory parallel backend (`numba.njit(parallel=True)` + `prange`,
flattened over `(i, j)`) with a batching optimization for track insertion,
and the top-level README documents a prior finding that adding threads
*hurts* wall time past 1 thread on this hardware. **That conclusion is not
hidden from opencode here** — treat this dataset as an opportunity to
independently re-derive and extend it with real artifacts (flamegraphs,
context-switch counts, per-layer comparisons), not to blindly re-confirm
prose that's already sitting in the repo.

## Scripts

| Script | What it produces |
|---|---|
| `common.py` | Shared config: the "converged, ~6 track radii" grid (`grid_size_um=5.0, sampled_radius_cm=0.012`, same `BEAM_KWARGS` as `examples/run_pulsed_proton_beam.py`) and `run_once(num_threads)`, which returns timing + OS resource-usage (context switches, user/sys time) + scientific (`k_s`, `f_t`) metrics for one run. |
| `sweep.py` + `run_sweep.sh` | Thread-count x threading-layer sweep. Each `(threads, layer)` point is launched as its own `srun --overlap --cpus-per-task=N --cpu-bind=none` step (see the top-level README's "cpuset gotcha": an interactive Slurm shell's own process is itself pinned to 1 CPU regardless of the job's allocation, so getting real N-core affinity for a single run requires its own step). Appends one row per point to `data/thread_sweep.csv`. |
| `diagnostics.py` | Numba's resolved threading layer, CPU affinity, SLURM job sizing, `numactl --hardware`/`lscpu` output, and `parallel_diagnostics(level=4)` for both hot `prange` kernels -> `data/diagnostics.txt`. (Needs `.recompile()` first: a `cache=True` dispatcher loaded from its on-disk cache doesn't retain the parfor metadata `parallel_diagnostics` needs.) |
| `cprofile_run.py` | cProfile of one run at a given thread count -> `data/cprofile_{N}threads.{pstats,txt}`. The two hot kernels are opaque compiled calls to cProfile, so this shows Python-level driver overhead (schedule building, the per-time-step loop, dispatch cost) rather than time inside the kernels themselves. |
| `pyspy_record.sh` | py-spy flamegraphs (`--native`, so OpenMP/Numba worker threads show up, not just the Python driver) at 1 and 96 threads -> `data/flamegraph_{N}threads.svg`. Sampling rate is scaled down at high thread count (200 Hz -> 20 Hz) because native stack-walking across many OS threads is itself expensive enough to perturb the run at full rate (measured: 200 Hz/96 threads inflated wall time from ~36 s to 102 s, with py-spy logging "behind in sampling" throughout). **Treat the flamegraphs as a qualitative call-stack picture only** — `thread_sweep.csv` is the timing evidence. |
| `bench_kernels.py` | **Phase 2.** Per-phase timing of one time step (sweep, copy-back, broadcast, boundary) on a chosen grid tier and thread count, each reported as GB/s of the DRAM traffic it must move. `--init serial\|parallel\|both` compares main-thread NUMA first touch against per-thread first touch, which is the single largest effect on this hardware. -> `data/bench_kernels_full_electrode.json` |
| `run_full_electrode_sweep.sh` | **Phase 2.** Whole-run thread scaling on the `full_electrode` tier, one `srun` step per thread count. Whole runs rather than kernels, because what limits the run now is the phases that were never threaded. -> `data/full_electrode_sweep/threads_N.json` |
| `scientific_validation.py` | Jaffe-theory single-track cross-check and the converged-grid `f(t)` curve, both at 1 and 96 threads -> `data/jaffe_validation.json`, `data/f_t_curve_{N}threads.csv`, `data/converged_grid_summary.json`. Confirms results are thread-count-invariant (to float non-associativity, ~1e-15) before any performance conclusion is trusted. |

`perf` and (initially) `py-spy` are not installed system-wide on this
Helios allocation; `py-spy` was pip-installed into the project's `venv`
instead (see `pyproject.toml`'s `profiling` extra). `perf stat`-style
hardware counters are not available here at all — `resource.getrusage()`
(context switches, user/sys time) in `common.run_once()` is the fallback
evidence for scheduling/fork-join overhead.

## Reproducing

Run from inside an existing Slurm allocation on Helios (`salloc ...` /
the interactive job this was developed under). `module load` must be
repeated in every new shell before activating the venv (the venv itself
doesn't remember which module built it), and every script below needs its
own `srun --overlap --cpus-per-task=N --cpu-bind=none` step -- **not** a
bare `python -m ...` -- because the interactive shell's own process is
itself cpuset-restricted to 1 CPU regardless of how many the job holds
(see the top-level README's "cpuset gotcha"; `os.sched_getaffinity(0)`
would silently report 1 otherwise, and every timing below would be
meaningless).

```bash
module load GCCcore/13.3.0 Python/3.12.3   # repeat in every new shell
source venv/bin/activate
pip install -e ".[profiling]"              # adds py-spy

# 1) thread x threading-layer sweep (~15-20 min; loops srun internally)
bash profiling/run_sweep.sh

# 2) numba parallel_diagnostics + numactl/lscpu + affinity dump
srun --overlap --ntasks=1 --cpus-per-task=8 --cpu-bind=none \
  python -m profiling.diagnostics

# 3) Jaffe cross-check + f(t) curves at 1 and 96 threads
srun --overlap --ntasks=1 --cpus-per-task=96 --cpu-bind=none \
  python -m profiling.scientific_validation

# 4) py-spy flamegraphs at 1 and 96 threads (loops srun internally)
bash profiling/pyspy_record.sh

# 5) cProfile at 1 and 96 threads
srun --overlap --ntasks=1 --cpus-per-task=1 --cpu-bind=none \
  python -m profiling.cprofile_run --threads 1 --out-prefix profiling/data/cprofile_1threads
srun --overlap --ntasks=1 --cpus-per-task=96 --cpu-bind=none \
  python -m profiling.cprofile_run --threads 96 --out-prefix profiling/data/cprofile_96threads
```

All of this overwrites `profiling/data/*` in place (`run_sweep.sh` even
`rm -f`s the CSV first) -- the versions already committed are the ones
described in this README and handed to opencode; re-running regenerates
them (numbers will vary run-to-run, especially past ~96 threads).

## The assignment for opencode

Given only `profiling/data/*` (raw CSVs, JSON, text, SVGs) plus this
README's description of what each file is:

1. Plot wall-time vs. thread count and speedup vs. thread count, per
   threading layer (`omp` vs `workqueue`), against an ideal-scaling
   reference line.
2. Explain the shape of that curve using `diagnostics.txt` (parallel
   region structure, launch count implied by `total_time_steps` in
   `common.run_once`'s output), the context-switch/user-vs-sys-time
   columns in `thread_sweep.csv`, and the flamegraphs — not by citing the
   top-level README.
3. Confirm (or refute) from `jaffe_validation.json` /
   `converged_grid_summary.json` that scientific correctness holds across
   thread counts, and report the Jaffe-theory agreement.
4. Compare the `omp` and `workqueue` threading-layer columns and say which
   one this workload should actually use, and why.
