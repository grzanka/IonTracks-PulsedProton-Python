# Phase 1: raw profiling & scientific-validation data

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
| `scientific_validation.py` | Jaffe-theory single-track cross-check and the converged-grid `f(t)` curve, both at 1 and 96 threads -> `data/jaffe_validation.json`, `data/f_t_curve_{N}threads.csv`, `data/converged_grid_summary.json`. Confirms results are thread-count-invariant (to float non-associativity, ~1e-15) before any performance conclusion is trusted. |

`perf` and (initially) `py-spy` are not installed system-wide on this
Helios allocation; `py-spy` was pip-installed into the project's `venv`
instead (see `pyproject.toml`'s `profiling` extra). `perf stat`-style
hardware counters are not available here at all — `resource.getrusage()`
(context switches, user/sys time) in `common.run_once()` is the fallback
evidence for scheduling/fork-join overhead.

## Reproducing

```bash
module load GCCcore/13.3.0 Python/3.12.3   # every new shell, before activating the venv
source venv/bin/activate
pip install -e ".[profiling]"

bash profiling/run_sweep.sh                        # ~15-20 min, writes data/thread_sweep.csv
python -m profiling.diagnostics                    # needs its own --cpus-per-task, see the script
python -m profiling.scientific_validation
bash profiling/pyspy_record.sh
python -m profiling.cprofile_run --threads 1  --out-prefix profiling/data/cprofile_1threads
python -m profiling.cprofile_run --threads 96 --out-prefix profiling/data/cprofile_96threads
```

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
