# Training plan: opencode performance-engineering exercise

**Goal**: use this repository as a hands-on exercise for an AI coding
agent (opencode, backed by open-weight models) to (1) independently
diagnose why a real, physically-correct HPC simulation scales badly
across CPU cores, and (2) attempt to actually fix it -- then compare its
reasoning and results against what's already known about this codebase.

This is a two-phase exercise. Phase 1 (diagnose) is done; this document
also specifies Phase 2 (optimize) so it can be run without re-deriving the
design from scratch.

## Why this codebase

`pulsed_ion_chamber` solves a real drift-diffusion-recombination PDE
(ion recombination in a pulsed-proton ionization chamber) with an
existing single-threaded baseline (`solver_numba.py`) and an existing
shared-memory parallel backend (`solver_numba_parallel.py`) that already
has a known, counter-intuitive performance story: an algorithmic batching
fix produced a real ~65x speedup, but naively adding more threads on top
of it made things *worse*, not better, past roughly one socket's worth of
cores (see the top-level `README.md`'s "Running on many cores" section).
That gap between "obvious next step" (more threads) and "what actually
works" (the batching fix, and, for using many cores, independent
replicas rather than one threaded run) is exactly the kind of reasoning
this exercise is meant to test in an agent, not just its ability to write
`prange` syntax.

Everything here runs on a real Cyfronet Helios allocation (dual AMD EPYC
9654, 192 physical cores, 8 NUMA domains) with real physics validation
(Jaffe theory) available as a correctness check, so both "is it fast" and
"is it still right" are answerable, not just assumed.

## Phase 1 -- Diagnose (done, see `profiling/`)

**Design decision**: this exercise runs directly on `master`, in the open
-- opencode has access to the same repo, including the top-level
`README.md`'s prose describing the batching fix and the "threads hurt"
finding. That conclusion is *not* hidden. The point of Phase 1 is not to
keep opencode ignorant of the answer; it's to make it **re-derive that
answer (or a better one) from raw evidence**, and to go further than the
prose does, rather than just reading and repeating it. `profiling/README.md`
states this explicitly as the assignment.

**What was built and run** (see `profiling/README.md` for full details
and exact reproduction commands):

- A harness (`profiling/common.py`, `sweep.py`/`run_sweep.sh`,
  `diagnostics.py`, `cprofile_run.py`, `pyspy_record.sh`,
  `scientific_validation.py`) that measures the existing
  `solver_numba_parallel` backend on the "converged grid" scenario across
  a thread-count x Numba-threading-layer (`omp`/`workqueue`) sweep, each
  point run as its own Slurm step for correct CPU affinity.
- Raw artifacts in `profiling/data/`: `thread_sweep.csv` (wall time, user/
  sys time, voluntary/involuntary context switches -- a `perf stat`
  substitute, since `perf` isn't installed on this allocation),
  `diagnostics.txt` (`numba.parallel_diagnostics`, NUMA topology,
  affinity), native py-spy flamegraphs at low and high thread count,
  cProfile traces, and scientific-correctness artifacts (`k_s`/`f(t)`
  invariance across thread counts, Jaffe-theory agreement).

**Handoff to opencode**: point it at `profiling/data/*` and
`profiling/README.md`'s "assignment" section. Ask for:
1. Plots of wall-time and speedup vs. thread count, per threading layer.
2. A written diagnosis of the scaling curve's shape, grounded in the
   context-switch counts, `parallel_diagnostics` output, and flamegraphs
   -- not in the top-level README's prose.
3. A correctness check: does `k_s`/`f(t)` actually hold constant across
   thread counts, and does it match Jaffe theory?
4. A recommendation on `omp` vs. `workqueue` and on what to actually do
   with ~190 cores (this is a leading question on purpose: "more threads"
   is the wrong answer, and a good response should notice that from the
   data rather than assume it).

**Grading Phase 1**: score against what's already known (the ~65x
batching fix, the fork-join/NUMA-launch-overhead explanation for why
threads hurt past ~1 socket, and "independent replicas, not threads" as
the actual way to use many cores -- all in the top-level README). Did
opencode reach the same conclusions from the raw data, on its own
reasoning? Did it notice anything the README doesn't mention (e.g. the
`workqueue`-vs-`omp` context-switch-count difference, which is new data
not in the original investigation)?

## Phase 2 -- Optimize (next)

**Setup**: give opencode its own Phase 1 diagnosis (not the grader's, not
the top-level README) as context, plus the profiling harness, and ask it
to make the converged-grid scenario meaningfully faster without changing
the physics. Suggested framing: "the current parallel backend is slower
past ~96 threads than at 1; make this scale, or explain why it
structurally can't and propose the right way to use ~190 cores instead."

**What "done" looks like** -- any of, in rising order of ambition:
1. Correctly recommends and implements **independent replicas**
   (`multiprocessing.Pool` or a Slurm job array of single-threaded runs)
   as the way to spend many cores, instead of trying to make one run
   scale -- this is the actual right answer per the existing
   investigation, so an agent that reaches it independently has done the
   core reasoning correctly even without new code.
2. Finds a further **algorithmic** improvement to `_accumulate_track_density_numba_parallel`/
   `_broadcast_density_numba_parallel`/`_lax_wendroff_step_numba_parallel` beyond the existing
   batching fix (e.g. reducing per-step parallel-region launch count
   further, larger time-step batching, a different data layout).
3. Gets real threaded scaling past 1 socket by addressing the fork-join
   cost directly (e.g. a persistent thread pool / manual OpenMP region
   spanning multiple time steps instead of one `prange` launch per step,
   which the existing docstring in `solver_numba_parallel.py` identifies
   as the actual bottleneck once batching is in place).
4. A GPU port (CuPy, as the original `IonTracks-Cython` project did) --
   out of scope for a first pass but worth naming as the honest ceiling.

**Process**:
1. opencode implements a candidate change.
2. Re-run `profiling/run_sweep.sh` (or a trimmed version -- see note
   below) against the new code; regenerate `scientific_validation.py`'s
   output to confirm `k_s`/`f(t)` still match Jaffe theory and the
   pre-change baseline (tolerance: float non-associativity only, ~1e-6
   relative, per `tests/test_backends_agree.py`'s existing
   tolerance).
3. Compare before/after plots; iterate.
4. Land the change with its own tests (extend
   `tests/test_backends_agree.py`'s pattern) and update
   `docs/PERFORMANCE.md` with the new numbers -- don't let the committed
   prose go stale relative to the code.

**Grading Phase 2**: did wall time actually improve on the converged
grid? Did it improve *for the right reason* (i.e. does the Phase 2
diagnosis match what actually changed, not just "I added more threads
and it happened to help this time")? Is scientific correctness still
intact? Is the fix explained in a way a future maintainer (human or
agent) could verify without re-running the whole sweep?

## Practical notes for whoever runs this (human or agent)

- Every Slurm step needs `srun --overlap --cpus-per-task=N --cpu-bind=none`
  -- a bare `python` in an interactive allocation's shell is itself
  cpuset-restricted to 1 CPU regardless of the job's size. See
  `docs/PERFORMANCE.md`'s "Slurm / cpuset caveat" and `profiling/README.md`'s
  "Reproducing" section for exact invocations.
- `module load GCCcore/13.3.0 Python/3.12.3` must be repeated in every
  new shell before `source venv/bin/activate`.
- The full 12-point x 2-layer sweep takes ~15-20 min; `profiling/
  run_sweep.sh`'s `THREADS` array can be trimmed (as done for the
  committed `profiling/data/thread_sweep.csv`, currently 1/16/64/128/190)
  for faster iteration during Phase 2 -- a coarser sweep is enough to
  tell "did this change help", a finer one is worth it only for the final
  reported numbers.
- `perf` and `py-spy` are not installed system-wide on this Helios
  allocation; `py-spy` is available via `pip install -e ".[profiling]"`
  into the project venv. There is no system-wide `perf`, so
  `resource.getrusage()` counters (in `profiling/common.py`) are the
  fallback evidence for scheduling overhead.
- Per the repo's stated longer-term plan: once this exercise's findings
  are solid, the intent is a clean refactor and a move to a new
  repository -- so treat `master` here as a working/experimental history,
  not the final shape of the code.
