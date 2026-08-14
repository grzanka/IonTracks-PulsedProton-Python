# Documentation

Five documents, split by the question they answer. Nothing about the history
of the code lives here — only its current state.

| | question | read it if |
|---|---|---|
| [`PHYSICS.md`](PHYSICS.md) | **What is modelled, and why?** | You want to know whether a result means what you think it means. One section per aspect — beam, track structure, dose, chamber, grid, transport, time integration, boundaries, scoring — each stating the assumption and justifying it. Ends with what is *not* modelled and the known systematics. |
| [`ALGORITHM.md`](ALGORITHM.md) | **How is it implemented?** | You are reading or changing the solver. Array layout, the two hot loops, the exact identities that make deposition cheap, what "batching" means and why it is exact, and where the parallelism is. |
| [`PERFORMANCE.md`](PERFORMANCE.md) | **What does it cost?** | You are sizing a run or wondering whether to add threads. Cost model and scaling laws, machine-independent; points at the two benchmark pages for wall times. |
| [`BENCHMARKS-LAPTOP.md`](BENCHMARKS-LAPTOP.md) | **What does it cost *here*?** | You are running on a laptop. CPU spec, tier timings, the full-electrode run, and why one thread is the right number. |
| [`HELIOS.md`](HELIOS.md) | **How do I run it on Helios?** | You are on a Cyfronet Helios node. Module setup, the Slurm cpuset trap, how many cores to ask for and what wall time to expect, thread and dose-rate scaling, why the big grid scales and the small one does not, and which optimisations made it slower. |

Two cross-cutting results that are easy to miss:

- The dominant error is the **finite simulated column**, and it falls only as
  `1/radius` — so it is cheaper to correct analytically than to simulate away
  (`PHYSICS.md` §14).
- The dominant cost is **memory bandwidth**, not arithmetic — so whether
  threads help is decided by whether the grid outgrows the machine's cache, not
  by how many cores are free (`PERFORMANCE.md` §6, `HELIOS.md`).
