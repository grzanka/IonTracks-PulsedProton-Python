# Documentation

Three documents, split by the question they answer. Nothing about the history
of the code lives here — only its current state.

| | question | read it if |
|---|---|---|
| [`PHYSICS.md`](PHYSICS.md) | **What is modelled, and why?** | You want to know whether a result means what you think it means. One section per aspect — beam, track structure, dose, chamber, grid, transport, time integration, boundaries, scoring — each stating the assumption and justifying it. Ends with what is *not* modelled and the known systematics. |
| [`ALGORITHM.md`](ALGORITHM.md) | **How is it implemented?** | You are reading or changing the solver. Array layout, the two hot loops, the exact identities that make deposition cheap, what "batching" means and why it is exact, and where the parallelism is. |
| [`PERFORMANCE.md`](PERFORMANCE.md) | **What does it cost?** | You are sizing a run or wondering whether to add threads. Cost model, measured timings, scaling laws, and the many-core guidance. |

Two cross-cutting results that are easy to miss:

- The dominant error is the **finite simulated column**, and it falls only as
  `1/radius` — so it is cheaper to correct analytically than to simulate away
  (`PHYSICS.md` §14).
- The dominant cost is **memory bandwidth**, not arithmetic — so more threads
  frequently do not help (`PERFORMANCE.md` §6).
