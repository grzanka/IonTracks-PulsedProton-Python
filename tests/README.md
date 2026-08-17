# Tests

`pytest` runs the lot in about 20 seconds. Every test uses a deliberately small
grid; correctness here does not depend on scale, and the scale-dependent claims
live in `docs/PERFORMANCE.md` instead.

| file | what it pins down |
|---|---|
| `test_single_track_vs_jaffe.py` | **The physics check.** In the single-track, low-dose limit the solver must reproduce analytic Jaffe theory for initial recombination. This is the only test that validates against something outside this codebase, and it exercises deposition, transport and recombination together. |
| `test_backends_agree.py` | The two CPU backends must agree to 1e-9 on `f(t)`, on `k_s`, on the final density field and on the track positions drawn. They differ in loop structure, in parallelism and in when the z-broadcast happens, so this is a real cross-implementation check, not a tautology. |
| `test_cuda_backend.py` | The GPU backend (`solver_cuda`) must reproduce the serial reference — `f(t)`, `k_s`, the full field, the track map, both wall conditions, both scoring regions, and the two-species stencil. The field matches to ~1e-15 (near machine epsilon; the per-voxel arithmetic is identical, only the FMA and the reduction order differ). Skips cleanly where there is no GPU, so the suite still passes on a CPU-only machine. |
| `test_grid_and_timing.py` | Derived grid geometry and pulse timing, including the degenerate configs that must raise rather than hang. |
| `test_rdd.py` | The tabulated-RDD track model: that the table's own radial integral reproduces an independent Bethe LET, that the stencil conserves energy exactly, that the in-domain fraction is *grid-independent* (what separates area-averaging from point-sampling), that the far field matches the closed-form area average, and that both CPU backends agree on the new path. |
| `test_v2_physics.py` | Every configurable physics option: two carrier species and the `dt` they imply, the von Neumann criterion, `chamber_fill_fraction`, the boundary and scoring modes, the deposition stencil, and the resource guards. Also asserts that all defaults reproduce the original single-species behaviour. |
| `test_output_and_diagnostics.py` | The per-step record: that it integrates back to `f(t)`, that injection stops with the pulse, that the gap empties, that the CSV is in ion pairs rather than densities, and that the plots are produced. |

## Conventions

- **Tolerances are 1e-9, not exact.** Reduction order differs between the
  backends, and float addition is not associative. Demanding bit equality would
  fail on that alone; anything looser would hide a real divergence.
- **Tests assert *why*, not just *what*.** Where a number looks arbitrary the
  docstring says where it comes from, so a future change that legitimately moves
  it can be told apart from a regression.
