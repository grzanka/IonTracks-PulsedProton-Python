# `pulsed_ion_chamber` — package layout

What each module is for. The physics is in [`../docs/PHYSICS.md`](../docs/PHYSICS.md),
the algorithm in [`../docs/ALGORITHM.md`](../docs/ALGORITHM.md).

## Setting up a run

| module | contents |
|---|---|
| `config.py` | `SimulationConfig`: every physical input, plus the derived grid, time step and track count. Validates aggressively — an impossible grid, an unstable `dt` or a run larger than available RAM is refused here rather than discovered mid-run. |
| `constants.py` | Kanai (1998) ion-transport constants for air, `W`, air densities. Single- and two-species values side by side. |
| `stopping_power.py` | LET from tabulated PSTAR data, the Rossomme track-radius fit, and dose-rate → fluence-rate. |
| `resources.py` | Host RAM and CPU discovery, and the guards built on them. |

## Running

| module | contents |
|---|---|
| `pulses.py` | When each track arrives and where it lands. |
| `solver_numba.py` | **Baseline backend.** Deposits one track at a time. Simpler; best when few tracks arrive per step. |
| `solver_numba_parallel.py` | **Batched backend.** Deposits a whole step's tracks in one pass, both hot loops under `prange`. Best for dense pulses and large grids on a multi-core CPU. |
| `solver_cuda.py` | **GPU backend.** The same physics on an NVIDIA GPU, carrier arrays resident on the device for the whole run. Best for grids too large for a CPU's cache (the 5 µm full electrode). Imports CuPy + Numba-CUDA lazily, so a CPU-only install is unaffected. See [`../docs/GPU.md`](../docs/GPU.md). |
| `state.py` | The run record (`Result`), the per-step `Diagnostics`, and the boundary update — the parts all backends share that are not JIT kernels. |

The three backends are not the same code: one broadcasts each track down the
gap, the other two sum a step's tracks first and broadcast once (on CPU threads
and on the GPU respectively). They agree on `k_s` to a relative tolerance —
`tests/test_backends_agree.py` holds the two CPU backends to 1e-9, and
`tests/test_cuda_backend.py` holds the GPU backend to the serial reference
(the field matches to ~1e-15, near machine epsilon).

## Reporting

| module | contents |
|---|---|
| `output.py` | Writes `collected_charge.csv`, converting the solver's voxel-density sums into absolute ion pairs. |
| `plots.py` | The four diagnostic figures: injection rate, carrier evolution, recombination rate, track cross-section. |
| `theory.py` | Jaffe (single track) and Boag (uniform density) analytic references. Jaffe is the meaningful one here; Boag assumes a uniformity this beam does not have. |
| `benchmark.py` | Times a few kernel calls and extrapolates, so a run can be sized before it is started. |

## Data

`data/stopping_power_air.csv` (PSTAR) and `data/LET_b.dat` (track radius vs LET),
both inherited from IonTracks.
