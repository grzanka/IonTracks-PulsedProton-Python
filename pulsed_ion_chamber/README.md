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
| `solver_numba_parallel.py` | **Batched backend.** Deposits a whole step's tracks in one pass, both hot loops under `prange`. Best for dense pulses and large grids. |
| `state.py` | The run record (`Result`), the per-step `Diagnostics`, and the boundary update — the parts both backends share that are not JIT kernels. |

The two backends are not the same code: one broadcasts each track down the gap,
the other sums a step's tracks first and broadcasts once. They agree to 1e-9,
and `tests/test_backends_agree.py` is what holds them to it.

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
