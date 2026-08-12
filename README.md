# pulsed-ion-chamber

A standalone, pure Python/NumPy simulation of ion recombination in a
parallel-plate, gas-filled ionization chamber exposed to a **pulsed proton
beam** (default: 540 us pulses, 50 Hz repetition rate, 60 Gy/s
time-averaged dose rate -- a FLASH-proton-therapy-like scenario).

It solves the coupled drift-diffusion-recombination equations for the
positive/negative ion charge-carrier densities `n+`, `n-` created by
individual proton tracks:

```
d n_+-/dt = D * laplacian(n_+-)  -+  mu * E * d(n_+-)/dz  -  alpha * n+ * n-
```

using the explicit **Lax-Wendroff** finite-difference scheme on a 3D grid,
with protons modeled as randomly-placed, randomly-timed Gaussian ion tracks
(amorphous track-structure theory) injected only during repeating
540 us-wide pulse windows.

## Why this exists / where it came from

This code is extracted and adapted from the `IonTracks-Cython` repository
(J.B. Christensen et al., contact jeppe.christensen@psi.ch), which implements the same physics across several
backends (Cython, Numba, CuPy) for **continuous** ion beams and for
**pulsed, spatially-uniform** electron/photon beams. Neither existing
solver directly covers *pulsed, non-uniform (track-structure) proton
beams*, so this repository combines:

- the Gaussian track-structure / Lax-Wendroff PDE solver from
  `hadrons/continuous_beam.py` (non-uniform ionization), with
- the pulse on/off timing idea from `electrons/common/pulsed_e_beam.py`
  (generalized here from a single pulse to a repeating pulse train).

It is deliberately written with **plain, explicit nested Python loops**
(see `pulsed_ion_chamber/solver.py`) rather than vectorized NumPy or a
compiled backend -- unlike the original repository, this is *not* meant to
be fast. It is meant to be an obviously-correct, easy-to-read starting
point for a multi-threading or GPU parallelization exercise. The two hot
loops, `_insert_track` (per-track Gaussian deposition) and
`_lax_wendroff_step` (per-voxel PDE update), are exactly where the original
project used Cython/Numba/CuPy for a 6-17x (or, on GPU, much larger)
speedup.

Physics validation (Jaffe theory of initial recombination, Boag theory for
uniform beams), physical constants (Kanai 1998), and the LET/track-radius
data tables are ported/reused from the original repository. See
`pulsed_ion_chamber/theory.py` for references.

**License**: GPLv3, inherited from IonTracks-Cython (see `LICENSE`), since
this is a derivative work.

## Physics validity and known simplifications

- **Recombination scoring is un-gated**: every track's charge and every
  step's recombination are counted from the very first time step,
  because this code targets a small number of pulses rather than a
  continuous steady-state beam. (The original `continuous_beam` solver
  discards an initial "build-up" period before scoring, which only makes
  sense when tracks arrive continuously for many separation times.)
- **Boag theory does not strictly apply** to proton track structure (it
  assumes a spatially uniform charge density); it is included in
  `theory.py` only as a rough cross-check for the general-recombination
  regime. **Jaffe theory** is the correct, and closely-matching (see
  `tests/test_single_track_vs_jaffe.py`), reference for the single-track
  (low-dose) limit.
- All constants (mobility, diffusion, recombination coefficient, `W`) are
  averaged, isotropic values for **air**, per Kanai et al. (1998), exactly
  as in the original repository.

## The reduced simulation volume (read this before trusting a `k_s` value)

A real ionization chamber's collecting volume is much bigger than what this
code (or the original IonTracks continuous-beam solver) actually
simulates. Because a single proton in air has such a low LET (~5e-4 to
0.03 keV/um, roughly 1000x lower than in water, since air is ~1000x less
dense), depositing a clinically-relevant dose in a 60 Gy/s pulse requires
a huge number of tracks -- so, as in the original repository, only a small,
statistically-representative cylindrical sub-volume of the chamber is
simulated explicitly (`sampled_radius_cm`), not the whole active area.

The catch: this port uses **plain Python loops**, which are ~10-100x
slower than the original's Cython/Numba backends. To keep the default
example runnable in under a minute, `SimulationConfig()`'s defaults use a
sampled radius of only about one Gaussian track radius and a coarse grid
(`grid_size_um=40`) -- good enough to validate the solver's mechanics and
to reproduce Jaffe theory in the single-track limit, but **not**
dosimetrically converged. `examples/run_pulsed_proton_beam.py` uses
`pulsed_ion_chamber.benchmark.estimate_full_runtime()` to show, without
actually running them, how much longer a properly converged sampled volume
(a few track radii across, per the convergence study documented in the
original repo's `documentation/README.md`) would take in serial Python --
typically hours to days. Closing that gap is the point of this repository.

## Installation

Requires Python >= 3.9. No compiler, no Cython, no GPU needed -- just
NumPy/SciPy/pandas/mpmath/matplotlib.

```bash
python -m venv venv
source venv/bin/activate
pip install -e .          # or: pip install -r requirements.txt
```

## Usage

```bash
python examples/run_pulsed_proton_beam.py
```

Takes roughly 1-2 minutes on a laptop. It runs a coarse-grid version of the
default pulsed-proton scenario, prints and plots (`pulsed_proton_beam_f_of_t.png`)
the collection efficiency `f(t)` and recombination correction factor
`k_s = 1/f`, validates against Jaffe theory in the single-track limit, and
prints the estimated (not actually run) cost of a converged version of the
same scenario. Expect output along these lines:

```
Final collection efficiency f = 0.6566
Recombination correction factor k_s = 1/f = 1.5229
...
k_s (PDE simulation) = 1.001336
k_s (Jaffe theory)   = 1.001234
...
  coarse (demo above)                              : ...  ~0.014 h serial-Python estimate
  ~6 track radii, converged grid (matches original) : ...  ~1.6e+02 h serial-Python estimate
```

### Running the tests

```bash
pip install -e ".[dev]"   # or: pip install pytest
pytest
```

All 8 tests finish in well under 10 seconds (they use even smaller grids
than the example) and include a check against analytic Jaffe theory in the
single-track limit.

Programmatic use:

```python
from pulsed_ion_chamber import SimulationConfig, run_simulation

config = SimulationConfig(
    E_MeV_u=150.0,          # proton energy
    voltage_V=200.0,        # chamber bias voltage
    electrode_gap_cm=0.2,   # electrode gap
    pulse_duration_s=540e-6,
    repetition_rate_hz=50.0,
    dose_rate_Gy_s=60.0,    # time-averaged dose rate
    n_pulses=1,
    seed=1,
)
print(config.summary())

result = run_simulation(config)
print(result.ks)      # recombination correction factor after full clearance
print(result.f_t)     # collection efficiency vs. time
```

## Configurable parameters (`SimulationConfig`)

| Parameter | Meaning |
|---|---|
| `E_MeV_u`, `particle` | Proton energy and species (sets LET via PSTAR data, and hence track radius and track density) |
| `voltage_V`, `electrode_gap_cm` | Chamber bias and electrode gap -> electric field |
| `pulse_duration_s`, `repetition_rate_hz` | Pulse train timing |
| `dose_rate_Gy_s` | Time-averaged dose rate -> number of tracks per pulse |
| `n_pulses` | How many pulses to simulate (>1 checks pulse-to-pulse residual buildup) |
| `grid_size_um` | Voxel size (smaller = finer, slower) |
| `sampled_radius_cm` | Radius of the simulated cylindrical sub-volume (bigger = more statistically representative, much slower) |
| `buffer_radius`, `no_z_electrode` | Margin voxels so charge never drifts off the array edge in one step |
| `n_clearance_separation_times` | How many ion-transit times to keep simulating after the last pulse, so charge can clear the gap |
| `seed` | RNG seed for track positions/arrival times |

## Repository layout

```
pulsed_ion_chamber/
  constants.py       Kanai (1998) air ion-transport constants
  stopping_power.py  LET lookup (PSTAR/libamtrack data), track-radius fit, dose-rate -> fluence-rate
  theory.py           Jaffe theory (single track) and Boag theory (uniform density) analytic references
  config.py           SimulationConfig: physical inputs + derived grid/timing quantities
  pulses.py           Pulse-train track scheduling (arrival times, xy positions)
  solver.py           The explicit-loop Lax-Wendroff PDE solver -- the code to parallelize
  benchmark.py        Extrapolates full-scenario runtime from a few measured loop iterations
  data/               Packaged LET/stopping-power tables
examples/
  run_pulsed_proton_beam.py
tests/
  test_single_track_vs_jaffe.py
  test_grid_and_timing.py
```

## Parallelization starting points

- `_insert_track` (in `solver.py`): independent per-voxel work for a fixed
  track; embarrassingly parallel over `(i, j, k)`, and independent tracks
  within the same time step could also be parallelized (they only read a
  shared array and add to it -- watch for the race on `+=`).
- `_lax_wendroff_step`: independent per-voxel stencil update; the classic
  structured-grid finite-difference parallelization target (domain
  decomposition, GPU kernel, or simple NumPy vectorization first).
- The time loop itself (`run_simulation`) is inherently sequential (each
  step depends on the previous one), so parallelism has to be found
  *within* each step, not across steps.

## References

- Christensen, J.B., Tolli, H., Bassler, N. (2016). "A general algorithm
  for calculation of recombination losses in ionization chambers exposed
  to ion beams." *Medical Physics* 43(10):5484-92.
- Christensen, J.B. et al. (2020). "Mapping initial and general
  recombination in scanning proton pencil beams." *Phys. Med. Biol.* 65,
  115003.
- Kanai, T. et al. (1998). Ion mobility/diffusion/recombination
  coefficients in air.
- Boag, J.W., Currant, J. (1980). "Current collection and ionic
  recombination in small cylindrical ionization chambers exposed to pulsed
  radiation." *Br. J. Radiol.* 53.
