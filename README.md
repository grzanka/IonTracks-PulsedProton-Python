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

It is deliberately written with **plain, explicit nested loops** rather
than vectorized NumPy, so that a later multi-threading/GPU port has
obvious per-voxel/per-track parallelism to exploit. There are two
implementations of the same algorithm:

- `pulsed_ion_chamber/solver.py` -- the plain pure-Python reference,
  mirroring the original IonTracks-Cython loop structure one-to-one, easiest
  to read, slowest.
- `pulsed_ion_chamber/solver_numba.py` -- **the baseline backend for this
  repository**: the same physics, JIT-compiled with `@numba.njit`, still
  single-threaded (no `parallel=True`, no `prange`), *and* algorithmically
  restructured (see its module docstring for the details): loop order
  matched to array memory layout, the per-track density computation hoisted
  out of the (redundant) z-loop, the 2D Gaussian factored into two 1D
  Gaussians (`exp(a+b) = exp(a)*exp(b)`, exact, not an approximation), and
  `sqrt()` replaced by squared-distance comparisons. This is what
  `examples/run_pulsed_proton_beam.py` and the package's top-level
  `run_simulation_numba` actually run.

The two hot loops, `_insert_track`/`_insert_track_numba` (per-track
Gaussian deposition) and `_lax_wendroff_step`/`_lax_wendroff_step_numba`
(per-voxel PDE update), are exactly where the original project used
Cython/Numba/CuPy for a 6-17x (or, on GPU, much larger) speedup, and where
your next parallelization step (numba `prange`, multiprocessing, or a GPU
port) should target.

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

The catch: even the single-threaded Numba baseline used here is much
slower than the original's Cython/Numba/CuPy backends (which are tuned,
and in CuPy's case run on GPU). `SimulationConfig()`'s own defaults use a
sampled radius of only about one Gaussian track radius and a coarse grid
(`grid_size_um=40`) -- good enough to validate the solver's mechanics and
to reproduce Jaffe theory in the single-track limit, but **not**
dosimetrically converged; `examples/run_pulsed_proton_beam.py` uses a
somewhat finer grid (~3.6 track radii, tuned to ~30s single-threaded).
`pulsed_ion_chamber.benchmark.estimate_full_runtime()` shows, without
actually running it, that reaching the fully converged sampled volume used
in the original repo's convergence study (`documentation/README.md`,
~6 track radii across) takes on the order of **tens of minutes**,
single-threaded Numba -- down from an estimated ~5 days for the plain
pure-Python backend on the same grid. Closing the remaining gap (minutes ->
seconds) is the point of this repository.

## Installation

Requires Python >= 3.9 and Numba (the baseline backend). No compiler, no
Cython, no GPU needed.

```bash
python -m venv venv
source venv/bin/activate
pip install -e .          # or: pip install -r requirements.txt
```

## Usage

```bash
python examples/run_pulsed_proton_beam.py
```

Takes about 30-45 seconds on a laptop, entirely through
`pulsed_ion_chamber.solver_numba` (single-threaded: no `parallel=True`, no
`prange`). It runs the pulsed-proton scenario on a grid about 3.6
ion-track-radii wide (tuned to take ~30 s single-threaded; still coarser
than the ~6-track-radii "converged" grid from the original IonTracks
study), prints and plots (`pulsed_proton_beam_f_of_t.png`) the collection
efficiency `f(t)` and recombination correction factor `k_s = 1/f`;
validates against Jaffe theory in the single-track limit; and prints the
*estimated* (not actually run) single-threaded-Numba cost of reaching a
fully converged grid. Expect output along these lines:

```
Numba compile time (one-off): 0.10 s
Wall time (numba, single-threaded): 28.1 s
Final collection efficiency f = 0.5391
Recombination correction factor k_s = 1/f = 1.8548
...
k_s (PDE simulation, numba) = 1.001336
k_s (Jaffe theory)          = 1.001234
...
  This demo's grid, numba estimate  : ~33 s (actually took 28 s -- the estimate is a sanity check on the cost model)
  ~3 track radii, finer grid                                       : ...  ~0.003 h single-threaded numba estimate
  ~6 track radii, converged grid (matches original IonTracks study): ...  ~0.4 h single-threaded numba estimate
```

Two actually-measured (not estimated) data points on the speedup, both
full runs, not extrapolations: `tests/test_solver_numba.py` compares the
Numba backend against the plain pure-Python reference on a small, fast
config (~10x on this development machine); separately, restructuring the
loops in `solver_numba.py` (described above) was measured at a further
~7x on top of that, before vs. after, on the same (previous, smaller)
demo grid with JIT compilation held constant. `pulsed_ion_chamber.benchmark`'s
cost model suggests the combined ratio grows substantially larger on
bigger grids (it estimated ~475x for the current demo grid), but that
number comes from timing only a handful of pure-Python loop iterations
and extrapolating -- actually running the pure-Python backend at this
grid size to verify it directly would itself take too long to be worth
doing, so treat it as a rough, unverified indication of scale rather than
a precise figure. Turning the remaining tens-of-minutes convergence cost
into seconds is the next step: `prange`, multiprocessing, or a GPU
backend.

### Running the tests

```bash
pip install -e ".[dev]"   # or: pip install pytest
pytest
```

All 11 tests finish in a few seconds (they use much smaller grids than the
default example) and include a check against analytic Jaffe theory in the
single-track limit, plus a numba-vs-pure-Python correctness check.

Programmatic use:

```python
from pulsed_ion_chamber import SimulationConfig, run_simulation_numba

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

result = run_simulation_numba(config)  # or run_simulation() for the plain pure-Python reference
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
| `rf_frequency_hz` | Optional: the accelerator's RF (e.g. a cyclotron's ~10-100 MHz extraction RF), purely diagnostic -- see below |
| `seed` | RNG seed for track positions/arrival times |

### Are protons injected all at once, and where does the accelerator's RF fit in?

No -- within a pulse, `pulses.py` spreads track arrival times
pseudo-uniformly across the *entire* `pulse_duration_s` window (a
cumulative-sum-of-uniforms trick inherited from the original
IonTracks-Cython `continuous_beam` model), not clustered at the start.

Real proton beams also have RF microstructure: a cyclotron only
accelerates/extracts protons in bunches spaced by its extraction RF period
(e.g. 26.26 MHz -> a bunch every ~38 ns), so a "540 us pulse" is really
~14,000 RF buckets, not a continuous stream. This code does **not** place
tracks at explicit RF-bucket times, and that's intentional: the
simulation's time step `dt` (set by the von Neumann stability criterion
for the chosen grid, not a free parameter) always comes out far longer
than one RF period -- typically ~13 RF cycles per `dt` for a 26.26 MHz
cyclotron on this repository's grids. Individual RF buckets fall inside
the same `dt` regardless of how arrival times are modeled within it, so
resolving them explicitly would take a `dt` (and correspondingly a grid)
tens of times finer, at a computational cost far beyond what even the
Numba-optimized backend can do in this exercise. Averaging over the RF
microstructure is the physically- and numerically-correct simplification
here, not a shortcut.

Pass `rf_frequency_hz` to `SimulationConfig` to make this explicit and
checked: `config.summary()` reports the resulting RF-cycles-per-time-step,
and a warning fires if it ever drops below 1 (meaning `dt` would actually
be fine enough that the averaging assumption deserves a second look) --
see `tests/test_grid_and_timing.py` for both cases.

## Repository layout

```
pulsed_ion_chamber/
  constants.py       Kanai (1998) air ion-transport constants
  stopping_power.py  LET lookup (PSTAR/libamtrack data), track-radius fit, dose-rate -> fluence-rate
  theory.py           Jaffe theory (single track) and Boag theory (uniform density) analytic references
  config.py           SimulationConfig: physical inputs + derived grid/timing quantities
  pulses.py           Pulse-train track scheduling (arrival times, xy positions)
  solver.py           The plain pure-Python explicit-loop Lax-Wendroff PDE solver (reference implementation)
  solver_numba.py     The baseline backend: same solver, two hot loops JIT-compiled with numba (single-threaded)
  benchmark.py        Extrapolates full-scenario runtime from a few measured loop iterations
  data/               Packaged LET/stopping-power tables
examples/
  run_pulsed_proton_beam.py
tests/
  test_single_track_vs_jaffe.py
  test_grid_and_timing.py
  test_solver_numba.py
```

## Parallelization starting points

`solver_numba.py` already takes the first steps: JIT compilation
(`@numba.njit`, no `parallel=True`/`prange`) plus the single-threaded
algorithmic restructuring described above (loop order, hoisting,
separable Gaussian, squared-distance checks) -- together worth roughly two
orders of magnitude over the plain pure-Python reference (see the
"actually-measured" numbers above), with no change in what's computed.
From here:

- `_insert_track_numba`: independent per-voxel work for a fixed track;
  embarrassingly parallel over `(i, j, k)` (e.g. `prange` over `i`), and
  independent tracks within the same time step could also be parallelized
  (they only read a shared array and add to it -- watch for the race on
  `+=`).
- `_lax_wendroff_step_numba`: independent per-voxel stencil update; the
  classic structured-grid finite-difference parallelization target
  (`prange` over `i`, domain decomposition, a GPU kernel, or NumPy
  vectorization).
- The time loop itself (`run_simulation_numba`) is inherently sequential
  (each step depends on the previous one), so parallelism has to be found
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
