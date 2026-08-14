# IFJ AIC-144 — `Markus 2 mm, macropulse, 10 Gy/s`

Findings from mapping one archived IonTracks v2 (FEniCSx) run onto this
repository, plus a full cross-code comparison of the physics and numerics
of the three IonTracks implementations involved — and the port of v2's
physics into this repository's rectangular-grid solver.

Sections 1–4 are the analysis. **Section 5 is what was implemented**,
section 6 is what was deliberately not, and **section 7 is the
execution-time report**. Run it with
`python examples/ifj_aic144/run_markus_2mm.py`.

Source of the reference case:
`IonTracks-experiments-archive/campaigns/ifj_aic144/markus_chambers_20260813/`
(`runs/markus_2mm_macropulse_10Gys/`), IonTracks commit `18f6b53-dirty`,
configs resolved at `611da70-dirty`.

---

## 1. The reference case

AIC-144 isochronous cyclotron at IFJ PAN, 60 MeV nominal degraded to
**56.2 MeV** at the measurement plane; PTW Markus 23343 plane-parallel chamber,
**2.0 mm** gap at **300 V** (150 kV/m). Macro-pulse **540 µs** repeating every
**20 ms** (duty cycle 1/37), one pulse simulated. Mean dose rate 10 Gy/s to
water → **0.1782 Gy to air per pulse** (water→air factor 0.891) → **54 239
tracks** over a 0.12 mm-radius gas column.

Archive result: **k_s = 1.1629** (exact, 1/f) / **1.1400** (first-order,
1 + R/I), 14.00 % of the injected charge recombined.

## 2. Parameter mapping onto `SimulationConfig`

| FEniCSx archive | value | this repo | agreement |
|---|---|---|---|
| particle / energy | 1H, 56.2 MeV (σ = 0.1 MeV) | `E_MeV_u=56.2` | LET **1.1994844e-3** vs archived **1.199470e-3** keV/µm — same PSTAR table |
| Gaussian track radius | 20 µm (config-set) | `calc_track_radius_cm` floors at **2e-3 cm = 20 µm** | numerically equal *by coincidence of the floor* — see §4.9 |
| bias / gap | 300 V, 2.0 mm | `voltage_V=300`, `electrode_gap_cm=0.2` | exact, E = 1500 V/cm |
| pulse structure | 540 µs, 20 ms period, 1 pulse | `pulse_duration_s=540e-6`, `repetition_rate_hz=50`, `n_pulses=1` | exact |
| gas column radius | 0.12 mm | `sampled_radius_cm` + `buffer_radius` | see §3 |
| track sampling radius | 0.084 mm (`chamber_fill_fraction=0.7`) | no equivalent — this code always fills the sampled disc | see §4.1 |
| mesh | uniform tets, `lc = 0.01 mm`, 366 002 cells / **72 165 DOF** | `grid_size_um=10.0` → 24×24×210 = **120 960 voxels** | same resolution, same domain |
| dose to air per pulse | 0.1782 Gy | `dose_rate_Gy_s=8.91` (mean, to air) | needs `air_density_kg_m3` — see §4.3 |
| W | 33 eV | hardcoded **34.2 eV** | see §4.2 |
| α | 1.6e-12 m³/s | 1.60e-6 cm³/s | identical |
| space-charge screening | off | not implemented | equal |
| magnetic field | off | not implemented | equal |
| RNG seed | 20260527 | `seed=20260527` | different generators, so not reproducible track-for-track |

### Grid quantisation

`config.py` computes `no_xy = int(2*r/u) + 2*buffer`. With
`sampled_radius_cm=0.0084` the truncation silently yields an **80 µm** sampled
radius while `area_cm2` — and hence the track count — is still computed from
π·(84 µm)². That is a 10 % areal-density error introduced by nothing but a
rounding choice.

Using **`sampled_radius_cm=0.008`, `grid_size_um=10.0`, `buffer_radius=4`,
`no_z_electrode=5`** makes every derived quantity integral:

```
no_xy        = 24        inner radius  =  8 voxels =  80 µm  (tracks + scoring)
                         outer radius  = 12 voxels = 120 µm  (= FEniCSx domain radius, exactly)
no_z         = 200       = 2.0 mm / 10 µm, exactly
voxels       = 24 × 24 × 210 = 120 960     (FEniCSx: 72 165 DOF)
dt           = 3.826e-7 s (von Neumann limited)
total steps  = 1 622  (1 412 injection + 210 clearance) = 621 µs
```

## 3. Measured runtimes and radius convergence

All rows measured on this machine with `solver_numba` (single-threaded), under
the **nominal** dose convention (§4.4), with the full v2 physics of §5 in
place (`buffer_radius=3`, which the reflecting wall makes sufficient):

| tier | `sampled_radius_cm` | grid | tracks/pulse | `solver_numba` | k_s |
|---|---|---|---|---|---|
| `dev` | 0.003 (30 µm) | 12²×210 | 3 157 | 0.2 s | 1.0580 |
| `archive` | **0.008 (80 µm)** | 22²×210 | 22 447 | **2.0 s** | **1.0929** |
| `converged` | 0.014 (140 µm) | 34²×210 | 68 744 | 17.4 s | 1.1011 |
| `production` | 0.018 (180 µm) | 42²×210 | 113 638 | 41.3 s | 1.1035 |

The pure-Python reference (`solver.py`) is ~550× slower: the `archive` tier is
**~0.9 h** there.

Two things follow.

**The archive-scale column is not an hours-long workload on the Numba
baseline.** It is 2.8 s. "Hours" describes `solver.py`, the pure-Python
reference (~0.9 h at this radius). The batching + JIT work already in this
repository has absorbed the entire gap; a ~1 min / ~hours dev-vs-production
split therefore only exists for `solver.py`, where it is dev r ≈ 20 µm
(~100 s) ↔ archive r = 80 µm (~0.9 h).

**The archive's own column is not radius-converged.** Under the as-run density
(§4.4) k_s runs 1.078 → 1.148 → 1.190 across 20 → 80 µm and only flattens by
120–140 µm (1.2044 → 1.2071). Both v1 and v2 inherit this systematic, because
both fix the column at r = 0.012 cm — a constant hardcoded in
`hadrons/python/continuous_beam.py` (`r_cm = 0.012`) and carried into the
FEniCSx configs as `geometry.radius`. It has never been a converged choice; it
is a value inherited from the v1 source.

So the sensible unit of work is r ≈ 180 µm (~1 min, converged with margin), and
the hours-scale target for parallelisation is the **campaign sweep** — 5 dose
rates × 2 pulse modes × 2 chambers × N seeds at that radius — not one larger
radius. Going to r = 400 µm would take ~1 h and buy nothing, since k_s is
already converged at 140 µm. The sweep is exactly the independent-replica
pattern the top-level README already identifies as the right way to spend a
190-core allocation.

---

## 4. Cross-code modelling differences

Three implementations of nominally the same physics:

- **v1 / Cython** — `IonTracks-Cython`, `hadrons/` (`continuous_beam.pyx` and its
  pure-Python twin `hadrons/python/continuous_beam.py`)
- **v2 / FEniCSx** — `IonTracks-FEniCSx`, `iontracks/`
- **this repo** — `pulsed_ion_chamber/`, extracted from v1

`this repo` inherits v1's physics wholesale, so nearly every v1-vs-v2 difference
below is also a this-repo-vs-v2 difference. Differences are ordered by expected
impact on k_s.

### 4.1 Charge-carrier mobility and diffusion: one averaged species vs. two

**The single largest structural difference.**

| | v1 / this repo | v2 / FEniCSx |
|---|---|---|
| µ⁺ | 1.65 cm²/(V·s) | 1.36 cm²/(V·s) (1.36e-4 m²/V·s) |
| µ⁻ | 1.65 cm²/(V·s) | 2.10 cm²/(V·s) |
| D⁺ | 3.7e-2 cm²/s | 2.82e-2 cm²/s |
| D⁻ | 3.7e-2 cm²/s | 4.35e-2 cm²/s |

v1 uses **one averaged mobility and one averaged diffusion coefficient for both
species** (`ion_mobility = 1.65 # averaged for positive and negative ions`);
v2 carries the two Kanai species separately. Consequences at 1500 V/cm across
2 mm:

| | v1 / this repo | v2 / FEniCSx |
|---|---|---|
| positive transit | 80.8 µs | **98.0 µs** |
| negative transit | 80.8 µs | **63.5 µs** |
| separation velocity (µ⁺+µ⁻)·E | 49.5 m/s | 51.9 m/s |

Two competing effects: v2's slow positive cloud lingers ~21 % longer (more
overlap time → more recombination), while its clouds separate ~5 % faster (less
overlap → less). The sign of the net effect is not obvious a priori. Averaging
also symmetrises the two clouds' diffusive spreading, which the two-species
model does not.

This is not a knob on either side — it is baked into v1's constants and into
v2's `carriers_const` block. Reproducing v2 exactly in this repo would require
carrying two species through the solver, i.e. doubling the state and splitting
the drift/diffusion coefficients per array. That is a real change, not a config
edit.

### 4.2 W (mean energy per ion pair) — three different values in two repos

| | value | where |
|---|---|---|
| v1 PDE solvers | **34.2** eV | `hadrons/python/continuous_beam.py`, `continuous_beam.pyx` |
| v1 Jaffe theory / `functions.py` | **33.9** eV | `hadrons/functions.py`, `common_properties.py` |
| v2 / archive config | **33** eV | `track_source.W` |
| this repo | **34.2** eV | `constants.py` (standardised on v1's PDE value so `theory.py` and the solver agree — documented there) |

W scales the linear charge density N₀ = LET/W directly, so 34.2 vs 33 is a
**3.6 % difference in charge per track**, and recombination goes as the square
of density. v1 is internally inconsistent between its own PDE solver and its
own analytic reference; this repo resolved that by picking one, and picked the
one v2 did not.

### 4.3 Air density: 1.225 vs 1.293 kg/m³

v1 and this repo use **1.225** kg/m³ (`doserate_to_fluence`,
`AIR_DENSITY_KG_M3`); v2's archive config uses **1.293** kg/m³ (0 °C,
1013 hPa). Track count goes as ρ/LET, so the same stated dose gives **5.6 %
more tracks** in v2. Neither value is wrong — they are different reference
conditions — but comparing k_s at "the same dose" compares two different
numbers of tracks unless this is reconciled.

### 4.4 Track-count area vs. track-placement area (the 0.7 fill fraction)

v2 computes the track count from the **full** chamber radius
(`compute_number_of_tracks` uses `chamber_radius_mm`) but places every track
inside `chamber_fill_fraction × radius = 0.7 × 0.12 = 0.084 mm`. The areal
track density actually simulated is therefore **1/0.7² = 2.04×** the nominal
one implied by the quoted dose. The archive's own `RESULTS.md` documents this
under "Caveats carried over from the reconstruction".

v1 and this repo are self-consistent: the count and the placement use the same
disc, so the simulated density always equals the nominal fluence.

Measured effect here — same geometry, same everything else, only the density
convention changed:

| convention | tracks in the 80 µm column | k_s |
|---|---|---|
| nominal (0.1782 Gy to air, uniform) | 24 104 | **1.1034** |
| as v2 actually ran it (2.04×) | 49 196 | **1.1903** |

The archive's own number is 1.1629. So the convention choice moves k_s by more
than the entire disagreement between the codes.

### 4.5 Lateral boundary condition: reflecting vs. absorbing

| | v1 / this repo | v2 / FEniCSx |
|---|---|---|
| side wall | outer voxel ring never updated → **absorbing** (clamped to 0) | natural BC, no advective boundary term → **reflecting** (zero flux) |
| electrodes | zero at the ends of a 5-voxel gas buffer *beyond* each electrode | **zero Dirichlet on the electrode plane itself** |

The diffusion length over the 667 µs run is √(2Dt) ≈ 70 µm, comparable to the
80–120 µm domain radius — so I expected the side-wall condition to be a
first-order effect. **It is not: measured, it changes k_s by 1.4e-5.** See
§5.2, which corrects this ranking and shows where the change does pay off
(a smaller buffer, and removing a frozen-charge artifact in v1).

The two codes still disagree on it in the strongest possible way: one absorbs,
the other reflects.

The electrode condition also differs: v2 clamps n = 0 exactly at the collecting
plane, which imposes an artificial steep gradient and an extra diffusive flux
out of the gap; v1 lets charge drift on into a buffer region and simply stops
scoring it.

### 4.6 Scoring domain

| | v1 / this repo | v2 / FEniCSx |
|---|---|---|
| injected charge | Σ of the Gaussian over voxels **inside the track disc only** | ∫ n dx over the **whole mesh** |
| recombination | Σ α n⁺n⁻ dt over voxels **inside the track disc**, gap layers only | ∫ α n⁺_old n⁻_old dt dx over the **whole mesh** |

Both are internally consistent ratios, but they measure different things. v1
measures a core column and discards a track's tails once they cross the scoring
radius — including the tails of tracks that landed near the edge, whose
injected charge is under-counted from the start. v2 scores everything,
including the initially-empty 0.084 → 0.12 mm annulus that fills by diffusion
and dilutes the density there.

**Measured: 0.8 % on k_s** (1.0927 → 1.0837), and it does *not* explain the
radius systematic of §3. Now selectable via `scoring_region`, defaulting to
v1's — see §5.3 for why v2's variant is a reproduction knob rather than an
improvement.

### 4.7 Time discretisation and stabilisation

| | v1 / this repo | v2 / FEniCSx |
|---|---|---|
| scheme | explicit **Lax-Wendroff** (2nd order) | **backward Euler**, CG1 Galerkin, drift + diffusion implicit, recombination lagged |
| dt | **not free** — largest dt satisfying von Neumann, 3.83e-7 s here | free — config-set, **7e-7 s** here |
| Courant number | 0.947 (at the stability limit by construction) | **1.43** (positive) / **2.21** (negative) |
| mesh Péclet `v·h/2D` | 33.4 | **36.2** (both species) |
| stabilisation | none (Lax-Wendroff is CFL-bounded) | **none** — plain Galerkin |

Both codes run at mesh Péclet ≈ 35, i.e. strongly advection-dominated. v1's
scheme is bounded by the CFL condition it solves for; v2 is unconditionally
stable and runs at Courant > 2 on an unstabilised CG1 discretisation, where
accuracy is not what stability guarantees — no SUPG, no upwinding. `determine_dt`
in v2 would suggest 3.17e-7 s from its own drift-CFL rule; the campaign used
7e-7 s, 2.2× larger, and that module's docstring explicitly notes it is
"a drift/CFL condition only", relied upon for accuracy rather than stability.

This is worth a direct check: rerun one archived case at v2's own suggested
dt and see how far k_s moves.

### 4.8 Track geometry and per-track sampling

| | v1 / this repo | v2 / FEniCSx |
|---|---|---|
| track direction | always exactly ‖ z | azimuth and tilt ~ N(0, 0.01 rad); RDD is distance to a **3D line** |
| LET | one deterministic value for all tracks | per-track, from per-track energy ~ N(56.2, 0.1) MeV |
| track radius b | one deterministic value | **also one deterministic value** — see below |
| arrival times | normalised cumulative sum of uniforms (ordered, spans the window exactly, fixed total) | i.i.d. uniform over the window |
| transverse positions | rejection sampling, areal-uniform | inverse-CDF radial sampling, areal-uniform |

The tilt spread is 0.01 rad, so the path-length effect is ~5e-5 — negligible
in magnitude, but it means v2's tracks are genuinely 3D lines while v1's are a
z-independent 2D cross-section, which is what lets v1 hoist the whole z-loop
out of track insertion. The distributional differences (per-track LET,
arrival-time construction) are negligible at N ≈ 5e4.

**The archived config's `beam_radius_distribution` is dead.** It reads
`{type: normal, mu: 20, sigma: 0.1}`, but `sample_energy_let_radius` only
samples it on the branch where no `particle_energy_distribution` is given;
this campaign supplies one, so the branch taken is *"use default `E_MeV_u`, and
`radius` is taken from RDD default radius"*. Confirmed in the run's own
`ion_tracks.csv`: all 54 239 tracks carry `radius = 20.0` exactly, min = max,
while `LET` does vary per track (1.19756e-3 … 1.19894e-3, mean 1.199477e-3).
So b is a single constant in both codes after all, and the `sigma: 0.1` in the
config never took effect.

### 4.8b How v2 actually deposits a track, and how much of it is negligible

`IonTrackSource.inject()` → `_evaluate_batch_gaussian()`
(`iontracks/tracks/ion_track_source.py`). Per time step:

1. Select the tracks whose arrival time falls in `[t, t+dt)`.
2. Take **every DOF coordinate on the rank**, `V.tabulate_dof_coordinates()`,
   as a `(3, n_dofs)` array — cached once.
3. Loop over the step's tracks. For each, compute the squared perpendicular
   distance to the track *line* (`|p-p0|² - ((p-p0)·u)²`, so oblique incidence
   is supported), then `np.exp(-r²/b²) * N0/(π b²)` over the whole DOF array
   and `result += …`.
4. Add `result` straight into the carrier's DOF array, and measure the injected
   charge as `assemble_scalar(form(n * dx))`.

The result is cached per time step and reused for the second carrier, so the
RDD is evaluated once per step, not twice.

**There is no cutoff anywhere in that path.** Every DOF gets an `np.exp` for
every track, however far away. For the archived mesh (r = 0.12 mm, h = 2 mm,
`lc = 0.01 mm` → 72 165 DOF, 366 002 tets) and b = 20 µm, so **b/lc = 2 nodes
per Gaussian radius**:

| within | exp(−k²) | DOF, centred track | DOF, mean over the 54 239 real track positions | share of mesh |
|---|---|---|---|---|
| 1b | 3.68e-1 | 2 005 | 2 005 | 2.8 % |
| 2b | 1.83e-2 | 8 018 | 8 011 | 11.1 % |
| 3b | 1.23e-4 | 18 041 | 17 330 | 24.0 % |
| 4b | 1.13e-7 | 32 073 | 28 466 | 39.4 % |
| 5b | 1.39e-11 | 50 115 | 40 091 | 55.6 % |
| whole mesh | — | 72 165 | 72 165 | 100 % |

So **~75 % of the mesh receives less than 1.2e-4 of a track's peak density, and
~89 % less than 1.8e-2.** The smallest exponent actually evaluated is
exp(−36) = 2.3e-16 (centred track to the wall), and for a track at the edge of
the sampling disc the far wall is 10.2b away: **exp(−104) = 6.6e-46**. Those
contributions are below float64 epsilon relative to the running sum — they
cannot change it, and are computed anyway. Over the run that is
54 239 × 72 165 = **3.9e9 `exp()` evaluations**.

**v1 / this repo does the same thing far more cheaply**, not by truncating but
by factorising. Because its tracks are exactly ‖ z, the 2D cross-section is
z-independent and the 2D Gaussian is separable, so `_insert_track_numba`
computes `2 × no_xy = 44` exponentials per track and reconstructs the rest with
multiplications — **1 640× fewer transcendentals per track** than v2's 72 165,
on the same physical domain at the same resolution. The subsequent broadcast
still touches the whole gap (96 800 voxels), so the *element* counts are
comparable; it is only the `exp()` count that differs, and it differs enormously.

Neither code truncates, and at this domain size that costs relatively little —
the domain is only 6b in radius, so there is at most a ~4× saving available.
The waste becomes catastrophic only when the domain grows: see §7.1, where a
±5b stencil covers 0.139 % of the full-electrode grid.

**Accuracy of the deposit.** b/lc = 2 puts only ~10 nodes inside the 1/e radius
in each transverse plane, which is coarse, and the campaign ran with
`renormalize_to_let: false`, so nothing corrects the interpolated charge back to
the LET. Measured from the run's own output: summing `injected_positive` over
all 953 steps gives 3.9314e6 against an analytic
`N0 × 2 mm × 54 239 = 3.9429e6`, i.e. **0.29 % low**. That deficit is P1
interpolation error plus the Gaussian tails that fall outside the domain — small
enough not to matter, and worth knowing rather than assuming.

### 4.9 Where the 20 µm track radius comes from

Both codes use b = 20 µm here, but for different reasons:

- v1 / this repo **derive** it: a quadratic fit in log₁₀(LET) to `LET_b.dat`
  (Rossomme et al.), then `max(b, 2e-3 cm)`. At LET = 1.2e-3 keV/µm the fit
  falls below the floor, so the floor is what is returned.
- v2 **sets** it from the config (`beam_radius_distribution.mu = 20 µm`).

They agree numerically only because the floor happens to equal the configured
value. At any higher LET — a lower-energy beam, a heavier ion — the two would
silently diverge, with v1 tracking the Rossomme fit and v2 holding whatever the
config says.

### 4.10 What both codes agree on

Worth stating explicitly, since it bounds the search for discrepancies:

- The Gaussian RDD normalisation is **identical**: `(N₀/π b²)·exp(-r²/b²)`
  with `N₀ = LET/W`. v2's `_radial_dose_gaussian` and v1's inner loop compute
  the same expression.
- The recombination coefficient α = 1.6e-6 cm³/s = 1.6e-12 m³/s.
- The recombination sink is **explicit / lagged in both** — v2's linear form
  uses `n_old` for both carriers, v1 uses the previous-step arrays.
- The applied field is uniform and analytic in both: v2 with
  `include_space_screening: false` selects `ConstantField`, not the Poisson
  solver, so no field solve happens in this campaign.
- LET comes from the same PSTAR table in both. (v2's configs carry
  `stopping_power_source: ICRU`, but `let_keV_um()` calls PSTAR regardless —
  the archive README documents this.)
- Tracks are infinite straight lines at constant LET; no Bragg peak in the gap.
- Neither models space charge, magnetic fields, RF micro-structure, or
  pulse-to-pulse accumulation.
- **Initial / columnar recombination is not a missing term in either.** The
  archive README lists it under "Not modelled", but the same α n⁺n⁻ term acting
  on the resolved Gaussian core *is* columnar recombination — that is precisely
  what `tests/test_single_track_vs_jaffe.py` in this repo validates against
  Jaffe theory in the single-track limit. What is actually limited is the
  resolution of the core: b/h = 2 in both codes, which is coarse. The
  distinction is one of resolution, not of physics included.

### 4.11 Collection-time tail

| | v1 / this repo | v2 / FEniCSx |
|---|---|---|
| rule | `n_clearance_separation_times × d/(2 µ E)` = 2 × 40.2 µs | `d / min(v⁺, v⁻) × 1.3` |
| value | **80.4 µs** | **127.45 µs** |

v1's default of 2.0 separation times is exactly one full-gap transit at its
averaged mobility, with **no safety margin** — charge still in flight at the
last step stops being tracked. v2 applies a 1.3× safety coefficient on top of
the slowest carrier's full transit. Truncating early slightly under-counts
recombination; raising `n_clearance_separation_times` to ~2.5 would match v2's
margin. (Confirmed in v2's `determine_simulation_time_one_track`: 2 mm /
20 400 mm/s × 1.3 = 127.45 µs, matching the archived
`simulation_time: 0.000667451` minus the 540 µs window exactly.)

### 4.12 v1-only: the continuous-beam build-up gate

v1's `continuous_beam` runs `n_separation_times = 3` and **discards all scoring
during the first `separation_time_steps`** to let a continuous beam reach steady
state. This repo deliberately removes that gate (it targets a small number of
pulses, not a steady state — documented in the top-level README). v2 has no
such gate either. Only relevant to the archive's `continuous` cases, not to the
macropulse one.

---


## 5. What was ported into this repository

Everything below is implemented in `pulsed_ion_chamber/` on this branch, on the
**regular rectangular grid** — no finite elements, no unstructured mesh. Every
new knob defaults to the original v1 behaviour, so existing configs, tests and
the `theory.py` Jaffe cross-check are bit-for-bit unchanged
(`tests/test_v2_physics.py::test_defaults_are_the_v1_single_averaged_species`).

| § | Change | New config field | Default |
|---|---|---|---|
| 4.1 | two carrier species, resolved separately | `mu_positive_cm2_Vs`, `mu_negative_cm2_Vs`, `D_positive_cm2_s`, `D_negative_cm2_s` | `None` → v1's averaged pair |
| 4.2 | W as an input rather than a constant | `W_eV` | `34.2` (v1's PDE value) |
| 4.3 | air reference density as an input | `air_density_kg_m3` | `1.225` (v1's ISA value) |
| 4.4 | v2's track-count / track-placement split | `chamber_fill_fraction` | `1.0` (self-consistent) |
| 4.5 | zero-flux chamber wall | `lateral_boundary` | `"absorbing"` |
| 4.6 | score the whole grid, not just the track disc | `scoring_region` | `"track_disc"` |
| 4.11 | collection tail from the slowest carrier | *(existing)* `n_clearance_separation_times` | `2.0`; use `2.6` for v2's rule |
| 2 | `no_xy` rounds instead of truncating | — | fixed outright |

### 5.1 Two carrier species (§4.1)

`SimulationConfig.scheme_coefficients()` now returns a *pair* of Lax-Wendroff
stencils, one per species, each built from its own `s = D dt/h²` and
`c = µ E dt/h`. The three solvers take eight scalars instead of four; the
per-voxel cost is unchanged (the same seven multiply-adds per carrier).

Two knock-on effects, both handled in `config.py`:

- **`dt` is now the stricter of the two von Neumann limits.** The negative ion
  binds — it is both faster (µ = 2.10 vs 1.36) and more diffusive (D = 4.35e-2
  vs 2.82e-2) — dropping `dt` from 3.826e-7 s to **3.043e-7 s** (−20 %) on the
  10 µm grid.
- **The clearance period is now sized by the *slowest* carrier**
  (`slowest_mobility_cm2_Vs`), because the positive ion is the one still in
  flight. Combined with `n_clearance_separation_times=2.6` this reproduces v2's
  127.45 µs tail.

Together: **1 622 → 2 194 time steps (+35 %)** for the archive case. See §7.

Measured effect on the answer: **k_s 1.0967 → 1.0927, −0.4 %.** The two
competing effects predicted in §4.1 do largely cancel, and the net sign is
*less* recombination — the faster separation wins over the lingering positive
cloud.

### 5.2 Zero-flux chamber wall (§4.5)

`solver.apply_lateral_boundary()` mirrors the interior into the outer ring
after every step, giving a zero-gradient wall. The z ends are untouched in both
modes: charge that drifts past the electrode buffer has been collected and
should leave.

This also fixes a genuine defect in v1. The Lax-Wendroff sweep runs
`i, j ∈ [1, no_xy-2]`, so the outer ring is **never updated** — but track
insertion *does* write to it. Gaussian tails therefore pile up on the ring and
are then frozen: they never drift, never diffuse, never recombine, and keep
feeding their inward neighbour for the rest of the run. In the archive case the
stranded ring density ends up **5×10⁵ times** the interior density it sits next
to. `tests/test_v2_physics.py::test_reflecting_wall_leaves_no_frozen_charge_on_the_outer_ring`
pins this down.

**Correction to §4.5's impact estimate.** I ranked this third by expected effect
on k_s. Measured, it is nearly irrelevant *to the answer* — 1.092708 vs
1.092693 at `buffer_radius=4`, a relative change of 1.4e-5 — because
recombination is dominated by fresh in-disc track density during the pulse, and
the wall sits 4 voxels beyond the track disc.

Where it does pay off is **cost**, by letting the buffer shrink:

| `buffer_radius` | grid | absorbing k_s | reflecting k_s | wall time |
|---|---|---|---|---|
| 8 (converged) | 32² | 1.092912 | 1.092912 | 5.8 s |
| 6 | 28² | 1.092825 | 1.092825 | 4.5 s |
| 4 | 24² | 1.092708 | 1.092693 | 2.6 s |
| 3 | 22² | 1.093308 | **1.092921** | 2.0 s |
| 2 | 20² | 1.099569 | 1.093302 | 1.5 s |
| 1 | 18² | **1.191254** | 1.095392 | 1.2 s |

The absorbing wall needs `buffer_radius` 4–6 to stay within 1e-4 of the
converged value and blows up by 9 % at 1, precisely because the frozen ring
ends up adjacent to the scored disc. The reflecting wall is within 1e-5 at 3.
Dropping 4 → 3 shrinks the grid 24² → 22², **−21 % wall time** at equal
accuracy.

### 5.3 Scoring domain (§4.6)

`scoring_region="full_grid"` widens the injected-charge and recombination
tallies from the track disc to every voxel in the gap, matching v2's
`∫ n dx` / `∫ α n⁺n⁻ dt dx` over its whole mesh. Implemented without touching
the kernels: `config.scoring_radius_sq` is simply set large enough to admit
every voxel, corners included.

**Ship it as a reproduction knob, not as an improvement — and this corrects
§4.6.** Measured, `full_grid` moves k_s from 1.0927 to 1.0837 (−0.8 %), and the
shift is buffer-converged (1.08532 / 1.08369 / 1.08373 / 1.08375 / 1.08400 at
`buffer_radius` 2 / 4 / 6 / 8 / 12), so it is a real, bounded edge-treatment
difference rather than divergence. But the annulus it adds contains Gaussian
*tails only* — no track centres — so it is systematically under-irradiated
relative to the uniformly irradiated chamber being modelled, and including it
biases k_s low. `track_disc` remains the better estimator and stays the
default. This is a plausible contributor to v2's k_s sitting below this repo's.

### 5.4 W, air density, fill fraction, rounding

`W_eV` and `air_density_kg_m3` are plain inputs now; `dose_rate_to_fluence_rate`
takes the density as an argument. Per your instruction the campaign config uses
**dry air at 20 °C, 101.325 kPa = 1.2041 kg/m³** (`AIR_DENSITY_20C_KG_M3`),
not v2's 1.293 (0 °C) and not v1's 1.225 (ISA, 15 °C). Relative to v1 that is
1.7 % fewer tracks for the same stated dose.

`chamber_fill_fraction` reproduces v2's split between the disc tracks are
*counted* over and the disc they are *placed* in. Left at 1.0 for the campaign,
per the nominal-dose decision.

`no_xy` now rounds. `sampled_radius_cm=0.0084` at 10 µm/voxel gave an 80 µm disc
while the track count used 84 µm; it now gives 84 µm, as asked for.

## 6. What was deliberately not ported

- **v2's time integration (§4.7).** Backward Euler would let `dt` float free of
  the von Neumann limit, which is the single biggest lever on run time — but
  v2 runs it at Courant 2.2 and mesh Péclet 36 on *unstabilised* CG1, where
  stability is not accuracy. Porting the freedom without porting SUPG or
  upwinding would buy speed by degrading the answer. If this is wanted later,
  the honest version is implicit + a stabilisation term, and it should be
  validated against the explicit scheme before being trusted.
- **Per-track LET and track radius, tilted tracks (§4.8).** The tilt spread is
  0.01 rad (a 5e-5 path-length effect) and the energy spread is 0.1 MeV out of
  56.2. Both are negligible at N ≈ 2×10⁴, and 3D tilted tracks would destroy
  the z-independence that lets track insertion hoist its entire z-loop — the
  optimisation the whole repository is built around.
- **v2's zero-Dirichlet electrode plane (§4.5).** v1's gas buffer beyond each
  electrode is the numerically gentler version of the same absorbing condition,
  and unlike v2's it does not impose an artificial density gradient right where
  the charge is being collected.
- **A configurable RDD.** v2 supports uniform and libamtrack models; the
  campaign used Gaussian, which both codes already normalise identically
  (§4.10).

### The one change worth making that is *not* in v2

The radius systematic (§3) survives everything above — measured, k_s still runs
1.028 → 1.093 across 20 → 120 µm with `full_grid` + `reflecting`, essentially
parallel to the v1 curve. So it is not caused by the wall or by the scoring
domain. It is intrinsic to drawing tracks in a finite disc: a track near the
edge has neighbours on one side only, so the local charge density — and hence
`α n⁺n⁻` — is genuinely lower there, and the deficit only vanishes as the edge
fraction does.

The fix is a **periodic** lateral boundary with tracks filling the whole
periodic cell, which makes the sampled volume a true unit cell of an infinite
uniformly irradiated slab with no edge at all. Neither v1 nor v2 does this, so
it is out of scope here — but it is the change that would let a much smaller
column give the converged answer, and on an r⁴ cost curve that is worth far
more than any of the ports above. Recommended as the next step.

## 7. Execution-time report

Cost model, `archive` tier, `solver_numba`: **72 % track insertion, 28 % PDE
steps** (22 447 tracks vs 2 194 steps). Insertion is
`O(n_tracks · no_xy² · no_z)` and the PDE sweep is
`O(n_steps · no_xy² · no_z)`, so the split is just `n_tracks / n_steps` — and
since `n_tracks ∝ r²` while `n_steps` is fixed, insertion dominates further as
the column grows and as the dose rate rises.

Ranked by execution-time impact:

| Change | Mechanism | Effect on wall time |
|---|---|---|
| **Two carrier species** (§5.1) | negative ion cuts `dt` 20 %; positive ion lengthens the tail → **+35 % time steps** | **+10 %** at macropulse dose (track-dominated); **+33 %** at 1/37 of the dose (PDE-dominated). Measured: 1.90 → 2.09 s and 0.48 → 0.64 s |
| **Reflecting wall** (§5.2) | +4.5 % per-step overhead for the mirror, but converges at `buffer_radius` 3 instead of 4–6 | **−21 % net** (2.64 s → 2.09 s at equal accuracy) |
| **`no_xy` rounding** (§5.4) | can move `no_xy` by ±1 | ±8 % on a 24² grid, config-dependent, one-off |
| **Air density 1.225 → 1.2041** (§5.4) | 1.7 % fewer tracks | **−1.7 %** |
| **W, `chamber_fill_fraction`, `scoring_region`** | none change the grid, the step count or the track count | **0 %** |

**The two-species change is the one to watch.** It is the only port that costs
time, it costs it as a fixed +35 % on the step count, and that surcharge lands
hardest exactly where there is least other work to hide it — the low-dose-rate
`continuous` cases of the campaign, where it is the full +33 %. It is also
unavoidable if the two species are to be resolved at all: `dt` is set by the
von Neumann criterion, not chosen, and the negative ion's mobility and
diffusion coefficient are both larger than the averaged values it replaces.

Net for the archive case, all ports enabled: **2.63 s → 2.02 s, 23 % faster**
than the v1 baseline, because the buffer saving more than covers the extra time
steps. The ported physics is not a performance regression.

### 7.1 Cost of simulating the full electrode

The Classic Markus PTW 23343 collecting electrode is **5.3 mm** in diameter
(r = 2.65 mm), giving 0.2206 cm² of area and 0.0441 cm³ of gas across the 2 mm
gap. **That diameter appears nowhere in any of the four repositories** — not in
`SimulationConfig`, not in the FEniCSx `geometry` block, not in v1's
`continuous_beam`. Only the gap and the bias are modelled; the archive states
this outright ("the real electrode diameter, the guard ring and edge effects are
**not** modelled"), and its `PHYSICS.md` mentions the electrode only as prose,
rounded to "the full 5 mm-diameter electrode". So the figure below is what the
full electrode *would* cost, not a configuration anyone has run.

At `grid_size_um=10.0` and `buffer_radius=3`:

```
grid          536 x 536 x 210 = 60.3 M voxels   (under the 1e8 max_voxels guard)
tracks/pulse  24 630 400                        (487x the archived 0.12 mm column)
time steps    2 194
memory        1.80 GiB  (four float64 arrays)
```

Track insertion is **98 %** of the work at this size, so the estimate is
essentially `n_tracks × no_xy²` divided by a measured rate. Anchored on directly
measured runs at r = 300/500/800 µm (11.5 s / 56.7 s / 178.7 s, batched backend,
one thread), where the insertion rate rises from 2.6e8 to 4.1e8 ops/s as the
grid grows and then flattens:

| backend | full-electrode estimate |
|---|---|
| `solver_numba` (unbatched) | **~16 days** |
| `solver_numba_parallel`, 1 thread (batched) | **~5 h** (3.4–7.7 h over the rate range) |
| batched **+ a local Gaussian stencil** (not implemented) | **~7 min** |

Note the measured scaling is *not* yet the asymptotic r⁴ over this range —
178.7/56.7 = 3.15 for r×1.6 where r⁴ predicts 6.55 — because the runs are still
partly broadcast-bound and the insertion rate itself improves with grid size.
The table uses the op-count model with the best measured rate rather than a
naive power-law fit, which is why it is quoted as a range.

**Two caveats before anyone runs this.**

First, it buys nothing physically *as the model stands*. The field is uniform,
there is no guard ring, and the track density is homogeneous, so the full
electrode is 350 identical copies of a column that already converged at 140 µm
(§3). It would return the same k_s with better statistics. Simulating the real
electrode would only be meaningful alongside the edge physics that motivates it
— field distortion at the guard ring — which is not in either code.

Second, the ~700× between the batched and local-stencil rows is the honest
headline. A track's Gaussian has b = 20 µm = 2 voxels, so a ±5b stencil covers
**0.139 %** of a 536² grid; the other 99.86 % of every track's insertion loop
adds `exp(-25)` and smaller. Both current backends evaluate the full grid per
track, which costs nothing at r = 80 µm (22² grid) and dominates completely at
r = 2.65 mm. If the full electrode is ever wanted, the local stencil — not more
cores — is the change that makes it a coffee break instead of a day.

## 8. Net effect on the comparison

With all of §5 enabled at the archive geometry this repository gives
**k_s = 1.0929** under the nominal dose convention. The archive reports
**1.1629**, but at 2.04× this areal track density (§4.4) — the two are not
comparable until that convention is settled, which remains the first thing to
resolve.

Contributions measured here, largest first:

1. **§4.4 dose convention** — ±8 % on k_s. Still the dominant term, and a
   documented inconsistency in v2 rather than a modelling choice.
2. **§3 finite column radius** — 6 % across 20 → 120 µm, unresolved by any port,
   fixable only by the periodic cell described in §6.
3. **§4.6 scoring domain** — 0.8 %, now selectable.
4. **§4.1 mobility model** — 0.4 %, now ported.
5. **§4.2 / §4.3 W and ρ_air** — now inputs; 3.6 % and 1.7 % on charge density
   and track count respectively.
6. **§4.5 lateral boundary** — 1.4e-5 on the answer, but worth 21 % of the run
   time.

Still open and untested: **§4.7**, v2's Courant-2.2 unstabilised time
integration. That one cannot be settled from this side — it needs one archived
case rerun in v2 at its own suggested `dt` of 3.17e-7 s.
