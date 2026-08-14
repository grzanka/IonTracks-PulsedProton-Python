# IFJ AIC-144 — `Markus 2 mm, macropulse, 10 Gy/s`

Findings from mapping one archived IonTracks v2 (FEniCSx) run onto this
repository, plus a full cross-code comparison of the physics and numerics
of the three IonTracks implementations involved.

**Nothing here is implemented yet** — this document is the analysis and the
plan. No code under `pulsed_ion_chamber/` has been changed on this branch.

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
the **nominal** dose convention (§4.1), `W` and `ρ_air` folded in to match the
archive:

| `sampled_radius_cm` | grid | tracks/pulse | `solver_numba` | `solver.py` (est.) | k_s |
|---|---|---|---|---|---|
| 0.003 (30 µm) | 14²×210 | ~3.4 k | ~0.2 s | ~220 s | — |
| 0.004 (40 µm) | 16²×210 | 12 299 | 0.70 s | ~430 s | — |
| **0.008 (80 µm)** | 24²×210 | 24 104 | **2.8 s** | **~0.9 h** | **1.1034** |
| 0.014 (140 µm) | 36²×210 | 73 820 | 20 s | — | **1.1129** |
| 0.018 (180 µm) | 44²×210 | ~122 k | ~60 s (est.) | — | — |

Two things follow.

**The archive-scale column is not an hours-long workload on the Numba
baseline.** It is 2.8 s. "Hours" describes `solver.py`, the pure-Python
reference (~0.9 h at this radius). The batching + JIT work already in this
repository has absorbed the entire gap; a ~1 min / ~hours dev-vs-production
split therefore only exists for `solver.py`, where it is dev r ≈ 20 µm
(~100 s) ↔ archive r = 80 µm (~0.9 h).

**The archive's own column is not radius-converged.** Under the as-run density
(§4.1) k_s runs 1.078 → 1.148 → 1.190 across 20 → 80 µm and only flattens by
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
80–120 µm domain radius — so the side-wall condition is a first-order effect
here, not a detail. The two codes disagree on it in the strongest possible way:
one absorbs, the other reflects.

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
| track radius b | one deterministic value | per-track, ~ N(20, 0.1) µm |
| arrival times | normalised cumulative sum of uniforms (ordered, spans the window exactly, fixed total) | i.i.d. uniform over the window |
| transverse positions | rejection sampling, areal-uniform | inverse-CDF radial sampling, areal-uniform |

The tilt spread is 0.01 rad, so the path-length effect is ~5e-5 — negligible
in magnitude, but it means v2's tracks are genuinely 3D lines while v1's are a
z-independent 2D cross-section, which is what lets v1 hoist the whole z-loop
out of track insertion. The distributional differences (per-track LET/b,
arrival-time construction) are negligible at N ≈ 5e4.

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

## 5. Net effect on the comparison

At the archive's own geometry, this repository gives **k_s = 1.1903** under the
as-run density convention, against the archive's **1.1629** — 2.4 % apart. That
residual is small relative to several of the individual differences above
(§4.1 mobility model, §4.5 boundary conditions, §4.7 Courant > 2), which
suggests partial cancellation rather than genuine agreement. Under the nominal
density convention the two are not comparable at all (1.1034 vs 1.1629, §4.4).

Ranked by what is worth resolving first:

1. **§4.4 dose convention** — moves k_s by 8 %, and is a documented
   inconsistency in v2 rather than a modelling choice. Resolve by agreeing which
   density is being compared before anything else.
2. **§4.7 Courant number** — cheap to test: rerun one archived case at v2's own
   suggested dt.
3. **§4.5 lateral boundary condition** — absorbing vs. reflecting, at a radius
   comparable to the diffusion length. Testable in this repo by varying
   `buffer_radius`.
4. **§4.1 mobility model** — the largest structural difference, and the most
   expensive to change.
5. **§4.2 / §4.3 W and ρ_air** — 3.6 % and 5.6 % on charge density and track
   count respectively; trivially reconcilable once someone decides which values
   the comparison should use.

## 6. Proposed implementation

Confirmed with the user: **nominal** dose convention (§4.4), and physics
constants promoted to explicit config knobs.

### `pulsed_ion_chamber/config.py`
New optional fields, defaults preserving current behaviour exactly:

- `air_density_kg_m3: float = AIR_DENSITY_KG_M3` (1.225) → 1.293 for this campaign
- `W_eV: float = W_EV_PER_ION_PAIR` (34.2) → 33 for this campaign; feeds `N0 = LET_eV_cm / W_eV`
- `chamber_fill_fraction: float = 1.0` — mirrors v2's `track_source.chamber_fill_fraction`:
  track count from the full scored column, positions drawn inside `fill × radius`.
  Not used by the primary config; needed to *reproduce* §4.4 as a documented
  sensitivity check rather than a magic dose number.
- `no_xy`: round with tolerance instead of truncating, so `sampled_radius_cm` is honoured
- derive `sampling_radius` / `sampling_radius_sq` (track placement) separately from
  `inner_radius_sq` (scoring)

### Other modules
- `stopping_power.py`: `dose_rate_to_fluence_rate(..., air_density_kg_m3=None)`
- `pulses.py`, `solver.py`, `solver_numba.py`, `solver_numba_parallel.py`: pass
  `config.sampling_radius` to `sample_xy_inside_cylinder`; scoring keeps
  `inner_radius_sq`. Four one-line call-site changes, no kernel signature changes.

### `examples/ifj_aic144/run_markus_2mm.py`

```python
MARKUS_2MM_MACROPULSE_10GYS = dict(
    E_MeV_u=56.2, voltage_V=300.0, electrode_gap_cm=0.2,   # 150 kV/m, as archived
    pulse_duration_s=540e-6, repetition_rate_hz=50.0,      # 540 us / 20 ms, duty 1/37
    n_pulses=1, rf_frequency_hz=26.26e6,
    dose_rate_Gy_s=8.91,          # 10 Gy/s to water x 0.891 water->air => 0.1782 Gy/pulse
    air_density_kg_m3=1.293, W_eV=33.0,                    # archive values
    seed=20260527,                                         # archive seed
)
```

with the four grid tiers of §3 (`DEV`, `ARCHIVE_GEOMETRY`, `CONVERGED`,
`PRODUCTION`). LET and the 20 µm track radius need no override — see §4.9 for
why that is luck rather than agreement.

### `tests/test_ifj_aic144_mapping.py`
Assert LET, track radius, `no_xy`, `no_z`, dt and tracks/pulse for
`ARCHIVE_GEOMETRY`, so the mapping cannot drift silently.
