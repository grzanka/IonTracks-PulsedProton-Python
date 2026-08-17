# Can we run a Cucinotta radial dose distribution instead of a Gaussian?

A feasibility study for the IonTracks pulsed-proton solver. The question is
whether a track's *radial dose distribution* (RDD) can be represented on the
solver's Cartesian grid, what it would cost in memory and wall clock, and what
physics it would buy.

**The answer, in one paragraph.** In **air** — the medium that actually matters,
since the sensitive volume is an air cavity — the Cucinotta RDD is a nearly
perfect `1/r²` over **5.7 decades**, from 100 nm out to 5 cm, with a steeper
core only below ~30 nm and a hard delta-ray cutoff at **180 mm**. That shape has
two awkward properties and one convenient one. Awkward: 24 % of the track's
energy sits inside 30 nm, which no gap-resolving grid will ever reach; and
**18.5 % of it is deposited beyond the electrode radius**, i.e. outside the
chamber entirely, so a local-deposition RDD does not conserve the energy the
solver currently assumes. Convenient: the mean distance between tracks at
10 Gy/s is **0.946 µm**, at which radius only 42 % of a track's energy has been
deposited — so the majority of every track's dose is laid down *past its
nearest neighbours*, and the outer profile is a self-averaging background rather
than a structure. The construction that follows exploits exactly that: an
analytic sub-voxel core, a voxel-averaged `1/r²` band on the grid, and an
exactly-integrable uniform background beyond the column. The grid then never
needs to resolve anything finer than ~1 µm — a size the current hardware
reaches comfortably.

Scenario throughout: AIC-144 Markus 2 mm chamber, 56.2 MeV/u protons, 300 V
across a 0.2 cm gap (E = 1500 V/cm), 540 µs macropulse, 10 Gy/s to water.
Two libamtrack RDD tabulations for that beam are used, one in liquid water and
one in air, both 100 points per decade.

---

## 1. The input data

### 1.1 The air distribution

`D(r)` in Gy against `r` in metres. Because the energy in an annulus is
`dE/dr = 2πr·D(r)`, the diagnostic that matters is **`r²·D(r)`**: it is flat
wherever `D ∝ 1/r²`, and in that region *each decade of radius carries equal
energy*.

| r | D(r) [Gy] | r²D [Gy·m²] |
|---|---|---|
| 10 nm | 134.60 | **1.346e-14** |
| 15 nm | 26.08 | 5.868e-15 |
| 20 nm | 7.890 | 3.143e-15 |
| 30 nm | 1.712 | 1.534e-15 |
| 50 nm | 0.5077 | 1.301e-15 |
| 100 nm | 0.1303 | 1.306e-15 |
| 1 µm | 1.309e-3 | 1.316e-15 |
| 10 µm | 1.307e-5 | 1.316e-15 |
| 100 µm | 1.304e-7 | **1.317e-15** |
| 1 mm | 1.301e-9 | 1.316e-15 |
| 1 cm | 1.269e-11 | 1.287e-15 |
| 5 cm | 5.143e-13 | 1.309e-15 |
| 10 cm | 1.340e-14 | 1.362e-16 |
| 180 mm | 2.896e-17 | 9.40e-19 |
| 182 mm | **0** | 0 |

Three features, and each one drives a design decision later:

1. **`r²D` is constant to four digits from 100 nm to 5 cm.** That is a pure
   `1/r²` law across **5.7 decades** — the delta-ray halo. Equal energy per
   decade, which is why the energy ends up spread so widely.
2. **Below ~30 nm the core turns on**, rising to 10.3× the plateau at 10 nm
   (`D ∝ r⁻⁴`-ish). This is the initial ionisation column, and it is the part
   no grid can hold.
3. **A hard cutoff at 180 mm** — the maximum delta-ray range in air. The
   tabulation is identically zero from 181.8 mm out to 1 m.

### 1.2 The tabulation reproduces the solver's LET to 0.6 %

```
2π ∫ r D(r) dr = 1.604e-13 Gy·m²
```

× ρ_air = 1.2041 kg/m³, then × 6.2415e9 (1 J/m → keV/µm):

| | value |
|---|---|
| LET implied by the air RDD | **1.206e-3 keV/µm** |
| `SimulationConfig.LET_keV_um` (air) used by the solver | 1.199e-3 keV/µm |
| ratio | **1.006** |

**Agreement to 0.6 %.** An independent libamtrack RDD and the solver's own
stopping-power lookup describe the same track to better than one percent, which
is a real cross-check: it means the RDD can be substituted for the Gaussian
without renormalising anything, and that any change in `k_s` afterwards is
attributable to *shape*, not to a bookkeeping error in the total.

### 1.3 The water distribution, and why density scaling only half works

The same beam in liquid water, for comparison:

| r | 10 nm | 100 nm | 1 µm | 10 µm | 100 µm | 199 µm |
|---|---|---|---|---|---|---|
| D(r) [Gy] | 435.2 | 0.1441 | 1.438e-3 | 1.398e-5 | 1.978e-8 | 0 |
| r²D | 4.35e-14 | 1.44e-15 | 1.44e-15 | 1.40e-15 | 2.05e-16 | — |

Integrating gives **1.114 keV/µm in water**, again consistent with the same beam
(and with the air figure through the density ratio). But the *shapes* differ in
a way that a naive scaling does not predict.

Delta-ray transport is a range in **mass** units (g/cm²), so the halo should
scale as `r_air = r_water × ρ_water/ρ_air = 830.5`. Testing that against the two
tabulations:

| feature | water | predicted air (× 830.5) | **measured air** | verdict |
|---|---|---|---|---|
| delta-ray cutoff | 199 µm | 165 mm | **182 mm** | ✅ good to 10 % |
| onset of the steeper core | 100 nm | 83 µm | **~30 nm** | ❌ **off by 2 800×** |
| extent of the `1/r²` region | 2 decades | 2 decades | **5.7 decades** | ❌ |

**The halo density-scales; the core does not.** This is the single most
important physical point in the study and it is easy to get wrong. The halo is
set by how far energetic delta rays travel, which is a mass range, so it
stretches by 830× in air. The **core radius is set by the initial
ionisation/excitation physics** — impact parameters, the adiabatic limit, the
medium's mean excitation energy — none of which is a density. It therefore stays
at tens of nanometres in *both* media. In water the core (100 nm) and the halo
(199 µm) are only 3.3 decades apart and the core dominates the energy budget;
in air the core is unchanged while the halo runs out to 180 mm, so the same core
is a much smaller share and the `1/r²` region opens up from 2 decades to 5.7.

Practical consequence: **never derive an air RDD by rescaling a water one.**
Ask libamtrack for the air tabulation, as was done here. The 2 800× error would
have put the "unresolvable core" at 83 µm — larger than the whole resolvable
band — and inverted every conclusion below.

---

## 2. Where the energy actually is (in air)

Cumulative energy fraction inside radius `r`, trapezoidal integration of `r²D`
in `ln r`, with every scale of the problem marked:

| r | what it is | energy inside |
|---|---|---|
| 30 nm | end of the core | **24.1 %** |
| 100 nm | start of the pure `1/r²` | 30.5 % |
| **0.946 µm** | **mean inter-track spacing** | **42.0 %** |
| 1 µm | a fine voxel | 42.3 % |
| 10 µm | | 54.2 % |
| **20 µm** | **the Gaussian `b` the code uses now** | **57.8 %** |
| 100 µm | | 66.1 % |
| 163 µm | radius of an R = 0.016 cm column | 68.6 % |
| 1 mm | half the gap | 77.9 % |
| **2 mm** | **the electrode gap** | **81.5 %** |
| **2.65 mm** | **the full-electrode radius** | **82.9 %** |
| 1 cm | | 89.7 % |
| 180 mm | delta-ray cutoff | 100 % |

Four readings, in order of how much they matter:

**(a) 17 % of every track's energy is deposited outside the chamber.** At the
full electrode radius of 2.65 mm only 82.9 % has been laid down; the rest is
carried by delta rays travelling up to 180 mm — 68× the electrode radius. The
solver currently deposits **100 %** of the unrestricted LET locally. So a
faithful RDD implementation *must* decide what to do with that 17 %: discard it
(physically right for a cavity, but then the deposited energy no longer equals
the LET the config was built from) or force it back in (conserves the config's
bookkeeping, but is wrong). This is exactly the territory cavity theory and the
air `W` value occupy, and it is a bigger effect than anything about grid
resolution. **It should be settled on paper before any code is written.**

**(b) The core is unresolvable and is a quarter of the track.** 24.1 % inside
30 nm. On a 2 mm gap, resolving 30 nm means 66 000 z-layers; §4 shows the grid
that implies. It has to be handled analytically, not on the grid — §5, Zone C.

**(c) The current Gaussian over-concentrates the charge.** `b = 20 µm` is a
reasonable *central* width — the RDD reaches 57.8 % there — but the solver's
Gaussian is normalised to deposit the whole LET inside roughly that radius,
where the true distribution puts only 58 % and spreads the other 42 % from
20 µm to 180 mm. Since recombination goes as `α·n₊·n₋`, over-concentration
over-estimates recombination. Combined with (a), both errors push the same way,
so this study makes a **signed prediction: switching to Cucinotta should lower
`k_s`.** A change in the other direction would mean the implementation is wrong.

**(d) The energy is spread over decades, not localised.** Only 42 % is inside a
1 µm voxel and 66 % inside 100 µm. This is the `1/r²` law doing what it does —
equal energy per decade — and it is why the "uniform background" treatment of
§5 Zone F carries such a large share.

---

## 3. The decisive number: tracks overlap, massively

We are never simulating one track.

| quantity | value | how |
|---|---|---|
| tracks per pulse (R = 0.265 cm) | 2.463e7 | solver, 10 Gy/s to water |
| areal track density | 1.116e8 cm⁻² | N / πR² |
| **mean inter-track spacing** | **0.946 µm** | 1/√density |
| energy inside that radius | **42.0 %** | §2 |
| Gaussian `b` in use | 20.0 µm | `calc_track_radius_cm(LET)` |
| lateral diffusion length over one run | 76.2 µm | √(2·D·T), D = 4.35e-2 cm²/s, T = 668 µs |
| delta-ray cutoff | 180 mm | tabulation |

**Every point in the gas receives dose from an enormous number of tracks.** The
count scales as `(R_profile/d)²`:

| profile extent | overlapping tracks at that radius |
|---|---|
| core, 30 nm | ~0.001 |
| inter-track spacing, 0.946 µm | 1 (by definition) |
| current Gaussian, 20 µm | ~450 |
| electrode radius, 2.65 mm | ~7.8e6 |
| full halo, 180 mm | ~3.6e10 |

Three consequences:

**(a) Beyond ~1 µm the RDD carries no resolvable structure.** With ≳450
contributions by 20 µm, the Poisson fluctuation of the summed dose is
≲1/√450 ≈ 5 %, and by the electrode radius it is 0.04 %. The outer profile is a
*mean level*, not a pattern. That is the licence to replace it with a uniform
background (§5 Zone F) and it is exact in the limit, not an approximation of
convenience.

**(b) `k_s` is therefore insensitive to the outer shape at this dose rate.**
Recombination is quadratic in density and so does care about concentration —
but once hundreds of profiles superpose, the local density is nearly smooth
regardless of the individual shape. **Prediction: at 10 Gy/s the Gaussian →
Cucinotta change moves `k_s` by well under a percent from shape alone**, with
the larger effect coming from the 17 % escaping energy of §2a, which is a
normalisation question rather than a shape one.

**(c) The shape matters where overlap does not happen.** Single-track
validation (Jaffé), low fluence, and high-LET ions — where the core is dense
enough that prompt in-core recombination is significant — are the cases that
justify the machinery. The pulsed 10 Gy/s proton case is the *least* promising
place to look for a difference, which is worth knowing before spending a
campaign on it.

---

## 4. What grids are available

Carrier arrays are `no_xy × no_xy × no_z_with_buffer`, four float64 grids =
**32 bytes per voxel**, with
`no_z_with_buffer = gap/h + 2·no_z_electrode` and
`no_xy = 2R/h + 2·buffer_radius`. Refining `h` also shrinks the stable time
step, so the step count grows too.

### 4.1 Hardware ceilings

| platform | usable budget | finest h at R=0.265 cm | at R=0.05 cm | at R=0.016 cm |
|---|---|---|---|---|
| Laptop, Core Ultra 5 225U (30.8 GiB) | 24.6 GiB | 4.0 µm | 1.32 µm | 0.62 µm |
| Ares node, 2× Xeon 8268 (192 GB) | 152 GiB | 2.2 µm | 0.73 µm | 0.34 µm |
| Helios CPU node, 2× EPYC 9654 (377 GiB) | 320 GiB | 1.74 µm | 0.57 µm | 0.27 µm |
| Athena, A100-40GB (HBM only) | 36 GiB | 3.6 µm | 1.18 µm | 0.55 µm |
| **Helios GH200 (HBM only)** | **85.5 GiB** | **2.7 µm** | **0.89 µm** | **0.42 µm** |
| Helios GH200 + node LPDDR5X (unified) | 468 GiB | 1.53 µm | 0.50 µm | 0.24 µm |

Two remarks a physicist should hear from the computing side:

- On **capacity** a Helios CPU node beats the GH200's HBM by 3.7×. The GPU's
  advantage is **bandwidth** (~4 TB/s vs ~0.9 TB/s aggregate), not size.
- The last row is not a working figure. Reaching host memory across NVLink-C2C
  was measured at **32 s/step** on a 212 GiB grid — 84× slower than HBM —
  because the solver streams every array every step, so page migration has no
  working set to exploit. **Treat ~85 GiB in HBM as the real ceiling**, i.e.
  0.42 µm on a 160 µm column.

### 4.2 Grid dimensions, explicitly

| R [cm] | h [µm] | no_xy | no_z | voxels | arrays | steps | tracks/pulse |
|---|---|---|---|---|---|---|---|
| 0.265 | 10 | 536 | 210 | 6.0e7 | 1.8 GiB | 2 194 | 2.46e7 |
| 0.265 | 5 | 1 066 | 410 | 4.7e8 | 13.9 GiB | 4 580 | 2.46e7 |
| 0.265 | 2 | 2 656 | 1 010 | 7.1e9 | 212 GiB | 13 019 | 2.46e7 |
| 0.265 | 1 | 5 306 | 2 010 | 5.7e10 | 1 687 GiB | 31 564 | 2.46e7 |
| 0.265 | 0.2 | 26 506 | 10 010 | 7.0e12 | 205 TiB | 463 394 | 2.46e7 |
| 0.05 | 2 | 506 | 1 010 | 2.6e8 | 7.7 GiB | 13 019 | 8.77e5 |
| **0.05** | **1** | **1 006** | **2 010** | **2.0e9** | **60.6 GiB** | 31 564 | 8.77e5 |
| 0.05 | 0.5 | 2 006 | 4 010 | 1.6e10 | 481 GiB | 89 725 | 8.77e5 |
| 0.025 | 1 | 506 | 2 010 | 5.2e8 | 15.3 GiB | 31 564 | 2.19e5 |
| 0.025 | 0.5 | 1 006 | 4 010 | 4.1e9 | 121 GiB | 89 725 | 2.19e5 |
| **0.016** | **1** | **326** | **2 010** | **2.1e8** | **6.4 GiB** | 31 564 | 8.98e4 |
| **0.016** | **0.5** | **646** | **4 010** | **1.7e9** | **49.9 GiB** | 89 725 | 8.98e4 |
| 0.016 | 0.2 | 1 606 | 10 010 | 2.6e10 | 769 GiB | 463 394 | 8.98e4 |
| 0.008 | 0.5 | 326 | 4 010 | 4.3e8 | 12.7 GiB | 89 725 | 2.24e4 |
| 0.008 | 0.2 | 806 | 10 010 | 6.5e9 | 194 GiB | 463 394 | 2.24e4 |

Memory goes as `R²/h³`, and the table shows what that means: at h = 0.5 µm,
R = 0.265 cm needs **13.4 TiB** while R = 0.016 cm needs **49.9 GiB** — a factor
of 275. **Refining the voxel is cheap; widening the column is not.**

And resolving the 30 nm core is not merely expensive, it is absurd: at
h = 0.03 µm on even the smallest useful column the grid would be
~5 400 × 5 400 × 66 700 = 1.9e12 voxels = **57 TiB**, with ~3e6 time steps.
This is the quantitative form of "the core must be analytic".

### 4.3 How small can the column be?

The column radius must exceed the **lateral correlation length** of the charge
distribution. Once the halo is a uniform background (§3a) that length is set by
diffusion, not by the 180 mm RDD extent:

```
sqrt(2 D T) = sqrt(2 × 4.35e-2 cm²/s × 6.68e-4 s) = 76.2 µm
```

With `lateral_boundary="reflecting"` — the right condition for a column sampled
from the interior of a large, uniformly irradiated chamber — and the measured
`buffer_radius=3` convergence, **R ≈ 150–250 µm (0.015–0.025 cm) is
defensible**, roughly 10× smaller than the full electrode.

Statistics are not the constraint: at R = 0.016 cm there are still **89 790
tracks per pulse**, arriving over 1 775 of the 2 194 steps, so each step lays
~50 tracks into a 326 × 326 column. The convergence test that matters is `k_s`
versus `R` at fixed `h`, not the track count.

**Headline of this section:** **R = 0.016 cm with h = 0.5 µm —
646 × 646 × 4 010, 1.7 G voxels, 49.9 GiB — fits in GH200 HBM**, and the same
column at h = 1 µm is 326 × 326 × 2 010, 6.4 GiB, and runs on a laptop.

---

## 5. The proposed construction: three zones

Pick a split radius `r_s` of order the column radius (~150 µm), and a voxel
size `h` (1 µm or 0.5 µm). The RDD is then handled in three zones, each by the
method appropriate to its physics.

### Zone C — core, `r < h`: analytic, off-grid

**42.3 % of the energy at h = 1 µm** (24.1 % of it inside 30 nm). Two framings:

1. **Prompt-recombination correction.** Inside the core the density is high
   enough that `α n₊ n₋` outruns diffusive transport. Solve `dn/dt = −α n²`
   analytically over `[0, t₀]` and inject only the survivors onto the grid.
2. **Diffusive start (do this first, it is nearly free).** Start the track at
   `t₀` with `b(t₀) = √(b₀² + 4 D t₀)`, choosing `t₀` so `b(t₀) ≈ 1–2` voxels.
   This is the physical justification for the Gaussian already in the code:
   after ~ns of diffusion the sub-micron structure is genuinely gone from the
   *physics*, not merely from the model.

### Zone M — middle, `h < r < r_s`: voxel-averaged RDD on the grid

**26.3 %** of the energy for `h = 1 µm, r_s = 163 µm`. Replace the Gaussian
shape with the RDD **integrated over each voxel's area**:

```
w_ij = (1/A_ij) ∫∫_voxel D(r) dx dy
```

Point-sampling a `1/r²` profile mis-estimates the near voxels badly and does
not conserve energy; area-averaging conserves it by construction. And because
this zone is *pure* `1/r²` (§1.1), the integral has a closed form — no numerical
quadrature per voxel. Precompute into the same `n_tracks × stencil` table
`_precompute_track_gaussians` already builds; the hot loop is untouched. **This
is a change to one function, not to the solver.**

### Zone F — far halo, `r > r_s`: uniform background, exactly integrable

**31.4 %** of the energy, of which **17.1 % lands beyond the electrode and
leaves the chamber**. By §3a it has no resolvable structure, so integrate once

```
E_tail = 2π ∫_{r_s}^{R_chamber} r D(r) dr
```

and add it as a spatially uniform source scaled by the track rate — **zero grid
cost, exact, no resolution requirement**. This is what removes the 180 mm outer
scale from the problem and lets `R` be 163 µm instead of metres. The energy
beyond `R_chamber` is the escape term of §2a and must be *explicitly* discarded
or retained, with the choice documented.

### Energy budget — the regression test

The three zones must account for 100 % of 1.206e-3 keV/µm. At `h = 1 µm`,
`r_s = 163 µm`, `R_chamber = 2.65 mm`:

| zone | radial band | share |
|---|---|---|
| C — analytic | 0 → 1 µm | **42.3 %** |
| M — on grid | 1 µm → 163 µm | **26.3 %** |
| F — uniform, in chamber | 163 µm → 2.65 mm | **14.3 %** |
| escape — leaves the chamber | 2.65 mm → 180 mm | **17.1 %** |
| | | **100.0 %** |

That identity is grid-independent, so it catches normalisation bugs a `k_s`
comparison would hide. Assert it in a test.

---

## 6. Recommended order of work

Cheapest decisive experiment first.

1. **Settle the escape question on paper** (§2a). 17 % of the LET is deposited
   outside the chamber. Decide whether the solver should be driven by a
   *restricted* LET, and whether the campaign's `W = 33 eV` calibration already
   absorbs it. **This is the largest single effect in the study and it needs no
   code at all.**
2. **Compute the prompt-recombination fraction inside 30 nm** (§5 Zone C).
   Analytic, an afternoon. If it is a few percent, the existing Gaussian is
   adequate and the programme stops here — which is a result worth publishing.
3. **Single-track Gaussian vs Cucinotta**, on the existing
   `tests/test_single_track_vs_jaffe.py` harness at R = 0.008 cm, h = 0.5 µm
   (326 × 326 × 4 010, 12.7 GiB — fits on any machine in §4.1). No overlap, so
   `k_s` is maximally sensitive to shape. This bounds the effect size the full
   implementation could ever deliver.
4. **Only if (3) is large**: implement Zones M and F, and run the convergence
   ladder R = 0.008 / 0.016 / 0.025 cm at h = 1 / 0.5 µm to demonstrate that
   column radius and voxel size are both converged.
5. **Then** the production comparison at 10 and 50 Gy/s, where §3b predicts a
   sub-percent shift from shape and §2a a larger one from normalisation. A null
   result on shape is a publishable statement about *when* RDD structure
   matters in a pulsed cavity — not a failed experiment.

---

## 7. What this study does not establish

- **No RDD code exists.** This is a design study, not an implementation report.
- The **prompt-recombination fraction** of §6.2 is not computed. Every
  recommendation about Zone C is conditional on it.
- The **`k_s` insensitivity** of §3b is a prediction from the overlap count, not
  a measurement. So is the *sign* predicted in §2c.
- The energy fractions use the supplied tabulations decimated to ~10 points per
  decade (trapezoid in `ln r`). The curves are smooth in log-log and the LET
  integral reproduces the solver's value to 0.6 %, so the error is well under a
  percent — but they are not the full 100-per-decade integrals.
- The **water → air scaling of §1.3 must not be used** for anything: it is good
  to 10 % on the halo and wrong by 2 800× on the core. Both tabulations were
  obtained from libamtrack directly, and any new medium needs the same.
- Step counts in §4.2 assume the present von Neumann-limited `dt`. A finer grid
  may need a smaller `dt` for *accuracy* rather than stability, which would make
  wall clock worse than tabulated.
- Everything is for one beam quality (56.2 MeV/u protons, LET 1.2e-3 keV/µm in
  air) at one dose rate. The overlap argument of §3 is dose-rate dependent by
  construction and weakens as fluence falls.
