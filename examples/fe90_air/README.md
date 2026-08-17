# 90 MeV/u iron in air — single ion, Cucinotta RDD

**Status: plan + sizing study. Not yet runnable** — the RDD deposition kernel
(§4.1) does not exist yet. Everything below the sizing section is what has to be
built; everything in the sizing section is measured on this laptop
(Intel Core Ultra 5 225U, 14 threads, 30.8 GiB RAM).

Cross-code target: a colleague's IonTracks-FEniCSx run on an unstructured
tetrahedral mesh — cylinder R = 120 µm × 2000 µm tall, 28,515 vertices /
164,612 tetrahedra, spacing varying quadratically from 5 µm on the axis to
50 µm at the wall.

## 1. The input data

[`data/rdd_cucinotta_fe90_air.csv`](data/rdd_cucinotta_fe90_air.csv) — libamtrack,
Cucinotta RDD with the Tabata electron-range model, Bethe stopping power, dry
air. 1000 points, `r` log-spaced from 10 nm to 0.1 m (verified to be exactly
`numpy.logspace(-8, -1, 1000)`), dose in Gy.

Its shape, from the log–log slope:

| region | radius | `dlnD/dlnr` |
|---|---|---|
| core | < ~50 nm | −3.5 → −3.9 |
| penumbra | 100 nm – 10 mm | −2.000 (exact `1/r²`) |
| δ-ray cutoff | → 0.1 m | steepening past −3 |

Two normalisation checks pass:

- `2π ρ ∫ D(r) r dr` over the whole table = **0.5546 keV/µm**, against
  **0.5611 keV/µm** in this repo's libamtrack air stopping-power table
  (`pulsed_ion_chamber/data/stopping_power_air.csv`, `iron_LET_keV_um` at
  90 MeV/u). 1.2 % apart, which is the trapezoid rule on a 143-points-per-decade
  grid — the RDD is normalised to the Bethe LET.
- `D` is strictly decreasing, and smooth in log–log everywhere except the
  core/penumbra join near 35 nm, which is a real feature of the model.

## 2. The consequence that drives everything else: the track is not local

The δ-ray range in air at 90 MeV/u is ~10 cm, so the track's energy is spread
over a radius 50× larger than the chamber gap. Because the penumbra is `1/r²`,
energy is deposited **uniformly per decade of radius**, and a 5-decade-wide
domain still misses a third of it:

| lateral cut-off | fraction of LET inside |
|---|---|
| 0.1 µm | 39.1 % |
| 1 µm | 49.4 % |
| 5 µm | 56.6 % |
| 120 µm — the FEniCS domain | **70.9 %** |
| 600 µm | 78.1 % |
| 1200 µm | 81.2 % |
| 2650 µm — full AIC-144 electrode | 84.7 % |
| 10 cm — whole δ-ray halo | 100 % |

Neither domain is "big enough"; enlarging it is not a converging strategy,
because each further decade adds another ~3 % of the LET. §4.2 says what to do
about it instead.

## 3. Sizing on this laptop — measured

A regular grid at `h = 5 µm` matches the FEniCS mesh **exactly along z**
(400 voxels × 5 µm = 2000 µm) and is 5× finer than its outer wall spacing
(50 µm) but equal to its on-axis spacing.

`240 × 240` in the transverse plane at 5 µm is a **1200 µm-wide column,
R = 600 µm** — five times the FEniCS cylinder radius, not a match to it. The
grid that matches R = 120 µm at 5 µm is `48 × 48 × 400` (`56 × 56 × 406` once
`SimulationConfig` adds its lateral buffer and electrode margins).

Memory is `4 × 8 bytes × no_xy² × no_z_with_buffer` (two species × current/next).
Wall times below are the existing Gaussian-track solver
(`solver_numba_parallel`), which has the same PDE cost per voxel-step the RDD
version will have:

The first two columns are the **simulated volume** (a square column, `W × W`
transverse by `Lz` axial); the next three are the **voxel size along each
axis**; the sixth is the **voxel count along each axis**.

| `W` [µm] | `Lz` [µm] | `hx` [µm] | `hy` [µm] | `hz` [µm] | `nx × ny × nz` | Mvoxels | RAM | `dt` [ns] | steps | 1 thread | 4 threads |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 280 | 2060 | 10 | 10 | 10 | 28×28×206 | 0.16 | 4.9 MiB | 444 | 331 | 0.3 s | 0.2 s |
| **280** | **2030** | **5** | **5** | **5** | **56×56×406** | **1.27** | **39 MiB** | **209** | **705** | **2.8 s ᵐ** | **1.4 s ᵐ** |
| 280 | 2015 | 2.5 | 2.5 | 2.5 | 112×112×806 | 10.1 | 309 MiB | 92 | 1596 | 85 s | 50 s |
| 280 | 2007.5 | 1.25 | 1.25 | 1.25 | 224×224×1606 | 80.6 | 2.40 GiB | 37 | 4025 | ~29 min | ~17 min |
| **1240** | **2030** | **5** | **5** | **5** | **248×248×406** | **25.0** | **762 MiB** | **209** | **705** | **93 s ᵐ** | **54 s ᵐ** |
| 2440 | 2030 | 5 | 5 | 5 | 488×488×406 | 96.7 | 2.88 GiB | 209 | 705 | ~6 min | ~4 min |
| 1240 | 2015 | 2.5 | 2.5 | 2.5 | 496×496×806 | 198 | 5.91 GiB | 92 | 1596 | ~28 min | ~16 min |

**The three voxel sizes are equal by construction, not by choice.** The
Lax-Wendroff stencil carries a single `unit_length_cm`, and the von Neumann
condition (§3.1) is written for `sx = sy = sz`; anisotropic spacing — e.g. a
coarse `hz` to exploit the fact that the track is uniform along `z` — would
need a new stencil and a new stability condition. It is the obvious first
optimisation to reach for and it is not currently available.

`Lz` exceeds the 2000 µm gap by `6 × hz`: three voxels of electrode margin at
each end (`no_z_electrode=3`), so charge leaves the gap before it leaves the
array. `W` likewise exceeds the scored disc by the lateral buffer — the three
distinct widths correspond to **R = 120 µm** (the FEniCS-matched column),
**600 µm** and **1200 µm**.

ᵐ = measured; the rest projected from the measured 5.29 ns per voxel-step at
1 thread. `dt` is the von Neumann limit for the two Kanai species; the step
count is one 0.1 µs "pulse" plus a clearance of 2 × the half-gap transit time of
the slowest carrier (µ⁺ = 1.36 cm²/V/s → 147 µs to cross 2 mm at 1000 V/cm).

**Answers to the sizing question:** yes, both fit comfortably — 39 MiB for the
FEniCS-matched column and 762 MiB for the 240²-class one, against 22 GiB free.
Memory is not the constraint here; **grid spacing is** (§4.3), and it costs
`h⁻⁴` — `h⁻³` in voxels plus `h⁻¹` in time steps.

Beyond ~2.5 µm at R = 600 µm the laptop runs out of room; the two entries above
6 GiB are cluster work (`docs/HELIOS.md`).

All timings here are the **CPU** backend on the laptop, which is what the
sizing question was about. `solver_cuda` (#21) and its unified-memory/1 µm-grid
support on GH200 (#22) landed on master after these were measured and are not
reflected in the table — they change the economics of the fine-`h` ladder in
§4.3 substantially, and that ladder is the natural first thing to re-time on a
GPU node.

### 3.1 Time step — we do not get to choose it

The FEniCS run uses **dt = 700 ns**, 987 steps, 691.18 µs total; 4 min 41 s in
the solve loop, 546 MB peak, one core.

We cannot use 700 ns, and should not want to. `dt` here is slaved to `h` by the
von Neumann condition on the explicit Lax-Wendroff scheme
(`6s + c² ≤ 1`, `s = D dt/h²`, `c = µE dt/h`):

| `h` [µm] | `dt_max` [ns] | Courant | `6s` | binding term |
|---|---|---|---|---|
| 1.25 | 36.6 | 0.614 | 0.611 | diffusion |
| 2.5 | 92.2 | 0.775 | 0.385 | drift |
| **5** | **208.5** | **0.876** | 0.218 | drift |
| 10 | 444.2 | 0.933 | 0.116 | drift |
| 15.4 | 700 | 0.954 | 0.078 | drift |
| 20 | 918.4 | 0.964 | 0.060 | drift |

Three things follow.

**Use the von Neumann limit, i.e. leave `dt_seconds=None`.** Lax-Wendroff's
leading dispersive error scales as `(1 − c²)` and vanishes at Courant 1, so
shrinking `dt` below the limit makes the *drift* term worse while costing
proportionally more steps. There is no accuracy to buy by backing off; the only
real knob is `h`.

**700 ns would force `h ≥ 15.4 µm`** — three times coarser than the FEniCS
on-axis spacing, and hopeless for a core that is still unresolved at 5 µm
(§4.3). Only worth doing as a deliberate coarse apples-to-apples cross-check.

**Their 700 ns at 5 µm is Courant 2.94.** That is fine for stability — an
implicit solve is unconditionally stable — but stability is not accuracy: the
ion cloud crosses ~3 cells per step, and backward Euler smears it. For this
problem that bias runs the same direction as the unresolved core, i.e. it
*lowers* `k_s`. Worth asking them for a `dt` halving check (700 → 350 → 175 ns)
at fixed mesh; if `k_s` moves, the step is doing physics.

**Run length.** Full-gap transit of the slowest carrier (µ⁺ = 1.36 cm²/V·s at
1000 V/cm) is **147 µs**, which is what our clearance criterion uses. Their
691 µs is 4.7× that. Worth asking what sets it — if it is only a safety factor,
their solve loop has a ~4.7× saving in it.

**Cost per degree of freedom**, since the two codes are otherwise hard to
compare:

| | FEniCS, implicit, unstructured | this code, explicit, regular |
|---|---|---|
| DOFs | 28,515 vertices | 1.27 M voxels (56×56×406) |
| steps | 987 | 705 |
| solve wall time | 281 s | 2.8 s |
| per DOF-step | ~10 µs | 3.1 ns |
| peak memory | 546 MB | 39 MiB |

The implicit solve buys a 3.4× larger `dt` for roughly 3200× the cost per
DOF-step and ~600× the memory per DOF. That is the whole trade: matrix-free
explicit stepping on a regular grid can afford 45× more unknowns and still
finish 100× sooner, which is exactly the budget the `h` ladder in §4.3 needs.

## 4. What has to be built

### 4.1 RDD deposition kernel

`solver_numba._insert_track_numba` hard-codes the Gaussian `exp(-r²/b²)` and
leans on it three times: separability (`2w` exponentials instead of `w²`),
`10σ` truncation, and a `k`-independent 2D cross-section. Only the third
survives for a tabulated RDD. The replacement:

- Build a lookup of the RDD **integrated over each voxel's area**, not sampled
  at the voxel centre. Area-averaging conserves the LET exactly and is the only
  defensible treatment of a core that no affordable grid resolves;
  point-sampling a `1/r²` profile misses charge and is spacing-dependent.
- Precompute one `w × w` stencil per run, not per track. A single ion means
  deposition cost is irrelevant, so a dense radial table beats any clever
  factorisation.
- Truncate at the grid, not at a fixed radius — the profile has no natural
  cut-off inside the domain.

`config.Gaussian_factor`, `track_radius_cm`, `track_sigma_cm` and
`track_cutoff_sigmas` all become inapplicable; the `LET_b.dat` Rossomme fit
(which returns b = 49.4 µm here) is a proton/light-ion parameterisation and
should not be used for iron at all.

### 4.2 Book-keeping for the charge outside the domain

The 29 % of the LET beyond 120 µm is real charge at very low density. It
recombines essentially not at all (§4.3), but it *is* collected, so it belongs
in the denominator of the collection efficiency. Scoring only what is inside the
grid inflates the loss by `1/0.709`.

So: score the injected total against the **full** 0.5546 keV/µm, adding the
out-of-domain fraction as fully collected. Equivalently

```
f_chamber = f_in_domain · F + (1 − F)
```

with `F` the in-domain LET fraction from §2. This is worth raising with the
FEniCS side too — the same correction applies to their R = 120 µm mesh, and it
is a ~40 % change in `k_s − 1`, not a rounding detail.

Boundary conditions follow from the same picture: `lateral_boundary="absorbing"`
(a single track sits in otherwise empty gas — nothing diffuses back in) with
`scoring_region="full_grid"`. Note the diffusion length over the full 147 µs
collection is ~36 µm, so at R = 120 µm a few percent of the in-domain charge
does leave sideways; R ≳ 300 µm makes that negligible.

### 4.3 The core is not resolved, and `k_s` will show it

Area-averaging the RDD onto the centre voxel gives, per voxel size:

| `h` [µm] | centre-voxel `n₀` [cm⁻³] | `1/(α n₀)` | `α n₀ dt` |
|---|---|---|---|
| 10 | 9.27e10 | 6.74 µs | 0.066 |
| 5 | 3.51e11 | 1.78 µs | 0.117 |
| 2.5 | 1.32e12 | 0.47 µs | 0.195 |
| 1.25 | ~4.9e12 | 0.13 µs | 0.291 |
| 1.0 | 7.60e12 | 0.08 µs | — |

Two things follow.

**`k_s` is not converged at 5 µm.** `n₀` grows as the grid refines and the
recombination integral goes as `⟨n²⟩`, not `⟨n⟩²`, so coarse-graining
systematically *under*-predicts the loss. The physical floor is diffusion:
√(2Dt) reaches 0.75 µm in 0.1 µs and 2.4 µm in 1 µs, so the initial condition
below ~1 µm is smeared away almost immediately and refining past that buys
nothing. Between 5 µm and 1 µm it buys a lot. **Run the ladder
h = 10 → 5 → 2.5 → 1.25 µm at R = 120 µm and report `k_s(h)`** — that is the
first result worth having, it costs under half an hour on 4 threads, and it
tells the FEniCS side whether their 5 µm on-axis spacing is adequate.

**The explicit recombination step is stiffer than usual, but not a problem.**
`_lax_wendroff_step_numba` subtracts `α·dt·p·n` using old values, and `α n₀ dt`
reaches 0.117 at 5 µm and 0.291 at 1.25 µm — far above anything the AIC-144
scenarios produce. Measured against the analytic solution of `dn/dt = −αn²`, in
a recombination-only decay of the centre voxel over 700 steps:

| `h` [µm] | `dt` [ns] | `α n₀ dt` | explicit `n(end)` | analytic `n(end)` | error on integrated loss |
|---|---|---|---|---|---|
| 10 | 444 | 0.066 | 1.957e9 | 1.967e9 | 0.01 % |
| 5 | 209 | 0.117 | 4.204e9 | 4.230e9 | 0.01 % |
| 2.5 | 92 | 0.195 | 9.544e9 | 9.612e9 | 0.01 % |
| 1.25 | 37 | 0.291 | 2.412e10 | 2.431e10 | 0.00 % |

The per-step splitting error is a percent or two on the *peak density*, but it
almost cancels in the *integrated* loss, which is what `k_s` is built from — the
core recombines away either way, only the trajectory differs. `α n dt` would
have to reach 1 (around `h ≈ 0.7 µm`) before the explicit form could go
negative, which is below anything affordable.

So the analytic update `n → n/(1 + α n dt)` — exact when `p = n`, which a fresh
track satisfies by construction, and unconditionally positive — is worth having
as cheap insurance, but it is **not** a prerequisite for the ladder.

### 4.4 Single ion on the axis

`SimulationConfig` derives the track count from a dose rate and
rejection-samples positions in the disc. For one deliberate ion, add an explicit
`n_tracks` / `track_placement="axis"` path rather than leaning on
`dose_rate_Gy_s=1e-12` (which floors to 1 track) plus a tiny
`chamber_fill_fraction` (which nudges it near, not onto, the axis). With one
ion there is no general recombination — the result is pure initial/columnar
recombination, comparable against Jaffé.

## 5. Reference point from the existing Gaussian model

Already runnable today, as a sanity anchor rather than a result — 90 MeV/u iron,
200 V, 2 mm gap, one track, `b = 49.4 µm` from the Rossomme fit:

| | `k_s` |
|---|---|
| simulation, 56×56×406 at 5 µm | 1.043479 |
| simulation, 248×248×406 at 5 µm | 1.043473 |
| Jaffé theory, same LET | 1.045789 |

The two grids agreeing to 6 digits confirms the domain width is not what limits
this — the Gaussian is 49 µm wide, so both grids resolve it fully.

That is also the measure of how much the RDD changes: the Gaussian's peak
density is 2.14e9 cm⁻³, against 3.5e11 cm⁻³ for the area-averaged Cucinotta core
at the same 5 µm spacing — **164× higher**. Expect `k_s − 1` to move by orders
of magnitude, upward, and to keep moving as `h` shrinks.

## 6. Order of work

1. RDD table loader + area-averaged stencil (§4.1), validated by checking the
   deposited charge against `2πρ∫D r dr` over the domain.
2. Axis placement and explicit track count (§4.4).
3. Out-of-domain correction (§4.2), reported alongside the raw in-domain number.
4. The `h` ladder at R = 120 µm (§4.3), each rung at its own von Neumann `dt`
   (§3.1), then one R = 600 µm run at the best affordable `h` to size the
   lateral truncation error.
5. Analytic recombination update (§4.3) — cheap insurance, not a prerequisite.
