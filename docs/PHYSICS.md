# Physics and numerics of `pulsed_ion_chamber`

What this code simulates, what it assumes at every step, and why each
assumption is defensible. How it is implemented is covered in
[ALGORITHM.md](ALGORITHM.md), and what it costs in
[PERFORMANCE.md](PERFORMANCE.md).

Every quantity below is a field of `SimulationConfig` unless stated otherwise.
Where a section quotes two numbers, the first is the library default and the
second is what `examples/ifj_aic144/` sets for the AIC-144 Markus scenario.

---

## 1. The equation being solved

Two coupled scalar fields, the positive and negative charge-carrier number
densities `n₊(x,t)` and `n₋(x,t)` in the chamber gas:

```
∂n±/∂t = D± ∇²n±  ∓  µ± E·∇n±  −  α n₊ n₋  +  S(x,t)
```

drift in the applied field, isotropic diffusion, bimolecular (general)
recombination, and a source term `S` that injects ion tracks. The output is the
collection efficiency

```
f = (N_injected − N_recombined) / N_injected      k_s = 1/f
```

**Why this form.** An ionisation chamber measures collected charge and assumes
it equals the charge liberated. Recombination breaks that assumption, and `k_s`
is the correction. The bimolecular term `α n₊ n₋` is the only sink: it is
quadratic in density, which is why pulse structure matters far more than mean
dose rate — packing the same charge into a shorter pulse raises the
instantaneous density and the loss goes as its square.

**Initial (columnar) recombination is included, not missing.** There is no
separate "initial recombination" term because none is needed: the same
`α n₊ n₋` acting on the resolved Gaussian track core *is* columnar
recombination. `tests/test_single_track_vs_jaffe.py` checks exactly this
against analytic Jaffe theory in the single-track limit. What limits its
accuracy is how well the core is resolved on the grid (§7), not a missing term.

---

## 2. Particle and stopping power

**Assumption.** A single ion species at a single energy, entering perpendicular
to the electrodes, traversing the gap as a straight line at constant LET.
Default `E_MeV_u=150.0`, `particle="proton"`; the AIC-144 example uses
56.2 MeV protons.

LET comes from tabulated PSTAR data for **dry air**
(`data/stopping_power_air.csv`, interpolated in energy). At 56.2 MeV this gives
**1.1995 eV/µm** (1.1994844e-3 keV/µm in the units the table is stored in).

**Why.** A 56 MeV proton has a range in air of tens of metres, so it loses a
negligible fraction of its energy crossing a 2 mm gap: LET is constant along the
track to well under a percent, and there is no Bragg peak inside the chamber.
That is what makes the track a uniform line source and lets the whole
transverse problem be solved once per track (§7, §11).

Air's stopping power is ~1000× lower than water's, which is the central
practical fact of this simulation: depositing a clinically relevant dose takes a
very large number of tracks (§5).

*Related codes:* the same PSTAR table and interpolation are used by
IonTracks-Cython and IonTracks-FEniCSx, so LET agrees to 6 digits across all
three.

---

## 3. Track structure — the radial dose distribution

**Assumption.** Each track deposits a 2D Gaussian of ion-pair density about its
axis, constant along the track:

```
n(r) = N₀/(π b²) · exp(−r²/b²)        N₀ = LET/W   [ion pairs per unit length]
```

`b` is the **Gaussian track radius**, computed from a quadratic fit in
log₁₀(LET) to the Rossomme et al. tabulation (`data/LET_b.dat`), floored at
**20 µm**. At the LETs relevant here the fit falls below the floor, so
**b = 20 µm**.

**Note the parametrisation.** `exp(−r²/b²)` is a Gaussian of standard deviation
**σ = b/√2 = 14.14 µm**, not `b`. This matters when reading or setting the
cutoff in §4. The normalisation is exact: ∫ n dA = N₀.

**Why an amorphous Gaussian.** The real radial dose distribution from
delta-electrons is a steep power law with a core cutoff and a sharp outer edge
at the delta-electron range. Resolving that would need a grid orders of
magnitude finer than the drift/diffusion problem needs. The Gaussian is the
standard amorphous-track-structure surrogate: it preserves the two quantities
recombination actually depends on — total charge per unit length, and the width
over which it is spread — and it is what the analytic Jaffe theory used as a
reference assumes, which is what makes the single-track validation meaningful.

**Why `b` is floored.** The Rossomme fit is calibrated over a range of
therapeutic LETs; extrapolated to air's very low LET it returns unphysically
small radii. The 20 µm floor keeps the track wider than a couple of voxels at
the grid sizes used, which is also what keeps the deposit resolvable at all
(§7).

*Related codes:* IonTracks-Cython derives `b` the same way. IonTracks-FEniCSx
instead takes `b` from its config; for the AIC-144 campaign it was set to
20 µm, which coincides with this floor. At higher LET the two would diverge.

---

## 4. Track deposition and the truncation stencil

**Assumption.** A track's Gaussian is deposited only within
`track_cutoff_sigmas` standard deviations of its axis — default **10 σ**, i.e.
`10·b/√2 = 141.4 µm = 14.14 voxels` on a 10 µm grid. Outside that square
bounding box the contribution is dropped. `track_cutoff_sigmas=None` disables
truncation.

**Why this is exact, not an approximation.** For the 2D Gaussian above, the
fraction of a track's charge beyond radius `R` is exactly `exp(−R²/b²)`, so an
`n`-sigma cutoff discards `exp(−n²/2)`:

| cutoff | radius | charge discarded |
|---|---|---|
| 3 σ | 2.12 b | 1.1e-2 |
| 6 σ | 4.24 b | 1.5e-8 |
| 8 σ | 5.66 b | 1.3e-14 |
| **10 σ** | **7.07 b** | **1.9e-22** |
| 12 σ | 8.49 b | 5.4e-32 |

At 10 σ the discarded charge is **six orders of magnitude below float64
epsilon** (2.2e-16). A contribution that small cannot change a double-precision
sum it is added to, so the truncation is lossless in the only sense that
matters numerically. This is verified, not assumed:
`tests/test_v2_physics.py::test_default_cutoff_is_exact_to_machine_precision`
requires the truncated and untruncated runs to agree, and on the AIC-144
`archive` tier they agree to all 8 printed digits of `k_s`.

**Why truncate at all.** Without it, every track's Gaussian is evaluated at
every grid point, so insertion costs `O(no_xy² · no_z)` per track no matter how
far away those points are. On a small column that is affordable; on a wide one
almost all of it is spent adding `exp(−100)` and smaller. The stencil makes the
per-track cost independent of the grid — see [PERFORMANCE.md](PERFORMANCE.md).

**Why 10 σ rather than something tighter.** 4 σ would still only discard 3.4e-4
and would be visibly faster, but it puts a real approximation into the physics
that has to be justified per scenario. 10 σ costs about 20 % more than 4 σ on a
small grid and nothing at all on a large one (the per-track cost is dominated by
the grid traversal, not the stencil, once the stencil is smaller than the grid),
and it needs no justification at all. Tighten it only with a measured reason.

---

## 5. Dose, fluence and the number of tracks

**Assumption.** `dose_rate_Gy_s` is the **time-averaged** dose rate **to air**.
The dose delivered per pulse is `dose_rate_Gy_s / repetition_rate_hz`, all of it
inside the pulse window, so the instantaneous rate inside a pulse is higher by
the inverse duty cycle. The track count follows from fluence:

```
Φ = D · ρ_air / LET        N = Φ · π·sampled_radius²
```

**Why dose to air, not water.** The chamber gas is what recombines. Beam
metrology usually quotes dose to water, so a conversion factor belongs in the
scenario, not in the solver: the AIC-144 example writes
`dose_rate_Gy_s=8.91` with a comment that this is 10 Gy/s to water × 0.891.
Keeping the factor visible at the call site prevents it from being applied
twice.

**`air_density_kg_m3`** selects the reference condition the dose is quoted at;
the track count scales linearly with it. Default `1.225` (ISA sea level, 15 °C);
the AIC-144 example uses **1.2041 kg/m³, dry air at 20 °C and 101.325 kPa**,
the condition dosimetry normally reports at. Nothing else in the model depends
on gas density — the transport constants of §8 are used as given.

**`chamber_fill_fraction`** (default 1.0) decouples the disc tracks are *counted*
over from the disc they are *placed* in. At 1.0 they are the same, so the
simulated areal density equals the nominal fluence and the model is
self-consistent. A value `< 1` concentrates the same track count into a smaller
area, raising the areal density by `1/fraction²`; it exists to reproduce
external configurations that do this, and should be left at 1.0 otherwise.

*Related codes:* IonTracks-FEniCSx computes its track count over the full
chamber radius while placing tracks within 0.7 of it, so its simulated density
is 2.04× the nominal one implied by the dose it quotes — reproducible here by
setting `chamber_fill_fraction=0.7`.

---

## 6. Beam time structure

**Assumption.** Tracks arrive only during repeating windows of
`pulse_duration_s`, spaced `1/repetition_rate_hz` apart, for `n_pulses` pulses
(default 1). Within a window, arrival times are spread pseudo-uniformly across
the whole window; they are not clustered at its start. Defaults are 540 µs at
50 Hz — a 2.7 % duty cycle.

**Why the RF microstructure is averaged over, not resolved.** A cyclotron only
extracts protons in RF buckets, so a 540 µs "pulse" is really ~14,000 bunches at
~26 MHz. The simulation's time step is fixed by numerical stability (§9) at a
few hundred nanoseconds, which is ~10 RF periods. Many buckets therefore fall
inside a single time step regardless of how arrival times are modelled within
it, and resolving them would require a `dt` — and hence a grid — tens of times
finer, for a quantity that is averaged away by the transport anyway. Averaging
is the correct simplification here, not a shortcut.

This is made checkable rather than left implicit: set `rf_frequency_hz` and
`config.summary()` reports the RF cycles per time step, with a warning if it
ever drops below 1 (at which point the assumption would deserve re-examination).

**Why one pulse is usually enough.** At the default 50 Hz, carriers are fully
collected within ~130 µs of the pulse ending (§12) while the next pulse is 20 ms
away — four orders of magnitude later. There is nothing left in the gap to
accumulate. Raise `n_pulses` only when the repetition rate approaches the
collection time.

---

## 7. Simulated volume and grid

**Assumption.** A cylindrical column of gas spanning the full electrode gap,
radius `sampled_radius_cm`, discretised on a **uniform rectangular voxel grid**
of spacing `grid_size_um`. The array is square in `x`,`y` with
`buffer_radius` voxels of margin outside the scored disc, and has
`no_z_electrode` voxels of gas margin beyond each electrode.

For the AIC-144 `archive` tier: 80 µm radius (8 voxels) inside a 22×22×210
array at 10 µm, `no_z = 200 = 2 mm / 10 µm`.

**Why a sub-volume and not the chamber.** The field is uniform, the track
density is statistically homogeneous, and there is no guard ring or edge physics
in the model — so the chamber interior is translation-invariant transversally.
A column is a representative sample of it, and the answer converges once the
column is large enough that edge tracks are a small fraction (§14). Simulating
the full 5.3 mm-diameter electrode would reproduce the same result hundreds of
times over at enormous cost ([PERFORMANCE.md](PERFORMANCE.md) §5).

**Why a rectangular grid.** The geometry is a uniform slab: nothing about the
solution is curved, so an unstructured mesh buys no geometric fidelity and costs
indirection on every access. A regular lattice also makes the drift direction
axis-aligned, which is what allows the second-order Lax-Wendroff stencil of §9
and the separable-Gaussian deposition of §4 with no interpolation anywhere.

**Grid spacing.** `grid_size_um` must resolve the track core, which is the
sharpest feature in the problem: at the default 10 µm and `b = 20 µm` there are
**2 voxels per Gaussian radius**, ~10 voxels inside the 1/e radius per
transverse plane. That is adequate but not generous — it is the resolution at
which the deposit's integral is right to a fraction of a percent — and it is the
knob to tighten first if the track core is suspected.

`buffer_radius` is numerical margin, not physics: it keeps the boundary
condition (§10) away from the scored region. With a reflecting wall it converges
at 3 voxels; with an absorbing wall it needs 4–6.

**Size guard.** The carrier arrays are `4 · no_xy² · no_z_with_buffer · 8` bytes
and grow as the square of the column radius, so a millimetre-scale column is
gigabytes. `SimulationConfig` estimates the peak footprint and refuses to build
a config exceeding `memory_budget_fraction` (default 0.8) of available RAM —
the alternative is finding out from the OOM killer part-way through a long run.
Set it to `None` to opt out.

---

## 8. Gas and charge-carrier transport

**Assumption.** Air at standard conditions, with Kanai (1998) transport
constants:

| | default (single averaged species) | two-species option |
|---|---|---|
| µ₊ | 1.65 cm²/(V·s) | 1.36 cm²/(V·s) |
| µ₋ | 1.65 cm²/(V·s) | 2.10 cm²/(V·s) |
| D₊ | 0.037 cm²/s | 0.0282 cm²/s |
| D₋ | 0.037 cm²/s | 0.0435 cm²/s |
| α | 1.60 × 10⁻⁶ cm³/s | same |
| W | 34.2 eV/ion pair (`W_eV`) | — |

**Single vs. two species.** By default both carriers share one averaged mobility
and one averaged diffusion coefficient. Setting `mu_positive_cm2_Vs` and friends
resolves the two Kanai species separately; the AIC-144 example does. Both are
defensible, and the difference is smaller than one might expect: at 1500 V/cm
across 2 mm the averaged model gives both carriers an 80.8 µs transit, while the
resolved model gives 98.0 µs (positive) and 63.5 µs (negative). The slow
positive cloud lingers 21 % longer, which raises recombination, but the two
clouds separate 5 % faster, which lowers it. Measured on the AIC-144 `archive`
tier the net effect on `k_s` is **−0.4 %**. Resolving them costs ~35 % more time
steps (§9), so the averaged model is a reasonable default and the resolved one
is the more faithful choice when the extra cost is affordable.

Both mobilities are supplied as **positive magnitudes**. The drift directions
are opposite by construction in the solver's z-stencil, so a negative value
would flip the sign twice and send both species to the same electrode; the
config rejects it.

**W = 34.2 eV/ion pair** is the proton-specific value, and is used consistently
by both the PDE solver and the analytic Jaffe reference in `theory.py`, so the
single-track validation compares like with like. It is exposed as `W_eV` because
published values differ (33–34 eV depending on particle and source) and it
scales the charge per track linearly — and recombination quadratically.

**`epsilon_r` does not appear** because space-charge screening is not modelled
(§13): the field is the applied field, so the gas permittivity never enters.

---

## 9. Time integration and stability

**Assumption.** Explicit **Lax-Wendroff** finite differences, second order in
space and time, with the recombination sink evaluated from the previous step's
densities. The time step is **not a free parameter**: it is the largest `dt`
satisfying the von Neumann stability criterion (Deghan 2004) for full 3D
diffusion and drift along z,

```
6·s + c² ≤ 1        s = D dt/h²      c = µ E dt/h
```

For the AIC-144 `archive` tier with two species this gives
**dt = 304.3 ns**, set by the negative ion (both faster and more diffusive, so
it binds); the averaged-species model gives 382.6 ns. This is why resolving
two species costs step count: `dt` drops 20 %, and the collection tail lengthens
(§12).

**Why explicit, given the step-size penalty.** The alternative — an implicit
scheme — decouples `dt` from stability, but stability is not accuracy. This
problem is strongly advection-dominated: the mesh Péclet number
`µ E h / 2D` is **36** on the standard grid. Unstabilised Galerkin or
centred-difference discretisations at that Péclet number produce spurious
oscillations, and the usual remedies (upwinding, SUPG) introduce numerical
diffusion that directly smears the carrier clouds — which is precisely the
quantity `α n₊ n₋` is most sensitive to. The explicit scheme sidesteps the
question: it is bounded by the CFL condition it solves for, running at Courant
number **0.96** (negative) and 0.62 (positive), where Lax-Wendroff is
well-behaved and second-order accurate. Buying larger steps by degrading the
carrier-cloud overlap would defeat the purpose of the calculation.

*Related codes:* IonTracks-Cython uses the same explicit scheme and the same
criterion. IonTracks-FEniCSx uses backward Euler on P1 elements with no
stabilisation, which is unconditionally stable and was run at Courant ≈ 2.2 at
the same mesh Péclet of 36; whether that costs accuracy has not been measured
from this side.

---

## 10. Boundary conditions

**Electrodes (z).** Charge that drifts past the gap into the `no_z_electrode`
buffer layers is no longer scored, and the outermost layer is held at zero, so
carriers leave the system on arrival. **Why a buffer rather than clamping the
density to zero at the collecting plane:** a hard Dirichlet condition at the
electrode imposes an artificial density gradient exactly where the charge is
being collected, adding a spurious diffusive flux. Letting the carriers drift on
into a margin and simply ceasing to count them is the gentler formulation of the
same physical statement — the electrode neutralises what reaches it.

**Chamber wall (x, y), `lateral_boundary`.**

- `"reflecting"` (used by the AIC-144 example): zero-gradient mirror, i.e. zero
  net flux through the wall. **This is the physically right condition for a
  column sampled from the interior of a large, uniformly irradiated chamber**:
  the gas outside the column that is not being simulated is statistically
  identical to the gas inside, so it returns as much charge as it takes.
- `"absorbing"` (default): the outer ring is held fixed and charge reaching it
  leaves. Appropriate only if the column really is isolated.

Measured effect on `k_s`: **1.4e-5** on the `archive` tier — negligible, because
recombination is dominated by fresh in-disc track density during the pulse while
the wall sits several voxels beyond it. Where the choice does matter is how much
margin is needed: reflecting converges at `buffer_radius=3`, absorbing needs
4–6, and absorbing at `buffer_radius=1` is wrong by 9 %.

The reason absorbing degrades so sharply at small margins is worth knowing: the
Lax-Wendroff sweep never updates the outer ring, but track insertion does write
to it, so Gaussian tails accumulate there and are then frozen — they never
drift, diffuse or recombine, and keep feeding their inward neighbour. With a
reflecting wall the ring is a mirror of the interior and no such reservoir
forms.

---

## 11. Scoring

**Assumption.** Both the injected charge and the recombined charge are
accumulated over the same region, so `f` is a ratio of like quantities.
`scoring_region` selects which:

- `"track_disc"` (default): voxels inside `sampled_radius_cm`, in the gap
  layers only — the disc the tracks were drawn in.
- `"full_grid"`: every voxel in the gap, including the track-free buffer
  annulus.

**Why `track_disc` is the default.** The buffer annulus receives Gaussian tails
but no track centres, so it is systematically under-irradiated relative to the
uniformly irradiated chamber being modelled. Including it dilutes the mean
density and biases `k_s` low — measured, **0.8 %** on the `archive` tier, and
that bias is buffer-converged rather than divergent. `full_grid` exists to
reproduce external configurations that score this way.

`k_s` is reported as **1/f** (the exact convention). The first-order
approximation `1 + N_recombined/N_injected` agrees to 4 digits below ~1 %
recombination and diverges from it by up to 15 % in the densest cases, so
whichever is quoted must be stated explicitly when comparing against other
codes or published values.

---

## 12. Run duration

**Assumption.** A run covers the injection window plus a clearance period long
enough for the gas to empty:

```
separation_time_steps = gap / (2 · µ_slowest · E · dt)          # half-gap transit
clearance = n_clearance_separation_times × separation_time_steps
```

`n_clearance_separation_times` defaults to **2.0**, exactly one full-gap transit
of the slowest carrier — complete collection with no margin. The AIC-144 example
uses **2.6**, adding a 30 % safety factor.

**Why the slowest carrier.** Collection is finished when the *last* carrier
arrives. With two resolved species that is the positive ion (µ = 1.36), needing
98.0 µs to cross a 2 mm gap against the negative ion's 63.5 µs. Sizing the tail
on an average would truncate the run while charge is still in flight, which
under-counts recombination.

For the AIC-144 `archive` tier: 540 µs injection + 127.5 µs clearance =
**667.7 µs in 2194 steps**.

---

## 13. What is not modelled

Each of these is a deliberate omission, listed with the condition under which it
would stop being safe.

- **Space-charge screening.** The field is the applied field, undistorted by the
  carriers. Valid while the space-charge field is small compared to `V/d`; it
  grows with dose per pulse, so this is the first assumption to re-examine at
  very high instantaneous rates.
- **Magnetic fields.** No `v × B` term. Irrelevant outside an MR-linac context.
- **RF microstructure** within a pulse — see §6.
- **Electrode edge effects, guard ring, field non-uniformity.** The chamber
  enters only through gap and bias. The simulated column is interior gas, where
  the field is uniform by construction.
- **Ion species chemistry.** Two lumped carrier species with effective constants,
  not a reaction network; clustering, attachment and ion–molecule reactions are
  folded into the Kanai effective values.
- **Energy loss along the track.** Constant LET across the gap — see §2.
- **Pulse-to-pulse accumulation** at the default `n_pulses=1` — see §6.

---

## 14. Known systematics and convergence

**Finite column radius is the dominant one, and it does not converge quickly.**
`k_s` rises with `sampled_radius_cm` throughout the accessible range:

| `sampled_radius_cm` | k_s | shortfall vs. an infinite column |
|---|---|---|
| 0.003 (30 µm) | 1.0580 | 4.9 % |
| 0.008 (80 µm) | 1.0929 | 1.7 % |
| 0.014 (140 µm) | 1.1011 | 1.0 % |
| 0.018 (180 µm) | 1.1035 | 0.8 % |
| 0.265 (2.65 mm, the full electrode) | 1.1111 | 0.1 % |

**Why.** A track near the rim of the sampled disc has neighbouring tracks on one
side only, so the local charge density — and hence `α n₊ n₋` — is genuinely
lower there. This is a property of sampling a finite disc from a uniformly
irradiated plane, not an artifact: it survives every combination of boundary
condition (§10) and scoring region (§11). The affected material is an annulus a
few diffusion lengths wide, so the deficit is a **perimeter-to-area** effect and
falls as `1/r`:

```
k_s(r) = k_∞ − A/r        k_∞ = 1.1119        A = 1.512 µm
```

for the AIC-144 Markus scenario. Fitting `k_∞` and `A` on the 140 µm and 180 µm
points alone predicts the *measured* 2.65 mm full-electrode result to
**3e-4 in k_s** — a 15× extrapolation in radius — and reproduces the 80 µm
point to 1e-4. Below ~80 µm the column is only a few track radii across and
other effects enter; the law should not be trusted there.

**How to use this.** Do not chase convergence by enlarging the column: the cost
grows as `r²` while the residual bias only falls as `1/r`, so buying the last
percent costs two orders of magnitude in time. Instead **run a cheap column and
correct**. An 80 µm column (1.8 s) plus `A/r` gives 1.1118 against the
full-electrode 1.1111 — closer than the 2.65 mm run itself, which is still
0.1 % low. Quote `k_∞`, and quote the raw value and the correction alongside it.

Both `A` and `k_∞` are scenario-specific. Re-fit them for a new beam, gap or
dose rate by running two radii in the 100–200 µm range and solving the two
equations; that costs about a minute.

**The structural fix**, not currently implemented, is a periodic lateral
boundary with tracks filling the whole periodic cell. That removes the rim
entirely — every track then has a full complement of neighbours — so a small
column would give `k_∞` directly with no extrapolation and no residual bias.

**Track-core resolution** is the next systematic down, and it is small.
Halving `grid_size_um` at fixed column radius moves `k_s` by −0.11 % (10 µm →
5 µm) and −0.27 % (20 µm → 10 µm), consistent with a second-order spatial
error; Richardson extrapolation puts the discretisation error at the default
10 µm at about **0.15 %**. It costs ~32× to check, since halving `h` quadruples
the transverse grid, doubles the axial one and halves `dt` through the von
Neumann condition.

**Statistical noise.** Track positions and arrival times are random; `seed` fixes
them. Repeat runs with different seeds to size the scatter before attributing a
small difference to a physics change.

---

## 15. Validation

- **`tests/test_single_track_vs_jaffe.py`** — in the single-track, low-dose
  limit the solver must reproduce analytic Jaffe theory for initial
  recombination. This is the strongest available independent check: it tests the
  track deposition, the transport and the recombination term together against a
  closed-form result derived under the same Gaussian-track assumption.
- **`tests/test_backends_agree.py`** — the two backends must agree to 1e-9 on
  `f(t)`, `k_s`, the final density field and the tracks drawn. They differ in
  loop structure, in parallelism and in when the z-broadcast happens, so this
  is a genuine cross-implementation check and an optimisation cannot silently
  change the physics.
- **`theory.py`** also provides Boag theory for uniform charge density. It does
  *not* strictly apply to track-structured beams and is included only as a
  rough cross-check in the general-recombination regime.
