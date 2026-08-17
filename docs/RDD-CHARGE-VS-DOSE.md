# A radial dose distribution is not a radial charge distribution

**The question.** `pulsed_ion_chamber` transports ion pairs. A radial dose
distribution gives energy. The conversion currently used everywhere in this code
is `n_pairs(r) = ρ D(r) / W`, applied voxel by voxel. This document asks whether
that is right, reading Cucinotta, Katz, Wilson & Dubey (1996) —
[`Cucinotta.pdf`](Cucinotta.pdf) — against the libamtrack tabulation in
[`../examples/fe90_air/data/`](../examples/fe90_air/data/).

**The answer.** It is not right, and the error is concentrated exactly where it
does the most damage. About a third of the tabulated dose sits in a core that is
*molecular excitation*, which creates no ion pair at all — and it sits at the
density peak, where recombination (`∝ n²`) is most sensitive. Correcting it also
removes the reason the `h`-ladder in
[`../examples/fe90_air/README.md`](../examples/fe90_air/README.md) §7 fails to
converge: rebuilt as a charge distribution, the same ladder settles to four
significant figures (§4.4) where the dose version was still climbing.

Companion documents: [`CUCINOTTA-RDD-STUDY.md`](CUCINOTTA-RDD-STUDY.md) is the
earlier feasibility study for the AIC-144 *proton* multi-track case — grids,
memory, track overlap — and its conclusions stand; this one is about what the
distribution *means*, which that study did not address.

---

## 1. What Cucinotta's RDD actually is

The paper is a **δ-ray theory**. Its Eq. (13) builds the radial dose by
integrating, over the secondary-electron spectrum `dnᵢ/dω`, the energy each
ejected electron deposits as it slows:

```
E(t) = (−1/2πt) Σᵢ ∫ dω  ∂/∂t [ η(t,ω) W(t,ω) ]  dnᵢ/dω
```

Everything in it is secondary electrons: `dn/dω` is the ejection spectrum
(Rudd's model for protons, scaled to heavy ions by effective charge, Eq. 8),
`W(r,t) = ω(r−t)` is the residual energy after penetrating `t`, and `η` is the
transmission factor. Energy the primary ion deposits *without* ejecting a free
electron is not in this integral.

The paper says so, twice, and both statements are quantitative.

**Page 256, on the core:**

> Close to the ion track (t < 10 nm) a contribution to the radial dose from
> molecular excitations […] is expected. **It is important to keep the
> contributions from excitation and ionizations distinct, since it is the
> secondary electron dose which is assumed to be responsible for most physical
> effects by heavy ions.**

In figures 7–14 the excitation contribution is drawn as a *separate* dash-dot
curve, always above the δ-ray curves, and always only at small `t`. Figure 13 is
90 MeV/amu ⁵⁶Fe — our exact projectile, in water.

**Page 263, on how much of the LET it is:**

> The calculations performed with the present model find about **55–70 percent
> of the linear energy transfer to be due to the secondary electrons** with only
> a small variation with ion velocity, except below 1 MeV/amu.

So **30–45 % of the LET is not δ-ray dose at all.** It is the primary ion's own
close-collision and excitation energy, deposited within nanometres of its path.

## 2. The tabulation, dissected

The libamtrack table for 90 MeV/u Fe in air splits cleanly in two. Using
`r²D(r)`, which is flat wherever `D ∝ 1/r²`:

| `r` | `r²D` | vs the plateau |
|---|---|---|
| 10 nm | 8.494e-8 | **×16.2** |
| 30 nm | 8.392e-9 | ×1.60 |
| 100 nm | 5.221e-9 | ×0.99 |
| 1 µm → 10 mm | 5.25e-9 | ×1.00 |

**The penumbra is an exact `1/r²` over 5.7 decades**, 100 nm to 10 cm, flat in
`r²D` to three significant figures. Because `dE/d(ln r) = 2πρ r²D` is then
constant, it carries **10.31 % of the LET per decade of radius** — the fact
behind every "the track is not local" statement in the Fe-90 study.

**The core is a distinct, superimposed term.** Energy per decade, actual against
what the `1/r²` continuation alone would give:

| decade | actual | if pure `1/r²` | enhancement |
|---|---|---|---|
| **10–100 nm** | **39.14 %** | 10.31 % | **×3.80** |
| 100 nm–1 µm | 10.29 % | 10.31 % | ×1.00 |
| 1–10 µm | 10.31 % | 10.31 % | ×1.00 |
| 10–100 µm | 10.31 % | 10.31 % | ×1.00 |

Subtracting the penumbra's share, the core carries **33.3 % of the LET** and
lives between 10 and 30 nm.

**33.3 % is inside Cucinotta's 30–45 % non-δ-ray band.** The tabulation is not a
renormalised δ-ray profile — it is a δ-ray penumbra *plus an explicit core term
carrying almost exactly the excitation share the paper measures*. The two
independent numbers agree, which is what makes the rest of this document a
decomposition rather than a guess.

## 3. Do excitations and ionisations make charge pairs the same way?

**No.** They are different processes with different products:

| | product | free charge? |
|---|---|---|
| **ionisation** | ion⁺ + free electron | **yes** — one pair, immediately |
| **excitation** | bound excited molecule | **no** — relaxes by photon emission, dissociation or collisional quenching |

In air a small fraction of excited N₂ can Penning-ionise O₂, but that is a
minor and slow channel, not a route to prompt charge in the track core.

### 3.1 The W-value is what makes this subtle

`W = 34.2 eV` per ion pair in air, against a mean ionisation potential of
~15 eV. The factor of ~2.3 between them **is** the excitation and
sub-excitation-electron energy: `W` is defined as *total* energy expended per
ion pair formed, averaged over the whole slowing-down cascade.

That has a precise consequence:

- `n = ρD/W` is correct **if and only if `D` is a total dose whose local
  composition matches the cascade average `W` was measured for.**
- In the **penumbra** that holds. The energy there is deposited by electrons
  slowing down — exactly the cascade `W` averages over. Pairs track dose. ✓
- In the **core** it fails. That energy is excitation by the primary ion's
  distant-collision field. Dividing it by `W` manufactures ion pairs that were
  never created — **33 % of the track's charge, placed at the density peak.**

Recombination goes as `n₊n₋`, so an error at the peak is the worst-placed error
available. It is also one-signed: it can only *overestimate* the loss.

### 3.2 The fix is one number

Every ion pair is made in the δ-ray cascade — in Cucinotta's Eq. (9) the
ejected electron's energy `ω = W + I` already carries the binding energy of the
ionisation that created it, so the primary ionisations are counted inside the
δ-ray term. So keep the δ-ray dose and convert it with an effective

```
W_δ = W × (δ-ray LET / total LET) ≈ 34.2 × 0.67 ≈ 23 eV
```

Total pair count unchanged; radial distribution corrected. Nothing else in the
solver has to move.

### 3.3 A second asymmetry, not quantified here

Speculative, but it points the same way. Each *primary* ionisation leaves its
positive ion at the impact parameter — the adiabatic radius, computed for this
projectile as

| medium | mean excitation energy | `b_ad = γβc/ω` |
|---|---|---|
| air | 85.7 eV | **1.04 nm** |
| water | 75 eV | 1.18 nm |

— while its electron carries the energy outward and thermalises far away. The
positive charge from primary ionisations therefore starts *on the axis*, and the
matching negative charge starts spread over the penumbra. Only the *secondary*
ionisations along δ-ray tracks are co-located.

In air roughly a third of all pairs come from primary ionisations, so
`n₊(r) ≠ n₋(r)` at `t = 0` — and this code assumes they are equal. Their overlap
integral `∫n₊n₋` is smaller than either `∫n²`, so **this error too makes the
current model over-predict initial recombination.** Quantifying it needs the
primary-ionisation spectrum and is not attempted here.

Note also that `b_ad` barely changes between air and water: the core is set by
impact-parameter physics, not by density. That is the same conclusion
[`CUCINOTTA-RDD-STUDY.md`](CUCINOTTA-RDD-STUDY.md) §1.3 reached from the
tabulations, and the reason an air RDD must never be rescaled from a water one.

## 4. Which RDD model should IonTracks use?

Ranked, worst to best.

### 4.1 Worst: the raw dose-RDD, as currently implemented

What `pulsed_ion_chamber/rdd.py` does today. Three defects, in order:

1. Converts the 33 % excitation core into charge that does not exist, at the
   density peak (§3.1).
2. **Has no `h → 0` limit.** The dose keeps rising into the core with no
   physical cutoff, so `k_s` creeps up with every refinement — measured, and
   still rising at 1.25 µm (Fe-90 README §7.1).
3. Not separable and not truncated, so deposition is a full-grid pass per
   track. Fine for one ion; unusable for the millions the IFJ campaign needs.

Its virtue is that it is faithful to the tabulation, which makes it the right
*reference* to measure the alternatives against.

### 4.2 Cheapest real improvement: keep the Gaussian, fix its radius

The Gaussian is separable and truncatable — the two properties that make
multi-track deposition affordable (`docs/ALGORITHM.md`), and they are exactly
what a tabulated profile cannot offer. The problem is not the shape, it is that
`b` comes from the Rossomme LET fit, a proton/light-ion parameterisation that
returns **49.4 µm** for iron with no physical claim behind it.

Replace `b` with a *physically derived* charge-cloud radius (§4.3) and the
existing hot loops are untouched. For the IFJ multi-track case this is very
likely enough, and it costs one function.

### 4.3 Best: a two-component charge-RDD

Structurally Chatterjee–Schaefer (1976), which the paper cites as ref. 18 —
core plus penumbra — but built for **charge** rather than dose:

| zone | radius | content |
|---|---|---|
| **core** | `r < r₀` | uniform (or Gaussian) disc; the δ-ray share only, converted at `W_δ` |
| **penumbra** | `r₀ < r < R_δ` | exact `K/r²`, 10.31 % of LET per decade |
| **halo** | `r > R_δ` (10 cm) | outside any grid; collected in full, enters only through `rdd.chamber_ks` |

Three things make this the right structure:

**It is analytic where it matters.** `∫ 2πρ (K/r²) r dr = 2πρK ln(r₂/r₁)`, so
the energy in any annulus, and the area-average over any voxel ring, is closed
form. No tabulation, no quadrature, no interpolation error.

**`r₀` is physical, so the ladder converges.** This is the whole point. A
profile with a finite inner cutoff has a finite peak density, so `k_s` settles
once `h < r₀` — and `r₀` is a property of the gas, not of the grid.

**In a gas `r₀` is micrometres, not nanometres.** This is the step that is easy
to get wrong. The *dose* core is at 1–30 nm (§2, §3.3), but charge is not
created where the energy is: sub-excitation electrons in air at 1 atm, below the
~7 eV N₂/O₂ excitation threshold, lose energy only through elastic and
rotational collisions and travel of order **1–10 µm** before thermalising and
attaching to O₂. That thermalisation distance — not the adiabatic radius — is
the smallest scale at which a charge cloud exists in air.

If that is right, then the 5 µm on-axis spacing used by both codes is not
under-resolving the physics: it is sitting *at* the physical smearing scale, and
the Fe-90 README §7 conclusion — that refining to 1.25 µm and beyond chases
model artefacts — has a physical explanation rather than only an empirical one.

**This number needs a literature value before anything is built on it.** It is
the one quantity here taken from general gas-detector reasoning rather than from
the paper or the tabulation.

## 4.4 Measured: the charge-RDD converges, the dose-RDD does not

Built by [`../examples/fe90_air/charge_rdd.py`](../examples/fe90_air/charge_rdd.py),
which is where every number below comes from.

**The construction validates itself.** §3.2 predicted `W_δ ≈ 23 eV` from
Cucinotta's 55–70 % δ-ray share. Rebuilding the profile — delete the excitation
core, redistribute the sub-`r₀` energy rather than discard it — and measuring
the share directly:

| `r₀` | charge-RDD LET | share of total | `W_δ` |
|---|---|---|---|
| 1–20 µm | 0.401–0.404 keV/µm | **71.2–71.6 %** | **24.4 eV** |

24.4 eV against a predicted 23 eV, and a 71 % δ-ray share against a predicted
67 %. The paper's independently measured share and the tabulation's core term
agree to a few percent.

**The ladder settles.** R = 120 µm, one ion, `r₀` = 5 µm, two cores:

| `h` [µm] | core `n₀` [cm⁻³] | `k_s` (chamber) | increment | ratio | dose-RDD, same grid |
|---|---|---|---|---|---|
| 10 | 6.583e10 ᵃ | 1.173117 ᵃ | | | 1.322332 |
| 5 | 8.129e10 | 1.137180 | | | 1.386873 |
| 2.5 | 8.181e10 | 1.142586 | +0.005406 | | 1.438139 |
| 1.25 | 8.375e10 | **1.143289** | **+0.000703** | **0.130** | 1.481347 |

ᵃ Not a valid rung: one voxel is wider than the core, so `r₀` is unresolved.

Read the last two columns against each other, because this is the whole
argument:

- **charge-RDD**: increments +0.005406, +0.000703 — a ratio of **0.130**,
  *falling*. Geometric extrapolation gives a limit of **1.1434**, which the
  1.25 µm rung already reaches to four significant figures.
- **dose-RDD**: increments +0.064541, +0.051266, +0.043208 — ratios 0.794,
  0.843, *rising*, extrapolating to ≥ 1.71 and possibly nowhere at all
  (Fe-90 README §7.1).

The peak density says the same thing more directly. Across a 4× refinement the
charge-RDD's `n₀` moves **3 %** (8.13e10 → 8.38e10) while the dose-RDD's moves
**14×** (3.56e11 → 5.05e12). A finite inner cutoff gives a finite peak density,
so refining past `r₀` stops changing the answer. That is the property the whole
construction exists to buy, and it is now measured rather than argued.

**Converged value: `k_s`(chamber) = 1.1434 at `r₀` = 5 µm**, against 1.481 and
climbing for the dose-RDD at the same spacing.

**`r₀` now carries the uncertainty, and that is the right place for it.** At
h = 2.5 µm:

| `r₀` [µm] | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| `k_s` (chamber) | 1.2116 | 1.1773 | 1.1426 | 1.1055 | 1.0634 |

`k_s − 1` varies 3.3× across 1–20 µm. The model is only as good as `r₀` — but
`r₀` is a measurable property of air, not a discretisation artefact, which is
exactly the trade §4.3 was after. Note that *every* row is far below the
dose-RDD's 1.481 at h = 1.25 µm and its extrapolated ≥ 1.71: removing the
excitation core moves the answer substantially, in the direction §3.1 predicted.

**`r₀` is now the only free parameter.** With `h` no longer able to move the
answer, the model reduces to a single measurable input — which is what §4.3 was
after and what §6 now turns on.

## 5. Why the multi-track case is the *easy* one

Counter-intuitively, the IFJ campaign is less exposed to all of this than the
single Fe ion is.

[`CUCINOTTA-RDD-STUDY.md`](CUCINOTTA-RDD-STUDY.md) §3 measured the mean
track-to-track spacing at 10 Gy/s as **0.946 µm**, at which radius only 42 % of
a track's energy has been laid down. So the cores **overlap**: the sub-`r₀`
structure that makes a single track's `k_s` grid-dependent is averaged over many
tracks before it can matter, and the outer `1/r²` tails superpose into a smooth
background rather than a structure.

Two practical consequences:

- The core treatment matters most where there is **one** track — initial
  recombination of a single heavy ion. That is the Fe-90 case, and it is the
  hard one.
- For the proton campaign, the largest RDD-related effect is not the core shape
  at all but **normalisation**: 30 % of a track's energy is deposited outside
  the chamber (`CUCINOTTA-RDD-STUDY.md` §2a, Fe-90 README §4.2), and whether the
  campaign's `W` calibration already absorbs that is a paper question, not a
  computational one.

## 6. Recommended order

1. ~~Measure the charge-RDD against the dose-RDD on the Fe-90 ladder~~ — done,
   §4.4. It converges to `k_s`(chamber) = 1.1434 at `r₀` = 5 µm; the dose-RDD
   does not converge at all.
2. **Find the electron thermalisation distance in air at 1 atm** from the
   literature (§4.3). This is now the *only* thing between the model and a
   number: §4.4 shows `r₀` moves `k_s − 1` by 3.3× across 1–20 µm, and needs no
   code to settle.
3. **Derive the Gaussian `b` from `r₀`** (§4.2) and check that
   the cheap model reproduces the two-component one for a single ion. If it
   does, the multi-track campaign needs no new deposition code at all.
4. Only then consider implementing the full two-component profile in the hot
   loops.

## 7. What this document does not establish

- The 55–70 % δ-ray share is Cucinotta's calculation for **water**; the
  agreement with the 33.3 % core in the **air** tabulation is a consistency
  check, not a derivation. libamtrack's actual core term has not been read.
- The thermalisation-distance argument of §4.3 is reasoning from gas-detector
  practice, not a cited measurement. §4.4 shows the answer depends on it
  strongly, so this is the load-bearing gap.
- The `n₊ ≠ n₋` asymmetry of §3.3 is argued, not quantified. The "roughly a
  third of pairs are primary" figure is a rule of thumb for air, not a
  calculation for this projectile.
- `W_δ`: predicted 23 eV, measured 24.4 eV from the rebuilt profile (§4.4).
  Both rest on the same 55-70 % share, so the agreement is a consistency check
  on the construction, not an independent determination of a δ-ray-only
  W-value.
- Everything is for 90 MeV/u Fe in air. The δ-ray share varies little with ion
  velocity above 1 MeV/u per the paper, but the core/penumbra *balance* does.
