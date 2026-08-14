# IFJ PAN AIC-144 — Markus 2 mm, macropulse

Recombination in a PTW Markus 23343 plane-parallel ionisation chamber on the
AIC-144 isochronous cyclotron proton beam at IFJ PAN.

```bash
python examples/ifj_aic144/run_markus_2mm.py [dev|archive|standard|wide|full_electrode]
python examples/ifj_aic144/run_markus_2mm.py full_electrode --threads 24   # batched backend
```

`--threads N` switches to the batched, multi-core backend and is what makes the
`full_electrode` tier affordable. On a Cyfronet Helios node it must be launched
as its own `srun` step; see [`docs/HELIOS.md`](../../docs/HELIOS.md) for that
and for how many cores to ask for.

For what each assumption means and why it is made, see
[`docs/PHYSICS.md`](../../docs/PHYSICS.md); for timings and scaling,
[`docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md).

## Scenario

| | value | set by |
|---|---|---|
| particle | 1H, **56.2 MeV** (60 MeV nominal, degraded at the measurement plane) | `E_MeV_u` |
| LET in air (PSTAR) | 1.1994844e-3 keV/µm | derived |
| Gaussian track radius | 20 µm (σ = 14.14 µm) | derived, at the fit floor |
| electrode gap | **2.0 mm** | `electrode_gap_cm` |
| bias | **300 V** → 150 kV/m | `voltage_V` |
| macro-pulse | **540 µs**, repeating every 20 ms → duty cycle 1/37 | `pulse_duration_s`, `repetition_rate_hz` |
| pulses simulated | 1 | `n_pulses` |
| mean dose rate | 10 Gy/s to water × 0.891 → **8.91 Gy/s to air**, 0.1782 Gy per pulse | `dose_rate_Gy_s` |
| air | dry, **20 °C**, 101.325 kPa → 1.2041 kg/m³ | `air_density_kg_m3` |
| W | 33 eV/ion pair | `W_eV` |
| carriers | two Kanai species resolved separately | `mu_*`, `D_*` |
| chamber wall | reflecting (zero-flux) | `lateral_boundary` |
| collection tail | 2.6 × half-gap transit of the slowest carrier = 127.5 µs | `n_clearance_separation_times` |
| deposition cutoff | 10 σ | `track_cutoff_sigmas` |
| grid | 10 µm voxels, `buffer_radius=3`, `no_z_electrode=5` | `grid_size_um` |
| time step | 3.043e-7 s (von Neumann limited, set by the negative ion) | derived |
| run length | 540 µs injection + 127.5 µs clearance = 667.7 µs, 2194 steps | derived |

Only the gap and the bias enter the chamber model — the 5.3 mm electrode
diameter, the guard ring and edge effects are not represented, and the field is
uniform and normal to the electrodes.

## Grid tiers and results

| tier | `sampled_radius_cm` | grid | tracks/pulse | wall time | k_s = 1/f | corrected |
|---|---|---|---|---|---|---|
| `dev` | 0.003 (30 µm) | 12²×210 | 3 157 | 0.2 s | 1.0580 | 1.1084 |
| `archive` | 0.008 (80 µm) | 22²×210 | 22 447 | 1.8 s | 1.0929 | 1.1118 |
| `standard` | 0.014 (140 µm) | 34²×210 | 68 744 | 8.8 s | 1.1011 | 1.1119 |
| `wide` | 0.018 (180 µm) | 42²×210 | 113 638 | 14.5 s | 1.1035 | 1.1119 |
| `full_electrode` | 0.265 (2.65 mm) | 536²×210 | 24 630 400 | 12.8 min¹ | 1.1111 | 1.1117 |

¹ single core on a laptop; 680 s on one Helios core. The same run takes 77 s
on 96 Helios cores, with `k_s` identical to six digits — see
[`docs/HELIOS.md`](../../docs/HELIOS.md). The full-electrode run record, with
its collected-charge curve, is archived in
[`results/full_electrode/`](results/full_electrode/README.md).

**No tier is converged in column radius.** A track at the rim of the sampled
disc has neighbours on one side only, so `k_s` is biased low by a
perimeter-to-area effect that falls only as `1/r`:

```
k_s(r) = 1.1119 − 1.512 µm / r
```

The "corrected" column applies it. Enlarging the column is the wrong way to
chase the residual — cost grows as `r²` while the bias falls as `1/r`, so the
80 µm run plus the correction is both faster and closer to the limit than the
full electrode. **Quote k_∞ = 1.1119** for this scenario, with the raw value and
correction stated alongside. See `docs/PHYSICS.md` §14; the full-electrode run
exists to verify the extrapolation, which it does across a 15× range in radius.

## Comparing against other IonTracks results

Published IonTracks v2 (FEniCSx) results for this case quote **k_s = 1.1629**.
Two things must be reconciled before that number and the ones above are
comparable:

1. **Areal track density.** Those runs count tracks over the full 0.12 mm
   column but place them within 0.7 of that radius, giving 2.04× the areal
   density implied by the quoted dose. Reproduce it with
   `chamber_fill_fraction=0.7`.
2. **Convention.** `k_s = 1/f` here. The first-order form
   `1 + N_recombined/N_injected` gives 1.1400 for the same v2 run — the two
   diverge as recombination grows.

Smaller contributors, all now explicit config fields: the air reference
condition (1.2041 vs 1.293 kg/m³, 1.7 % on track count), `W` (33 vs 34.2 eV,
3.6 % on charge per track), and the scoring region (0.8 %, `scoring_region`).
