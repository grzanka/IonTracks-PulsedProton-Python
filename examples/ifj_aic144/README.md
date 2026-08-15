# IFJ PAN AIC-144 — Markus 2 mm, macropulse

Recombination in a PTW Markus 23343 plane-parallel ionisation chamber on the
AIC-144 isochronous cyclotron proton beam at IFJ PAN.

Two scripts, cleanly split by whether they touch the solver. Both take a tier
name (`dev`, `archive`, `standard`, `wide`, `full_electrode` — see the results
table below) that sets how much of the chamber is simulated.

### Layer 1: `run_markus_2mm.py` — runs the simulation

Always prints the config summary and `k_s`. Writes nothing to disk unless
asked; three independent flags each save something different.

```bash
python examples/ifj_aic144/run_markus_2mm.py [dev|archive|standard|wide|full_electrode]
python examples/ifj_aic144/run_markus_2mm.py full_electrode --threads 2    # batched backend
python examples/ifj_aic144/run_markus_2mm.py full_electrode --dry-run     # memory only, instant
python examples/ifj_aic144/run_markus_2mm.py full_electrode --threads 2 --estimate-runtime-seconds 5
python examples/ifj_aic144/run_markus_2mm.py archive --json out.json           # metrics-only JSON
python examples/ifj_aic144/run_markus_2mm.py archive --save-run out/my_run     # full record, for plotting
```

`--threads N` switches to the batched, multi-core backend and is what makes the
`full_electrode` tier affordable — 2 threads is the sweet spot on a laptop
(~354 s vs. ~562 s single-threaded, same `k_s` to six digits; see
[`docs/BENCHMARKS-LAPTOP.md`](../../docs/BENCHMARKS-LAPTOP.md) sec. 4). On a
Cyfronet Helios node it must be launched as its own `srun` step, and the
thread count that pays is larger than on a laptop; see
[`docs/HELIOS.md`](../../docs/HELIOS.md) for how many cores to ask for. This
flag lives here and nowhere else in the pipeline, because this is the only
script that runs the solver.

`--dry-run` prints the grid's peak memory allocation against what this machine
actually has free, then exits without allocating anything — the way to find
out a grid is too big for this machine before finding out the hard way. See
[`docs/BENCHMARKS-LAPTOP.md`](../../docs/BENCHMARKS-LAPTOP.md) sec. 3 for how
that memory estimate compares to measured peak RSS.

`--estimate-runtime-seconds N` is the runtime counterpart: it actually
allocates the grid and runs the real backend (respecting `--threads`) for
`~N` seconds, then extrapolates and exits without doing the full run. Slower
than `--dry-run` but far more trustworthy on a large, batched run than a
guess built from isolated single-track timings — see
[`docs/BENCHMARKS-LAPTOP.md`](../../docs/BENCHMARKS-LAPTOP.md) sec. 7 and
[`docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md) sec. 7. `--dry-run` and
`--estimate-runtime-seconds` are mutually exclusive.

`--json FILE` writes one lightweight JSON — tier, timing, peak RSS, `k_s` —
what a Slurm job array or a scaling sweep collects.

`--save-run DIR` writes what Layer 2 needs and nothing else:

| file | contents |
|---|---|
| `collected_charge.csv` | per time step: time, n_positive, n_negative, injected_positive, injected_negative, recombination |
| `track_density_xy.npy` | 2D array, tracks landed per xy voxel |
| `run_meta.json` | the config scalars the plots read, plus `f`, `ks` and charge totals |

### Layer 2: `plot.py` — draws the figures

Reads a `--save-run` directory and writes four PNGs back into it (or a
separate output directory) by default. No solver import, no Numba, no
`--threads` — plotting doesn't run anything that threads could speed up, so
the flag simply isn't there. A run performed once can be replotted any
number of times, on a different machine, without ever repeating the
simulation.

| file | contents |
|---|---|
| `injection_rate.png` | beam arrival rate vs time |
| `carrier_evolution.png` | carriers present vs time (positive/negative) |
| `recombination_rate.png` | ion pairs lost per time step |
| `track_density_cross_section.png` | where the tracks landed, in the xy plane |

```bash
python examples/ifj_aic144/plot.py out/my_run                 # figures land in out/my_run/
python examples/ifj_aic144/plot.py out/full out/full/figures  # or a separate output_dir
```

A committed run record from this pipeline lives in
[`results/full_electrode/`](results/full_electrode/README.md).

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
| `full_electrode` | 0.265 (2.65 mm) | 536²×210 | 24 630 400 | 562 s¹ | 1.1111 | 1.1117 |

¹ single core on a laptop — the fastest of the three machines this code has
been measured on, edging out a Helios core (572 s) and beating an Ares core by
1.7× (see [`docs/BENCHMARKS-LAPTOP.md`](../../docs/BENCHMARKS-LAPTOP.md) §3).
The same run takes 47 s on 32 Helios cores, with `k_s` identical to six
digits — see [`docs/HELIOS.md`](../../docs/HELIOS.md). The full-electrode run
record, with its collected-charge curve, is archived in
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
