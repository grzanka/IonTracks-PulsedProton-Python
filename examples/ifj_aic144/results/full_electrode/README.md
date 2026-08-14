# Full Classic Markus electrode — run record

The AIC-144 Markus 2 mm macropulse scenario at the full PTW 23343 collecting
electrode (r = 2.65 mm), reproduced with:

```bash
python examples/ifj_aic144/report.py full_electrode examples/ifj_aic144/results/full_electrode
```

| | |
|---|---|
| grid | 536 × 536 × 210 = 60.3 M voxels, 10 µm |
| tracks per pulse | 24 630 400 |
| time step | 304.3 ns (von Neumann limit, Courant 0.959) |
| steps | 2 194 = 540 µs injection + 127.5 µs clearance |
| wall time | 843 s (14.05 min), single thread |
| peak RSS | ~2.0 GiB |
| injected (each sign) | 1.7826e9 ion pairs |
| recombined | 1.7819e8 ion pairs (10.00 %) |
| **f** | **0.900037** |
| **k_s = 1/f** | **1.111065** |
| k_s, edge-corrected | 1.111636 |

## Files

| file | contents |
|---|---|
| `collected_charge.csv` | per time step: `time`, `n_positive`, `n_negative`, `injected_positive`, `injected_negative`, `recombination`, all in ion pairs |
| `injection_rate.png` | beam arrival vs time |
| `carrier_evolution.png` | carriers present vs time, both species |
| `recombination_rate.png` | pairs lost per time step |
| `track_density_cross_section.png` | where the track centres landed |

`track_density_xy.npy` is not kept here; it is regenerable from `seed` and
costs about a minute to replay.

## What the record shows

- **Injection** is flat at 3.30 M ion pairs/µs for exactly 540 µs, then zero.
  Integrated, 1.783e9 pairs — 99.6 % of the analytic
  `N_tracks × LET/W × gap`, the remainder being Gaussian tails outside the
  scored disc.
- **Carrier populations** plateau at 144 M (positive) and 92.5 M (negative), a
  ratio of 1.557 against the 1.544 predicted by µ₋/µ₊: in steady state the
  population is injection rate × transit time, and transit time goes as 1/µ.
  Each species fills on its own transit time (98.0 µs and 63.5 µs) and clears
  on it again after the pulse — an independent check that the two-species
  transport behaves as specified.
- **Recombination** tracks the carrier population, plateauing once both clouds
  are full and falling away as they clear.
- **Track density** is uniform at 112 M cm⁻² out to a sharp edge at the scored
  radius, matching `N_tracks / πr²`.
