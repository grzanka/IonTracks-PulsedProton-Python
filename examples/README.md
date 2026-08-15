# Examples

| | |
|---|---|
| `ifj_aic144/` | A real measurement campaign — the AIC-144 cyclotron at IFJ PAN with a PTW Markus 23343 chamber. Start here: how a concrete scenario is set up, run and reported. |

## `ifj_aic144/`

Two entry points, both selecting how much of the chamber to simulate via a
tier name (`dev`, `archive`, `standard`, `wide`, `full_electrode` — 0.2 s to
several minutes; see [`ifj_aic144/README.md`](ifj_aic144/README.md) for the
scenario parameters and per-tier results):

**Step 1 — `run_markus_2mm.py`: check a tier cheaply, terminal output only.**
Prints the config summary and `k_s`; writes nothing to disk unless `--json`
is given. Use this to explore tiers, or to size a big one with `--dry-run` /
`--estimate-runtime-seconds` before committing to it (below).

```bash
python examples/ifj_aic144/run_markus_2mm.py dev                     # smoke test, ~0.2 s, prints only
python examples/ifj_aic144/run_markus_2mm.py archive                 # prints k_s, ~2 s
python examples/ifj_aic144/run_markus_2mm.py archive --json out.json # same, also writes out.json
```

**Step 2 — `report.py`: the full run, with CSV and plots on disk.** Runs the
same simulation in full (it does not reuse Step 1's run — Step 1 is for
sizing/checking first, this is the one that produces files), always
single-threaded, and writes, to `out/ifj_aic144/markus_2mm_<tier>/` by
default:

| file | contents |
|---|---|
| `collected_charge.csv` | per time step: time, n_positive, n_negative, injected_positive, injected_negative, recombination |
| `track_density_xy.npy` | 2D array, tracks landed per xy voxel — the data behind `track_density_cross_section.png` |
| `injection_rate.png` | beam arrival rate vs time |
| `carrier_evolution.png` | carriers present vs time (positive/negative) |
| `recombination_rate.png` | ion pairs lost per time step |
| `track_density_cross_section.png` | where the tracks landed, in the xy plane |

```bash
python examples/ifj_aic144/report.py archive                 # k_s + CSV + four plots, ~2 s
python examples/ifj_aic144/report.py full_electrode          # same, full chamber radius, ~9-13 min on a laptop
```

`report.py` deliberately has no `--threads` flag — plotting itself is cheap
and single-threaded regardless, and a threading option on a script named
"report" would misleadingly suggest the *reporting* needs cores. What
actually benefits from threads is the simulation `report.py` runs internally
before it can plot. To get that speed-up (2 threads is the laptop sweet
spot: ~354 s vs. ~562 s single-threaded for `full_electrode`, same `k_s` to
six digits — see [`docs/BENCHMARKS-LAPTOP.md`](../docs/BENCHMARKS-LAPTOP.md)
sec. 4) while still getting plots, call the same three functions `report.py`
calls, yourself, with a thread count:

```python
from pulsed_ion_chamber.output import write_collected_charge_csv
from pulsed_ion_chamber.plots import save_diagnostic_plots
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel, warmup_parallel
from run_markus_2mm import build_config  # examples/ifj_aic144/run_markus_2mm.py

config = build_config("full_electrode")
warmup_parallel()
result = run_simulation_numba_parallel(config, progress=True, num_threads=2)
write_collected_charge_csv(result, "out/my_run/collected_charge.csv")
save_diagnostic_plots(result, "out/my_run")
```

See `results/full_electrode/` for a committed `report.py` run record, and
[`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md) sec. 7 for what
`--dry-run` / `--estimate-runtime-seconds` on `run_markus_2mm.py` report.
