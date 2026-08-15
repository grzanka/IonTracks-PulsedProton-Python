# Examples

| | |
|---|---|
| `ifj_aic144/` | A real measurement campaign — the AIC-144 cyclotron at IFJ PAN with a PTW Markus 23343 chamber. Start here: how a concrete scenario is set up, run and reported. |

## `ifj_aic144/`

Two layers, cleanly split: `run_markus_2mm.py` runs the simulation (with the
performance and sizing flags that involves), `plot.py` only draws figures
from what was saved — it never touches the solver and has no thread count of
its own. Both take a tier name (`dev`, `archive`, `standard`, `wide`,
`full_electrode` — 0.2 s to several minutes; see
[`ifj_aic144/README.md`](ifj_aic144/README.md) for the scenario parameters
and per-tier results) where relevant.

**Layer 1 — `run_markus_2mm.py`: run the simulation.** Always prints the
config summary and `k_s`. Writes nothing to disk by default; two flags each
save something different:

```bash
python examples/ifj_aic144/run_markus_2mm.py dev  # prints only, ~0.2 s
```

```bash
python examples/ifj_aic144/run_markus_2mm.py archive --json out.json  # + a metrics-only JSON
```

```bash
python examples/ifj_aic144/run_markus_2mm.py archive --save-run out/my_run  # + the full record, for plotting
```

```bash
python examples/ifj_aic144/run_markus_2mm.py full_electrode --threads 2 --save-run out/full  # same, faster
```

`--json FILE` writes one lightweight JSON (tier, timing, peak RSS, `k_s`) —
what a Slurm job array or a scaling sweep collects; see
[`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md) sec. 7.

`--save-run DIR` writes what **Layer 2** needs and nothing else:

| file | contents |
|---|---|
| `collected_charge.csv` | per time step: time, n_positive, n_negative, injected_positive, injected_negative, recombination |
| `track_density_xy.npy` | 2D array, tracks landed per xy voxel |
| `run_meta.json` | the handful of config scalars the plots read, plus `f`, `ks` and charge totals |

`--threads N` (default 1, batched backend once N > 1) is what makes
`full_electrode` affordable here — 2 threads is the laptop sweet spot
(~354 s vs. ~562 s single-threaded, same `k_s` to six digits — see
[`docs/BENCHMARKS-LAPTOP.md`](../docs/BENCHMARKS-LAPTOP.md) sec. 4). It lives
on this script because it's the one that runs the solver; `plot.py` below has
no such flag because it never does.

`--dry-run` and `--estimate-runtime-seconds N` size a bigger tier before
committing to it — see [`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md) sec. 7.

**Layer 2 — `plot.py`: draw the figures.** Reads a `--save-run` directory —
no solver import, no Numba, no `--threads` — and writes, back into that same
directory by default:

| file | contents |
|---|---|
| `injection_rate.png` | beam arrival rate vs time |
| `carrier_evolution.png` | carriers present vs time (positive/negative) |
| `recombination_rate.png` | ion pairs lost per time step |
| `track_density_cross_section.png` | where the tracks landed, in the xy plane |

```bash
python examples/ifj_aic144/plot.py out/my_run  # figures land in out/my_run/
```

```bash
python examples/ifj_aic144/plot.py out/full out/full/figures  # or a separate output_dir
```

Because plotting only reads files, the same `--save-run` directory can be
replotted any number of times, moved to another machine, or handed to
someone who never runs the simulation at all — none of which was possible
while a single script did both jobs. See `results/full_electrode/` for a
committed run record.
