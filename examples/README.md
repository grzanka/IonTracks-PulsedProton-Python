# Examples

| | |
|---|---|
| `ifj_aic144/` | A real measurement campaign — the AIC-144 cyclotron at IFJ PAN with a PTW Markus 23343 chamber. Start here: how a concrete scenario is set up, run and reported. |

## `ifj_aic144/`

```bash
python examples/ifj_aic144/run_markus_2mm.py dev        # smoke test, ~0.2 s
python examples/ifj_aic144/run_markus_2mm.py archive     # just k_s
python examples/ifj_aic144/report.py archive             # k_s + CSV + four plots
```

Both `run_markus_2mm.py` and `report.py` take a tier name (`dev`, `archive`,
`standard`, `wide`, `full_electrode`) selecting how much of the chamber is
simulated — 0.2 s to 9 minutes. `dev` is the fastest way to see the machinery
run end to end; see [`ifj_aic144/README.md`](ifj_aic144/README.md) for the
scenario parameters and results, and `results/full_electrode/` for a
committed run record.

`run_markus_2mm.py --dry-run` and `--estimate-runtime-seconds N` size a
bigger tier before committing to it — see
[`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md) sec. 7.
