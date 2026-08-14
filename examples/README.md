# Examples

| | |
|---|---|
| `run_pulsed_proton_beam.py` | Generic demonstration: a FLASH-like 150 MeV pulsed beam, plots `f(t)`, cross-checks the single-track limit against Jaffe theory, and estimates what a larger grid would cost. Start here to see the machinery work. |
| `ifj_aic144/` | A real measurement campaign — the AIC-144 cyclotron at IFJ PAN with a PTW Markus 23343 chamber. Start here for how a concrete scenario is set up, run and reported. |

## `ifj_aic144/`

```bash
python examples/ifj_aic144/run_markus_2mm.py archive   # just k_s
python examples/ifj_aic144/report.py archive           # k_s + CSV + four plots
```

Both take a tier name (`dev`, `archive`, `standard`, `wide`, `full_electrode`)
selecting how much of the chamber is simulated — 0.2 s to 9 minutes. See
[`ifj_aic144/README.md`](ifj_aic144/README.md) for the scenario parameters and
results, and `results/full_electrode/` for a committed run record.
