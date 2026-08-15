# pulsed-ion-chamber

**How much charge does an ionisation chamber lose to recombination when the
beam arrives in short, intense pulses?**

An ionisation chamber measures dose by collecting the charge a beam liberates
in its gas, and assumes the charge it collects is the charge that was created.
Recombination breaks that assumption: positive and negative ions that meet
before reaching an electrode annihilate and are never counted. The correction
factor is `k_s`, and computing it is what this code does.

It matters most for **pulsed beams**. Recombination goes as the *square* of the
local carrier density, so packing a pulse's worth of dose into 540 µs instead
of spreading it over 20 ms raises the loss by more than an order of magnitude
at the same average dose rate — which is the regime FLASH radiotherapy and
cyclotron-based proton therapy operate in, and where the standard analytic
corrections are least reliable.

The approach is to simulate it directly: solve the coupled
drift–diffusion–recombination equations

```
dn±/dt = D± grad² n±  ∓  µ± E·grad n±  −  alpha n+ n−  +  tracks
```

on a regular voxel grid, entering protons as individual Gaussian ion tracks
rather than as a smooth charge density — because at these dose rates it is the
granularity of individual tracks that drives the recombination.

## Quick start

```bash
python -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install -e ".[dev]"
```

```bash
pytest  # ~20 s
```

```bash
python examples/ifj_aic144/run_markus_2mm.py archive --save-run out/archive  # ~2 s, writes a CSV
```

```bash
python examples/ifj_aic144/plot.py out/archive  # + four plots
```

```python
from pulsed_ion_chamber import SimulationConfig, run_simulation_numba

config = SimulationConfig(
    E_MeV_u=56.2, voltage_V=300.0, electrode_gap_cm=0.2,   # beam and chamber
    pulse_duration_s=540e-6, repetition_rate_hz=50.0,      # pulse structure
    dose_rate_Gy_s=8.91,                                   # time-averaged, to air
    grid_size_um=10.0, sampled_radius_cm=0.008,            # grid
    seed=1,
)
print(config.summary())
result = run_simulation_numba(config)
print(result.ks)      # recombination correction factor
```

## Documentation

Index: [`docs/README.md`](docs/README.md) — what each document answers.

| | |
|---|---|
| [`docs/PHYSICS.md`](docs/PHYSICS.md) | Every physical and numerical assumption, and why it is made. **Start here** for what is modelled. |
| [`docs/ALGORITHM.md`](docs/ALGORITHM.md) | Data layout, the two hot loops, what batching means, where the parallelism is. **Start here** for how. |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | Cost model, measured timings, scaling, using many cores. |
| [`docs/BENCHMARKS-LAPTOP.md`](docs/BENCHMARKS-LAPTOP.md) | Measured wall times on a laptop, and why one thread is the right number there. |
| [`docs/HELIOS.md`](docs/HELIOS.md) | Running on a Cyfronet Helios node: setup, how many cores to ask for, what to expect. |
| [`docs/BENCHMARKS-ARES.md`](docs/BENCHMARKS-ARES.md) | The same on Cyfronet Ares — and why its cores lose to Helios's by 1.7× despite a higher clock. |
| [`pulsed_ion_chamber/README.md`](pulsed_ion_chamber/README.md) | What each module is for. |
| [`examples/README.md`](examples/README.md) | What each example does. |
| [`tests/README.md`](tests/README.md) | What each test file pins down. |
| [`profiling/README.md`](profiling/README.md) | Profiling harness and the raw data it produced. |
| [`profiling/cluster_scaling/README.md`](profiling/cluster_scaling/README.md) | The cluster thread-scaling study (Helios, Ares) — `./submit.sh`. |
| [`profiling/laptop_scaling/README.md`](profiling/laptop_scaling/README.md) | The laptop scaling benchmark — `./bench_laptop.sh`. |

Two results worth knowing before trusting a number:

- **`k_s` is biased low by the finite simulated column**, falling only as
  `1/radius`, and no affordable radius removes it. Correct for it rather than
  enlarging the grid — `docs/PHYSICS.md` §14.
- **Whether threads help depends on whether the grid fits in cache.** Both hot
  loops are memory-bandwidth-bound. A small grid is already saturated on one
  core and gets slower with more; a grid larger than the machine's L3 scales
  well — the full electrode is 572 s on one Helios core and 47 s on 32, same
  `k_s` to six digits. `docs/PERFORMANCE.md` §6, `docs/HELIOS.md`.

## Provenance and scope

Extracted and adapted from the IonTracks family of codes (J.B. Christensen et
al.), which implement the same physics for *continuous* ion beams and for
*pulsed, spatially uniform* electron beams. Neither covers pulsed beams with
resolved proton track structure, which is what this adds. Physical constants
(Kanai et al. 1998), the LET and track-radius tables, and the Jaffe/Boag
analytic references are reused from it.

Deliberately **not** modelled: space-charge screening, magnetic fields, RF
microstructure within a pulse, electrode edge effects and the guard ring. Each
omission, and the condition under which it stops being safe, is listed in
`docs/PHYSICS.md` §13.

## References

- Christensen, J.B., Tölli, H., Bassler, N. (2016). "A general algorithm for
  calculation of recombination losses in ionization chambers exposed to ion
  beams." *Medical Physics* 43(10):5484–92.
- Christensen, J.B. et al. (2020). "Mapping initial and general recombination
  in scanning proton pencil beams." *Phys. Med. Biol.* 65, 115003.
- Kanai, T. et al. (1998). Ion mobility, diffusion and recombination
  coefficients in air.
- Boag, J.W., Currant, J. (1980). "Current collection and ionic recombination
  in small cylindrical ionization chambers exposed to pulsed radiation."
  *Br. J. Radiol.* 53.

**License**: GPLv3, inherited from IonTracks (see `LICENSE`).
