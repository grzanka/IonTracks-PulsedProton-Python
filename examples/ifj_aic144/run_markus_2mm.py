#!/usr/bin/env python3
"""IFJ PAN AIC-144: PTW Markus 23343 (2 mm gap), macropulse, 10 Gy/s.

Reproduces the beam and chamber of the archived IonTracks v2 (FEniCSx) run
`ifj_aic144/markus_chambers_20260813/runs/markus_2mm_macropulse_10Gys`, with
the physics this repository ported over from that code: the two Kanai carrier
species resolved separately, W and the air reference density taken from the
campaign, a reflecting (zero-flux) chamber wall, and v2's collection-tail rule.

See examples/ifj_aic144/README.md for the full parameter mapping, the
measured effect of each ported change, and what was deliberately *not*
ported (and why).

Run:  python examples/ifj_aic144/run_markus_2mm.py [dev|archive|converged|production]
"""

import sys
import time

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import (
    AIR_DENSITY_20C_KG_M3,
    ION_DIFFUSION_NEGATIVE_CM2_S,
    ION_DIFFUSION_POSITIVE_CM2_S,
    ION_MOBILITY_NEGATIVE_CM2_VS,
    ION_MOBILITY_POSITIVE_CM2_VS,
)
from pulsed_ion_chamber.solver_numba import run_simulation_numba, warmup

# --- beam and chamber, straight from the archived source_config.yaml ---------
BEAM_KWARGS = dict(
    E_MeV_u=56.2,  # 60 MeV nominal, degraded to 56.2 at the measurement plane
    voltage_V=300.0,
    electrode_gap_cm=0.2,  # PTW Markus 23343 -> E = 1500 V/cm = 150 kV/m
    pulse_duration_s=540e-6,  # macropulse
    repetition_rate_hz=50.0,  # every 20 ms -> duty cycle 1/37
    n_pulses=1,
    rf_frequency_hz=26.26e6,  # diagnostic only; see config.py on RF averaging
    # 10 Gy/s to water x 0.891 (water->air, calibrated for this beam) gives
    # 0.1782 Gy to air per pulse. This is the *nominal* dose: tracks fill the
    # scored column uniformly. The archive instead counted tracks over the full
    # 0.12 mm column while confining them to 0.7 of it, i.e. 2.04x this areal
    # density -- set chamber_fill_fraction=0.7 to reproduce that (README 4.4).
    dose_rate_Gy_s=8.91,
    seed=20260527,  # archive seed (different RNG, so not track-for-track equal)
)

# --- physics ported from IonTracks-FEniCSx (README sections 4.1-4.3, 4.5, 4.11) ---
V2_PHYSICS_KWARGS = dict(
    # Two Kanai species instead of one averaged pair. Sets dt on the negative
    # ion (faster and more diffusive) and the collection tail on the positive.
    mu_positive_cm2_Vs=ION_MOBILITY_POSITIVE_CM2_VS,
    mu_negative_cm2_Vs=ION_MOBILITY_NEGATIVE_CM2_VS,
    D_positive_cm2_s=ION_DIFFUSION_POSITIVE_CM2_S,
    D_negative_cm2_s=ION_DIFFUSION_NEGATIVE_CM2_S,
    W_eV=33.0,  # campaign value (this repo defaults to v1's 34.2)
    air_density_kg_m3=AIR_DENSITY_20C_KG_M3,  # dry air at 20 degC
    # Zero-flux wall: the right model for a column sampled from the interior of
    # a large uniformly irradiated chamber, and it removes v1's frozen
    # never-updated outer ring. Converges at buffer_radius=3 where the
    # absorbing wall needs 4-6.
    lateral_boundary="reflecting",
    # v2 sizes its collection tail as (gap / slowest carrier) x 1.3; here
    # separation_time_steps is the half-gap transit of the slowest carrier, so
    # 2 x 1.3 = 2.6 reproduces it.
    n_clearance_separation_times=2.6,
)

# --- grid tiers (all at grid_size_um=10.0 = the FEniCSx lc of 0.01 mm) -------
# Wall times measured single-threaded with solver_numba on one development
# machine; k_s from those same runs.
GRID_TIERS = {
    # name:          (sampled_radius_cm, buffer_radius)
    "dev": (0.003, 3),  # 12^2 x 210,   3 157 tracks,  0.2 s,  k_s = 1.0580
    "archive": (0.008, 3),  # 22^2 x 210,  22 447 tracks,  2.0 s,  k_s = 1.0929
    "converged": (0.014, 3),  # 34^2 x 210,  68 744 tracks, 17.4 s,  k_s = 1.1011
    "production": (0.018, 3),  # 42^2 x 210, 113 638 tracks, 41.3 s,  k_s = 1.1035
}
DEFAULT_TIER = "archive"


def build_config(tier: str = DEFAULT_TIER) -> SimulationConfig:
    """SimulationConfig for one grid tier of the Markus 2 mm macropulse case."""
    if tier not in GRID_TIERS:
        raise ValueError(f"Unknown tier {tier!r}; expected one of {sorted(GRID_TIERS)}.")
    sampled_radius_cm, buffer_radius = GRID_TIERS[tier]
    return SimulationConfig(
        **BEAM_KWARGS,
        **V2_PHYSICS_KWARGS,
        grid_size_um=10.0,
        sampled_radius_cm=sampled_radius_cm,
        buffer_radius=buffer_radius,
        no_z_electrode=5,
    )


def main(tier: str = DEFAULT_TIER) -> None:
    config = build_config(tier)
    print(f"=== IFJ AIC-144, Markus 2 mm, macropulse, 10 Gy/s -- '{tier}' grid ===")
    print(config.summary())
    print()

    warmup()  # one-off JIT compilation, excluded from the timing below
    t0 = time.perf_counter()
    result = run_simulation_numba(config, progress=True)
    elapsed_s = time.perf_counter() - t0

    print(f"\nWall time (numba, single-threaded): {elapsed_s:.1f} s")
    print(f"Collection efficiency f = {result.f_t[-1]:.4f}")
    print(f"Recombination correction k_s = 1/f = {result.ks:.4f}")
    print(
        "\nArchived IonTracks v2 (FEniCSx) result for this case: k_s = 1.1629 (exact, 1/f)"
        "\n-- but at 2.04x this areal track density (README section 4.4), so the two are"
        "\nnot directly comparable until the dose convention is agreed."
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TIER)
