"""Tests for the physics ported from IonTracks-FEniCSx (v2).

Two things are checked: that every new knob defaults to the original
IonTracks-Cython (v1) behaviour, and that the new code paths are consistent
between the pure-Python reference and the two Numba backends.
"""

import numpy as np
import pytest

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import (
    AIR_DENSITY_KG_M3,
    ION_DIFFUSION_CM2_S,
    ION_DIFFUSION_NEGATIVE_CM2_S,
    ION_DIFFUSION_POSITIVE_CM2_S,
    ION_MOBILITY_CM2_VS,
    ION_MOBILITY_NEGATIVE_CM2_VS,
    ION_MOBILITY_POSITIVE_CM2_VS,
    W_EV_PER_ION_PAIR,
)
from pulsed_ion_chamber.solver import run_simulation
from pulsed_ion_chamber.solver_numba import run_simulation_numba
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel

# Small enough to run all three backends in a few seconds.
SMALL = dict(
    E_MeV_u=56.2,
    voltage_V=300.0,
    electrode_gap_cm=0.2,
    pulse_duration_s=2e-6,
    repetition_rate_hz=50.0,
    dose_rate_Gy_s=8.91,
    grid_size_um=40.0,
    sampled_radius_cm=0.004,
    buffer_radius=3,
    no_z_electrode=3,
    seed=7,
)
TWO_SPECIES = dict(
    mu_positive_cm2_Vs=ION_MOBILITY_POSITIVE_CM2_VS,
    mu_negative_cm2_Vs=ION_MOBILITY_NEGATIVE_CM2_VS,
    D_positive_cm2_s=ION_DIFFUSION_POSITIVE_CM2_S,
    D_negative_cm2_s=ION_DIFFUSION_NEGATIVE_CM2_S,
)


def test_defaults_are_the_v1_single_averaged_species():
    config = SimulationConfig(**SMALL)
    assert not config.two_carrier_species
    assert config.mu_positive == config.mu_negative == ION_MOBILITY_CM2_VS
    assert config.D_positive == config.D_negative == ION_DIFFUSION_CM2_S
    assert config.W_eV == W_EV_PER_ION_PAIR
    assert config.air_density_kg_m3 == AIR_DENSITY_KG_M3
    assert config.lateral_boundary == "absorbing"
    assert config.scoring_region == "track_disc"
    assert config.chamber_fill_fraction == 1.0
    # With one species both stencils must be identical.
    pos, neg = config.scheme_coefficients()
    assert pos == neg
    # and tracks are drawn in the scored disc itself
    assert config.sampling_radius == config.inner_radius
    assert config.scoring_radius_sq == config.inner_radius_sq


def test_two_species_dt_is_set_by_the_negative_ion():
    single = SimulationConfig(**SMALL)
    both = SimulationConfig(**SMALL, **TWO_SPECIES)
    assert both.two_carrier_species
    # The negative ion is both faster and more diffusive, so it binds; dt must
    # drop relative to the averaged model, and more steps are needed.
    assert both.dt < single.dt
    assert both.total_time_steps > single.total_time_steps
    # A run with only the negative species' constants must give the same dt.
    negative_only = SimulationConfig(
        **SMALL,
        mu_positive_cm2_Vs=ION_MOBILITY_NEGATIVE_CM2_VS,
        mu_negative_cm2_Vs=ION_MOBILITY_NEGATIVE_CM2_VS,
        D_positive_cm2_s=ION_DIFFUSION_NEGATIVE_CM2_S,
        D_negative_cm2_s=ION_DIFFUSION_NEGATIVE_CM2_S,
    )
    assert both.dt == negative_only.dt
    # ... while the clearance period is sized by the *slowest* carrier.
    assert both.slowest_mobility_cm2_Vs == ION_MOBILITY_POSITIVE_CM2_VS


def test_von_neumann_criterion_holds_for_both_species():
    config = SimulationConfig(**SMALL, **TWO_SPECIES)
    for diffusion, mobility in (
        (config.D_positive, config.mu_positive),
        (config.D_negative, config.mu_negative),
    ):
        s = diffusion * config.dt / config.unit_length_cm**2
        cz = mobility * config.Efield_V_cm * config.dt / config.unit_length_cm
        assert 6 * s + cz**2 <= 1.0


def test_no_xy_rounds_so_the_sampled_radius_is_honoured():
    # 2 * 0.0084 / 0.0010 = 16.8: truncating gave an 80 um disc while the track
    # count still used 84 um. Rounding keeps the two consistent.
    config = SimulationConfig(**{**SMALL, "grid_size_um": 10.0, "sampled_radius_cm": 0.0084})
    assert config.no_xy == 17 + 2 * config.buffer_radius


def test_chamber_fill_fraction_shrinks_placement_not_scoring():
    full = SimulationConfig(**SMALL)
    partial = SimulationConfig(**SMALL, chamber_fill_fraction=0.7)
    # Same number of tracks (counted over the full disc), packed into 0.49 of
    # the area -- exactly what IonTracks-FEniCSx does.
    assert partial.number_of_tracks_per_pulse == full.number_of_tracks_per_pulse
    assert partial.scoring_radius_sq == full.scoring_radius_sq
    assert partial.sampling_radius == pytest.approx(0.7 * full.inner_radius)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_invalid_chamber_fill_fraction_rejected(fraction):
    with pytest.raises(ValueError, match="chamber_fill_fraction"):
        SimulationConfig(**SMALL, chamber_fill_fraction=fraction)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(lateral_boundary="sticky"), "lateral_boundary"),
        (dict(scoring_region="everything"), "scoring_region"),
        (dict(mu_negative_cm2_Vs=-2.1), "mu_negative_cm2_Vs"),
    ],
)
def test_invalid_options_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SimulationConfig(**SMALL, **kwargs)


def test_full_grid_scoring_admits_every_voxel():
    config = SimulationConfig(**SMALL, scoring_region="full_grid")
    # The furthest voxel from the centre is a corner of the square.
    corner_sq = 2 * max(config.mid_xy, config.no_xy - 1 - config.mid_xy) ** 2
    assert config.scoring_radius_sq > corner_sq


def test_reflecting_wall_leaves_no_frozen_charge_on_the_outer_ring():
    """v1's outer ring is never updated by the Lax-Wendroff sweep, so track
    tails deposited there are stranded: they never drift, diffuse or recombine,
    and keep feeding their inward neighbour for the rest of the run. The
    reflecting wall replaces that with a mirror of the interior, so the ring
    tracks whatever the gas next to it is doing."""
    absorbing = run_simulation_numba(SimulationConfig(**SMALL), progress=False)
    reflecting = run_simulation_numba(
        SimulationConfig(**SMALL, lateral_boundary="reflecting"), progress=False
    )
    mid = absorbing.config.mid_xy
    k = absorbing.config.no_z_electrode + absorbing.config.no_z // 2

    # Charge really is stranded on the absorbing ring, and sits out of
    # equilibrium with the voxel just inside it.
    assert absorbing.positive_array[0, mid, k] > 0.0
    assert absorbing.positive_array[0, mid, k] != pytest.approx(
        absorbing.positive_array[1, mid, k], rel=1e-6
    )
    # The reflecting ring is a zero-gradient mirror by construction.
    assert reflecting.positive_array[0, mid, k] == pytest.approx(
        reflecting.positive_array[1, mid, k]
    )


@pytest.mark.parametrize(
    "extra",
    [
        TWO_SPECIES,
        dict(lateral_boundary="reflecting"),
        dict(scoring_region="full_grid"),
        dict(chamber_fill_fraction=0.7),
        {**TWO_SPECIES, "lateral_boundary": "reflecting", "scoring_region": "full_grid"},
    ],
)
def test_backends_agree_on_the_ported_physics(extra):
    """Every new code path must give the same answer in the pure-Python
    reference and both Numba backends (same seed, same RNG stream)."""
    config = SimulationConfig(**SMALL, **extra)
    reference = run_simulation(config, progress=False)
    numba_result = run_simulation_numba(config, progress=False)
    parallel_result = run_simulation_numba_parallel(config, progress=False, num_threads=1)

    assert numba_result.ks == pytest.approx(reference.ks, rel=1e-9)
    assert parallel_result.ks == pytest.approx(reference.ks, rel=1e-9)
    np.testing.assert_allclose(numba_result.f_t, reference.f_t, rtol=1e-9)
    np.testing.assert_allclose(parallel_result.f_t, reference.f_t, rtol=1e-9)
