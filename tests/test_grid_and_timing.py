import pytest

from pulsed_ion_chamber.config import SimulationConfig


def test_default_config_builds_and_is_small():
    config = SimulationConfig(seed=0)
    assert config.no_xy > 2 * config.buffer_radius
    assert config.no_z > 0
    assert config.dt > 0
    assert config.number_of_tracks_per_pulse >= 1
    # defaults are meant to run in well under a minute
    assert config.no_xy * config.no_xy * config.no_z_with_buffer < 1e6


def test_pulse_period_must_fit_pulse_duration():
    with pytest.raises(ValueError):
        SimulationConfig(
            pulse_duration_s=1e-2,  # 10 ms pulse
            repetition_rate_hz=1000.0,  # but pulses requested every 1 ms
            n_pulses=2,
        )


def test_grid_size_guard_raises_for_huge_grid():
    with pytest.raises(ValueError):
        SimulationConfig(grid_size_um=0.1, sampled_radius_cm=1.0, electrode_gap_cm=1.0)


def test_more_pulses_take_more_total_time_steps():
    one_pulse = SimulationConfig(n_pulses=1)
    two_pulses = SimulationConfig(n_pulses=2)
    assert two_pulses.total_time_steps > one_pulse.total_time_steps


def test_degenerate_inner_radius_raises_instead_of_hanging():
    # sampled_radius_cm too small relative to buffer_radius/grid_size_um
    # makes inner_radius <= 0, which would make pulses.sample_xy_inside_cylinder's
    # rejection-sampling loop spin forever instead of raising.
    with pytest.raises(ValueError, match="inner_radius"):
        SimulationConfig(grid_size_um=100.0, sampled_radius_cm=0.001, buffer_radius=2, no_z_electrode=2)
