import numpy as np
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


def test_rf_frequency_is_opt_in_and_defaults_to_none():
    config = SimulationConfig(seed=0)
    assert config.rf_frequency_hz is None
    assert config.rf_cycles_per_time_step is None
    assert "RF" not in config.summary()


def test_rf_cycles_per_time_step_for_a_real_cyclotron_rf():
    # dt is set by the von Neumann stability criterion, always far coarser
    # than a real cyclotron's RF period -- individual RF buckets can't be
    # resolved regardless, so this should be well above 1 with no warning.
    config = SimulationConfig(rf_frequency_hz=26.26e6, seed=0)
    assert config.rf_cycles_per_time_step > 1.0
    assert config.rf_cycles_per_time_step == pytest.approx(config.dt * 26.26e6)
    assert "RF microstructure" in config.summary()


def test_rf_period_longer_than_dt_warns():
    # a deliberately absurd (very low) "RF frequency" so that one RF period
    # is longer than dt, to check the warning path actually fires.
    with pytest.warns(UserWarning, match="RF"):
        SimulationConfig(rf_frequency_hz=1.0, seed=0)


def test_summary_omits_the_gap_note_for_float_noise_alone():
    """summary()'s "simulated as ..." note must not fire on ~1e-16 float
    round-off alone: no_z * unit_length_cm and the original electrode_gap_cm
    can differ by float noise even when no_z is the exact right voxel count
    (0.7 / 0.001 rounds to 700, but 700 * 0.001 == 0.7000000000000001, not
    bit-identical to the literal 0.7). Comparing with exact `==` would flag
    this as a "simulated as ..." mismatch on every such call (PR #20 review)."""
    config = SimulationConfig(grid_size_um=10.0, electrode_gap_cm=0.7, seed=0)
    assert config.no_z == 700
    assert config.effective_gap_cm != config.electrode_gap_cm  # float noise is real here
    assert abs(config.effective_gap_cm - config.electrode_gap_cm) < 1e-9
    assert "simulated as" not in config.summary()


def test_summary_reports_the_gap_note_for_a_genuine_mismatch():
    # 0.05 cm / 40 um doesn't divide evenly: no_z = round(0.05/0.004) = 12
    # (0.05/0.004 == 12.5 exactly, and round() rounds half to even), so the
    # simulated gap is 0.048 cm -- 4% off, large enough that a warning also
    # fires (see the P3 warning tests above), and nowhere near a tolerance
    # that could be mistaken for float round-off.
    with pytest.warns(UserWarning, match="electrode_gap_cm"):
        config = SimulationConfig(grid_size_um=40.0, electrode_gap_cm=0.05, seed=0)
    assert config.no_z == 12
    assert "simulated as 12 voxels" in config.summary()


# --- track sampling ---------------------------------------------------------


def test_block_sampler_draws_exactly_what_the_per_track_sampler_would():
    """CylinderSampler exists to be 166x faster than sample_xy_inside_cylinder
    *without* moving a single result, so that published k_s values stay tied to
    their seeds. Bit equality is the whole contract, not a nicety: any drift in
    how much of the RNG stream a batch consumes silently reseeds everything
    that follows it.

    Uneven batch sizes on purpose -- the leftover of one call has to be carried
    into the next for the stream to line up.
    """
    from pulsed_ion_chamber.pulses import CylinderSampler, sample_xy_inside_cylinder

    mid_xy, radius, no_xy = 30, 24.0, 60
    batches = [7, 1, 5000, 3, 900]

    reference = np.random.default_rng(11)
    expected = np.array(
        [sample_xy_inside_cylinder(reference, mid_xy, radius, no_xy) for _ in range(sum(batches))]
    )

    sampler = CylinderSampler(np.random.default_rng(11), mid_xy, radius, no_xy)
    got = np.vstack([np.column_stack(sampler.sample(n)) for n in batches])

    np.testing.assert_array_equal(got, expected)


def test_block_sampler_stays_inside_the_disc():
    from pulsed_ion_chamber.pulses import CylinderSampler

    mid_xy, radius, no_xy = 30, 24.0, 60
    xs, ys = CylinderSampler(np.random.default_rng(2), mid_xy, radius, no_xy).sample(20_000)
    assert len(xs) == len(ys) == 20_000
    assert np.all((xs - mid_xy) ** 2 + (ys - mid_xy) ** 2 <= radius**2)
