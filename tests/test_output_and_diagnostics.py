"""Per-step diagnostics, the CSV record, and the diagnostic plots."""

import numpy as np
import pytest

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.output import (
    COLLECTED_CHARGE_COLUMNS,
    collected_charge_table,
    track_density_per_cm2,
    write_collected_charge_csv,
)
from pulsed_ion_chamber.solver import run_simulation
from pulsed_ion_chamber.solver_numba import run_simulation_numba
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel

SMALL = dict(
    E_MeV_u=56.2,
    voltage_V=300.0,
    electrode_gap_cm=0.2,
    pulse_duration_s=4e-6,
    repetition_rate_hz=50.0,
    dose_rate_Gy_s=8.91,
    grid_size_um=40.0,
    sampled_radius_cm=0.004,
    buffer_radius=3,
    no_z_electrode=3,
    seed=11,
)


@pytest.fixture(scope="module")
def result():
    return run_simulation_numba(SimulationConfig(**SMALL), progress=False)


def test_diagnostics_reproduce_the_collection_efficiency(result):
    """f(t) is derived from running totals inside the solver; the per-step
    series must integrate to the same thing, or one of them is wrong."""
    injected = result.injected_positive.cumsum()
    recombined = result.recombination.cumsum()
    expected = (injected - recombined) / injected
    np.testing.assert_allclose(result.f_t[injected > 0], expected[injected > 0], rtol=1e-12)
    assert result.ks == pytest.approx(1.0 / expected[-1], rel=1e-12)


def test_both_signs_are_injected_equally(result):
    # every track liberates one positive and one negative carrier
    np.testing.assert_allclose(result.injected_positive, result.injected_negative, rtol=0)


def test_injection_is_confined_to_the_pulse(result):
    pulse_steps = result.config.pulse_time_steps
    assert result.injected_positive[:pulse_steps].sum() > 0
    assert result.injected_positive[pulse_steps:].sum() == 0


def test_carriers_are_collected_by_the_end_of_the_run(result):
    """The clearance period exists so the gap empties; if it did not, the
    reported k_s would be missing recombination that had yet to happen.

    n_clearance_separation_times defaults to 2.0, which is exactly one full-gap
    transit of the slowest carrier and so leaves no margin: a residual of order
    1e-5 of the peak is expected, not a bug. Lax-Wendroff is not
    positivity-preserving either, so that residual can be slightly negative.
    """
    peak = result.n_positive.max()
    assert abs(result.n_positive[-1]) < 1e-3 * peak
    assert abs(result.n_negative[-1]) < 1e-3 * peak


def test_extra_clearance_empties_the_gap_without_moving_k_s():
    """The 30 % margin the AIC-144 scenario uses is worth ~7 orders of
    magnitude in residual charge, and changes the answer not at all -- so the
    default is adequate for k_s, and the margin is for peace of mind."""
    default = run_simulation_numba(SimulationConfig(**SMALL), progress=False)
    with_margin = run_simulation_numba(
        SimulationConfig(**SMALL, n_clearance_separation_times=2.6), progress=False
    )
    peak = default.n_positive.max()
    assert abs(default.n_positive[-1]) > 1e-9 * peak  # no margin: something is left
    assert abs(with_margin.n_positive[-1]) < 1e-9 * peak  # margin: nothing is
    assert with_margin.ks == pytest.approx(default.ks, rel=1e-6)


def test_all_track_centres_are_counted_inside_the_sampling_disc(result):
    config = result.config
    assert result.track_density_xy.sum() == config.number_of_tracks_per_pulse
    i, j = np.nonzero(result.track_density_xy)
    radius = np.hypot(i - config.mid_xy, j - config.mid_xy)
    # a centre lands in the voxel whose lower corner it falls in, so allow one
    assert radius.max() <= config.sampling_radius + np.sqrt(2)


def test_track_areal_density_matches_the_nominal_fluence(result):
    config = result.config
    density = track_density_per_cm2(result)
    total = density.sum() * config.unit_length_cm**2
    assert total == pytest.approx(config.number_of_tracks_per_pulse)
    mean_over_disc = total / (np.pi * config.sampled_radius_cm**2)
    assert mean_over_disc == pytest.approx(
        config.number_of_tracks_per_pulse / (np.pi * config.sampled_radius_cm**2), rel=1e-9
    )


def test_csv_round_trip(result, tmp_path):
    path = write_collected_charge_csv(result, tmp_path / "collected_charge.csv")
    rows = path.read_text().strip().split("\n")
    assert rows[0].split(",") == list(COLLECTED_CHARGE_COLUMNS)
    assert len(rows) == result.config.total_time_steps + 1
    table = collected_charge_table(result)
    first = [float(v) for v in rows[1].split(",")]
    for value, name in zip(first, COLLECTED_CHARGE_COLUMNS):
        assert value == pytest.approx(table[name][0], rel=1e-11)


def test_csv_is_in_ion_pairs_not_densities(result):
    """The solver accumulates densities; the record must be absolute counts,
    or it cannot be compared against another code's output."""
    table = collected_charge_table(result)
    voxel_volume = result.config.unit_length_cm**3
    assert table["injected_positive"][0] == pytest.approx(
        result.injected_positive[0] * voxel_volume
    )
    # and the total should be within a factor of order 1 of N_tracks * LET/W * gap
    config = result.config
    analytic = (
        config.number_of_tracks_per_pulse
        * (config.LET_keV_um * 1e7 / config.W_eV)
        * config.electrode_gap_cm
    )
    # scoring is restricted to the track disc, so tails are lost: expect < 1
    assert 0.3 < table["injected_positive"].sum() / analytic < 1.0


@pytest.mark.parametrize("backend", [run_simulation, run_simulation_numba, run_simulation_numba_parallel])
def test_backends_agree_on_the_diagnostics(backend):
    config = SimulationConfig(**SMALL)
    reference = run_simulation_numba(config, progress=False)
    other = backend(config, progress=False)
    for name in ("n_positive", "n_negative", "injected_positive", "recombination"):
        np.testing.assert_allclose(getattr(other, name), getattr(reference, name), rtol=1e-9)
    np.testing.assert_allclose(other.track_density_xy, reference.track_density_xy, rtol=0)


def test_plots_are_written(result, tmp_path):
    from pulsed_ion_chamber.plots import save_diagnostic_plots

    paths = save_diagnostic_plots(result, tmp_path, title="test")
    assert {p.name for p in paths} == {
        "injection_rate.png",
        "carrier_evolution.png",
        "recombination_rate.png",
        "track_density_cross_section.png",
    }
    assert all(p.stat().st_size > 5000 for p in paths)
