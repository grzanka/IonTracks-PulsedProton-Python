"""In the single-track (very low dose-rate) limit, the PDE simulation should
reproduce the analytic Jaffe theory of initial recombination. This is the
same cross-check the original IonTracks-Cython test suite uses
(tests/ks_initial/), just against a deliberately coarse/fast grid here, so
the tolerance is loose -- this is a sanity/regression check, not a
convergence study.
"""

import numpy as np
import pytest

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.solver_numba import run_simulation_numba
from pulsed_ion_chamber.theory import jaffe_ks


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_single_track_matches_jaffe_theory(seed):
    config = SimulationConfig(
        E_MeV_u=1.0,
        voltage_V=10.0,
        electrode_gap_cm=0.02,
        pulse_duration_s=1e-6,
        dose_rate_Gy_s=1e-8,  # low enough that number_of_tracks_per_pulse floors to 1
        n_pulses=1,
        grid_size_um=10.0,
        sampled_radius_cm=0.01,
        buffer_radius=5,
        no_z_electrode=4,
        seed=seed,
    )
    assert config.number_of_tracks_per_pulse == 1

    result = run_simulation_numba(config, progress=False)
    # Passed explicitly (even though they equal jaffe_ks's own defaults here)
    # so this stays correct if the config above ever stops using the module
    # defaults -- see test_jaffe_ks_honours_a_non_default_w_ev below for what
    # goes wrong when a caller forgets to (issue #19 P4).
    ks_jaffe = jaffe_ks(
        config.LET_keV_um,
        config.voltage_V,
        config.electrode_gap_cm,
        W_eV=config.W_eV,
        mu=config.mu_positive,
        D=config.D_positive,
    )

    # Compare recombination *loss* (ks - 1) rather than ks itself, since
    # ks is close to 1 and the loss is what the two methods actually predict.
    sim_loss = result.ks - 1.0
    jaffe_loss = ks_jaffe - 1.0
    assert sim_loss / jaffe_loss == pytest.approx(1.0, rel=0.3)


def test_jaffe_ks_honours_a_non_default_w_ev():
    """jaffe_ks must not silently fall back to the module's W_eV constant when
    a config overrides it -- recombination scales as N0**2, so a ~4% change in
    W_eV (34.2 -> 33.0, examples/ifj_aic144's value) moves the *loss* (ks - 1)
    by roughly 2x that fraction, comfortably outside float noise."""
    from pulsed_ion_chamber.constants import W_EV_PER_ION_PAIR

    args = (60.0, 300.0, 0.2)  # LET_keV_um, voltage_V, electrode_gap_cm
    default_loss = jaffe_ks(*args) - 1.0
    overridden_loss = jaffe_ks(*args, W_eV=33.0) - 1.0
    assert 33.0 != W_EV_PER_ION_PAIR
    assert overridden_loss != pytest.approx(default_loss, rel=1e-6)
    # N0 (and so the loss, to leading order) scales as 1/W_eV.
    assert overridden_loss / default_loss == pytest.approx(W_EV_PER_ION_PAIR / 33.0, rel=0.05)


def test_f_t_is_monotonically_non_increasing():
    """Collection efficiency can only go down as more recombination accumulates."""
    config = SimulationConfig(
        E_MeV_u=1.0,
        voltage_V=10.0,
        electrode_gap_cm=0.02,
        pulse_duration_s=1e-6,
        dose_rate_Gy_s=1e-8,
        grid_size_um=10.0,
        sampled_radius_cm=0.01,
        buffer_radius=5,
        no_z_electrode=4,
        seed=1,
    )
    result = run_simulation_numba(config, progress=False)
    assert np.all(np.diff(result.f_t) <= 1e-6)
