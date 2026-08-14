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
    ks_jaffe = jaffe_ks(config.LET_keV_um, config.voltage_V, config.electrode_gap_cm)

    # Compare recombination *loss* (ks - 1) rather than ks itself, since
    # ks is close to 1 and the loss is what the two methods actually predict.
    sim_loss = result.ks - 1.0
    jaffe_loss = ks_jaffe - 1.0
    assert sim_loss / jaffe_loss == pytest.approx(1.0, rel=0.3)


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
