"""The single-threaded Numba path (the baseline backend for this
repository, solver_numba.py) must reproduce the plain pure-Python
reference implementation (solver.py) exactly -- same RNG draws, same
arithmetic, just JIT-compiled.
"""

import numpy as np
import pytest

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.solver import run_simulation
from pulsed_ion_chamber.solver_numba import run_simulation_numba, warmup


def _fast_config(seed):
    # Same small/fast config used for the Jaffe cross-check in
    # test_single_track_vs_jaffe.py -- keeps this test in the sub-second
    # range rather than running the (~20,000-track) default demo config
    # twice.
    return SimulationConfig(
        E_MeV_u=1.0,
        voltage_V=10.0,
        electrode_gap_cm=0.02,
        pulse_duration_s=1e-6,
        dose_rate_Gy_s=1e-8,
        grid_size_um=10.0,
        sampled_radius_cm=0.01,
        buffer_radius=5,
        no_z_electrode=4,
        seed=seed,
    )


def test_numba_matches_pure_python():
    result_py = run_simulation(_fast_config(3), progress=False)
    result_nb = run_simulation_numba(_fast_config(3), progress=False)

    np.testing.assert_allclose(result_py.f_t, result_nb.f_t)
    assert result_py.ks == pytest.approx(result_nb.ks)


def test_warmup_does_not_hang_or_raise():
    warmup()  # should compile and return in well under a second
