"""The parallel Numba backend (solver_numba_parallel.py) must reproduce the
plain pure-Python reference (solver.py) up to floating-point reduction
reordering: `prange` sums the `inserted`/`recombined` reductions across
threads in a different order than the serial loop, so results can differ
in the last few ULPs -- hence `rtol` looser than the exact match used in
test_solver_numba.py, but still tight.
"""

import numpy as np
import pytest

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.solver import run_simulation
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel, warmup_parallel


def _fast_config(seed):
    # Same small/fast config used in test_solver_numba.py.
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


def test_numba_parallel_matches_pure_python():
    result_py = run_simulation(_fast_config(3), progress=False)
    result_nb = run_simulation_numba_parallel(_fast_config(3), progress=False)

    np.testing.assert_allclose(result_py.f_t, result_nb.f_t, rtol=1e-9)
    assert result_py.ks == pytest.approx(result_nb.ks, rel=1e-9)


def test_warmup_parallel_does_not_hang_or_raise():
    warmup_parallel()
