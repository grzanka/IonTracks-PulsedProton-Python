"""The GPU backend must produce the same physics as the CPU reference.

solver_cuda runs the identical per-voxel update as solver_numba, in the same
order, on a CUDA device -- so the density field matches the serial backend to
near machine epsilon (the only differences are last-bit: CUDA contracts a*b+c
to an FMA, and the scored reduction sums in a different order). The scalar
summaries therefore agree to a relative tolerance, exactly as the two CPU
backends already do with each other in test_backends_agree.py.

These tests skip cleanly when there is no GPU (or CuPy/Numba-CUDA is not
installed), so the suite still passes on a CPU-only machine; they only run
where run_simulation_cuda can actually run.
"""

import numpy as np
import pytest

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.solver_numba import run_simulation_numba, warmup

# Skip the whole module unless a CUDA device and the GPU stack are present.
cp = pytest.importorskip("cupy", reason="GPU backend needs CuPy")


def _cuda_available() -> bool:
    try:
        from numba import cuda

        return cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="no CUDA device available")

RTOL = 1e-9  # observed agreement is ~1e-15; this leaves six orders of margin


def _fast_config(seed=3, **overrides):
    """Small enough to run in well under a second, same as test_backends_agree."""
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
        **overrides,
    )


def test_cuda_matches_serial_on_collection_efficiency():
    from pulsed_ion_chamber.solver_cuda import run_simulation_cuda, warmup_cuda

    warmup()
    warmup_cuda()
    serial = run_simulation_numba(_fast_config(), progress=False)
    gpu = run_simulation_cuda(_fast_config(), progress=False)
    np.testing.assert_allclose(gpu.f_t, serial.f_t, rtol=RTOL)
    assert gpu.ks == pytest.approx(serial.ks, rel=RTOL)


def test_cuda_matches_serial_on_the_final_density_field():
    """f(t) is a scalar summary; the field itself is the stronger check."""
    from pulsed_ion_chamber.solver_cuda import run_simulation_cuda

    serial = run_simulation_numba(_fast_config(), progress=False)
    gpu = run_simulation_cuda(_fast_config(), progress=False)
    peak = serial.positive_array.max()
    np.testing.assert_allclose(gpu.positive_array, serial.positive_array, atol=peak * RTOL)
    np.testing.assert_allclose(gpu.negative_array, serial.negative_array, atol=peak * RTOL)


def test_cuda_draws_the_same_tracks():
    """Same seed, same RNG stream (the same CylinderSampler): the track centres
    must be bit-identical, so a difference can never be blamed on sampling."""
    from pulsed_ion_chamber.solver_cuda import run_simulation_cuda

    serial = run_simulation_numba(_fast_config(), progress=False)
    gpu = run_simulation_cuda(_fast_config(), progress=False)
    np.testing.assert_array_equal(gpu.track_density_xy, serial.track_density_xy)


@pytest.mark.parametrize("wall", ["absorbing", "reflecting"])
def test_cuda_matches_serial_under_each_wall_condition(wall):
    """The GPU backend swaps buffers and handles the outer ring on-device with
    the same helpers the batched CPU backend uses; both wall routes must land on
    the serial answer, and the field -- not just k_s -- is what shows it."""
    from pulsed_ion_chamber.solver_cuda import run_simulation_cuda

    serial = run_simulation_numba(_fast_config(lateral_boundary=wall), progress=False)
    gpu = run_simulation_cuda(_fast_config(lateral_boundary=wall), progress=False)
    peak = serial.positive_array.max()
    np.testing.assert_allclose(gpu.positive_array, serial.positive_array, atol=peak * RTOL)
    np.testing.assert_allclose(gpu.negative_array, serial.negative_array, atol=peak * RTOL)
    assert gpu.ks == pytest.approx(serial.ks, rel=RTOL)


@pytest.mark.parametrize("scoring", ["track_disc", "full_grid"])
def test_cuda_matches_serial_under_each_scoring_region(scoring):
    """full_grid scoring exercises the outer-ring exclusion in both the sweep
    kernel and the injected-charge mask (issue #19 P1/P2), which track_disc
    never reaches -- the two must give the same k_s as the serial reference."""
    from pulsed_ion_chamber.solver_cuda import run_simulation_cuda

    serial = run_simulation_numba(_fast_config(scoring_region=scoring), progress=False)
    gpu = run_simulation_cuda(_fast_config(scoring_region=scoring), progress=False)
    assert gpu.ks == pytest.approx(serial.ks, rel=RTOL)


def test_cuda_matches_serial_with_two_carrier_species():
    """The default single averaged carrier hides the swapped z-neighbour drift
    of the negative species; resolve the two Kanai species so the negative
    stencil's coefficients actually differ from the positive one's."""
    from pulsed_ion_chamber.solver_cuda import run_simulation_cuda

    cfg = dict(mu_positive_cm2_Vs=1.36, mu_negative_cm2_Vs=2.10, D_positive_cm2_s=2.82e-2, D_negative_cm2_s=4.35e-2)
    serial = run_simulation_numba(_fast_config(**cfg), progress=False)
    gpu = run_simulation_cuda(_fast_config(**cfg), progress=False)
    peak = serial.positive_array.max()
    np.testing.assert_allclose(gpu.positive_array, serial.positive_array, atol=peak * RTOL)
    assert gpu.ks == pytest.approx(serial.ks, rel=RTOL)


def test_warmup_cuda_does_not_hang_or_raise():
    from pulsed_ion_chamber.solver_cuda import warmup_cuda

    warmup_cuda()
