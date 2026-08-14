"""The two backends must produce identical physics.

They are not trivially the same code: `solver_numba` deposits one track at a
time and broadcasts each down the gap, while `solver_numba_parallel` sums a
whole time step's tracks into a 2D array first and broadcasts once, under
`prange`. Sum-then-broadcast is exact by associativity, but the *order* of the
float additions differs, so the two agree to a relative tolerance rather than
bit-for-bit. Anything looser than that would hide a real divergence; anything
tighter would fail on reduction reordering alone.

This pairing is the suite's cross-implementation check. It is a real one: the
two differ in loop structure, in parallelism, and in when the z-broadcast
happens. The independent *physics* check is test_single_track_vs_jaffe.py.
"""

import numpy as np
import pytest

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.solver_numba import run_simulation_numba, warmup
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel, warmup_parallel

RTOL = 1e-9


def _fast_config(seed=3, **overrides):
    """Small enough to run both backends in well under a second."""
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


def test_backends_agree_on_collection_efficiency():
    serial = run_simulation_numba(_fast_config(), progress=False)
    batched = run_simulation_numba_parallel(_fast_config(), progress=False)
    np.testing.assert_allclose(batched.f_t, serial.f_t, rtol=RTOL)
    assert batched.ks == pytest.approx(serial.ks, rel=RTOL)


def test_backends_agree_on_the_final_density_field():
    """f(t) is a scalar summary; the field itself is the stronger check."""
    serial = run_simulation_numba(_fast_config(), progress=False)
    batched = run_simulation_numba_parallel(_fast_config(), progress=False)
    peak = serial.positive_array.max()
    np.testing.assert_allclose(batched.positive_array, serial.positive_array, atol=peak * RTOL)
    np.testing.assert_allclose(batched.negative_array, serial.negative_array, atol=peak * RTOL)


def test_backends_draw_the_same_tracks():
    """Same seed, same RNG stream: the track positions must be identical, so a
    difference in results can never be blamed on different sampling."""
    serial = run_simulation_numba(_fast_config(), progress=False)
    batched = run_simulation_numba_parallel(_fast_config(), progress=False)
    np.testing.assert_array_equal(batched.track_density_xy, serial.track_density_xy)


@pytest.mark.parametrize("wall", ["absorbing", "reflecting"])
def test_backends_agree_under_each_wall_condition(wall):
    """The batched backend swaps its buffers where the serial one copies the
    interior back, so the outer ring reaches the next step by a different route
    in each: rewritten from the interior (reflecting) or carried across
    explicitly (absorbing). Both routes must land on the serial answer, and the
    field -- not just f(t) -- is what shows it, because the ring is exactly
    where the two could differ without moving k_s much.
    """
    serial = run_simulation_numba(_fast_config(lateral_boundary=wall), progress=False)
    batched = run_simulation_numba_parallel(_fast_config(lateral_boundary=wall), progress=False)
    peak = serial.positive_array.max()
    np.testing.assert_allclose(batched.positive_array, serial.positive_array, atol=peak * RTOL)
    np.testing.assert_allclose(batched.negative_array, serial.negative_array, atol=peak * RTOL)
    assert batched.ks == pytest.approx(serial.ks, rel=RTOL)


@pytest.mark.parametrize("num_threads", [1, 2])
def test_batched_backend_is_thread_count_independent(num_threads):
    """Thread count changes the reduction order, nothing else."""
    single = run_simulation_numba_parallel(_fast_config(), progress=False, num_threads=1)
    threaded = run_simulation_numba_parallel(_fast_config(), progress=False, num_threads=num_threads)
    assert threaded.ks == pytest.approx(single.ks, rel=RTOL)


def test_warmup_does_not_hang_or_raise():
    warmup()
    warmup_parallel()
