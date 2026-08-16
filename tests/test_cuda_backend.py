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


# --- unified memory, block size, truncation (the GH200 work) -----------------
# docs/BENCHMARKS-HELIOS-GH200.md. The allocator must be invisible to the
# physics: these check that claim on the same small config the rest of the
# module uses, so a regression in the managed/host paths fails here and not
# only on a 200 GiB grid nobody runs in CI.


@pytest.mark.parametrize("memory,advise", [("device", "device"), ("managed", "device"), ("managed", "none")])
def test_memory_modes_agree_with_device_memory(memory, advise):
    """cudaMalloc, cudaMallocManaged and the placement hints are an allocation
    detail; k_s and the density field must not notice."""
    from pulsed_ion_chamber.solver_cuda import run_simulation_cuda

    reference = run_simulation_cuda(_fast_config(), progress=False, memory="device")
    other = run_simulation_cuda(_fast_config(), progress=False, memory=memory, advise=advise)
    assert other.ks == pytest.approx(reference.ks, rel=RTOL)
    np.testing.assert_allclose(other.positive_array, reference.positive_array, atol=reference.positive_array.max() * RTOL)


def test_host_shared_memory_agrees_or_is_refused():
    """memory="host" needs a GPU that walks the host page tables (GH200). Where
    it does, the answer is identical; where it does not, the refusal must be a
    clear RuntimeError rather than a kernel fault."""
    from pulsed_ion_chamber.solver_cuda import _host_page_tables, run_simulation_cuda

    reference = run_simulation_cuda(_fast_config(), progress=False, memory="device")
    if not _host_page_tables(cp):
        with pytest.raises(RuntimeError, match="host page tables"):
            run_simulation_cuda(_fast_config(), progress=False, memory="host")
        return
    host = run_simulation_cuda(_fast_config(), progress=False, memory="host")
    assert host.ks == pytest.approx(reference.ks, rel=RTOL)


def test_max_steps_truncates_and_reports_per_step_cost():
    """A grid too large to finish is still benchmarkable: max_steps stops the
    loop, records how far it got, and refuses to report a k_s it did not earn."""
    from pulsed_ion_chamber.solver_cuda import run_simulation_cuda

    full = run_simulation_cuda(_fast_config(), progress=False)
    truncated = run_simulation_cuda(_fast_config(), progress=False, max_steps=5)
    assert truncated.steps_completed == 5
    assert truncated.loop_elapsed_s > 0
    assert np.isnan(truncated.ks)
    # The first steps are the same run, so the efficiency series must match.
    np.testing.assert_allclose(truncated.f_t[:5], full.f_t[:5], rtol=RTOL)


def test_return_fields_false_drops_only_the_snapshot():
    """The time series is what a large run is after; the two grid-sized host
    arrays are what kills it on a memory-limited job."""
    from pulsed_ion_chamber.solver_cuda import run_simulation_cuda

    full = run_simulation_cuda(_fast_config(), progress=False)
    lean = run_simulation_cuda(_fast_config(), progress=False, return_fields=False)
    assert lean.positive_array.size == 0 and lean.negative_array.size == 0
    assert lean.ks == pytest.approx(full.ks, rel=RTOL)
    np.testing.assert_allclose(lean.f_t, full.f_t, rtol=RTOL)


def test_rejects_impossible_options():
    from pulsed_ion_chamber.solver_cuda import run_simulation_cuda, set_threads_per_block

    with pytest.raises(ValueError, match="memory must be"):
        run_simulation_cuda(_fast_config(), progress=False, memory="hbm")
    with pytest.raises(ValueError, match="advise must be"):
        run_simulation_cuda(_fast_config(), progress=False, advise="somewhere")
    with pytest.raises(ValueError, match="power of two"):
        set_threads_per_block(300)


def test_block_size_is_tunable_and_does_not_move_the_answer():
    """The block size is a machine tuning knob (256 measured best on both the
    A100 and Hopper), so changing it must recompile and return the same physics."""
    from pulsed_ion_chamber import solver_cuda

    original = solver_cuda.THREADS_PER_BLOCK
    reference = solver_cuda.run_simulation_cuda(_fast_config(), progress=False)
    try:
        solver_cuda.set_threads_per_block(128)
        assert solver_cuda.run_simulation_cuda(_fast_config(), progress=False).ks == pytest.approx(
            reference.ks, rel=RTOL
        )
    finally:
        solver_cuda.set_threads_per_block(original)
