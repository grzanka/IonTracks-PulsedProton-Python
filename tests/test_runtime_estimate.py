"""pulsed_ion_chamber.benchmark.estimate_full_runtime_empirical and the
max_wall_s early-exit it relies on in both solver backends.
"""

import pytest

from pulsed_ion_chamber.benchmark import estimate_full_runtime_empirical
from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.solver_numba import run_simulation_numba

# Small enough that a full run finishes in well under a second -- fast tests,
# and (with a generous max_wall_s) the "whole run finished inside the
# budget" / exact=True branch is what actually gets exercised.
SMALL = dict(
    E_MeV_u=56.2,
    voltage_V=300.0,
    electrode_gap_cm=0.2,
    pulse_duration_s=2e-6,
    repetition_rate_hz=50.0,
    dose_rate_Gy_s=8.91,
    grid_size_um=40.0,
    sampled_radius_cm=0.004,
    buffer_radius=3,
    no_z_electrode=3,
    seed=7,
)


def test_max_wall_s_none_is_unaffected():
    """Default behaviour (no budget given) must be untouched: no early-exit
    attributes on the Result, same as before this parameter existed."""
    config = SimulationConfig(**SMALL)
    result = run_simulation_numba(config, progress=False)
    assert not hasattr(result, "steps_completed")
    assert not hasattr(result, "loop_elapsed_s")


def test_max_wall_s_stops_early_and_reports_progress():
    config = SimulationConfig(**SMALL)
    # 0 steps would need an impossibly small budget; use something the first
    # step or two will exceed but the full run (well under a second) won't.
    result = run_simulation_numba(config, progress=False, max_wall_s=1e-9)
    assert 0 < result.steps_completed <= config.total_time_steps
    assert result.loop_elapsed_s >= 0


def test_estimate_full_runtime_empirical_exact_when_run_completes():
    config = SimulationConfig(**SMALL)
    est = estimate_full_runtime_empirical(config, num_threads=1, max_wall_s=30.0)
    assert est["exact"] is True
    assert est["steps_measured"] == config.total_time_steps
    assert est["estimated_seconds"] == est["elapsed_measured_s"]
    assert est["backend"] == "solver_numba"
    assert est["num_threads"] == 1


def test_estimate_full_runtime_empirical_extrapolates_when_budget_is_tight():
    config = SimulationConfig(**SMALL)
    est = estimate_full_runtime_empirical(config, num_threads=1, max_wall_s=1e-9)
    assert est["exact"] is False
    assert 0 < est["steps_measured"] < config.total_time_steps
    assert est["estimated_seconds"] >= est["elapsed_measured_s"]


def test_estimate_full_runtime_empirical_uses_batched_backend_above_one_thread():
    config = SimulationConfig(**SMALL)
    est = estimate_full_runtime_empirical(config, num_threads=2, max_wall_s=30.0)
    assert est["backend"] == "solver_numba_parallel"
    assert est["exact"] is True


def test_estimate_full_runtime_empirical_rejects_a_budget_too_small_for_one_step(monkeypatch):
    """steps_measured == 0 can only happen if the config has no steps at
    all -- simulated here rather than hunted for on real hardware, since a
    real first step reliably completes in far less than any wall clock can
    measure as zero elapsed time."""
    import pulsed_ion_chamber.benchmark as benchmark_module

    class _FakeResult:
        steps_completed = 0
        loop_elapsed_s = 0.0

    monkeypatch.setattr(benchmark_module, "run_simulation_numba", lambda *a, **k: _FakeResult())
    config = SimulationConfig(**SMALL)
    with pytest.raises(RuntimeError, match="max_wall_s"):
        estimate_full_runtime_empirical(config, num_threads=1, max_wall_s=1e-12)
