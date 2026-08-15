"""Shared harness config for the Phase-1 profiling/diagnostic exercise.

Uses the "converged, ~6 track radii" grid (grid_size_um=5.0,
sampled_radius_cm=0.012) -- the one config on which the batching fix and the
"more threads = slower" pathology were originally observed, so numbers
collected here are directly comparable to each other across runs.
"""

import resource
import time

import numpy as np

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel, warmup_parallel

BEAM_KWARGS = dict(
    E_MeV_u=150.0,
    voltage_V=200.0,
    electrode_gap_cm=0.2,
    pulse_duration_s=540e-6,
    repetition_rate_hz=50.0,
    dose_rate_Gy_s=60.0,
    n_pulses=1,
    rf_frequency_hz=26.26e6,
)

CONVERGED_GRID_KWARGS = dict(grid_size_um=5.0, sampled_radius_cm=0.012)


def build_converged_config(seed: int = 1) -> SimulationConfig:
    return SimulationConfig(**BEAM_KWARGS, **CONVERGED_GRID_KWARGS, seed=seed)


def run_once(num_threads: int, seed: int = 1, progress: bool = False) -> dict:
    """Run the converged-grid scenario once at a given thread count.

    Returns timing, OS-level resource-usage (context switches, kernel vs.
    user time -- a free proxy for fork-join/scheduling overhead when perf
    and py-spy's kernel-level counters aren't available), and scientific
    (f_t/k_s) metrics. JIT warmup is timed separately and excluded from
    wall_s, but is not itself excluded from the process (a fresh process
    may still need to load/verify the on-disk numba cache).
    """
    config = build_converged_config(seed=seed)
    rng = np.random.default_rng(config.seed)

    t0 = time.perf_counter()
    warmup_parallel()
    warmup_s = time.perf_counter() - t0

    ru_before = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    result = run_simulation_numba_parallel(config, rng=rng, progress=progress, num_threads=num_threads)
    wall_s = time.perf_counter() - t0
    ru_after = resource.getrusage(resource.RUSAGE_SELF)

    return {
        "num_threads": num_threads,
        "warmup_s": warmup_s,
        "wall_s": wall_s,
        "user_s": ru_after.ru_utime - ru_before.ru_utime,
        "sys_s": ru_after.ru_stime - ru_before.ru_stime,
        "voluntary_ctx_switches": ru_after.ru_nvcsw - ru_before.ru_nvcsw,
        "involuntary_ctx_switches": ru_after.ru_nivcsw - ru_before.ru_nivcsw,
        "max_rss_kb": ru_after.ru_maxrss,
        "f_t_final": float(result.f_t[-1]),
        "ks": float(result.ks),
        "no_xy": config.no_xy,
        "no_z_with_buffer": config.no_z_with_buffer,
        "total_time_steps": config.total_time_steps,
        "tracks_per_pulse": config.number_of_tracks_per_pulse,
    }
