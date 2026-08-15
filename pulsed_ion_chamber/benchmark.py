"""Cost model: size a run before committing to it.

Two ways to ask "how long will this take?" without running the whole thing:

* :func:`estimate_full_runtime` times a handful of *isolated* track
  insertions and PDE steps -- fast (a few ms) and allocates nothing but small
  scratch arrays, but on a batched (``--threads > 1``) run its per-track
  number comes from the unbatched kernel, which can be orders of magnitude
  pessimistic on a large grid (see its docstring).
* :func:`estimate_full_runtime_empirical` instead runs the real backend on
  the real grid -- real allocation, real JIT warmup, the real batched or
  unbatched code path -- for a short wall-clock budget and extrapolates from
  what that actually measured. Slower to call (default ~5 s, plus it commits
  the full grid's memory) but the number it produces is far closer to what a
  real run of this config would actually do -- same order of magnitude, not
  a proxy off by 2-3 orders of it.

A full-electrode run is a quarter of an hour and a badly chosen grid can be
much longer, so being able to ask either question in seconds is worth having.

Air's stopping power is ~1000x lower than water's, so a clinically relevant
dose needs an enormous number of tracks -- 24.6 million per pulse for the
full Markus electrode -- which is why deposition cost, not just grid size,
decides how long a run takes. See docs/PERFORMANCE.md.
"""

import time

import numpy as np

from pulsed_ion_chamber.constants import RECOMBINATION_ALPHA_CM3_S
from pulsed_ion_chamber.pulses import sample_xy_inside_cylinder
from pulsed_ion_chamber.resources import clamp_thread_count
from pulsed_ion_chamber.solver_numba import _insert_track_numba, _lax_wendroff_step_numba, run_simulation_numba, warmup
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel, warmup_parallel


def estimate_full_runtime(config, n_track_samples=10, n_step_samples=3, rng=None):
    """Time a few track insertions and PDE steps on the config's actual grid,
    then extrapolate to the full run without performing it.

    The two measured costs are also the diagnosis: comparing
    ``total_tracks * t_per_track_s`` against
    ``total_time_steps * t_per_pde_step_s`` says whether a given scenario is
    deposition-bound (raise the dose rate or shrink the grid) or PDE-bound
    (fewer steps, coarser grid), and therefore which knob is worth turning.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    shape = (config.no_xy, config.no_xy, config.no_z_with_buffer)
    positive_array = np.zeros(shape)
    negative_array = np.zeros(shape)
    positive_next = np.zeros(shape)
    negative_next = np.zeros(shape)

    (p_lat, p_zm, p_zp, p_cen), (n_lat, n_zm, n_zp, n_cen) = config.scheme_coefficients()

    warmup()  # one-off JIT compile, excluded from the timing below
    h2 = config.unit_length_cm**2
    b2 = config.track_radius_cm**2
    alpha_dt = RECOMBINATION_ALPHA_CM3_S * config.dt

    def insert_once(x, y):
        return _insert_track_numba(
            positive_array,
            negative_array,
            x,
            y,
            config.no_xy,
            config.no_z,
            config.no_z_electrode,
            h2,
            b2,
            config.Gaussian_factor,
            config.mid_xy,
            config.scoring_radius_sq,
            config.track_cutoff_voxels,
        )

    def step_once():
        return _lax_wendroff_step_numba(
            positive_array,
            negative_array,
            positive_next,
            negative_next,
            config.no_xy,
            config.no_z_with_buffer,
            config.no_z_electrode,
            config.no_z,
            config.mid_xy,
            config.scoring_radius_sq,
            p_lat,
            p_zm,
            p_zp,
            p_cen,
            n_lat,
            n_zm,
            n_zp,
            n_cen,
            alpha_dt,
        )

    t0 = time.perf_counter()
    for _ in range(n_track_samples):
        x, y = sample_xy_inside_cylinder(rng, config.mid_xy, config.sampling_radius, config.no_xy)
        insert_once(x, y)
    t_per_track = (time.perf_counter() - t0) / n_track_samples

    t0 = time.perf_counter()
    for _ in range(n_step_samples):
        step_once()
    t_per_step = (time.perf_counter() - t0) / n_step_samples  # step_once returns (recomb, n+, n-)

    total_tracks = config.number_of_tracks_per_pulse * config.n_pulses
    estimated_seconds = total_tracks * t_per_track + config.total_time_steps * t_per_step

    return {
        "t_per_track_s": t_per_track,
        "t_per_pde_step_s": t_per_step,
        "total_tracks": total_tracks,
        "total_time_steps": config.total_time_steps,
        "estimated_seconds": estimated_seconds,
        "estimated_hours": estimated_seconds / 3600.0,
        "estimated_days": estimated_seconds / 86400.0,
    }


def estimate_full_runtime_empirical(config, num_threads=1, max_wall_s=5.0, rng=None):
    """Run the *real* backend on the *real* grid for a short wall-clock
    budget, then extrapolate linearly from the measured per-step cost --
    instead of :func:`estimate_full_runtime`'s isolated single-track/single-step
    samples, which underrepresent the batched backend badly on a large grid
    (that function's own docstring situation: on `full_electrode` its
    per-track number implies ~21 h single-threaded, when the batched backend
    at 2 threads actually takes ~6.5 min -- a >150x gap, because the batched
    backend deposits a whole pulse's tracks as one blocked density draw, not
    one kernel call per track).

    This commits real memory (the four carrier arrays, same as a production
    run -- see ``config.estimated_memory_bytes``) and pays the one-off Numba
    JIT compile in full (excluded from the timing, same as a real run pays it
    once and amortises it). ``num_threads == 1`` uses the unbatched backend
    (:func:`~pulsed_ion_chamber.solver_numba.run_simulation_numba`); anything
    higher uses the batched one
    (:func:`~pulsed_ion_chamber.solver_numba_parallel.run_simulation_numba_parallel`)
    -- the same "auto" rule ``run_markus_2mm.py`` uses -- so the measured cost
    is the one the requested thread count would actually pay.

    The extrapolation is linear in step count and does not separate
    deposition-heavy (in-pulse) steps from PDE-only (clearance) ones the way
    :func:`estimate_full_runtime` does -- a short sample runs entirely inside
    the first pulse, so it cannot see the cheaper clearance-only steps that
    follow. Deposition is a minority of a step's cost on this backend
    (~15-20% at `full_electrode`, see docs/BENCHMARKS-LAPTOP.md sec. 3), so
    that effect alone would make this a mild over-estimate. In practice a
    second effect can outweigh it in the other direction: the sample is drawn
    from the very start of the run, on whatever machine it is called on. If
    that machine throttles under sustained load -- a laptop, not a cluster
    node (docs/BENCHMARKS-LAPTOP.md sec. 4) -- the first few seconds run on
    the coolest, fastest clock the run will ever see, and the real run's
    steady-state average is slower than that sample suggests. Measured on
    `full_electrode`, 2 threads, 10 Gy/s: a 5 s sample from the first 31 steps
    extrapolated to 356 s against an actual 382 s -- a 7% *under*-estimate,
    thermal throttling winning out over the deposition-share effect. Treat the
    result as a same-order-of-magnitude estimate, not a bound in either
    direction.

    If the whole run finishes inside ``max_wall_s`` (a small tier, or a
    generous budget), the result is exact -- an extrapolation of a complete
    run is just that run -- and ``exact`` is True.
    """
    rng = rng if rng is not None else np.random.default_rng(config.seed)
    batched = num_threads != 1

    if batched:
        threads_used = clamp_thread_count(num_threads)
        warmup_parallel()  # one-off JIT compile, excluded from the timing below
        t0 = time.perf_counter()
        result = run_simulation_numba_parallel(
            config, rng=rng, progress=False, num_threads=num_threads, max_wall_s=max_wall_s
        )
    else:
        threads_used = 1
        warmup()  # one-off JIT compile, excluded from the timing below
        t0 = time.perf_counter()
        result = run_simulation_numba(config, rng=rng, progress=False, max_wall_s=max_wall_s)
    setup_and_run_s = time.perf_counter() - t0  # includes schedule build; loop_elapsed_s below does not

    steps_measured = result.steps_completed
    elapsed_measured_s = result.loop_elapsed_s
    if steps_measured == 0:
        raise RuntimeError(
            f"max_wall_s={max_wall_s:.3g} s wasn't enough for even one step to complete "
            f"(checked only after each full step) -- raise max_wall_s."
        )
    exact = steps_measured >= config.total_time_steps
    estimated_seconds = (
        elapsed_measured_s if exact else elapsed_measured_s / steps_measured * config.total_time_steps
    )

    return {
        "backend": "solver_numba_parallel" if batched else "solver_numba",
        "num_threads": threads_used,
        "steps_measured": steps_measured,
        "total_time_steps": config.total_time_steps,
        "elapsed_measured_s": elapsed_measured_s,
        "ms_per_step_measured": elapsed_measured_s / steps_measured * 1e3,
        "setup_and_run_s": setup_and_run_s,
        "exact": exact,
        "estimated_seconds": estimated_seconds,
        "estimated_hours": estimated_seconds / 3600.0,
    }
