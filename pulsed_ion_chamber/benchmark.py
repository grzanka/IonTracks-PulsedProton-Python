"""Cost model: measure the actual per-track and per-time-step cost of the
hot loops on a given grid, then extrapolate the total wall-clock time a
full simulation would take -- without running it.

Why this exists: for a proton pulsed beam at clinically-relevant dose rates,
the number of tracks needed per pulse in a chamber-gas (air) volume even a
few ion-track-radii wide is large (air's stopping power is tiny, so a lot
of protons are needed to deposit a given dose), and inserting each track
costs O(no_xy^2 * no_z). Depending on the chosen sampled_radius_cm/
grid_size_um, a physically-realistic, dosimetrically-converged scenario can
still take minutes to hours even with the single-threaded Numba backend
that is the baseline for this repository (solver_numba.py). Rather than
silently shrinking the physics until it "fits" in a demo, this module gives
you a fast, honest estimate so you can decide whether to wait, reduce the
scope, or -- the actual point of this repository -- parallelize further
(numba prange, multiprocessing, or a GPU port).
"""

import time

import numpy as np

from pulsed_ion_chamber.constants import RECOMBINATION_ALPHA_CM3_S
from pulsed_ion_chamber.pulses import sample_xy_inside_cylinder
from pulsed_ion_chamber.solver import _insert_track, _lax_wendroff_step
from pulsed_ion_chamber.solver_numba import _insert_track_numba, _lax_wendroff_step_numba, warmup


def estimate_full_runtime(config, n_track_samples=10, n_step_samples=3, rng=None, backend="numba"):
    """Time a handful of track insertions and PDE steps on the config's
    actual grid, then extrapolate to the full total_time_steps/track count.

    backend: "numba" (the default/baseline backend, solver_numba.py) or
    "python" (the plain pure-Python reference, solver.py, for comparison).

    Returns a dict with the measured per-call costs and the extrapolated
    total wall-clock time (seconds/hours/days) for the full simulation.
    """
    if backend not in ("numba", "python"):
        raise ValueError(f"Unknown backend: {backend!r} (expected 'numba' or 'python')")

    rng = rng if rng is not None else np.random.default_rng(0)
    shape = (config.no_xy, config.no_xy, config.no_z_with_buffer)
    positive_array = np.zeros(shape)
    negative_array = np.zeros(shape)
    positive_next = np.zeros(shape)
    negative_next = np.zeros(shape)

    pos_weights, neg_weights = config.scheme_coefficients()
    (p_lat, p_zm, p_zp, p_cen), (n_lat, n_zm, n_zp, n_cen) = pos_weights, neg_weights

    if backend == "numba":
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

    else:  # backend == "python"

        def insert_once(x, y):
            return _insert_track(positive_array, negative_array, x, y, config)

        def step_once():
            return _lax_wendroff_step(
                positive_array, negative_array, positive_next, negative_next, config, pos_weights, neg_weights
            )

    t0 = time.perf_counter()
    for _ in range(n_track_samples):
        x, y = sample_xy_inside_cylinder(rng, config.mid_xy, config.sampling_radius, config.no_xy)
        insert_once(x, y)
    t_per_track = (time.perf_counter() - t0) / n_track_samples

    t0 = time.perf_counter()
    for _ in range(n_step_samples):
        step_once()
    t_per_step = (time.perf_counter() - t0) / n_step_samples

    total_tracks = config.number_of_tracks_per_pulse * config.n_pulses
    estimated_seconds = total_tracks * t_per_track + config.total_time_steps * t_per_step

    return {
        "backend": backend,
        "t_per_track_s": t_per_track,
        "t_per_pde_step_s": t_per_step,
        "total_tracks": total_tracks,
        "total_time_steps": config.total_time_steps,
        "estimated_seconds": estimated_seconds,
        "estimated_hours": estimated_seconds / 3600.0,
        "estimated_days": estimated_seconds / 86400.0,
    }
