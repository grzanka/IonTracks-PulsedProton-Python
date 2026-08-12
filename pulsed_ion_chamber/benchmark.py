"""Cost model: measure the actual per-track and per-time-step cost of the
explicit Python loops on a given grid, then extrapolate the total
wall-clock time a full simulation would take -- without running it.

Why this exists: for a proton pulsed beam at clinically-relevant dose rates,
the number of tracks needed per pulse in a chamber-gas (air) volume even a
few ion-track-radii wide is large (air's stopping power is tiny, so a lot
of protons are needed to deposit a given dose), and inserting each track
costs O(no_xy^2 * no_z) in this explicit-loop implementation. Depending on
the chosen sampled_radius_cm/grid_size_um, a physically-realistic scenario
can take from minutes to weeks in serial CPython. Rather than silently
shrinking the physics until it "fits" in a demo, this module gives you a
fast, honest estimate so you can decide whether to wait, reduce the scope,
or -- the actual point of this repository -- parallelize the two loops in
solver.py.
"""

import time

import numpy as np

from pulsed_ion_chamber.constants import ION_DIFFUSION_CM2_S, ION_MOBILITY_CM2_VS
from pulsed_ion_chamber.pulses import sample_xy_inside_cylinder
from pulsed_ion_chamber.solver import _insert_track, _lax_wendroff_step


def estimate_full_runtime(config, n_track_samples=10, n_step_samples=3, rng=None):
    """Time a handful of track insertions and PDE steps on the config's
    actual grid, then extrapolate to the full total_time_steps/track count.

    Returns a dict with the measured per-call costs and the extrapolated
    total wall-clock time (seconds/hours/days) for the full simulation.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    shape = (config.no_xy, config.no_xy, config.no_z_with_buffer)
    positive_array = np.zeros(shape)
    negative_array = np.zeros(shape)
    positive_next = np.zeros(shape)
    negative_next = np.zeros(shape)

    sx = ION_DIFFUSION_CM2_S * config.dt / config.unit_length_cm**2
    cz = ION_MOBILITY_CM2_VS * config.Efield_V_cm * config.dt / config.unit_length_cm
    sc_pos_z = sx + cz * (cz + 1.0) / 2.0
    sc_neg_z = sx + cz * (cz - 1.0) / 2.0
    sc_center = 1.0 - cz * cz - 6.0 * sx

    t0 = time.perf_counter()
    for _ in range(n_track_samples):
        x, y = sample_xy_inside_cylinder(rng, config.mid_xy, config.inner_radius, config.no_xy)
        _insert_track(positive_array, negative_array, x, y, config)
    t_per_track = (time.perf_counter() - t0) / n_track_samples

    t0 = time.perf_counter()
    for _ in range(n_step_samples):
        _lax_wendroff_step(
            positive_array, negative_array, positive_next, negative_next, config, sx, sc_pos_z, sc_neg_z, sc_center
        )
    t_per_step = (time.perf_counter() - t0) / n_step_samples

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
