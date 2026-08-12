"""Pulse-train track scheduling: when (which time step) and where (which
xy position) each ion track is injected into the grid.

Track arrival times within a pulse are spread out using the same
cumulative-sum-of-uniforms trick as hadrons/*/continuous_beam.py* (an easy
way to get an increasing, roughly-Poisson-like sequence of arrival times
without rejection sampling in time); xy positions are rejection-sampled
uniformly inside the sampled cylinder, exactly as in the original code.
"""

from math import sqrt

import numpy as np


def build_track_schedule(config, rng: np.random.Generator) -> np.ndarray:
    """Return an int array of length config.total_time_steps: the number of
    new tracks to insert at each time step, repeated every pulse_period_steps
    for config.n_pulses pulses.
    """
    schedule = np.zeros(config.total_time_steps, dtype=np.int64)
    counts = _sample_pulse_arrival_histogram(config, rng)
    for pulse_index in range(config.n_pulses):
        start = pulse_index * config.pulse_period_steps
        schedule[start : start + len(counts)] += counts
    return schedule


def _sample_pulse_arrival_histogram(config, rng: np.random.Generator) -> np.ndarray:
    n_tracks = config.number_of_tracks_per_pulse
    randomized = rng.random(n_tracks)
    summed = np.cumsum(randomized)
    distributed_times = summed / summed[-1] * config.pulse_duration_s
    counts, _ = np.histogram(distributed_times, config.pulse_time_bins)
    return counts.astype(np.int64)


def sample_xy_inside_cylinder(rng: np.random.Generator, mid_xy: int, inner_radius: float, no_xy: int):
    """Rejection-sample a grid coordinate (x, y) uniformly inside the sampled
    cylinder (radius inner_radius, centered at (mid_xy, mid_xy))."""
    while True:
        x = rng.uniform(0.0, 1.0) * no_xy
        y = rng.uniform(0.0, 1.0) * no_xy
        if sqrt((x - mid_xy) ** 2 + (y - mid_xy) ** 2) <= inner_radius:
            return x, y
