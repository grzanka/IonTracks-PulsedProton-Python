"""Pulse-train track scheduling: when (which time step) and where (which
xy position) each ion track is injected into the grid.

Track arrival times within a pulse are spread out using the same
cumulative-sum-of-uniforms trick as hadrons/*/continuous_beam.py* (an easy
way to get an increasing, roughly-Poisson-like sequence of arrival times
without rejection sampling in time); xy positions are rejection-sampled
uniformly inside the sampled cylinder, exactly as in the original code.

Note this spreads tracks pseudo-uniformly across the *whole* pulse, not at
exact accelerator-RF-bucket times (e.g. a cyclotron's ~10-100 MHz
extraction RF) -- see SimulationConfig.rf_frequency_hz/rf_cycles_per_time_step
in config.py for why that's the correct simplification here rather than a
shortcut: the simulation's time step is always far coarser than one RF
period, so individual buckets can't be resolved regardless.
"""

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
    summed = np.cumsum(rng.random(n_tracks))
    # Rescaled in place: at tens of millions of tracks each full-size float64
    # temporary is hundreds of megabytes, and this runs before the carrier
    # arrays are allocated, so it sets the run's peak footprint.
    summed /= summed[-1]
    summed *= config.pulse_duration_s
    counts, _ = np.histogram(summed, config.pulse_time_bins)
    return counts.astype(np.int64)


def sample_xy_inside_cylinder(
    rng: np.random.Generator, mid_xy: int, inner_radius: float, no_xy: int
) -> tuple[float, float]:
    """Rejection-sample a grid coordinate (x, y) uniformly inside the sampled
    cylinder (radius inner_radius, centered at (mid_xy, mid_xy))."""
    inner_radius_sq = inner_radius * inner_radius  # compare squared distances, avoid sqrt() per attempt
    while True:
        x = rng.uniform(0.0, 1.0) * no_xy
        y = rng.uniform(0.0, 1.0) * no_xy
        if (x - mid_xy) ** 2 + (y - mid_xy) ** 2 <= inner_radius_sq:
            return x, y
