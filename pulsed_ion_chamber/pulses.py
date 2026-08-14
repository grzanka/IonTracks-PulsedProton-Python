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
    cylinder (radius inner_radius, centered at (mid_xy, mid_xy)).

    One track at a time, at Python speed: ~2.4 us per track, which is 60 s of a
    24.6 M-track full-electrode run. :class:`CylinderSampler` draws the same
    numbers in blocks instead and is what the solvers use; this stays as the
    readable definition of what that class has to reproduce, and as the
    reference the tests check it against.
    """
    inner_radius_sq = inner_radius * inner_radius  # compare squared distances, avoid sqrt() per attempt
    while True:
        x = rng.uniform(0.0, 1.0) * no_xy
        y = rng.uniform(0.0, 1.0) * no_xy
        if (x - mid_xy) ** 2 + (y - mid_xy) ** 2 <= inner_radius_sq:
            return x, y


class CylinderSampler:
    """Vectorised rejection sampling that draws *exactly* the same numbers as
    calling :func:`sample_xy_inside_cylinder` once per track.

    Why not simply vectorise it
    ---------------------------
    The obvious batch version -- draw ``1.3 n`` candidate pairs, keep the ones
    inside the disc -- advances the RNG past what the scalar loop would have
    consumed, so every subsequent draw differs and a run's ``k_s`` changes in
    the 4th digit. The published tier results in ``examples/ifj_aic144`` are
    tied to a seed, so that is not an acceptable price for a speed-up.

    How exactness is kept
    ---------------------
    Both versions consume one flat stream of ``rng.random()`` doubles, two per
    rejection attempt, in order (``Generator.uniform(0.0, 1.0)`` is
    ``0.0 + 1.0 * next_double``, i.e. bit-for-bit ``Generator.random()``). So
    this class buffers that stream in blocks and consumes it *sequentially*,
    carrying whatever is left over into the next call rather than discarding it.
    Same doubles, same order, same accept/reject decisions -- the only thing
    that changes is that the accept test is a NumPy comparison over a whole
    block instead of a Python ``if`` per attempt.

    The RNG object itself ends the run further ahead than the scalar loop would
    leave it, by up to one unconsumed block. That is invisible here because
    nothing else draws from the generator once sampling starts (the pulse
    schedule is built first).
    """

    def __init__(self, rng: np.random.Generator, mid_xy: int, inner_radius: float, no_xy: int):
        self.rng = rng
        self.mid_xy = mid_xy
        self.inner_radius_sq = inner_radius * inner_radius
        self.no_xy = no_xy
        # Fraction of candidate pairs the disc accepts, used only to size the
        # blocks; a wrong guess costs an extra block, never correctness.
        area_ratio = np.pi * self.inner_radius_sq / (no_xy * no_xy)
        self._accept_rate = min(1.0, max(0.05, area_ratio))
        self._pool = np.empty(0)  # unconsumed doubles, still in draw order
        self._pool_pos = 0  # how much of _pool has been consumed

    def sample(self, n_tracks: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(xs, ys)``, ``n_tracks`` positions inside the disc."""
        if n_tracks == 0:
            return np.empty(0), np.empty(0)

        xs_parts: list[np.ndarray] = []
        ys_parts: list[np.ndarray] = []
        remaining = n_tracks

        while remaining > 0:
            pairs = self._pairs(remaining)
            x = pairs[:, 0] * self.no_xy
            y = pairs[:, 1] * self.no_xy
            accepted = (x - self.mid_xy) ** 2 + (y - self.mid_xy) ** 2 <= self.inner_radius_sq

            # How many attempts does it take to reach `remaining` acceptances?
            # cumsum + searchsorted answers that without a Python loop; if the
            # block does not contain enough, take all of it and draw again.
            n_accepted = np.cumsum(accepted)
            if n_accepted[-1] >= remaining:
                attempts = int(np.searchsorted(n_accepted, remaining) + 1)
            else:
                attempts = len(accepted)

            keep = accepted[:attempts]
            xs_parts.append(x[:attempts][keep])
            ys_parts.append(y[:attempts][keep])
            remaining -= int(keep.sum())
            self._pool_pos += 2 * attempts  # two doubles per attempt, consumed in order

        if len(xs_parts) == 1:
            return xs_parts[0], ys_parts[0]
        return np.concatenate(xs_parts), np.concatenate(ys_parts)

    def _pairs(self, n_wanted: int) -> np.ndarray:
        """A view of the unconsumed stream as candidate ``(x, y)`` pairs,
        refilling from the RNG when it has run dry."""
        available = (len(self._pool) - self._pool_pos) // 2
        if available == 0:
            # Enough attempts for `n_wanted` acceptances plus 20 %, floor 4096
            # so tiny steps do not pay a draw call each. Leftovers are kept, so
            # over-drawing costs nothing but a little memory.
            n_pairs = max(4096, int(n_wanted / self._accept_rate * 1.2) + 1)
            self._pool = self.rng.random(2 * n_pairs)
            self._pool_pos = 0
            available = n_pairs
        return self._pool[self._pool_pos : self._pool_pos + 2 * available].reshape(available, 2)
