"""Cost model: size a run before committing to it.

Times a handful of track depositions and PDE steps on the config's actual
grid, then extrapolates. A full-electrode run is a quarter of an hour and a
badly chosen grid can be much longer, so being able to ask "how long?" in a
second is worth having.

Air's stopping power is ~1000x lower than water's, so a clinically relevant
dose needs an enormous number of tracks -- 24.6 million per pulse for the
full Markus electrode -- which is why deposition cost, not just grid size,
decides how long a run takes. See docs/PERFORMANCE.md.
"""

import time

import numpy as np

from pulsed_ion_chamber.constants import RECOMBINATION_ALPHA_CM3_S
from pulsed_ion_chamber.pulses import sample_xy_inside_cylinder
from pulsed_ion_chamber.solver_numba import _insert_track_numba, _lax_wendroff_step_numba, warmup


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
