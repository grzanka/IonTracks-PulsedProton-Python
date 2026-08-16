"""Checks the kernels' own "scored" running totals -- `inserted` from
deposition, `recombined`/`total_positive`/`total_negative` from the
Lax-Wendroff sweep -- against an independent NumPy reduction over the same
arrays, using a voxel-set definition written from scratch here rather than
imported from the solver.

This is the conservation check issue #19 calls out as missing: every other
cross-check in this suite (`test_backends_agree.py`, the two-backend parity
tests in `test_v2_physics.py`) compares two *implementations* of the same
scoring logic, so a bug shared by both sides -- exactly what P1 (recombination
scored one fewer gap layer than deposition) and P2 (`full_grid` counting the
never-swept outer ring as injected) were -- is invisible to them. These tests
instead compare each kernel's running total against ground truth computed
directly from the array it just wrote.

Deliberately *not* a strict "injected - recombined == still present" identity
across a step: even immediately after a single deposit, a track's Gaussian
tails already extend past the scored disc/interior boundary (that is what
`docs/PHYSICS.md` sec. 11 calls `track_disc`'s documented cross-disc leak --
present from the moment of deposition, not something that only accumulates
over many steps), so "inserted" and "still present after one sweep" are
expected to differ by a few percent even with no bug at all. What must not
differ -- and is what P1/P2 actually broke -- is each kernel's own scored
total against ground truth over the voxel set *that kernel claims* to score.
"""

import numpy as np
import pytest

from pulsed_ion_chamber.config import SCORING_REGIONS, SimulationConfig
from pulsed_ion_chamber.constants import RECOMBINATION_ALPHA_CM3_S
from pulsed_ion_chamber.pulses import CylinderSampler
from pulsed_ion_chamber.solver_numba import _insert_track_numba, _lax_wendroff_step_numba

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
    seed=11,
)
N_TRACKS = 30


def _scored_mask(config) -> np.ndarray:
    """The (no_xy, no_xy) boolean voxel-column mask deposition and the sweep
    are both supposed to score against -- built from scratch (not imported
    from config.py or the solvers) so a shared bug in their predicate cannot
    also creep into the expected value here.
    """
    i = np.arange(config.no_xy)[:, None]
    j = np.arange(config.no_xy)[None, :]
    disc = (i - config.mid_xy) ** 2 + (j - config.mid_xy) ** 2 < config.scoring_radius_sq
    # The Lax-Wendroff sweep never writes the outer ring (it loops
    # i, j in [1, no_xy - 2]), so no column outside that interior can ever be
    # scored as recombined -- injection must not count it either (issue #19 P2).
    interior = np.zeros_like(disc)
    interior[1:-1, 1:-1] = True
    return disc & interior


def _deposit_and_sweep(config):
    """Deposit N_TRACKS tracks and run one Lax-Wendroff step directly through
    the jitted kernels -- the same two calls run_simulation_numba makes each
    step, but with the arrays kept around afterwards for inspection.
    """
    rng = np.random.default_rng(config.seed)
    sampler = CylinderSampler(rng, config.mid_xy, config.sampling_radius, config.no_xy)
    xs, ys = sampler.sample(N_TRACKS)

    shape = (config.no_xy, config.no_xy, config.no_z_with_buffer)
    positive = np.zeros(shape)
    negative = np.zeros(shape)
    positive_next = np.zeros(shape)
    negative_next = np.zeros(shape)

    h2 = config.unit_length_cm**2
    b2 = config.track_radius_cm**2
    inserted = 0.0
    for x, y in zip(xs, ys):
        inserted += _insert_track_numba(
            positive,
            negative,
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

    (p_lat, p_zm, p_zp, p_cen), (n_lat, n_zm, n_zp, n_cen) = config.scheme_coefficients()
    alpha_dt = RECOMBINATION_ALPHA_CM3_S * config.dt
    recombined, total_positive, total_negative = _lax_wendroff_step_numba(
        positive,
        negative,
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
    return dict(
        positive=positive,
        negative=negative,
        positive_next=positive_next,
        negative_next=negative_next,
        alpha_dt=alpha_dt,
        inserted=inserted,
        recombined=recombined,
        total_positive=total_positive,
        total_negative=total_negative,
    )


@pytest.mark.parametrize("scoring_region", SCORING_REGIONS)
def test_injected_charge_matches_an_independent_reduction_over_the_scored_voxels(scoring_region):
    """`_insert_track_numba`'s running `inserted` total must equal a plain
    NumPy sum over exactly the gap layers deposition wrote
    (`[no_z_electrode, no_z_electrode + no_z)`, unconditionally, for every
    track), restricted to `_scored_mask`. Directly catches P2: under
    `full_grid` scoring, `scoring_radius_sq` alone admits every column
    including the outer ring deposition also writes, so without the interior
    restriction `inserted` would exceed this independent total."""
    config = SimulationConfig(**SMALL, scoring_region=scoring_region)
    run = _deposit_and_sweep(config)

    mask = _scored_mask(config)
    gap = slice(config.no_z_electrode, config.no_z_electrode + config.no_z)
    expected_inserted = run["positive"][:, :, gap][mask].sum()

    assert run["inserted"] == pytest.approx(expected_inserted, rel=1e-9)


@pytest.mark.parametrize("scoring_region", SCORING_REGIONS)
def test_recombination_and_present_charge_are_scored_over_the_same_gap_layers_as_injection(
    scoring_region,
):
    """The sweep's `recombined`/`total_positive`/`total_negative` running
    totals must equal an independent NumPy reduction over *the same* voxel
    set injection uses -- same `_scored_mask`, same gap layers. Directly
    catches P1: with the pre-fix `no_z_electrode < k < ...` bound, the sweep
    scores one fewer gap layer than deposition wrote, so its totals fall
    short of this independent reduction by exactly the missing layer's
    contribution."""
    config = SimulationConfig(**SMALL, scoring_region=scoring_region)
    run = _deposit_and_sweep(config)

    mask = _scored_mask(config)
    gap = slice(config.no_z_electrode, config.no_z_electrode + config.no_z)
    expected_recombined = (run["alpha_dt"] * run["positive"] * run["negative"])[:, :, gap][mask].sum()
    expected_total_positive = run["positive_next"][:, :, gap][mask].sum()
    expected_total_negative = run["negative_next"][:, :, gap][mask].sum()

    assert run["recombined"] == pytest.approx(expected_recombined, rel=1e-9)
    assert run["total_positive"] == pytest.approx(expected_total_positive, rel=1e-9)
    assert run["total_negative"] == pytest.approx(expected_total_negative, rel=1e-9)
