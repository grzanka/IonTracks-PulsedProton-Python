"""What the tabulated-RDD track model has to get right.

The Gaussian model is self-checking in a way this one is not: it has a closed
form, so a wrong stencil shows up as a wrong `k_s` against Jaffe. A tabulated
RDD has no analytic partner, so the properties below are the whole safety net.
The load-bearing one is `test_in_domain_fraction_is_grid_independent` -- it is
what separates area-averaging from point-sampling, and point-sampling a `1/r^2`
profile passes every other test here while silently losing charge as `h` grows.
"""

import numpy as np
import pytest

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.constants import (
    ION_DIFFUSION_NEGATIVE_CM2_S,
    ION_DIFFUSION_POSITIVE_CM2_S,
    ION_MOBILITY_NEGATIVE_CM2_VS,
    ION_MOBILITY_POSITIVE_CM2_VS,
)
from pulsed_ion_chamber.rdd import (
    KEV_PER_JOULE,
    RadialDoseDistribution,
    build_track_stencil,
    chamber_ks,
)
from pulsed_ion_chamber.solver_numba import run_simulation_numba
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel
from pulsed_ion_chamber.stopping_power import E_MeV_u_to_LET_keV_um

RDD_CSV = "examples/fe90_air/data/rdd_cucinotta_fe90_air.csv"
AIR_G_CM3 = 1.225e-3
W_EV = 34.2

TWO_SPECIES = dict(
    mu_positive_cm2_Vs=ION_MOBILITY_POSITIVE_CM2_VS,
    mu_negative_cm2_Vs=ION_MOBILITY_NEGATIVE_CM2_VS,
    D_positive_cm2_s=ION_DIFFUSION_POSITIVE_CM2_S,
    D_negative_cm2_s=ION_DIFFUSION_NEGATIVE_CM2_S,
)


@pytest.fixture(scope="module")
def fe90():
    return RadialDoseDistribution.from_csv(RDD_CSV, density_g_cm3=AIR_G_CM3, r_unit="m")


def power_law_rdd(exponent=-2.0, r_lo=1e-6, r_hi=1e-1, n=2000, density_g_cm3=AIR_G_CM3):
    """A pure `D ~ r^exponent` table, for the checks that want a known answer."""
    r = np.geomspace(r_lo, r_hi, n)
    return RadialDoseDistribution(r_cm=r, dose_Gy=r**exponent, density_g_cm3=density_g_cm3)


# --- the table itself -------------------------------------------------------


def test_table_integrates_to_the_stopping_power_let(fe90):
    """The RDD's own radial integral must reproduce the LET from an independent
    Bethe stopping-power table. This is the one check that catches a wrong
    `r_unit` or a wrong gas density, both of which are silent otherwise."""
    bethe = E_MeV_u_to_LET_keV_um(90.0, "iron")
    assert fe90.LET_keV_um == pytest.approx(bethe, rel=0.02)


def test_metres_and_centimetres_disagree_by_the_right_factor(tmp_path):
    """Reading metres as centimetres shrinks every radius 100x, which scales the
    energy integral by 1e-4 -- large enough that the LET check above would
    catch it, which is the point of having both."""
    table = tmp_path / "rdd.csv"
    table.write_text('"r","D"\n1e-6,1e3\n1e-5,1e1\n1e-4,1e-1\n')
    in_m = RadialDoseDistribution.from_csv(table, density_g_cm3=AIR_G_CM3, r_unit="m")
    in_cm = RadialDoseDistribution.from_csv(table, density_g_cm3=AIR_G_CM3, r_unit="cm")
    assert in_m.LET_keV_cm == pytest.approx(in_cm.LET_keV_cm * 1e4, rel=1e-12)


def test_fraction_within_is_monotone_and_bracketed(fe90):
    radii = np.geomspace(1e-6, 1e1, 40)
    fractions = np.array([fe90.fraction_within(r) for r in radii])
    assert np.all(np.diff(fractions) >= 0)
    assert fractions[0] >= 0.0
    assert fractions[-1] == pytest.approx(1.0, abs=1e-12)


def test_penumbra_deposits_uniformly_per_decade():
    """For an exact 1/r^2 profile, 2*pi*r*D is 1/r, so equal decades of radius
    carry equal energy. This is the structural fact behind the whole
    out-of-domain correction, so it is worth pinning."""
    rdd = power_law_rdd(exponent=-2.0, r_lo=1e-6, r_hi=1e-1)
    per_decade = [rdd.fraction_within(10.0**k) for k in range(-6, 0)]
    steps = np.diff(per_decade)
    assert np.allclose(steps, steps[0], rtol=2e-3)


def test_rejects_malformed_tables():
    with pytest.raises(ValueError, match="strictly increasing"):
        RadialDoseDistribution(r_cm=np.array([1.0, 0.5]), dose_Gy=np.array([1.0, 1.0]), density_g_cm3=1e-3)
    with pytest.raises(ValueError, match="at least 2 points"):
        RadialDoseDistribution(r_cm=np.array([1.0]), dose_Gy=np.array([1.0]), density_g_cm3=1e-3)
    with pytest.raises(ValueError, match="same length"):
        RadialDoseDistribution(r_cm=np.array([1.0, 2.0]), dose_Gy=np.array([1.0]), density_g_cm3=1e-3)


# --- the stencil ------------------------------------------------------------


@pytest.mark.parametrize("h_um,no_xy", [(10, 28), (5, 56), (2.5, 112), (5, 248)])
def test_stencil_conserves_energy_exactly(fe90, h_um, no_xy):
    """Summing the stencil back to keV/cm must return exactly what the builder
    says it deposited -- the conversion to density is the only step between
    them, and a bug there would rescale every run."""
    h = h_um * 1e-4
    stencil = build_track_stencil(fe90, h, no_xy, (no_xy / 2.0, no_xy / 2.0), W_eV=W_EV)
    recovered = stencil.density_cm3.sum() * h**2 * W_EV / 1e3
    assert recovered == pytest.approx(stencil.deposited_keV_per_cm, rel=1e-12)


@pytest.mark.parametrize("h_um,no_xy", [(10, 28), (5, 56), (2.5, 112), (1.25, 224)])
def test_in_domain_fraction_is_grid_independent(fe90, h_um, no_xy):
    """Four grids covering the *same* 280 um column at four spacings must all
    hold the same share of the track's energy.

    This is the property area-averaging exists for. Point-sampling `D()` at
    voxel centres instead makes this fraction drift with `h` on a `1/r^2`
    profile, which would show up as `k_s` converging to the wrong thing for a
    purely numerical reason.
    """
    assert no_xy * h_um == pytest.approx(280.0)
    stencil = build_track_stencil(fe90, h_um * 1e-4, no_xy, (no_xy / 2.0, no_xy / 2.0), W_eV=W_EV)
    assert stencil.in_domain_fraction == pytest.approx(0.7205, abs=2e-3)


def test_far_field_density_matches_the_analytic_area_average():
    """Where the profile varies little across a voxel, the area-average reduces
    to a closed form: a voxel at radius r holds `rho*D(r)*h^2` of energy per cm
    of track, so its carrier density is `rho*D(r)/W` -- independent of `h`.

    An independent check of the angular quadrature: it constrains where the
    energy is put, not just how much of it there is, which conservation alone
    cannot do.

    Compared shell by shell rather than voxel by voxel. Voxels binned at the
    same nominal radius sit at genuinely different distances -- across one
    shell at 5 voxels out, `1/r^2` varies by 50 % -- so individual voxels
    scatter around the closed form for a physical reason, not a numerical one.
    The shell mean is the quantity the formula actually predicts.
    """
    rdd = power_law_rdd(exponent=-2.0, r_lo=1e-7, r_hi=1e0, n=4000)
    h, no_xy = 1e-3, 200
    stencil = build_track_stencil(rdd, h, no_xy, (no_xy / 2.0, no_xy / 2.0), W_eV=W_EV)

    centre = no_xy // 2
    i, j = np.mgrid[0:no_xy, 0:no_xy]
    distance = np.hypot(i - centre, j - centre)
    for offset in (20, 40, 80):
        expected = rdd.density_g_cm3 * 1e-3 * ((offset * h) ** -2.0) * KEV_PER_JOULE * 1e3 / W_EV
        shell = stencil.density_cm3[(distance >= offset - 0.5) & (distance < offset + 0.5)]
        assert shell.mean() == pytest.approx(expected, rel=0.01)


def test_refining_the_grid_concentrates_the_core(fe90):
    """Halving `h` must raise the peak density, and by roughly 4x once the
    profile is locally `1/r^2` -- the reason `k_s` cannot be converged at 5 um."""
    peaks = []
    for h_um, no_xy in ((10, 28), (5, 56), (2.5, 112), (1.25, 224)):
        stencil = build_track_stencil(fe90, h_um * 1e-4, no_xy, (no_xy / 2.0, no_xy / 2.0), W_eV=W_EV)
        peaks.append(stencil.density_cm3.max())
    ratios = np.array(peaks[1:]) / np.array(peaks[:-1])
    assert np.all(ratios > 3.0)
    assert np.all(ratios < 4.5)


def test_stencil_is_centred_where_it_was_asked_to_be(fe90):
    no_xy = 56
    stencil = build_track_stencil(fe90, 5e-4, no_xy, (no_xy / 2.0, no_xy / 2.0), W_eV=W_EV)
    peak = np.unravel_index(np.argmax(stencil.density_cm3), stencil.density_cm3.shape)
    assert peak == (no_xy // 2, no_xy // 2)


# --- the chamber correction -------------------------------------------------


def test_chamber_ks_is_identity_when_nothing_is_missing():
    assert chamber_ks(1.63, 1.0) == pytest.approx(1.63)


def test_chamber_ks_moves_the_loss_not_the_efficiency():
    """Charge outside the grid is collected in full, so it dilutes the loss by
    exactly its share: (1 - 1/ks) must scale by the in-domain fraction."""
    ks_domain, fraction = 1.629450, 0.7205
    ks_total = chamber_ks(ks_domain, fraction)
    assert (1.0 - 1.0 / ks_total) == pytest.approx(fraction * (1.0 - 1.0 / ks_domain), rel=1e-12)
    assert 1.0 < ks_total < ks_domain


def test_chamber_ks_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        chamber_ks(1.5, 0.0)
    with pytest.raises(ValueError):
        chamber_ks(1.5, 1.5)


# --- config wiring ----------------------------------------------------------


def fe90_config(**overrides):
    kwargs = dict(
        E_MeV_u=90.0,
        particle="iron",
        voltage_V=200.0,
        electrode_gap_cm=0.2,
        pulse_duration_s=1e-7,
        repetition_rate_hz=50.0,
        dose_rate_Gy_s=1e-12,
        n_pulses=1,
        grid_size_um=20.0,
        sampled_radius_cm=0.012,
        buffer_radius=2,
        no_z_electrode=3,
        memory_budget_fraction=None,
        rdd_csv=RDD_CSV,
        n_tracks=1,
        track_placement="axis",
        lateral_boundary="absorbing",
        scoring_region="full_grid",
        seed=1,
    )
    kwargs.update(TWO_SPECIES)
    kwargs.update(overrides)
    return SimulationConfig(**kwargs)


def test_n_tracks_overrides_the_dose_rate_derivation():
    assert fe90_config(n_tracks=7).number_of_tracks_per_pulse == 7


def test_rdd_requires_axis_placement():
    with pytest.raises(ValueError, match="track_placement='axis'"):
        fe90_config(track_placement="random")


def test_rdd_refuses_a_pulse_it_would_be_wrong_for():
    with pytest.raises(ValueError, match="rdd_max_tracks"):
        fe90_config(n_tracks=1000)


def test_gaussian_path_is_untouched_by_the_new_fields():
    """No rdd_csv means no stencil, no correction, and the Gaussian quantities
    still resolved -- the default run must not have moved."""
    config = SimulationConfig(E_MeV_u=150.0, grid_size_um=40.0, memory_budget_fraction=None)
    assert config.track_stencil is None
    assert config.rdd is None
    assert config.in_domain_let_fraction == 1.0
    assert config.track_placement == "random"
    assert config.Gaussian_factor > 0


# --- solver integration -----------------------------------------------------


def test_injected_charge_matches_the_stencil():
    """The books have to close: what the solver reports as injected must be the
    stencil summed over every gap layer -- over the swept interior only, since
    the never-swept outer ring cannot recombine and so is not scored."""
    config = fe90_config()
    result = run_simulation_numba(config, progress=False)
    expected = config.track_stencil.density_cm3[1:-1, 1:-1].sum() * config.no_z
    assert result.injected_positive.sum() == pytest.approx(expected, rel=1e-10)


def test_both_backends_agree_on_the_rdd_path():
    """The batched backend reaches the same broadcast kernel by a different
    route; if the two ever disagree, one of the two deposition branches is
    wrong."""
    config = fe90_config()
    serial = run_simulation_numba(config, progress=False)
    batched = run_simulation_numba_parallel(fe90_config(), progress=False, num_threads=2)
    assert batched.ks == pytest.approx(serial.ks, rel=1e-9)


def test_axis_placement_puts_every_track_on_the_centre_column():
    config = fe90_config(n_tracks=3)
    result = run_simulation_numba(config, progress=False)
    assert result.track_density_xy[config.no_xy // 2, config.no_xy // 2] == 3.0
    assert result.track_density_xy.sum() == 3.0


def test_rdd_recombines_far_more_than_the_gaussian_of_the_same_track():
    """Same ion, same grid, same field -- only the radial profile differs. The
    Cucinotta core is orders of magnitude denser than the Rossomme Gaussian, so
    the loss must be much larger, and it is the whole reason for this module."""
    rdd_ks = run_simulation_numba(fe90_config(), progress=False).ks
    gauss_ks = run_simulation_numba(
        fe90_config(rdd_csv=None, track_placement="axis"), progress=False
    ).ks
    assert rdd_ks - 1.0 > 5.0 * (gauss_ks - 1.0)
