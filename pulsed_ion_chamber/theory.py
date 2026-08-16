"""Analytic reference theories used to sanity-check the PDE simulation.

- Jaffe theory: closed-form initial recombination for a single ion track
  (valid in the low-dose / single-track limit). Ported from
  hadrons/functions.py::Jaffe_theory.
- Boag theory: closed-form recombination for *uniform* charge-carrier
  densities (pulsed and continuous beams). Ported from
  electrons/Boag_theory.py. Strictly speaking this applies to electron/
  photon beams with uniform ionization density, not to the non-uniform
  track structure of a proton beam -- it is included only as a rough,
  order-of-magnitude cross-check for the "general recombination"
  (multi-track overlap) regime, not as an exact reference for protons.

References (see original IonTracks README/documentation):
  Jaffe, G. (1913) theory of initial recombination.
  Boag, J.W. and Currant, J. (1980) "Current collection and ionic
    recombination in small cylindrical ionization chambers exposed to
    pulsed radiation", Br. J. Radiol. 53.
  Boutillon, M. (1998) "Volume recombination parameter in ionization
    chambers", Phys. Med. Biol. 43.
  Liszka et al (2018) "Ion recombination and polarity correction factors
    for a plane-parallel ionization chamber in a proton scanning beam".
"""

from math import log, pi

import mpmath
import numpy as np

from pulsed_ion_chamber.constants import ION_DIFFUSION_CM2_S, ION_MOBILITY_CM2_VS, W_EV_PER_ION_PAIR
from pulsed_ion_chamber.stopping_power import calc_track_radius_cm

mpmath.mp.dps = 50  # precision for the exponential-integral evaluation

# Boag theory constants (separate positive/negative ion mobilities, unlike
# the "averaged" mobility used in the track-structure PDE solver).
_ALPHA_CM3_S = 1.60e-6
_K1_CM2_VS = 1.36  # positive ions
_K2_CM2_VS = 2.1  # negative ions
_E_CHARGE_C = 1.60217662e-19


def jaffe_ks(LET_keV_um, voltage_V, electrode_gap_cm, W_eV=None, mu=None, D=None, alpha=None):
    """Jaffe theory recombination correction factor k_s = 1/f for a single track.

    ``W_eV``, ``mu``, ``D`` and ``alpha`` default to this module's constants
    (the original IonTracks-Cython single-averaged-species values), which
    reproduces every call site that predates these parameters exactly. Pass a
    ``SimulationConfig``'s own ``W_eV``/``mu_positive``/``D_positive``
    whenever the solver run being cross-checked overrode them -- recombination
    scales as ``N0**2`` (via ``W_eV``), so comparing against a reference that
    silently kept the default is a real error, not a rounding one: a config
    with ``W_eV=33.0`` (~3.6% below the 34.2 default) recombines ~7% more than
    a Jaffe reference computed with the default would predict (issue #19 P4).
    ``alpha`` is this module's own averaged-species Boag/Jaffe constant
    (``_ALPHA_CM3_S``), independent of ``constants.RECOMBINATION_ALPHA_CM3_S``
    the PDE solver uses -- pass it explicitly to compare like with like if a
    config ever diverges from the default there too.
    """
    W_eV = W_EV_PER_ION_PAIR if W_eV is None else W_eV
    mu = ION_MOBILITY_CM2_VS if mu is None else mu
    D = ION_DIFFUSION_CM2_S if D is None else D
    alpha = _ALPHA_CM3_S if alpha is None else alpha

    LET_eV_cm = LET_keV_um * 1e7
    electric_field = voltage_V / electrode_gap_cm
    b_cm = calc_track_radius_cm(LET_keV_um)

    N0 = LET_eV_cm / W_eV
    g = alpha * N0 / (8.0 * pi * D)

    factor = mpmath.exp(-1.0 / g) * mu * b_cm**2 * electric_field / (2.0 * g * electrode_gap_cm * D)
    first_term = mpmath.ei(1.0 / g + log(1.0 + (2.0 * electrode_gap_cm * D / (mu * b_cm**2 * electric_field))))
    second_term = mpmath.ei(1.0 / g)
    f = factor * (first_term - second_term)
    return float(1.0 / f)


def boag_pulsed_f(charge_density_C_cm3, electrode_gap_cm, voltage_V):
    """Collection efficiency f for one pulse of *uniform* charge density (Boag & Currant 1980)."""
    mu = _ALPHA_CM3_S / (_E_CHARGE_C * (_K1_CM2_VS + _K2_CM2_VS))
    u = mu * charge_density_C_cm3 * electrode_gap_cm**2 / voltage_V
    return float(np.log(1 + u) / u)


def boag_continuous_f(charge_density_rate_C_cm3_s, electrode_gap_cm, voltage_V):
    """Collection efficiency f for a *uniform*, continuous charge-density rate (Boutillon 1998)."""
    mu_c = _ALPHA_CM3_S / (6 * _E_CHARGE_C * _K1_CM2_VS * _K2_CM2_VS)
    xi_squared = mu_c * (electrode_gap_cm**4 / voltage_V**2) * charge_density_rate_C_cm3_s
    return float(1.0 / (1 + xi_squared))
