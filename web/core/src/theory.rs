//! Jaffe theory: closed-form initial recombination for a single ion track,
//! valid in the low-dose / single-track limit. Ports the parts of
//! `pulsed_ion_chamber/theory.py` this crate's test suite needs as an
//! independent (RNG-free) correctness anchor -- not exposed to the wasm
//! surface, native-only (`#[cfg(test)]` is the only caller today).

use crate::constants::{
    ION_DIFFUSION_CM2_S, ION_MOBILITY_CM2_VS, RECOMBINATION_ALPHA_CM3_S, W_EV_PER_ION_PAIR,
};
use crate::stopping_power::track_radius_cm;
use std::f64::consts::PI;

const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Exponential integral `Ei(x)` for `x > 0`, via the convergent series
/// `Ei(x) = gamma + ln(x) + sum_{k>=1} x^k / (k * k!)` (Abramowitz & Stegun
/// 5.1.11). Every term is positive for `x > 0`, so there is no cancellation
/// to worry about -- just a loop until terms stop mattering, which for the
/// `x` this module evaluates (tens, from `1/g` below) converges in well
/// under 200 terms.
fn ei(x: f64) -> f64 {
    debug_assert!(x > 0.0, "Ei(x) here is only implemented for x > 0");
    let mut term = 1.0; // x^k / k!, updated incrementally
    let mut sum = 0.0;
    let mut k = 1u32;
    loop {
        term *= x / k as f64;
        let addend = term / k as f64;
        sum += addend;
        if addend < sum.max(1.0) * 1e-17 || k > 2000 {
            break;
        }
        k += 1;
    }
    EULER_GAMMA + x.ln() + sum
}

/// Jaffe-theory recombination correction factor `k_s = 1/f` for a single
/// track, using the averaged mobility/diffusion pair (matches this crate's
/// single-species solver, not the two-species Kanai values `theory.py` also
/// supports).
pub fn jaffe_ks(let_kev_um: f64, voltage_v: f64, electrode_gap_cm: f64) -> f64 {
    let let_ev_cm = let_kev_um * 1e7;
    let electric_field = voltage_v / electrode_gap_cm;
    let b_cm = track_radius_cm(let_kev_um);

    let n0 = let_ev_cm / W_EV_PER_ION_PAIR;
    let g = RECOMBINATION_ALPHA_CM3_S * n0 / (8.0 * PI * ION_DIFFUSION_CM2_S);

    let factor = (-1.0 / g).exp() * ION_MOBILITY_CM2_VS * b_cm * b_cm * electric_field
        / (2.0 * g * electrode_gap_cm * ION_DIFFUSION_CM2_S);
    let first_term = ei(1.0 / g
        + (1.0
            + (2.0 * electrode_gap_cm * ION_DIFFUSION_CM2_S)
                / (ION_MOBILITY_CM2_VS * b_cm * b_cm * electric_field))
            .ln());
    let second_term = ei(1.0 / g);
    let f = factor * (first_term - second_term);
    1.0 / f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ei_matches_known_values() {
        // Reference values computed with mpmath (mp.dps=30): `mpmath.ei(x)`.
        assert!((ei(1.0) - 1.895_117_816_355_936_8).abs() < 1e-12);
        assert!((ei(5.0) - 40.185_275_355_803_18).abs() < 1e-9);
        assert!((ei(20.0) - 25_615_652.664_056_59).abs() / 25_615_652.664_056_59 < 1e-9);
    }

    #[test]
    fn jaffe_ks_matches_python_reference() {
        // `pulsed_ion_chamber.theory.jaffe_ks(E_MeV_u_to_LET_keV_um(1.0, "proton"), 10.0, 0.02)`
        // == 1.0012341579794355, computed directly from this repo's Python
        // implementation (mpmath, 50 digits of precision) -- the scenario
        // `tests/test_single_track_vs_jaffe.py` uses.
        let let_ = crate::stopping_power::let_kev_um(1.0);
        let ks = jaffe_ks(let_, 10.0, 0.02);
        assert!((ks - 1.001_234_157_979_435_5).abs() < 1e-9, "got {ks}");
    }
}
