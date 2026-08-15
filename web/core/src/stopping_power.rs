//! LET lookup, Gaussian track-radius estimate, and dose-rate -> fluence-rate
//! conversion. Ports `pulsed_ion_chamber/stopping_power.py`, protons in dry
//! air only (the other particles/materials in the original CSV aren't
//! carried over -- see the crate root docs for what this port narrows).

use crate::constants::{AIR_DENSITY_KG_M3, JOULE_TO_KEV};
use std::sync::OnceLock;

/// `E_MeV_u, proton_LET_keV_um` pairs, extracted from
/// `pulsed_ion_chamber/data/stopping_power_air.csv` (Libamtrack PSTAR table,
/// dry air). Sorted ascending by energy, which `let_kev_um` relies on for its
/// binary search.
const PSTAR_PROTON_AIR_CSV: &str = include_str!("data/pstar_proton_air.csv");

fn pstar_table() -> &'static Vec<(f64, f64)> {
    static TABLE: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        PSTAR_PROTON_AIR_CSV
            .lines()
            .skip(1) // header
            .filter(|line| !line.is_empty())
            .map(|line| {
                let mut parts = line.split(',');
                let e: f64 = parts.next().unwrap().parse().unwrap();
                let let_: f64 = parts.next().unwrap().parse().unwrap();
                (e, let_)
            })
            .collect()
    })
}

/// Linear interpolation of proton LET in dry air, matching
/// `scipy.interpolate.interp1d`'s default (linear) kind. Clamps to the
/// table's endpoints rather than extrapolating outside `[0.1, 500]` MeV/u.
pub fn let_kev_um(e_mev_u: f64) -> f64 {
    let table = pstar_table();
    let n = table.len();
    if e_mev_u <= table[0].0 {
        return table[0].1;
    }
    if e_mev_u >= table[n - 1].0 {
        return table[n - 1].1;
    }
    // First index whose energy is >= e_mev_u.
    let hi = table.partition_point(|&(e, _)| e < e_mev_u);
    let (e0, l0) = table[hi - 1];
    let (e1, l1) = table[hi];
    l0 + (l1 - l0) * (e_mev_u - e0) / (e1 - e0)
}

/// Quadratic fit of Gaussian track radius `b` [cm] vs. `log10(LET [keV/um])`,
/// from `pulsed_ion_chamber/data/LET_b.dat` (Rossomme et al.), floored at
/// 20 um. Coefficients below are `numpy.polyfit(log10(LET_keV_um), b_um, 2)`
/// on that table, computed once and hardcoded rather than re-fit at
/// startup -- ten points, refit only if `LET_b.dat` changes upstream.
const TRACK_RADIUS_FIT: [f64; 3] = [-0.389_496_85, 0.736_894_79, 5.153_815_26];
const TRACK_RADIUS_FLOOR_CM: f64 = 2e-3; // 20 um

pub fn track_radius_cm(let_kev_um: f64) -> f64 {
    let x = let_kev_um.log10();
    let [a, b, c] = TRACK_RADIUS_FIT;
    let b_um = a * x * x + b * x + c;
    (b_um * 1e-3).max(TRACK_RADIUS_FLOOR_CM)
}

/// Convert a dose-rate in dry air \[Gy/s\] to a fluence-rate \[cm^-2 s^-1\].
pub fn dose_rate_to_fluence_rate(dose_rate_gy_s: f64, e_mev_u: f64, air_density_kg_m3: f64) -> f64 {
    let let_kev_cm = let_kev_um(e_mev_u) * 1e4;
    let density_kg_cm3 = air_density_kg_m3 * 1e-6;
    dose_rate_gy_s * JOULE_TO_KEV * density_kg_cm3 / let_kev_cm
}

pub fn dose_rate_to_fluence_rate_default_air(dose_rate_gy_s: f64, e_mev_u: f64) -> f64 {
    dose_rate_to_fluence_rate(dose_rate_gy_s, e_mev_u, AIR_DENSITY_KG_M3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn let_matches_documented_reference_point() {
        // docs/PHYSICS.md sec. 2: 56.2 MeV protons in dry air -> 1.1995 eV/um.
        let let_ = let_kev_um(56.2);
        assert!((let_ - 1.1994844e-3).abs() < 2e-6, "got {let_}");
    }

    #[test]
    fn track_radius_floors_at_therapeutic_energy() {
        // docs/PHYSICS.md sec. 3: at 56.2 MeV/u the fit falls below the floor.
        let b = track_radius_cm(let_kev_um(56.2));
        assert!((b - 20e-4).abs() < 1e-9, "got {b}");
    }

    #[test]
    fn track_radius_above_floor_at_1_mev() {
        // At 1 MeV/u the fit sits above the floor (~30 um) -- this is the
        // energy tests/test_single_track_vs_jaffe.py uses, so the floor must
        // NOT silently kick in here or that cross-check is meaningless.
        let b = track_radius_cm(let_kev_um(1.0));
        assert!(b > 21e-4, "expected b above the 20um floor, got {b}");
    }
}
