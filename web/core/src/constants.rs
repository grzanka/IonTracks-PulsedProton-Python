//! Physical constants for ion transport in air, Kanai et al. (1998).
//!
//! Mirrors `pulsed_ion_chamber/constants.py`'s *averaged* pair (one mobility,
//! one diffusion coefficient shared by both carriers) -- the two-species
//! option that file also defines is not ported here; see the crate root docs.

/// eV, mean energy to create an ion pair in air (protons).
pub const W_EV_PER_ION_PAIR: f64 = 34.2;
/// cm^2 / (V s), averaged over positive/negative ions.
pub const ION_MOBILITY_CM2_VS: f64 = 1.65;
/// cm^2 / s, averaged over positive/negative ions.
pub const ION_DIFFUSION_CM2_S: f64 = 3.7e-2;
/// cm^3 / s, recombination coefficient.
pub const RECOMBINATION_ALPHA_CM3_S: f64 = 1.60e-6;

/// dry air, standard conditions (ISA sea level, 15 degC).
pub const AIR_DENSITY_KG_M3: f64 = 1.225;
/// 1 J expressed in keV.
pub const JOULE_TO_KEV: f64 = 6.241e15;
