//! Derived simulation configuration -- the Rust analogue of
//! `SimulationConfig.__post_init__` in `pulsed_ion_chamber/config.py`,
//! narrowed to what this browser prototype exposes. See the crate root docs
//! for the full list of what is fixed vs. adjustable and why.

use crate::constants::{
    ION_DIFFUSION_CM2_S, ION_MOBILITY_CM2_VS, RECOMBINATION_ALPHA_CM3_S, W_EV_PER_ION_PAIR,
};
use crate::stopping_power::{dose_rate_to_fluence_rate_default_air, let_kev_um, track_radius_cm};
use std::f64::consts::PI;

// --- fixed knobs (docs/PERFORMANCE.md's cost-model levers, not physics
// intuition -- see issue #6 sec. 6 for why these are hidden rather than
// exposed in v1) ---
pub const GRID_SIZE_UM: f64 = 10.0;
pub const BUFFER_RADIUS: i64 = 3;
pub const NO_Z_ELECTRODE: i64 = 3;
pub const TRACK_CUTOFF_SIGMAS: f64 = 10.0;
pub const REPETITION_RATE_HZ: f64 = 50.0;
pub const N_CLEARANCE_SEPARATION_TIMES: f64 = 2.0;

// --- hard browser-safety ceilings, independent of the UI's own slider
// ranges (defense in depth -- see issue #6 sec. 5) ---
pub const MAX_TOTAL_VOXELS: i64 = 2_000_000; // ~5x the "wide" tier
pub const MAX_MEMORY_BYTES: f64 = 256.0 * 1024.0 * 1024.0;
pub const MAX_TRACKS_PER_PULSE: i64 = 2_000_000;
pub const MAX_TOTAL_TIME_STEPS: i64 = 200_000;

/// The knobs this prototype exposes to the user. Everything else
/// (`particle`, grid spacing, boundary condition, single- vs. two-species,
/// ...) is fixed -- see module docs.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub e_mev_u: f64,
    pub voltage_v: f64,
    pub electrode_gap_cm: f64,
    pub dose_rate_gy_s: f64,
    pub pulse_duration_s: f64,
    pub sampled_radius_cm: f64,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Lax-Wendroff stencil weights for one carrier species: `(lateral, z_minus,
/// z_plus, centre)`. Both species use the same weights here since this
/// prototype only implements the single averaged-species model (config.py's
/// default) -- see `docs/PHYSICS.md` sec. 8 for the two-species option this
/// intentionally leaves out.
#[derive(Clone, Copy, Debug)]
pub struct SchemeCoefficients {
    pub lateral: f64,
    pub z_minus: f64,
    pub z_plus: f64,
    pub centre: f64,
}

// Several fields below aren't read anywhere in this crate yet -- they're
// derived quantities kept on the struct because they're the natural unit of
// documentation for what a config means (and candidates for a future,
// richer `estimate()` payload), not because something consumes them today.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Config {
    pub params: Params,

    pub unit_length_cm: f64,
    pub let_kev_um: f64,
    pub track_radius_cm: f64,
    pub efield_v_cm: f64,

    pub no_xy: i64,
    pub no_z: i64,
    pub no_z_with_buffer: i64,
    pub mid_xy: i64,
    pub inner_radius: f64,
    pub inner_radius_sq: f64,
    pub sampling_radius: f64,

    pub dt: f64,
    pub pulse_time_steps: i64,
    pub clearance_time_steps: i64,
    pub total_time_steps: i64,

    pub gaussian_factor: f64,
    pub track_cutoff_voxels: f64,

    pub number_of_tracks_per_pulse: i64,
    pub coefficients: SchemeCoefficients,

    pub carrier_array_bytes: f64,
    pub track_schedule_bytes: f64,
    pub estimated_memory_bytes: f64,
}

/// Largest `dt` (starting from 1 s and shrinking) satisfying the von Neumann
/// stability criterion for the explicit Lax-Wendroff scheme (Deghan 2004),
/// full 3D diffusion, drift along z only: `6s + c^2 <= 1`. Single (D, mu)
/// pair, unlike `config.py`'s version which also handles two species.
fn von_neumann_dt(
    diffusion_cm2_s: f64,
    grid_spacing_cm: f64,
    mobility_cm2_vs: f64,
    efield_v_cm: f64,
) -> f64 {
    let mut dt = 1.0;
    loop {
        dt /= 1.01;
        let s = diffusion_cm2_s * dt / (grid_spacing_cm * grid_spacing_cm);
        let c = mobility_cm2_vs * efield_v_cm * dt / grid_spacing_cm;
        if 6.0 * s + c * c <= 1.0 {
            return dt;
        }
    }
}

impl Config {
    pub fn build(params: Params) -> Result<Config, ConfigError> {
        if params.e_mev_u <= 0.0
            || params.voltage_v <= 0.0
            || params.electrode_gap_cm <= 0.0
            || params.dose_rate_gy_s < 0.0
            || params.pulse_duration_s <= 0.0
            || params.sampled_radius_cm <= 0.0
        {
            return Err(ConfigError("All physical inputs must be positive.".into()));
        }

        let unit_length_cm = GRID_SIZE_UM * 1e-4;
        let let_kev_um_ = let_kev_um(params.e_mev_u);
        let track_radius_cm_ = track_radius_cm(let_kev_um_);
        let efield_v_cm = params.voltage_v / params.electrode_gap_cm;
        let area_cm2 = PI * params.sampled_radius_cm * params.sampled_radius_cm;

        let no_xy =
            (2.0 * params.sampled_radius_cm / unit_length_cm).round() as i64 + 2 * BUFFER_RADIUS;
        let no_z = (params.electrode_gap_cm / unit_length_cm) as i64; // truncation, matches int(...) in config.py
        let no_z_with_buffer = 2 * NO_Z_ELECTRODE + no_z;
        if no_z <= 0 {
            return Err(ConfigError(
                "electrode_gap_cm is too small for the fixed 10 um grid spacing.".into(),
            ));
        }
        if no_xy * no_xy * no_z > MAX_TOTAL_VOXELS {
            return Err(ConfigError(format!(
                "Grid too large for this browser prototype: {no_xy}x{no_xy}x{no_z} \
                 ({} voxels, cap {MAX_TOTAL_VOXELS}). Reduce the radius.",
                no_xy * no_xy * no_z
            )));
        }
        let mid_xy = no_xy / 2;
        let outer_radius = no_xy as f64 / 2.0;
        let inner_radius = outer_radius - BUFFER_RADIUS as f64;
        if inner_radius <= 0.0 {
            return Err(ConfigError(
                "sampled_radius_cm is too small relative to the fixed buffer.".into(),
            ));
        }
        let inner_radius_sq = inner_radius * inner_radius;
        let sampling_radius = inner_radius; // chamber_fill_fraction fixed at 1.0

        let dt = von_neumann_dt(
            ION_DIFFUSION_CM2_S,
            unit_length_cm,
            ION_MOBILITY_CM2_VS,
            efield_v_cm,
        );

        let pulse_time_steps = (params.pulse_duration_s / dt).ceil().max(1.0) as i64;
        let separation_time_steps =
            (params.electrode_gap_cm / (2.0 * ION_MOBILITY_CM2_VS * efield_v_cm * dt)) as i64;
        let clearance_time_steps =
            (N_CLEARANCE_SEPARATION_TIMES * separation_time_steps as f64).round() as i64;
        let total_time_steps = pulse_time_steps + clearance_time_steps;
        if total_time_steps > MAX_TOTAL_TIME_STEPS {
            return Err(ConfigError(format!(
                "This configuration needs {total_time_steps} time steps (cap {MAX_TOTAL_TIME_STEPS}). \
                 Try a larger electrode gap or lower voltage."
            )));
        }

        let let_ev_cm = let_kev_um_ * 1e7;
        let n0 = let_ev_cm / W_EV_PER_ION_PAIR;
        let gaussian_factor = n0 / (PI * track_radius_cm_ * track_radius_cm_);

        let track_sigma_cm = track_radius_cm_ / std::f64::consts::SQRT_2;
        let track_cutoff_cm = TRACK_CUTOFF_SIGMAS * track_sigma_cm;
        let track_cutoff_voxels = track_cutoff_cm / unit_length_cm;

        let dose_per_pulse_gy = params.dose_rate_gy_s / REPETITION_RATE_HZ;
        let instantaneous_dose_rate_gy_s = dose_per_pulse_gy / params.pulse_duration_s;
        let fluence_rate_inst_cm2_s =
            dose_rate_to_fluence_rate_default_air(instantaneous_dose_rate_gy_s, params.e_mev_u);
        let number_of_tracks_per_pulse =
            ((fluence_rate_inst_cm2_s * params.pulse_duration_s * area_cm2).round() as i64).max(1);
        if number_of_tracks_per_pulse > MAX_TRACKS_PER_PULSE {
            return Err(ConfigError(format!(
                "This configuration injects {number_of_tracks_per_pulse} tracks/pulse \
                 (cap {MAX_TRACKS_PER_PULSE}). Reduce the dose rate or the radius."
            )));
        }

        let s = ION_DIFFUSION_CM2_S * dt / (unit_length_cm * unit_length_cm);
        let c = ION_MOBILITY_CM2_VS * efield_v_cm * dt / unit_length_cm;
        let coefficients = SchemeCoefficients {
            lateral: s,
            z_minus: s + c * (c + 1.0) / 2.0,
            z_plus: s + c * (c - 1.0) / 2.0,
            centre: 1.0 - c * c - 6.0 * s,
        };

        let voxels = (no_xy * no_xy * no_z_with_buffer) as f64;
        let carrier_array_bytes = 4.0 * voxels * 8.0;
        let track_schedule_bytes = 2.0 * number_of_tracks_per_pulse as f64 * 8.0;
        let scratch_bytes = (no_xy * no_xy) as f64 * 8.0;
        let estimated_memory_bytes = track_schedule_bytes.max(carrier_array_bytes + scratch_bytes);
        if estimated_memory_bytes > MAX_MEMORY_BYTES {
            return Err(ConfigError(format!(
                "Estimated peak memory {:.1} MiB exceeds this prototype's {:.0} MiB cap.",
                estimated_memory_bytes / (1024.0 * 1024.0),
                MAX_MEMORY_BYTES / (1024.0 * 1024.0)
            )));
        }

        Ok(Config {
            params,
            unit_length_cm,
            let_kev_um: let_kev_um_,
            track_radius_cm: track_radius_cm_,
            efield_v_cm,
            no_xy,
            no_z,
            no_z_with_buffer,
            mid_xy,
            inner_radius,
            inner_radius_sq,
            sampling_radius,
            dt,
            pulse_time_steps,
            clearance_time_steps,
            total_time_steps,
            gaussian_factor,
            track_cutoff_voxels,
            number_of_tracks_per_pulse,
            coefficients,
            carrier_array_bytes,
            track_schedule_bytes,
            estimated_memory_bytes,
        })
    }

    /// `RECOMBINATION_ALPHA_CM3_S * dt`, the sink coefficient the solver
    /// multiplies `p * n` by every step.
    pub fn alpha_dt(&self) -> f64 {
        RECOMBINATION_ALPHA_CM3_S * self.dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aic144_like(sampled_radius_cm: f64) -> Params {
        Params {
            e_mev_u: 56.2,
            voltage_v: 300.0,
            electrode_gap_cm: 0.2,
            dose_rate_gy_s: 8.91,
            pulse_duration_s: 540e-6,
            sampled_radius_cm,
            seed: 1,
        }
    }

    #[test]
    fn dev_tier_grid_matches_documented_shape() {
        // docs/BENCHMARKS-LAPTOP.md sec. 2: "dev" tier is 12^2 x 210.
        let config = Config::build(aic144_like(0.003)).unwrap();
        assert_eq!(config.no_xy, 12);
        assert_eq!(config.no_z, 200);
        assert_eq!(config.no_z_with_buffer, 206);
    }

    #[test]
    fn archive_tier_track_count_matches_documented_value() {
        // docs/BENCHMARKS-LAPTOP.md sec. 2: "archive" tier draws 22,447
        // tracks/pulse. Track count depends on dose, energy and area but not
        // on dt or carrier species, so the single-averaged-species model used
        // here doesn't move it -- but this crate fixes air_density_kg_m3 at
        // the library default (1.225, ISA sea level) rather than the
        // archived campaign's 20 degC value (1.2041), a deliberate 1.7%
        // difference (docs/PHYSICS.md sec. 5), hence the wider tolerance.
        let config = Config::build(aic144_like(0.008)).unwrap();
        assert_eq!(config.no_xy, 22);
        let relative_diff = (config.number_of_tracks_per_pulse as f64 - 22447.0) / 22447.0;
        assert!(
            relative_diff.abs() < 0.03,
            "got {}",
            config.number_of_tracks_per_pulse
        );
    }

    #[test]
    fn oversized_radius_is_refused() {
        let err = Config::build(aic144_like(1.0));
        assert!(
            err.is_err(),
            "a 1cm column should exceed the browser voxel cap"
        );
    }
}
