//! Derived simulation configuration -- the Rust analogue of
//! `SimulationConfig.__post_init__` in `pulsed_ion_chamber/config.py`,
//! narrowed to what this browser prototype exposes. See the crate root docs
//! for the full list of what is fixed vs. adjustable and why.

use crate::constants::{
    ION_DIFFUSION_CM2_S, ION_MOBILITY_CM2_VS, RECOMBINATION_ALPHA_CM3_S, W_EV_PER_ION_PAIR,
};
use crate::stopping_power::{dose_rate_to_fluence_rate_default_air, let_kev_um, track_radius_cm};
use std::f64::consts::PI;

/// `1234567` -> `"1,234,567"`. Only ever called on non-negative counts.
fn with_commas(n: i64) -> String {
    let digits = n.to_string();
    let bytes = digits.as_bytes();
    let mut out = String::with_capacity(bytes.len() + bytes.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

fn mib(bytes: f64) -> String {
    format!("{:.1} MiB", bytes / (1024.0 * 1024.0))
}

/// Measured (desktop Chromium, this wasm build, release + wasm-opt) cost of
/// one voxel touched once: both the Lax-Wendroff sweep and this crate's
/// batched-deposition broadcast (solver.rs) are `O(voxels)` per step, and
/// held to ~this rate consistently across a 25x grid-size range (2.3M to
/// 59M voxels). Used only for the *rough*, instant, always-on estimate --
/// `estimate_running_time` (lib.rs) measures the real number for whichever
/// device is actually running this, which is what the hard cap
/// (MAX_TOTAL_VOXEL_STEPS) leaves room for by budgeting a much slower
/// device than this constant assumes.
const ROUGH_NS_PER_VOXEL_STEP: f64 = 8.2;

// --- fixed knobs (docs/PERFORMANCE.md's cost-model levers, not physics
// intuition -- see issue #6 sec. 6 for why these are hidden rather than
// exposed in v1). grid_size_um used to live here too but is now a `Params`
// field -- the hard ceilings below are expressed in derived quantities
// (voxels, bytes, tracks, steps), so they don't care which input parameter
// drove a config there, and stay the real safety backstop regardless. ---
pub const BUFFER_RADIUS: i64 = 3;
pub const NO_Z_ELECTRODE: i64 = 3;
pub const TRACK_CUTOFF_SIGMAS: f64 = 10.0;
pub const REPETITION_RATE_HZ: f64 = 50.0;
pub const N_CLEARANCE_SEPARATION_TIMES: f64 = 2.0;

// --- hard browser-safety ceilings, independent of the UI's own slider
// ranges (defense in depth -- see issue #6 sec. 5). Two genuinely different
// resources, two genuinely different kinds of cap:
//
// RAM is a real byte count (MAX_MEMORY_BYTES) -- exhausting it can crash the
// tab, so it stays a hard, accurately-computed limit.
//
// CPU is wall time, and there is no way to know a visitor's device speed in
// advance, so MAX_TOTAL_VOXEL_STEPS is deliberately not a raw grid-size
// limit (an earlier version of this cap was exactly that -- reachable at a
// modest 106x106x200 grid using only 71 MiB, which is the bug this comment
// replaces the reasoning for). It caps `total_time_steps * voxels` instead:
// both the Lax-Wendroff sweep and this crate's batched-deposition broadcast
// (see solver.rs's module docs) cost `O(voxels)` *per step*, measured at a
// consistent ~8.2 ns/voxel-step across a 25x grid-size range (2.3M to 59M
// voxels) in this wasm build, in desktop Chromium. The cap below assumes a
// device up to ~12x slower than that (~100 ns/voxel-step) and budgets about
// 10 minutes of wall time even there -- generous on purpose, since the
// `estimate_running_time` escape hatch (see lib.rs) gives a visitor the real
// number for their own device before they commit to a long run.
pub const MAX_TOTAL_VOXEL_STEPS: f64 = 6_000_000_000.0;
// Independent of the above: an absolute ceiling on raw grid dimensions,
// purely to stay inside i64 multiplication range and refuse an allocation
// attempt before it happens. Not meant to be the resource-meaningful cap --
// for any realistic config, MAX_MEMORY_BYTES or MAX_TOTAL_VOXEL_STEPS is hit
// far below this.
pub const ABSOLUTE_MAX_GRID_DIM: i64 = 20_000;
pub const MAX_MEMORY_BYTES: f64 = 1024.0 * 1024.0 * 1024.0; // 1 GiB

// Batched deposition (solver.rs) made per-track cost independent of no_z, so
// this is no longer a CPU gate -- it's a backstop against the arrival-time
// schedule's own memory (2 f64s/track) and against unbounded schedule-build
// time, both of which scale with track count regardless of grid size.
pub const MAX_TRACKS_PER_PULSE: i64 = 50_000_000;
pub const MAX_TOTAL_TIME_STEPS: i64 = 200_000;

/// The knobs this prototype exposes to the user. Everything else
/// (`particle`, boundary condition, single- vs. two-species, ...) is fixed --
/// see module docs.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub e_mev_u: f64,
    pub voltage_v: f64,
    pub electrode_gap_cm: f64,
    pub dose_rate_gy_s: f64,
    pub pulse_duration_s: f64,
    pub sampled_radius_cm: f64,
    pub grid_size_um: f64,
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
    pub mid_xy: f64,
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
    /// `total_time_steps * voxels` -- the CPU-time proxy MAX_TOTAL_VOXEL_STEPS
    /// caps. See [`Config::rough_wall_seconds_estimate`].
    pub voxel_steps: f64,
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
            || params.grid_size_um <= 0.0
        {
            return Err(ConfigError("All physical inputs must be positive.".into()));
        }
        // let_kev_um() clamps silently outside the PSTAR table's range,
        // unlike pulsed_ion_chamber.stopping_power's `interp1d(...,
        // bounds_error=True)`, which raises there. The UI's own energy
        // slider (10-250 MeV/u) stays inside [0.1, 500] MeV/u today, but
        // estimate()/WasmSimulation are exported from the wasm module with
        // no such guard, so check explicitly here rather than let a
        // future/direct caller silently get the wrong LET (issue #19 W4).
        let (e_min, e_max) = crate::stopping_power::table_energy_bounds_mev_u();
        if params.e_mev_u < e_min || params.e_mev_u > e_max {
            return Err(ConfigError(format!(
                "e_mev_u={} MeV/u is outside the PSTAR stopping-power table's range \
                 [{e_min}, {e_max}] MeV/u.",
                params.e_mev_u
            )));
        }

        let unit_length_cm = params.grid_size_um * 1e-4;
        let let_kev_um_ = let_kev_um(params.e_mev_u);
        let track_radius_cm_ = track_radius_cm(let_kev_um_);
        let efield_v_cm = params.voltage_v / params.electrode_gap_cm;
        let area_cm2 = PI * params.sampled_radius_cm * params.sampled_radius_cm;

        // `as i64` on a float saturates (Rust 1.45+: NaN -> 0, out-of-range ->
        // MIN/MAX) rather than wrapping, but the `+ 2 * BUFFER_RADIUS` right
        // after it does not -- release builds have no overflow-checks (see
        // Cargo.toml), so an extreme grid_size_um/sampled_radius_cm pair that
        // saturates the cast to i64::MAX would silently wrap the sum negative,
        // and the ABSOLUTE_MAX_GRID_DIM guard a few lines down would never
        // see it (issue #19 W6). saturating_add keeps the guard reachable.
        let no_xy = (2.0 * params.sampled_radius_cm / unit_length_cm).round() as i64;
        let no_xy = no_xy.saturating_add(2 * BUFFER_RADIUS);
        // Round rather than truncate: bare truncation both loses a whole
        // voxel to float representation error (0.7 / 0.001 is
        // 699.9999999999999 in f64) and silently shortens the modelled gap
        // whenever electrode_gap_cm isn't an exact multiple of
        // unit_length_cm, while efield_v_cm keeps using the requested
        // (longer) gap (issue #19 P3, mirrors config.py).
        let no_z = (params.electrode_gap_cm / unit_length_cm).round() as i64;
        let no_z_with_buffer = (2 * NO_Z_ELECTRODE).saturating_add(no_z);
        if no_z <= 0 {
            return Err(ConfigError(
                "electrode_gap_cm is too small for this grid_size_um -- the gap must span at \
                 least one voxel."
                    .into(),
            ));
        }
        if no_xy > ABSOLUTE_MAX_GRID_DIM || no_z_with_buffer > ABSOLUTE_MAX_GRID_DIM {
            // Refuses before any multiplication that could overflow i64 or
            // any allocation attempt -- MAX_MEMORY_BYTES/MAX_TOTAL_VOXEL_STEPS
            // reject everything realistic long before this triggers.
            return Err(ConfigError(
                "Grid dimensions absurdly large for this browser prototype -- reduce the \
                 radius or the electrode gap, or coarsen grid_size_um."
                    .into(),
            ));
        }
        // mid_xy must equal outer_radius exactly -- both are the grid's true
        // continuous centre, no_xy as f64 / 2.0 -- rather than mid_xy being
        // independently integer-divided (`no_xy / 2`). The two agree only
        // when no_xy is even; for odd no_xy that would centre every radius
        // test and the sampler half a voxel off from the point outer_radius
        // and inner_radius are actually measured from (issue #19 P6, mirrors
        // config.py).
        let outer_radius = no_xy as f64 / 2.0;
        let mid_xy = outer_radius;
        let inner_radius = outer_radius - BUFFER_RADIUS as f64;
        if inner_radius <= 0.0 {
            return Err(ConfigError(
                "sampled_radius_cm is too small relative to the fixed buffer at this \
                 grid_size_um -- increase the radius, or coarsen grid_size_um."
                    .into(),
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
                "This configuration needs {} time steps (cap {}) -- the Lax-Wendroff sweep re-reads \
                 the whole grid on every one of them, so wall time scales directly with this number. \
                 Try a larger electrode gap or lower voltage (both raise the stable dt).",
                with_commas(total_time_steps),
                with_commas(MAX_TOTAL_TIME_STEPS),
            )));
        }
        let voxels = (no_xy * no_xy * no_z_with_buffer) as f64;
        let voxel_steps = total_time_steps as f64 * voxels;
        if voxel_steps > MAX_TOTAL_VOXEL_STEPS {
            // The real CPU-time gate -- see the MAX_TOTAL_VOXEL_STEPS doc
            // comment for the ~8.2 ns/voxel-step measurement and the safety
            // margin behind this cap. Reports an estimated wall time at that
            // measured rate (this build, a fast desktop) so the message
            // means something concrete rather than a bare ratio of two big
            // numbers.
            let estimated_seconds_here = voxel_steps * ROUGH_NS_PER_VOXEL_STEP * 1e-9;
            let cap_seconds_here = MAX_TOTAL_VOXEL_STEPS * ROUGH_NS_PER_VOXEL_STEP * 1e-9;
            return Err(ConfigError(format!(
                "This configuration needs about {:.0}s on the machine this was benchmarked on \
                 (cap {:.0}s there; could be much longer on a slower device) -- {no_xy}x{no_xy}x{no_z} \
                 voxels swept {} times. Both the Lax-Wendroff sweep and this crate's batched track \
                 deposition cost one full grid pass per time step, so wall time scales with grid size \
                 times step count together, not either alone. Reduce the radius, coarsen grid_size_um, \
                 or use a larger electrode gap/lower voltage to cut the step count -- or use \
                 \"Estimate running time\" to measure the real number for your own device before \
                 deciding.",
                estimated_seconds_here,
                cap_seconds_here,
                with_commas(total_time_steps),
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
            // This crate batches deposition (solver.rs: sum a step's tracks
            // into a 2D scratch, broadcast down z once), which makes a
            // single track's own cost small and grid-independent -- the
            // voxel_steps cap above already accounts for the (grid-size-
            // dependent) broadcast pass. What track count alone still costs,
            // unboundedly, is the arrival-time schedule built before the
            // grid is even touched: two f64 arrays, so this cap is really a
            // memory/schedule-build backstop, not a per-step CPU gate.
            let schedule_bytes = 2.0 * number_of_tracks_per_pulse as f64 * 8.0;
            let schedule_bytes_at_cap = 2.0 * MAX_TRACKS_PER_PULSE as f64 * 8.0;
            return Err(ConfigError(format!(
                "This configuration injects {} tracks/pulse (cap {}). The arrival-time schedule \
                 alone would need about {} (at the cap: {}) before the grid is even touched. \
                 Reduce the dose rate or the radius.",
                with_commas(number_of_tracks_per_pulse),
                with_commas(MAX_TRACKS_PER_PULSE),
                mib(schedule_bytes),
                mib(schedule_bytes_at_cap),
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

        let carrier_array_bytes = 4.0 * voxels * 8.0;
        let track_schedule_bytes = 2.0 * number_of_tracks_per_pulse as f64 * 8.0;
        let scratch_bytes = (no_xy * no_xy) as f64 * 8.0;
        let estimated_memory_bytes = track_schedule_bytes.max(carrier_array_bytes + scratch_bytes);
        if estimated_memory_bytes > MAX_MEMORY_BYTES {
            return Err(ConfigError(format!(
                "Estimated peak memory {} exceeds this prototype's {} cap.",
                mib(estimated_memory_bytes),
                mib(MAX_MEMORY_BYTES)
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
            voxel_steps,
        })
    }

    /// Rough, instant, allocation-free wall-time guess -- `voxel_steps` at
    /// [`ROUGH_NS_PER_VOXEL_STEP`], a desktop-class measurement. Meant to be
    /// shown alongside a clear "measure it for real" escape hatch
    /// (`estimate_running_time` in lib.rs), not trusted on its own -- device
    /// speed varies far more than this single constant can capture.
    pub fn rough_wall_seconds_estimate(&self) -> f64 {
        self.voxel_steps * ROUGH_NS_PER_VOXEL_STEP * 1e-9
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
            grid_size_um: 10.0,
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
    fn grid_size_um_changes_grid_width_as_expected() {
        // no_xy = round(2*radius/grid_size_um) + 2*BUFFER_RADIUS. The buffer
        // term (6 voxels at BUFFER_RADIUS=3) is a *constant* added on top of
        // the disc's diameter, so doubling the radius does not double no_xy:
        // 30um -> 6+6=12, 60um -> 12+6=18 (1.5x, not 2x), even though the
        // disc term itself (6 -> 12) doubles exactly.
        let mut params = aic144_like(0.003); // 30 um
        let at_30um = Config::build(params).unwrap();
        assert_eq!(at_30um.no_xy, 12);

        params.sampled_radius_cm = 0.006; // 60 um
        let at_60um = Config::build(params).unwrap();
        assert_eq!(at_60um.no_xy, 18);
    }

    #[test]
    fn coarser_grid_size_um_shrinks_the_grid() {
        let mut params = aic144_like(0.008); // "archive" tier radius
        params.grid_size_um = 10.0;
        let fine = Config::build(params).unwrap();
        params.grid_size_um = 20.0;
        let coarse = Config::build(params).unwrap();
        assert!(
            coarse.no_xy < fine.no_xy,
            "coarse={}, fine={}",
            coarse.no_xy,
            fine.no_xy
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

    #[test]
    fn grid_too_large_error_reports_estimated_time_and_cap() {
        let params = aic144_like(2.0); // absurdly wide, forces the voxel-steps cap
        let err = Config::build(params).unwrap_err();
        assert!(err.0.contains("voxels"), "{}", err.0);
        assert!(err.0.contains("s on the machine"), "{}", err.0);
        assert!(err.0.contains("cap"), "{}", err.0);
    }

    #[test]
    fn energy_outside_the_pstar_table_is_refused() {
        // Matches pulsed_ion_chamber.stopping_power's interp1d(...,
        // bounds_error=True): the Python backend raises outside [0.1, 500]
        // MeV/u, and this crate used to clamp silently instead (issue #19 W4).
        let mut params = aic144_like(0.008);
        params.e_mev_u = 0.05; // below the table's minimum
        let err = Config::build(params).unwrap_err();
        assert!(err.0.contains("PSTAR"), "{}", err.0);

        params.e_mev_u = 600.0; // above the table's maximum
        let err = Config::build(params).unwrap_err();
        assert!(err.0.contains("PSTAR"), "{}", err.0);
    }

    #[test]
    fn absurd_inputs_are_refused_rather_than_overflowing_no_xy() {
        // A grid_size_um/sampled_radius_cm pair extreme enough that
        // (2*radius/unit_length_cm).round() as i64 saturates to i64::MAX --
        // the regression case for issue #19 W6. A plain `+ 2*BUFFER_RADIUS`
        // right after that cast wraps negative in a release build (no
        // overflow-checks, see Cargo.toml), which used to let a bogus
        // negative no_xy slip past the ABSOLUTE_MAX_GRID_DIM guard a few
        // lines below and get refused with a misleading error instead
        // ("sampled_radius_cm is too small"). saturating_add fixes that.
        let mut params = aic144_like(0.008);
        params.sampled_radius_cm = 1e300;
        // unit_length_cm = 1e-300 * 1e-4 = 1e-304 -- still representable (an
        // f64 normal goes down to ~2.2e-308), so this isn't a division by
        // zero. What overflows is the *ratio*: 2*radius/unit_length_cm is
        // ~2e604, past f64::MAX, so it evaluates to +inf, and `.round() as
        // i64` on infinity is what saturates to i64::MAX (PR #20 review).
        params.grid_size_um = 1e-300;
        let err = Config::build(params).unwrap_err();
        assert!(err.0.contains("absurdly large"), "{}", err.0);
    }

    #[test]
    fn modest_wide_grid_is_not_rejected_on_voxel_count_alone() {
        // Regression test for the bug this cap redesign fixes: a
        // 106x106x200 grid (71 MiB, well under the 1 GiB memory cap) used to
        // be refused purely for its raw voxel count (the old, since-removed
        // MAX_TOTAL_VOXELS). It should be accepted now -- the real CPU cost
        // (~3.65e9 voxel-steps, comfortably under MAX_TOTAL_VOXEL_STEPS) is
        // what actually gates it, not grid size by itself.
        let params = aic144_like(0.05);
        let config = Config::build(params).unwrap();
        assert_eq!(config.no_xy, 106);
        assert!(config.estimated_memory_bytes < 100.0 * 1024.0 * 1024.0);
    }

    #[test]
    fn too_many_tracks_error_reports_memory_and_reason() {
        let mut params = aic144_like(0.008); // "archive" radius -- grid stays tiny
        params.dose_rate_gy_s = 25_000.0; // forces the track-count cap, not the voxel-steps cap
        let err = Config::build(params).unwrap_err();
        assert!(err.0.contains("tracks/pulse"), "{}", err.0);
        assert!(err.0.contains("MiB"), "{}", err.0);
        assert!(err.0.contains("schedule"), "{}", err.0);
    }
}
