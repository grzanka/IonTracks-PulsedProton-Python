//! Rust/WASM port of `pulsed_ion_chamber`'s drift-diffusion-recombination
//! solver, scoped down for an interactive browser prototype. See
//! `https://github.com/grzanka/IonTracks-PulsedProton-Python/issues/6` for
//! the feasibility writeup this crate implements the first milestone of.
//!
//! ## What this narrows relative to the Python package
//!
//! - **Single averaged carrier species only** (config.py's default). The
//!   two-Kanai-species option (`docs/PHYSICS.md` sec. 8) is not ported.
//! - **`lateral_boundary` is always `"reflecting"`**, `scoring_region` is
//!   always `"track_disc"`, `n_pulses` is always `1`, `particle` is always
//!   `"proton"`, `chamber_fill_fraction` is always `1.0`.
//! - `buffer_radius`, `no_z_electrode`, `track_cutoff_sigmas`,
//!   `repetition_rate_hz` and `n_clearance_separation_times` are fixed
//!   constants (`config` module) -- cost-model levers, not physics intuition,
//!   per issue #6 sec. 6. `grid_size_um` *is* adjustable (see below); the
//!   others stayed fixed because they're numerical margin, not something a
//!   user has physical intuition for.
//! - The track-density cross-section (a full 2D field) is not scored; only
//!   the four scalar time series (`n+`, `n-`, `injected`, `recombination`)
//!   this prototype's live plots need.
//!
//! Adjustable: beam energy, voltage, electrode gap, dose rate, pulse
//! duration, the sampled column radius, and the grid spacing (`grid_size_um`)
//! -- each checked against a hard browser-safety ceiling in
//! [`config::Config::build`] before any grid is allocated. The ceilings are
//! expressed in the *derived* quantities (voxel count, byte count, track
//! count, step count), not in the input parameters themselves, so they catch
//! an expensive config regardless of which knob (a wide radius, a fine
//! grid_size_um, or some combination) drove it there.
//!
//! ## Layout
//!
//! - [`config`] -- derived quantities and the hard safety ceilings.
//! - [`stopping_power`] -- LET lookup and Gaussian track radius.
//! - [`sampler`] -- track scheduling and placement (seeded PCG64, *not* the
//!   same generator NumPy uses -- see `sampler`'s module docs).
//! - [`solver`] -- the two hot loops and the [`solver::Simulation`] driver.
//! - [`theory`] -- Jaffe theory, used only by this crate's own tests as an
//!   RNG-independent correctness check (mirrors
//!   `tests/test_single_track_vs_jaffe.py`).
//!
//! This module (`lib.rs`) is the thin wasm-bindgen wrapper: a JS Web Worker
//! is expected to call [`estimate`] on every parameter change (instant, no
//! allocation) before ever constructing a [`WasmSimulation`], then drive its
//! `step()` loop and read the scalar getters back -- the same
//! call-native-`tick()`-and-read-state idiom as wasm-bindgen's own
//! "Game of Life" example.

mod config;
mod sampler;
mod solver;
mod stopping_power;
// Native-test-only: an RNG-independent correctness anchor (Jaffe theory), not
// part of the wasm surface -- see solver::tests::single_track_matches_jaffe_theory.
#[cfg(test)]
mod theory;

pub mod constants;

use config::{Config, Params};
use wasm_bindgen::prelude::*;

/// Call once, before anything else, so a Rust panic prints a real message
/// (with file/line) to the browser console instead of an opaque
/// `RuntimeError: unreachable`.
#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Instant, allocation-free sizing of a configuration -- the wasm analogue of
/// `SimulationConfig.estimated_memory_bytes` / `run_markus_2mm.py --dry-run`
/// (`docs/PERFORMANCE.md` sec. 7). Call this on every slider change; only
/// construct a [`WasmSimulation`] once the user approves what it reports.
#[wasm_bindgen(getter_with_clone)]
pub struct Estimate {
    pub ok: bool,
    pub error: Option<String>,
    pub no_xy: i32,
    pub no_z: i32,
    pub total_time_steps: i32,
    pub number_of_tracks_per_pulse: f64,
    pub estimated_memory_bytes: f64,
    pub dt_ns: f64,
    pub let_kev_um: f64,
    pub track_radius_um: f64,
}

#[allow(clippy::too_many_arguments)]
fn build_params(
    e_mev_u: f64,
    voltage_v: f64,
    electrode_gap_cm: f64,
    dose_rate_gy_s: f64,
    pulse_duration_s: f64,
    sampled_radius_cm: f64,
    grid_size_um: f64,
    seed: f64,
) -> Params {
    Params {
        e_mev_u,
        voltage_v,
        electrode_gap_cm,
        dose_rate_gy_s,
        pulse_duration_s,
        sampled_radius_cm,
        grid_size_um,
        seed: seed as u64,
    }
}

#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn estimate(
    e_mev_u: f64,
    voltage_v: f64,
    electrode_gap_cm: f64,
    dose_rate_gy_s: f64,
    pulse_duration_s: f64,
    sampled_radius_cm: f64,
    grid_size_um: f64,
) -> Estimate {
    let params = build_params(
        e_mev_u,
        voltage_v,
        electrode_gap_cm,
        dose_rate_gy_s,
        pulse_duration_s,
        sampled_radius_cm,
        grid_size_um,
        1.0,
    );
    match Config::build(params) {
        Ok(c) => Estimate {
            ok: true,
            error: None,
            no_xy: c.no_xy as i32,
            no_z: c.no_z as i32,
            total_time_steps: c.total_time_steps as i32,
            number_of_tracks_per_pulse: c.number_of_tracks_per_pulse as f64,
            estimated_memory_bytes: c.estimated_memory_bytes,
            dt_ns: c.dt * 1e9,
            let_kev_um: c.let_kev_um,
            track_radius_um: c.track_radius_cm * 1e4,
        },
        Err(e) => Estimate {
            ok: false,
            error: Some(e.0),
            no_xy: 0,
            no_z: 0,
            total_time_steps: 0,
            number_of_tracks_per_pulse: 0.0,
            estimated_memory_bytes: 0.0,
            dt_ns: 0.0,
            let_kev_um: 0.0,
            track_radius_um: 0.0,
        },
    }
}

/// A running simulation, driven one time step at a time by the caller (see
/// module docs). Construction validates and sizes the config the same way
/// [`estimate`] does -- `ok()`/`error()` report the outcome instead of
/// throwing, so the Worker can show the same message either way.
#[wasm_bindgen]
pub struct WasmSimulation {
    inner: Option<solver::Simulation>,
    error: Option<String>,
}

#[wasm_bindgen]
impl WasmSimulation {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        e_mev_u: f64,
        voltage_v: f64,
        electrode_gap_cm: f64,
        dose_rate_gy_s: f64,
        pulse_duration_s: f64,
        sampled_radius_cm: f64,
        grid_size_um: f64,
        seed: f64,
    ) -> WasmSimulation {
        let params = build_params(
            e_mev_u,
            voltage_v,
            electrode_gap_cm,
            dose_rate_gy_s,
            pulse_duration_s,
            sampled_radius_cm,
            grid_size_um,
            seed,
        );
        match Config::build(params) {
            Ok(config) => WasmSimulation {
                inner: Some(solver::Simulation::new(config)),
                error: None,
            },
            Err(e) => WasmSimulation {
                inner: None,
                error: Some(e.0),
            },
        }
    }

    pub fn ok(&self) -> bool {
        self.inner.is_some()
    }

    pub fn error(&self) -> Option<String> {
        self.error.clone()
    }

    /// Advance one time step. Returns `true` once the run is complete (or
    /// immediately, if construction failed -- check `ok()` first).
    pub fn step(&mut self) -> bool {
        match &mut self.inner {
            Some(sim) => sim.step(),
            None => true,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.inner.as_ref().map(|s| s.is_finished()).unwrap_or(true)
    }

    pub fn total_steps(&self) -> f64 {
        self.inner
            .as_ref()
            .map(|s| s.config.total_time_steps as f64)
            .unwrap_or(0.0)
    }

    pub fn step_index(&self) -> f64 {
        self.inner
            .as_ref()
            .map(|s| s.step_index() as f64)
            .unwrap_or(0.0)
    }

    pub fn time_s(&self) -> f64 {
        self.inner.as_ref().map(|s| s.time_s()).unwrap_or(0.0)
    }

    pub fn last_injected(&self) -> f64 {
        self.inner.as_ref().map(|s| s.last_injected).unwrap_or(0.0)
    }

    pub fn last_recombined(&self) -> f64 {
        self.inner
            .as_ref()
            .map(|s| s.last_recombined)
            .unwrap_or(0.0)
    }

    pub fn last_total_positive(&self) -> f64 {
        self.inner
            .as_ref()
            .map(|s| s.last_total_positive)
            .unwrap_or(0.0)
    }

    pub fn last_total_negative(&self) -> f64 {
        self.inner
            .as_ref()
            .map(|s| s.last_total_negative)
            .unwrap_or(0.0)
    }

    pub fn ks(&self) -> f64 {
        self.inner.as_ref().map(|s| s.ks()).unwrap_or(1.0)
    }
}
