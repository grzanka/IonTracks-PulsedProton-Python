//! The two hot loops (track deposition, Lax-Wendroff update) and the
//! `Simulation` struct that drives them one time step at a time. Deposition
//! ports `pulsed_ion_chamber/solver_numba_parallel.py`'s *batching* (not its
//! threading, which this single-threaded prototype has no use for): sum a
//! step's tracks into a 2D scratch array first, then broadcast down every
//! gap layer once per step instead of once per track (`ALGORITHM.md` sec. 5).
//! This crate's first version ported the simpler unbatched backend instead,
//! reasoning that this prototype's capped grids meant few tracks per step --
//! true at the original 30-250um radius range, but measurably false once
//! larger radii were allowed: a ~600-tracks/step config measured 120ms/step
//! unbatched (each track's deposit re-walking all `no_z` layers), batching
//! cut that to single-digit ms/step. The Lax-Wendroff update stays a direct
//! port of `solver_numba.py`'s serial version either way -- it has no
//! per-track structure to batch.
//!
//! Grid layout matches the Python code on purpose: one flat buffer per
//! carrier, `(no_xy, no_xy, no_z_with_buffer)` with `k` (along the field)
//! fastest-varying, so `idx(i, j, k) = (i * no_xy + j) * no_z_with_buffer + k`
//! walks memory sequentially in the `k` loop, same as the NumPy arrays it
//! mirrors.

use crate::config::{Config, SchemeCoefficients};
use crate::sampler::{build_schedule, sample_xy_inside_cylinder, seeded_rng};
use rand_pcg::Pcg64;

struct Grid {
    no_xy: usize,
    no_z_with_buffer: usize,
    no_z: usize,
    no_z_electrode: usize,
    mid_xy: i64,
    scoring_radius_sq: f64,
}

impl Grid {
    #[inline(always)]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.no_xy + j) * self.no_z_with_buffer + k
    }
}

/// Phase 1 of batched deposition: accumulate one track's separable,
/// truncated Gaussian into the *step's* 2D scratch density -- not yet
/// broadcast down `z`. Costs `2w` exponentials and `w^2` additions (no `no_z`
/// factor, unlike the unbatched form) -- see `ALGORITHM.md` sec. 4 for the
/// separable-Gaussian identity and sec. 5 for why summing 2D profiles first
/// and broadcasting once is exact, not approximate (addition is
/// associative: `sum_t broadcast(g_t) == broadcast(sum_t g_t)`).
#[allow(clippy::too_many_arguments)]
fn accumulate_track_2d(
    grid: &Grid,
    total_density: &mut [f64],
    x: f64,
    y: f64,
    h2: f64,
    b2: f64,
    gaussian_factor: f64,
    cutoff_voxels: f64,
    gauss_i: &mut Vec<f64>,
    gauss_j: &mut Vec<f64>,
) {
    let no_xy = grid.no_xy as i64;
    let i_lo = ((x - cutoff_voxels).ceil() as i64).max(0);
    let i_hi = (((x + cutoff_voxels).floor() as i64) + 1).min(no_xy);
    let j_lo = ((y - cutoff_voxels).ceil() as i64).max(0);
    let j_hi = (((y + cutoff_voxels).floor() as i64) + 1).min(no_xy);
    if i_lo >= i_hi || j_lo >= j_hi {
        return;
    }

    // Reused across every track in the step (and across steps): `.clear()`
    // keeps the underlying heap allocation, so after the first call (with
    // the widest stencil this config produces) this is allocation-free.
    gauss_i.clear();
    gauss_i.extend((i_lo..i_hi).map(|i| (-((i as f64 - x).powi(2)) * h2 / b2).exp()));
    gauss_j.clear();
    gauss_j.extend((j_lo..j_hi).map(|j| (-((j as f64 - y).powi(2)) * h2 / b2).exp()));

    for (ii, i) in (i_lo..i_hi).enumerate() {
        let gi = gauss_i[ii];
        let iu = i as usize;
        let row_lo = iu * grid.no_xy + j_lo as usize;
        let row_hi = iu * grid.no_xy + j_hi as usize;
        for (cell, &gj) in total_density[row_lo..row_hi].iter_mut().zip(gauss_j.iter()) {
            *cell += gaussian_factor * gi * gj;
        }
    }
}

/// Phase 2 of batched deposition: broadcast the step's accumulated 2D
/// density down every gap layer of both carrier arrays, and score the charge
/// that landed inside the scored disc. `O(no_xy^2 * no_z)` regardless of how
/// many tracks contributed this step -- the caller skips calling this at all
/// when no tracks arrived, so a clearance-phase step (no deposition) costs
/// nothing here, same as the unbatched version.
fn broadcast_and_score(
    grid: &Grid,
    total_density: &[f64],
    positive: &mut [f64],
    negative: &mut [f64],
) -> f64 {
    let mut inserted = 0.0;
    let k_lo = grid.no_z_electrode;
    let k_hi = grid.no_z_electrode + grid.no_z;
    for i in 0..grid.no_xy {
        let di_sq = ((i as i64 - grid.mid_xy) as f64).powi(2);
        let row_base = i * grid.no_xy;
        for j in 0..grid.no_xy {
            let density = total_density[row_base + j];
            if density == 0.0 {
                continue; // most columns are untouched at low-to-moderate dose rates
            }
            let base = (row_base + j) * grid.no_z_with_buffer;
            let p_slice = &mut positive[base + k_lo..base + k_hi];
            let n_slice = &mut negative[base + k_lo..base + k_hi];
            for (p, n) in p_slice.iter_mut().zip(n_slice.iter_mut()) {
                *p += density;
                *n += density;
            }
            let dj_sq = ((j as i64 - grid.mid_xy) as f64).powi(2);
            if di_sq + dj_sq < grid.scoring_radius_sq {
                inserted += density * grid.no_z as f64;
            }
        }
    }
    inserted
}

/// Advance both carrier densities by one time step. Ports
/// `solver_numba._lax_wendroff_step_numba` -- see that function's docstring
/// for the stencil weights and the sign convention that makes the two
/// species drift in opposite directions along z despite sharing one
/// coefficient set (this crate's single-averaged-species model).
#[allow(clippy::too_many_arguments)]
fn lax_wendroff_step(
    grid: &Grid,
    positive: &[f64],
    negative: &[f64],
    positive_next: &mut [f64],
    negative_next: &mut [f64],
    coeffs: &SchemeCoefficients,
    alpha_dt: f64,
) -> (f64, f64, f64) {
    let mut recombined = 0.0;
    let mut total_positive = 0.0;
    let mut total_negative = 0.0;

    for i in 1..grid.no_xy - 1 {
        let di_sq = ((i as i64 - grid.mid_xy) as f64).powi(2);
        for j in 1..grid.no_xy - 1 {
            let dj_sq = ((j as i64 - grid.mid_xy) as f64).powi(2);
            let inside = di_sq + dj_sq < grid.scoring_radius_sq;
            for k in 1..grid.no_z_with_buffer - 1 {
                let idx = grid.idx(i, j, k);
                let p = positive[idx];
                let n = negative[idx];

                let p_new = coeffs.z_minus * positive[grid.idx(i, j, k - 1)]
                    + coeffs.z_plus * positive[grid.idx(i, j, k + 1)]
                    + coeffs.lateral
                        * (positive[grid.idx(i, j - 1, k)]
                            + positive[grid.idx(i, j + 1, k)]
                            + positive[grid.idx(i - 1, j, k)]
                            + positive[grid.idx(i + 1, j, k)])
                    + coeffs.centre * p;
                // Negative ions drift opposite to positive ions, so the
                // z-neighbour weights are swapped relative to p_new.
                let n_new = coeffs.z_minus * negative[grid.idx(i, j, k + 1)]
                    + coeffs.z_plus * negative[grid.idx(i, j, k - 1)]
                    + coeffs.lateral
                        * (negative[grid.idx(i, j - 1, k)]
                            + negative[grid.idx(i, j + 1, k)]
                            + negative[grid.idx(i - 1, j, k)]
                            + negative[grid.idx(i + 1, j, k)])
                    + coeffs.centre * n;

                let recomb = alpha_dt * p * n;
                let p_out = p_new - recomb;
                let n_out = n_new - recomb;
                positive_next[idx] = p_out;
                negative_next[idx] = n_out;

                if inside && k > grid.no_z_electrode && k < grid.no_z + grid.no_z_electrode {
                    recombined += recomb;
                    total_positive += p_out;
                    total_negative += n_out;
                }
            }
        }
    }
    (recombined, total_positive, total_negative)
}

/// Copy the interior of `src` (indices `1..len-1` on every axis) into `dst`,
/// leaving `dst`'s outer shell untouched -- the sweep never writes it, so
/// `apply_reflecting_boundary` is what gives it a value.
fn copy_interior(grid: &Grid, src: &[f64], dst: &mut [f64]) {
    for i in 1..grid.no_xy - 1 {
        for j in 1..grid.no_xy - 1 {
            for k in 1..grid.no_z_with_buffer - 1 {
                let idx = grid.idx(i, j, k);
                dst[idx] = src[idx];
            }
        }
    }
}

/// Zero-gradient (zero-flux) mirror on the `x, y` ring -- the only lateral
/// boundary mode this prototype implements (see `docs/PHYSICS.md` sec. 10 for
/// why it's the physically right choice for a column sampled from the
/// interior of a large, uniformly irradiated chamber, and issue #6 sec. 6 for
/// why `absorbing` is left out of v1). The `z` ends are never touched, in
/// either mode -- charge past the electrode buffer has been collected.
fn apply_reflecting_boundary(grid: &Grid, array: &mut [f64]) {
    for j in 0..grid.no_xy {
        for k in 0..grid.no_z_with_buffer {
            let lo_src = grid.idx(1, j, k);
            let lo_dst = grid.idx(0, j, k);
            array[lo_dst] = array[lo_src];
            let hi_src = grid.idx(grid.no_xy - 2, j, k);
            let hi_dst = grid.idx(grid.no_xy - 1, j, k);
            array[hi_dst] = array[hi_src];
        }
    }
    for i in 0..grid.no_xy {
        for k in 0..grid.no_z_with_buffer {
            let lo_src = grid.idx(i, 1, k);
            let lo_dst = grid.idx(i, 0, k);
            array[lo_dst] = array[lo_src];
            let hi_src = grid.idx(i, grid.no_xy - 2, k);
            let hi_dst = grid.idx(i, grid.no_xy - 1, k);
            array[hi_dst] = array[hi_src];
        }
    }
}

/// A running simulation: owns the carrier arrays and RNG, and advances one
/// time step per call to [`Simulation::step`]. The caller (the wasm-bindgen
/// wrapper in `lib.rs`, driven by a JS Web Worker) owns the *loop* and reads
/// the per-step scalars back after each call -- this struct never buffers a
/// full time series itself, so its memory footprint is just the four carrier
/// arrays plus the schedule.
pub struct Simulation {
    pub config: Config,
    grid: Grid,
    positive: Vec<f64>,
    negative: Vec<f64>,
    positive_next: Vec<f64>,
    negative_next: Vec<f64>,
    rng: Pcg64,
    schedule: Vec<i64>,
    // Reusable per-track scratch space -- see accumulate_track_2d's doc comment.
    gauss_i_scratch: Vec<f64>,
    gauss_j_scratch: Vec<f64>,
    // The step's batched 2D density, before broadcast_and_score spreads it
    // down z -- see the module docs on why this crate batches deposition.
    total_density_scratch: Vec<f64>,
    step_index: i64,
    no_initialised: f64,
    no_recombined: f64,
    pub last_injected: f64,
    pub last_recombined: f64,
    pub last_total_positive: f64,
    pub last_total_negative: f64,
}

impl Simulation {
    pub fn new(config: Config) -> Simulation {
        let no_xy = config.no_xy as usize;
        let no_z_with_buffer = config.no_z_with_buffer as usize;
        let grid = Grid {
            no_xy,
            no_z_with_buffer,
            no_z: config.no_z as usize,
            no_z_electrode: crate::config::NO_Z_ELECTRODE as usize,
            mid_xy: config.mid_xy,
            scoring_radius_sq: config.inner_radius_sq,
        };
        let size = no_xy * no_xy * no_z_with_buffer;
        let mut rng = seeded_rng(config.params.seed);
        let schedule = build_schedule(
            &mut rng,
            config.number_of_tracks_per_pulse,
            config.params.pulse_duration_s,
            config.dt,
            config.pulse_time_steps,
        );
        // Generous enough for the widest stencil this config can produce
        // (track_cutoff_voxels doesn't change once built) -- sized once so
        // insert_track's clear()+extend() never needs to grow the backing
        // allocation.
        let scratch_capacity = config.track_cutoff_voxels.ceil() as usize * 2 + 4;

        Simulation {
            grid,
            positive: vec![0.0; size],
            negative: vec![0.0; size],
            positive_next: vec![0.0; size],
            negative_next: vec![0.0; size],
            rng,
            schedule,
            gauss_i_scratch: Vec::with_capacity(scratch_capacity),
            gauss_j_scratch: Vec::with_capacity(scratch_capacity),
            total_density_scratch: vec![0.0; no_xy * no_xy],
            step_index: 0,
            no_initialised: 0.0,
            no_recombined: 0.0,
            last_injected: 0.0,
            last_recombined: 0.0,
            last_total_positive: 0.0,
            last_total_negative: 0.0,
            config,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.step_index >= self.config.total_time_steps
    }

    /// Advance one time step; returns `true` once the run (pulse + clearance)
    /// is complete. A no-op returning `true` if called again after that.
    pub fn step(&mut self) -> bool {
        if self.is_finished() {
            return true;
        }
        let step_idx = self.step_index as usize;
        let n_tracks_this_step = self.schedule.get(step_idx).copied().unwrap_or(0);

        let h2 = self.config.unit_length_cm * self.config.unit_length_cm;
        let b2 = self.config.track_radius_cm * self.config.track_radius_cm;

        // Batched: accumulate every track this step into the 2D scratch
        // first, then broadcast+score once -- skipped entirely (both the
        // clear and the O(no_xy^2 * no_z) broadcast) when nothing arrives,
        // which is most steps during the clearance phase.
        let injected = if n_tracks_this_step > 0 {
            self.total_density_scratch.iter_mut().for_each(|v| *v = 0.0);
            for _ in 0..n_tracks_this_step {
                let (x, y) = sample_xy_inside_cylinder(
                    &mut self.rng,
                    self.config.mid_xy,
                    // Tracks are drawn inside sampling_radius (==
                    // inner_radius at the fixed chamber_fill_fraction=1.0
                    // this crate uses).
                    self.config.inner_radius_sq,
                    self.config.no_xy,
                );
                accumulate_track_2d(
                    &self.grid,
                    &mut self.total_density_scratch,
                    x,
                    y,
                    h2,
                    b2,
                    self.config.gaussian_factor,
                    self.config.track_cutoff_voxels,
                    &mut self.gauss_i_scratch,
                    &mut self.gauss_j_scratch,
                );
            }
            broadcast_and_score(
                &self.grid,
                &self.total_density_scratch,
                &mut self.positive,
                &mut self.negative,
            )
        } else {
            0.0
        };
        self.no_initialised += injected;

        let (recombined, total_p, total_n) = lax_wendroff_step(
            &self.grid,
            &self.positive,
            &self.negative,
            &mut self.positive_next,
            &mut self.negative_next,
            &self.config.coefficients,
            self.config.alpha_dt(),
        );
        self.no_recombined += recombined;

        copy_interior(&self.grid, &self.positive_next, &mut self.positive);
        copy_interior(&self.grid, &self.negative_next, &mut self.negative);
        apply_reflecting_boundary(&self.grid, &mut self.positive);
        apply_reflecting_boundary(&self.grid, &mut self.negative);

        self.last_injected = injected;
        self.last_recombined = recombined;
        self.last_total_positive = total_p;
        self.last_total_negative = total_n;
        self.step_index += 1;
        self.is_finished()
    }

    pub fn step_index(&self) -> i64 {
        self.step_index
    }

    pub fn time_s(&self) -> f64 {
        self.step_index as f64 * self.config.dt
    }

    /// Collection efficiency so far: `(injected - recombined) / injected`,
    /// `1.0` before any charge has been injected.
    pub fn f_t(&self) -> f64 {
        if self.no_initialised == 0.0 {
            1.0
        } else {
            (self.no_initialised - self.no_recombined) / self.no_initialised
        }
    }

    pub fn ks(&self) -> f64 {
        1.0 / self.f_t()
    }

    #[cfg(test)]
    fn run_to_completion(&mut self) {
        while !self.step() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Params;
    use crate::theory::jaffe_ks;

    fn single_track_params(seed: u64) -> Params {
        // Same scenario as tests/test_single_track_vs_jaffe.py: dose rate low
        // enough that number_of_tracks_per_pulse floors to 1.
        Params {
            e_mev_u: 1.0,
            voltage_v: 10.0,
            electrode_gap_cm: 0.02,
            dose_rate_gy_s: 1e-8,
            pulse_duration_s: 1e-6,
            sampled_radius_cm: 0.01,
            grid_size_um: 10.0,
            seed,
        }
    }

    #[test]
    fn single_track_matches_jaffe_theory() {
        for seed in [1, 2, 3] {
            let params = single_track_params(seed);
            let config = Config::build(params).unwrap();
            assert_eq!(config.number_of_tracks_per_pulse, 1);

            let ks_jaffe = jaffe_ks(config.let_kev_um, params.voltage_v, params.electrode_gap_cm);

            let mut sim = Simulation::new(config);
            sim.run_to_completion();

            let sim_loss = sim.ks() - 1.0;
            let jaffe_loss = ks_jaffe - 1.0;
            let ratio = sim_loss / jaffe_loss;
            assert!(
                (ratio - 1.0).abs() < 0.3,
                "seed {seed}: sim ks={}, jaffe ks={ks_jaffe}, ratio={ratio}",
                sim.ks()
            );
        }
    }

    #[test]
    fn f_t_is_monotonically_non_increasing() {
        let config = Config::build(single_track_params(1)).unwrap();
        let mut sim = Simulation::new(config);
        let mut previous = 1.0;
        while !sim.step() {
            let current = sim.f_t();
            assert!(
                current <= previous + 1e-6,
                "f_t rose: {previous} -> {current}"
            );
            previous = current;
        }
    }

    #[test]
    fn dev_tier_runs_to_completion_and_recombines_something() {
        // docs/BENCHMARKS-LAPTOP.md "dev" tier, single averaged species.
        let params = Params {
            e_mev_u: 56.2,
            voltage_v: 300.0,
            electrode_gap_cm: 0.2,
            dose_rate_gy_s: 8.91,
            pulse_duration_s: 540e-6,
            sampled_radius_cm: 0.003,
            grid_size_um: 10.0,
            seed: 1,
        };
        let config = Config::build(params).unwrap();
        let mut sim = Simulation::new(config);
        sim.run_to_completion();
        // Deterministic (seed=1): 1.0586388... measured with the unbatched
        // deposition this crate started with, and unchanged after switching
        // to batched deposition -- batching sums each step's tracks before
        // broadcasting rather than depositing them individually, and
        // addition is associative, so this is a genuine invariance, not a
        // coincidence (see the module docs on why batching was added, and
        // accumulate_track_2d's on why summing first is exact).
        assert!(
            (sim.ks() - 1.058639).abs() < 1e-5,
            "ks={} (expected 1.058639, seed=1 deterministic)",
            sim.ks()
        );
    }
}
