//! Track scheduling: when (which time step) and where (which xy position)
//! each track is injected. Ports `pulsed_ion_chamber/pulses.py`'s reference
//! (non-batched) versions -- this prototype runs few enough tracks per step
//! that CylinderSampler's block-buffered speed-up buys nothing (see
//! `ALGORITHM.md` sec. 5 on when batching pays), and a plain rejection loop
//! is exact and easy to read.
//!
//! Uses a PCG64 PRNG seeded deterministically. This is a *different*
//! generator from NumPy's default (PCG64 with a different seeding scheme),
//! so runs are not bit-for-bit reproducible against the Python backends --
//! only statistically comparable, same as comparing two different seeds of
//! the same generator. See issue #6 sec. 7.

use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Number of tracks arriving at each time step of the pulse, length
/// `pulse_time_steps`. Draws `n_tracks` uniforms, takes the running sum,
/// rescales to `[0, pulse_duration_s)`, and histograms into `dt`-wide bins --
/// the same "cumulative sum of uniforms" trick as
/// `pulses._sample_pulse_arrival_histogram`.
pub fn build_schedule(
    rng: &mut Pcg64,
    n_tracks: i64,
    pulse_duration_s: f64,
    dt: f64,
    pulse_time_steps: i64,
) -> Vec<i64> {
    let n = n_tracks.max(0) as usize;
    let mut cumulative = Vec::with_capacity(n);
    let mut running = 0.0;
    for _ in 0..n {
        running += rng.gen::<f64>();
        cumulative.push(running);
    }
    let mut schedule = vec![0i64; pulse_time_steps.max(0) as usize];
    if let Some(&last) = cumulative.last() {
        for arrival in cumulative {
            let scaled = arrival / last * pulse_duration_s;
            let bin = ((scaled / dt) as i64).clamp(0, pulse_time_steps - 1);
            schedule[bin as usize] += 1;
        }
    }
    schedule
}

/// Rejection-sample one track position `(x, y)`, in fractional voxel units,
/// uniformly inside the disc of radius `sqrt(inner_radius_sq)` centred at
/// `(mid_xy, mid_xy)`.
pub fn sample_xy_inside_cylinder(
    rng: &mut Pcg64,
    mid_xy: i64,
    inner_radius_sq: f64,
    no_xy: i64,
) -> (f64, f64) {
    loop {
        let x = rng.gen::<f64>() * no_xy as f64;
        let y = rng.gen::<f64>() * no_xy as f64;
        let dx = x - mid_xy as f64;
        let dy = y - mid_xy as f64;
        if dx * dx + dy * dy <= inner_radius_sq {
            return (x, y);
        }
    }
}

pub fn seeded_rng(seed: u64) -> Pcg64 {
    Pcg64::seed_from_u64(seed)
}
