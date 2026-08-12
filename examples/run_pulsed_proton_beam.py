#!/usr/bin/env python3
"""Example: recombination in a parallel-plate ionization chamber exposed to
a pulsed proton beam (540 us pulses, 50 Hz, 60 Gy/s average dose rate).

Single-threaded Numba (pulsed_ion_chamber.solver_numba) is the baseline
backend for this repository -- everything below runs through it. The
plain pure-Python reference implementation (pulsed_ion_chamber.solver) is
what solver_numba.py was JIT-compiled from; it's ~10x slower (see
tests/test_solver_numba.py for a direct comparison on a small config) and
is not run here.

1. Runs the pulsed-proton scenario on a grid about 2.85 ion-track-radii
   wide (still coarser than the ~6-track-radii "converged" grid from the
   original IonTracks study, but noticeably finer than a 1-track-radius
   smoke test) -- tuned to take about 30 seconds single-threaded. Plots
   the collection efficiency f(t) and the final recombination correction
   factor k_s = 1/f.
2. Validates the solver in the single-track limit against the analytic
   Jaffe theory (a fast, independent correctness check).
3. Uses pulsed_ion_chamber.benchmark to *estimate* -- without running it --
   how long a fully converged grid would take, still single-threaded
   Numba. That gap (a few hours, vs. the ~30s demo) is your starting
   point for a multi-threaded or GPU port.
"""

import time

import matplotlib.pyplot as plt

from pulsed_ion_chamber.benchmark import estimate_full_runtime
from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.solver_numba import run_simulation_numba, warmup
from pulsed_ion_chamber.theory import jaffe_ks

# The physical scenario requested: 150 MeV protons, 200 V across a 0.2 cm
# gap, delivered in 540 us pulses at 50 Hz, 60 Gy/s time-averaged dose rate.
BEAM_KWARGS = dict(
    E_MeV_u=150.0,
    voltage_V=200.0,
    electrode_gap_cm=0.2,
    pulse_duration_s=540e-6,
    repetition_rate_hz=50.0,
    dose_rate_Gy_s=60.0,
    n_pulses=1,
)

# Tuned (empirically, on one machine) so run_simulation_numba() takes about
# 30 s single-threaded: sampled_radius_cm is roughly 2.85x the (floored)
# Gaussian track radius of a 150 MeV proton in air (~20 um) -- finer than a
# 1-track-radius smoke test, still coarser than the ~6-track-radii
# convergence-study grid from the original IonTracks repository.
DEMO_GRID_KWARGS = dict(grid_size_um=14.5, sampled_radius_cm=0.0057)


def run_demo():
    print("=" * 70)
    print("1) Pulsed-proton scenario, single-threaded Numba, ~30 s")
    print("=" * 70)
    config = SimulationConfig(**BEAM_KWARGS, **DEMO_GRID_KWARGS, seed=1)
    print(config.summary())
    print()

    t0 = time.perf_counter()
    warmup()  # one-off JIT compilation, excluded from the timing below
    print(f"Numba compile time (one-off): {time.perf_counter() - t0:.2f} s")

    t0 = time.perf_counter()
    result = run_simulation_numba(config, progress=True)
    elapsed_s = time.perf_counter() - t0
    print(f"\nWall time (numba, single-threaded): {elapsed_s:.1f} s")
    print(f"Final collection efficiency f = {result.f_t[-1]:.4f}")
    print(f"Recombination correction factor k_s = 1/f = {result.ks:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.time_s * 1e6, result.f_t)
    ax.axvline(config.pulse_duration_s * 1e6, color="gray", ls="--", label="pulse ends")
    ax.set_xlabel("time [us]")
    ax.set_ylabel("collection efficiency f(t)")
    ax.set_title("Charge collection during and after one proton pulse\n(~2.85 track radii, single-threaded numba, NOT fully converged)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("pulsed_proton_beam_f_of_t.png", dpi=150)
    print("Saved plot to pulsed_proton_beam_f_of_t.png")
    return config, result, elapsed_s


def validate_against_jaffe_theory():
    print()
    print("=" * 70)
    print("2) Single-track limit vs. analytic Jaffe theory")
    print("=" * 70)
    config = SimulationConfig(
        E_MeV_u=1.0,
        voltage_V=10.0,
        electrode_gap_cm=0.02,
        pulse_duration_s=1e-6,
        dose_rate_Gy_s=1e-8,  # forces number_of_tracks_per_pulse == 1
        grid_size_um=10.0,
        sampled_radius_cm=0.01,
        buffer_radius=5,
        no_z_electrode=4,
        seed=1,
    )
    result = run_simulation_numba(config, progress=False)
    ks_jaffe = jaffe_ks(config.LET_keV_um, config.voltage_V, config.electrode_gap_cm)
    print(f"k_s (PDE simulation, numba) = {result.ks:.6f}")
    print(f"k_s (Jaffe theory)          = {ks_jaffe:.6f}")


def estimate_converged_grid_cost(demo_config, demo_elapsed_s):
    print()
    print("=" * 70)
    print("3) Cost of a fully converged grid, still single-threaded Numba")
    print("=" * 70)
    demo_est = estimate_full_runtime(demo_config, backend="numba")
    print(
        f"  This demo's grid, numba estimate  : ~{demo_est['estimated_seconds']:.0f} s "
        f"(actually took {demo_elapsed_s:.0f} s -- the estimate is a sanity check on the cost model)"
    )
    for label, sampled_radius_cm, grid_size_um in [
        ("~3 track radii, finer grid", 0.006, 10.0),
        ("~6 track radii, converged grid (matches original IonTracks study)", 0.012, 5.0),
    ]:
        config = SimulationConfig(
            **BEAM_KWARGS, grid_size_um=grid_size_um, sampled_radius_cm=sampled_radius_cm, seed=1
        )
        est = estimate_full_runtime(config, backend="numba")
        print(
            f"  {label:65s}: grid {config.no_xy}x{config.no_xy}x{config.no_z_with_buffer}, "
            f"{est['total_tracks']:>10,d} tracks/pulse -> "
            f"~{est['estimated_hours']:.2g} h single-threaded numba estimate"
        )
    print(
        "\nA fully converged grid is now hours, not days/weeks, single-threaded "
        "-- Numba already did the heavy lifting (see tests/test_solver_numba.py "
        "for a direct comparison against the plain pure-Python reference, "
        "solver.py). Turning hours into minutes is the next step: numba "
        "prange, multiprocessing, or a GPU port of the two hot loops "
        "(_insert_track_numba and _lax_wendroff_step_numba in solver_numba.py)."
    )


if __name__ == "__main__":
    config, result, elapsed_s = run_demo()
    validate_against_jaffe_theory()
    estimate_converged_grid_cost(config, elapsed_s)
