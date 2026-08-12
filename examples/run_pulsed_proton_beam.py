#!/usr/bin/env python3
"""Example: recombination in a parallel-plate ionization chamber exposed to
a pulsed proton beam (540 us pulses, 50 Hz, 60 Gy/s average dose rate).

This script does four things:

1. Runs a quick, coarse-grid version of the requested scenario end to end
   (~1 minute) and plots the collection efficiency f(t) and the final
   recombination correction factor k_s = 1/f.
2. Re-runs the *same* scenario with the two hot loops JIT-compiled by
   Numba (still single-threaded: no parallel=True, no prange) and reports
   the speedup and a correctness check against the pure-Python result --
   the smallest possible first step of the parallelization exercise.
3. Validates the solver in the single-track limit against the analytic
   Jaffe theory (a fast, independent correctness check).
4. Uses pulsed_ion_chamber.benchmark to *estimate* -- without running it --
   how long a dosimetrically-converged version of the same scenario (a
   sampled volume several ion-track-radii wide, at fine grid resolution)
   would take in this serial, explicit-loop Python implementation. That
   gap is the reason this repository exists: it is your starting point for
   a multi-threaded or GPU port.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from pulsed_ion_chamber.benchmark import estimate_full_runtime
from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.solver import run_simulation
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


def run_quick_demo():
    print("=" * 70)
    print("1) Quick demo: coarse grid, runs in well under a minute")
    print("=" * 70)
    config = SimulationConfig(**BEAM_KWARGS, grid_size_um=40.0, sampled_radius_cm=0.002, seed=1)
    print(config.summary())
    print()
    t0 = time.perf_counter()
    result = run_simulation(config, progress=True)
    elapsed_s = time.perf_counter() - t0
    print(f"\nWall time (pure Python): {elapsed_s:.1f} s")
    print(f"Final collection efficiency f = {result.f_t[-1]:.4f}")
    print(f"Recombination correction factor k_s = 1/f = {result.ks:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.time_s * 1e6, result.f_t)
    ax.axvline(config.pulse_duration_s * 1e6, color="gray", ls="--", label="pulse ends")
    ax.set_xlabel("time [us]")
    ax.set_ylabel("collection efficiency f(t)")
    ax.set_title("Charge collection during and after one proton pulse\n(coarse grid demo, NOT dosimetrically converged)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("pulsed_proton_beam_f_of_t.png", dpi=150)
    print("Saved plot to pulsed_proton_beam_f_of_t.png")
    return config, result, elapsed_s


def run_numba_speedup_check(config, result_py, elapsed_py_s):
    print()
    print("=" * 70)
    print("2) Same scenario, single-threaded Numba JIT vs. pure Python")
    print("=" * 70)
    try:
        from pulsed_ion_chamber.solver_numba import run_simulation_numba, warmup
    except ImportError:
        print("numba is not installed (pip install numba) -- skipping this section.")
        return

    t0 = time.perf_counter()
    warmup()  # trigger JIT compilation on tiny dummy arrays, excluded from the timing below
    print(f"Numba compile time (one-off): {time.perf_counter() - t0:.2f} s")

    t0 = time.perf_counter()
    result_nb = run_simulation_numba(config, progress=False)
    elapsed_nb_s = time.perf_counter() - t0
    print(f"Wall time (numba, single-threaded, warm): {elapsed_nb_s:.2f} s")
    print(f"Speedup vs. pure Python: {elapsed_py_s / elapsed_nb_s:.1f}x")

    max_diff = float(np.max(np.abs(result_py.f_t - result_nb.f_t)))
    print(f"Max |f_t difference| vs. pure Python: {max_diff:.3g} (k_s: {result_py.ks:.6f} vs {result_nb.ks:.6f})")


def validate_against_jaffe_theory():
    print()
    print("=" * 70)
    print("3) Single-track limit vs. analytic Jaffe theory")
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
    result = run_simulation(config, progress=False)
    ks_jaffe = jaffe_ks(config.LET_keV_um, config.voltage_V, config.electrode_gap_cm)
    print(f"k_s (PDE simulation) = {result.ks:.6f}")
    print(f"k_s (Jaffe theory)   = {ks_jaffe:.6f}")


def estimate_converged_runtime():
    print()
    print("=" * 70)
    print("4) Cost of a dosimetrically-converged version of the same scenario")
    print("=" * 70)
    for label, sampled_radius_cm, grid_size_um in [
        ("coarse (demo above)", 0.002, 40.0),
        ("~2 track radii, coarser grid", 0.004, 20.0),
        ("~3 track radii, finer grid", 0.006, 10.0),
        ("~6 track radii, converged grid (matches original IonTracks study)", 0.012, 5.0),
    ]:
        config = SimulationConfig(
            **BEAM_KWARGS, grid_size_um=grid_size_um, sampled_radius_cm=sampled_radius_cm, seed=1
        )
        est = estimate_full_runtime(config)
        print(
            f"  {label:65s}: grid {config.no_xy}x{config.no_xy}x{config.no_z_with_buffer}, "
            f"{est['total_tracks']:>10,d} tracks/pulse -> "
            f"~{est['estimated_hours']:.2g} h serial-Python estimate"
        )
    print(
        "\nSingle-threaded Numba (section 2) already buys a solid speedup for "
        "free, but even that likely isn't enough to close this gap -- "
        "the explicit loops in solver.py (_insert_track and "
        "_lax_wendroff_step) are the target for your next step: numba "
        "prange, multiprocessing, or a GPU port."
    )


if __name__ == "__main__":
    config, result, elapsed_s = run_quick_demo()
    run_numba_speedup_check(config, result, elapsed_s)
    validate_against_jaffe_theory()
    estimate_converged_runtime()
