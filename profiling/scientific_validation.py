"""Scientific-correctness artifacts to hand over alongside the performance
data: does the parallel backend's result still match Jaffe theory (single-
track limit) and stay physically consistent across thread counts (not just
at 1 thread, where solver_numba_parallel.py's own tests already check it)?
"""

import csv
import json
from pathlib import Path

import numpy as np

from profiling.common import BEAM_KWARGS, CONVERGED_GRID_KWARGS, build_converged_config
from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel, warmup_parallel
from pulsed_ion_chamber.theory import jaffe_ks

OUT_DIR = Path("profiling/data")

THREAD_COUNTS = (1, 96)


def jaffe_single_track_check(num_threads: int) -> dict:
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
    rng = np.random.default_rng(config.seed)
    result = run_simulation_numba_parallel(config, rng=rng, progress=False, num_threads=num_threads)
    ks_jaffe = jaffe_ks(
        config.LET_keV_um,
        config.voltage_V,
        config.electrode_gap_cm,
        W_eV=config.W_eV,
        mu=config.mu_positive,
        D=config.D_positive,
    )
    return {
        "num_threads": num_threads,
        "ks_simulation": float(result.ks),
        "ks_jaffe": ks_jaffe,
        "relative_error": abs(result.ks - ks_jaffe) / ks_jaffe,
    }


def converged_grid_run(num_threads: int, seed: int = 1):
    config = build_converged_config(seed=seed)
    rng = np.random.default_rng(config.seed)
    result = run_simulation_numba_parallel(config, rng=rng, progress=False, num_threads=num_threads)
    return config, result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warmup_parallel()

    print("=== Jaffe single-track cross-check, by thread count ===")
    jaffe_rows = [jaffe_single_track_check(n) for n in THREAD_COUNTS]
    for row in jaffe_rows:
        print(row)
    with open(OUT_DIR / "jaffe_validation.json", "w") as f:
        json.dump(jaffe_rows, f, indent=2)

    print("\n=== Converged-grid f(t) curve, by thread count ===")
    print(f"config: {BEAM_KWARGS} + {CONVERGED_GRID_KWARGS}")
    summary_rows = []
    for n in THREAD_COUNTS:
        config, result = converged_grid_run(n)
        with open(OUT_DIR / f"f_t_curve_{n}threads.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "f_t"])
            for t, f_t in zip(result.time_s, result.f_t):
                writer.writerow([t, f_t])
        summary_rows.append({"num_threads": n, "ks": float(result.ks), "f_t_final": float(result.f_t[-1])})
        print(f"threads={n}: ks={result.ks:.6f}, f_final={result.f_t[-1]:.6f}")

    with open(OUT_DIR / "converged_grid_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    ks_values = [row["ks"] for row in summary_rows]
    max_spread = max(ks_values) - min(ks_values)
    print(f"\nk_s spread across thread counts {THREAD_COUNTS}: {max_spread:.3e} (float non-associativity only, expected ~1e-6 or smaller)")


if __name__ == "__main__":
    main()
