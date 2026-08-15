"""CLI-level checks for the example scripts under examples/.

Not physics tests -- those live in test_v2_physics.py and friends. This just
checks that the command-line surface behaves the way its own --help and
docs/BENCHMARKS-LAPTOP.md promise, run as a subprocess the way a user
actually invokes it.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_MARKUS_2MM = REPO_ROOT / "examples" / "ifj_aic144" / "run_markus_2mm.py"
PLOT = REPO_ROOT / "examples" / "ifj_aic144" / "plot.py"


def test_dry_run_exits_without_simulating():
    """--dry-run on the smallest tier should print only the memory sizing and
    return well under the ~0.2 s it would take to actually run 'dev' -- this
    is the whole point of a dry run: sizing a run that never allocates the
    grid, let alone starts it."""
    result = subprocess.run(
        [sys.executable, str(RUN_MARKUS_2MM), "dev", "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "dry run: memory" in out
    assert "Estimated peak allocation" in out
    assert "Fits within budget" in out or "Currently available RAM   : unknown" in out
    assert "No simulation was run (--dry-run)." in out
    # No runtime estimate of any kind belongs in --dry-run any more -- an
    # accurate one requires actually running the backend, which
    # --estimate-runtime-seconds does instead (see below).
    assert "runtime estimate" not in out
    # The real run prints per-step progress and a final "Wall time" line;
    # neither should appear when --dry-run skipped the simulation.
    assert "Wall time (" not in out
    assert "step 1/" not in out  # progress.py's "  step N/total  f = ..." lines


def test_estimate_runtime_seconds_runs_a_real_short_sample():
    """--estimate-runtime-seconds actually allocates the grid and runs the
    real backend for a bounded sample, then extrapolates -- unlike --dry-run,
    which never touches the solver at all. 'dev' is small enough to finish
    inside the budget, so this also exercises the exact/non-extrapolated
    branch."""
    result = subprocess.run(
        [sys.executable, str(RUN_MARKUS_2MM), "dev", "--estimate-runtime-seconds", "5"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "empirical runtime estimate" in out
    assert "steps measured" in out
    assert "estimated total wall time" in out
    assert "No full run was performed (--estimate-runtime-seconds)." in out
    # 'dev' finishes in well under 5s, so this is the exact-completion branch,
    # not an extrapolation.
    assert "finished inside the 5s budget" in out
    # Still must not run (and print the output of) the full, unbounded run.
    assert "Wall time (" not in out
    assert "\nwrote " not in out


def test_estimate_runtime_seconds_respects_explicit_backend_override():
    """--backend batched --threads 1 must sample the batched backend, not the
    num_threads==1 default of unbatched -- regression test for the mismatch
    where the header text (built from the resolved --backend/--threads
    decision) and the actual sampled backend (previously re-derived from
    num_threads alone inside estimate_full_runtime_empirical) could disagree.
    See test_runtime_estimate.py for the same check below the CLI layer."""
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_MARKUS_2MM),
            "dev",
            "--backend",
            "batched",
            "--threads",
            "1",
            "--estimate-runtime-seconds",
            "5",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "real solver_numba_parallel backend, 1 thread(s)" in result.stdout


def test_dry_run_and_estimate_runtime_are_mutually_exclusive():
    result = subprocess.run(
        [sys.executable, str(RUN_MARKUS_2MM), "dev", "--dry-run", "--estimate-runtime-seconds", "5"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr


def test_save_run_then_plot_is_the_documented_two_step_pipeline(tmp_path):
    """run_markus_2mm.py --save-run DIR, then plot.py DIR: the two-layer
    replacement for the old single-script report.py, run as subprocesses the
    way examples/README.md tells a user to run them."""
    run_dir = tmp_path / "run"
    run_result = subprocess.run(
        [sys.executable, str(RUN_MARKUS_2MM), "dev", "--save-run", str(run_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert (run_dir / "collected_charge.csv").exists()
    assert (run_dir / "track_density_xy.npy").exists()
    assert (run_dir / "run_meta.json").exists()
    assert f"python examples/ifj_aic144/plot.py {run_dir}" in run_result.stdout

    plot_result = subprocess.run(
        [sys.executable, str(PLOT), str(run_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert plot_result.returncode == 0, plot_result.stderr
    for name in (
        "injection_rate.png",
        "carrier_evolution.png",
        "recombination_rate.png",
        "track_density_cross_section.png",
    ):
        path = run_dir / name
        assert path.exists()
        assert path.stat().st_size > 5000


def test_plot_reports_a_missing_run_directory():
    result = subprocess.run(
        [sys.executable, str(PLOT), "/nonexistent/run/dir"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "collected_charge.csv" in result.stderr
