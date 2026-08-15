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


def test_dry_run_exits_without_simulating():
    """--dry-run on the smallest tier should print the memory/runtime sizing
    and return well under the ~0.2 s it would take to actually run 'dev' --
    this is the whole point of a dry run: sizing a run that has not started
    yet, not measuring a fast one."""
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
    assert "dry run: rough runtime estimate" in out
    assert "No simulation was run (--dry-run)." in out
    # The real run prints per-step progress and a final "Wall time" line;
    # neither should appear when --dry-run skipped the simulation.
    assert "Wall time (" not in out
    assert "step 1/" not in out  # progress.py's "  step N/total  f = ..." lines


def test_dry_run_warns_about_batched_backend_when_threads_requested():
    """The single-track kernel timed by --dry-run is the unbatched one, so
    when --threads > 1 would select the batched backend the estimate is not
    a wall-time prediction -- the CLI must say so rather than print a
    number that looks authoritative."""
    result = subprocess.run(
        [sys.executable, str(RUN_MARKUS_2MM), "dev", "--dry-run", "--threads", "2"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "CAUTION" in result.stdout
    assert "batched backend" in result.stdout
