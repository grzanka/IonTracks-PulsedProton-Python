"""One-shot dump of Numba's parallel-region diagnostics, resolved threading
layer, and node/affinity/NUMA info -- context for interpreting
thread_sweep.csv on a machine where perf and py-spy's kernel-level counters
either aren't installed or need elevated privileges."""

import io
import os
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

import numba

from profiling.common import run_once
from pulsed_ion_chamber.solver_numba_parallel import (
    _accumulate_track_density_numba_parallel,
    _broadcast_density_numba_parallel,
    _lax_wendroff_step_numba_parallel,
)


def main() -> None:
    out_path = Path("profiling/data/diagnostics.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_once(num_threads=8)  # trigger JIT compilation + threading-layer resolution

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("=== numba.threading_layer() ===")
        print(numba.threading_layer())
        print()

        print("=== os.sched_getaffinity(0) ===")
        affinity = sorted(os.sched_getaffinity(0))
        print(f"cpus = {affinity}")
        print(f"len = {len(affinity)}")
        print()

        print("=== SLURM env (job/cpu sizing only) ===")
        for k, v in sorted(os.environ.items()):
            if k.startswith("SLURM_") and "BIND_LIST" not in k and "GTIDS" not in k:
                print(f"{k}={v}")
        print()

        print("=== numactl --hardware ===")
        proc = subprocess.run(["numactl", "--hardware"], capture_output=True, text=True)
        print(proc.stdout)

        print("=== lscpu ===")
        proc = subprocess.run(["lscpu"], capture_output=True, text=True)
        print(proc.stdout)

        # parallel_diagnostics() needs parfor metadata that a cache=True
        # dispatcher doesn't retain when loaded from an on-disk cache hit
        # (only a fresh compile keeps it) -- force one via recompile().
        for name, kernel in (
            ("_accumulate_track_density_numba_parallel", _accumulate_track_density_numba_parallel),
            ("_broadcast_density_numba_parallel", _broadcast_density_numba_parallel),
        ):
            print(f"=== parallel_diagnostics: {name} ===")
            kernel.recompile()
            kernel.parallel_diagnostics(level=4)
        print()
        print("=== parallel_diagnostics: _lax_wendroff_step_numba_parallel ===")
        _lax_wendroff_step_numba_parallel.recompile()
        _lax_wendroff_step_numba_parallel.parallel_diagnostics(level=4)

    out_path.write_text(buf.getvalue())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
