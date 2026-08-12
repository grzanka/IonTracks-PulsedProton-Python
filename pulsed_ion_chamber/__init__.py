"""Standalone finite-difference simulation of ion recombination in a
parallel-plate ionization chamber exposed to a pulsed proton beam.

Ported and adapted from the IonTracks-Cython repository (J.B. Christensen
et al., see README for references) as a clear starting point for a
multi-threading or GPU parallelization exercise. The baseline backend is
single-threaded Numba (`run_simulation_numba`, in solver_numba.py): the
same explicit loops, JIT-compiled, no `parallel=True`/`prange`. The plain
pure-Python reference it was compiled from (`run_simulation`, in
solver.py) is kept for comparison/readability but is ~10x slower.
`run_simulation_numba_parallel` (solver_numba_parallel.py) is the
shared-memory multi-core backend: same algorithm, both hot loops
parallelized across CPU cores with Numba prange.
"""

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.solver import Result, run_simulation
from pulsed_ion_chamber.solver_numba import run_simulation_numba
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel

__all__ = ["SimulationConfig", "Result", "run_simulation", "run_simulation_numba", "run_simulation_numba_parallel"]
