"""Ion recombination in a plane-parallel ionisation chamber exposed to a
pulsed proton beam.

Solves the coupled drift-diffusion-recombination equations for the positive and
negative charge-carrier densities on a regular voxel grid, with protons entered
as individual Gaussian ion tracks, and reports the recombination correction
factor ``k_s``.

Two interchangeable backends, both JIT-compiled with Numba and agreeing to
1e-9:

* :func:`run_simulation_numba` -- deposits one track at a time. The simpler
  reference; best when few tracks arrive per time step.
* :func:`run_simulation_numba_parallel` -- deposits a whole time step's tracks
  in one pass and runs both hot loops under ``prange``. Best for dense pulses
  and large grids.

See docs/PHYSICS.md for what is modelled and why, docs/ALGORITHM.md for how,
and docs/PERFORMANCE.md for what it costs.
"""

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.solver_numba import run_simulation_numba
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel
from pulsed_ion_chamber.state import Result

__all__ = [
    "SimulationConfig",
    "Result",
    "run_simulation_numba",
    "run_simulation_numba_parallel",
]
