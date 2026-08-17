"""Ion recombination in a plane-parallel ionisation chamber exposed to a
pulsed proton beam.

Solves the coupled drift-diffusion-recombination equations for the positive and
negative charge-carrier densities on a regular voxel grid, with protons entered
as individual Gaussian ion tracks, and reports the recombination correction
factor ``k_s``.

Three interchangeable backends, all agreeing on k_s to a relative tolerance:

* :func:`run_simulation_numba` -- deposits one track at a time. The simpler
  reference; best when few tracks arrive per time step.
* :func:`run_simulation_numba_parallel` -- deposits a whole time step's tracks
  in one pass and runs both hot loops under ``prange``. Best for dense pulses
  and large grids on a multi-core CPU.
* :func:`run_simulation_cuda` -- the same physics on an NVIDIA GPU, with the
  carrier arrays resident on the device for the whole run. Best for grids too
  large to fit in a CPU's cache (the 5 um full electrode); a small grid is
  faster on one CPU core. Needs CuPy and Numba's CUDA target, imported lazily
  so a CPU-only install is unaffected. See docs/GPU.md.

See docs/PHYSICS.md for what is modelled and why, docs/ALGORITHM.md for how,
and docs/PERFORMANCE.md / docs/GPU.md for what it costs.
"""

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.rdd import RadialDoseDistribution, chamber_ks
from pulsed_ion_chamber.solver_cuda import run_simulation_cuda
from pulsed_ion_chamber.solver_numba import run_simulation_numba
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel
from pulsed_ion_chamber.state import Result

__all__ = [
    "SimulationConfig",
    "Result",
    "RadialDoseDistribution",
    "chamber_ks",
    "run_simulation_numba",
    "run_simulation_numba_parallel",
    "run_simulation_cuda",
]
