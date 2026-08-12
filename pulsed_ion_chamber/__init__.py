"""Standalone finite-difference simulation of ion recombination in a
parallel-plate ionization chamber exposed to a pulsed proton beam.

Ported and adapted from the IonTracks-Cython repository
(J.B. Christensen et al., see README for references), stripped of the
Cython/Numba/CuPy backends and rewritten as plain explicit-loop
NumPy/Python so it can serve as a clear starting point for a
multi-threading or GPU parallelization exercise.
"""

from pulsed_ion_chamber.config import SimulationConfig
from pulsed_ion_chamber.solver import Result, run_simulation

__all__ = ["SimulationConfig", "Result", "run_simulation"]
