"""Physical constants for ion transport in air, from Kanai et al. (1998),
as used throughout the original IonTracks code (hadrons/continuous_beam.pyx).

Note: the original repository uses W = 33.9 eV/ion-pair in its Jaffe-theory
reference (hadrons/functions.py) but W = 34.2 eV/ion-pair (a proton-specific
value) in its PDE solvers (hadrons/*/continuous_beam.py*). We standardize on
the proton-specific value everywhere in this port so the PDE solver and the
analytic cross-checks in theory.py are internally consistent.
"""

W_EV_PER_ION_PAIR = 34.2  # eV, mean energy to create an ion pair in air (protons)
ION_MOBILITY_CM2_VS = 1.65  # cm^2 / (V s), averaged over positive/negative ions
ION_DIFFUSION_CM2_S = 3.7e-2  # cm^2 / s, averaged over positive/negative ions
RECOMBINATION_ALPHA_CM3_S = 1.60e-6  # cm^3 / s, recombination coefficient

AIR_DENSITY_KG_M3 = 1.225  # dry air, standard conditions
JOULE_TO_KEV = 6.241e15  # 1 J expressed in keV

DEFAULT_BUFFER_RADIUS = 10  # voxels of margin around the sampled cylinder
DEFAULT_NO_Z_ELECTRODE = 5  # voxels of margin at each electrode
