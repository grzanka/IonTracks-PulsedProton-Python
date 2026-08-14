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

# Per-species Kanai (1998) values. The averaged constants above lump both
# carriers into one effective species, as IonTracks-Cython does; other codes
# such as IonTracks-FEniCSx carry the
# two species separately. SimulationConfig defaults to the averaged pair; pass
# the four below to resolve them. See docs/PHYSICS.md sec. 8 for the trade-off.
ION_MOBILITY_POSITIVE_CM2_VS = 1.36  # cm^2 / (V s)  (1.36e-4 m^2/(V s))
ION_MOBILITY_NEGATIVE_CM2_VS = 2.10  # cm^2 / (V s)  (2.10e-4 m^2/(V s))
ION_DIFFUSION_POSITIVE_CM2_S = 2.82e-2  # cm^2 / s  (2.82e-6 m^2/s)
ION_DIFFUSION_NEGATIVE_CM2_S = 4.35e-2  # cm^2 / s  (4.35e-6 m^2/s)

AIR_DENSITY_KG_M3 = 1.225  # dry air, standard conditions
# Dry air at 20 degC, 101.325 kPa -- the reference condition dosimetry
# normally quotes. 1.225 above is the ISA sea-level value (15 degC) inherited
# from IonTracks-Cython; the IonTracks-FEniCSx ifj_aic144 configs used 1.293
# (0 degC). All three are the same gas at different reference conditions and
# the track count scales linearly with whichever is chosen.
AIR_DENSITY_20C_KG_M3 = 1.2041
JOULE_TO_KEV = 6.241e15  # 1 J expressed in keV

DEFAULT_BUFFER_RADIUS = 10  # voxels of margin around the sampled cylinder
DEFAULT_NO_Z_ELECTRODE = 5  # voxels of margin at each electrode
