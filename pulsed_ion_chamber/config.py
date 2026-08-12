"""Simulation configuration: physical inputs plus all derived grid/timing
quantities needed by the PDE solver.

The grid layout (sampled cylinder + electrode buffer) and the time-step
search follow hadrons/cython_files/continuous_beam.pyx (and its pure-Python
twin hadrons/python/continuous_beam.py) in the IonTracks-Cython repository.
What's new here is the *pulse-train* timing: instead of spreading tracks
uniformly over the whole simulated window (continuous beam), tracks are
only injected during repeating pulse_duration_s windows spaced
1/repetition_rate_hz apart (pulsed beam), followed by a clearance period
long enough for ions to drift out of the gap.
"""

import warnings
from dataclasses import dataclass
from math import pi
from typing import Optional

import numpy as np

from pulsed_ion_chamber.constants import ION_DIFFUSION_CM2_S, ION_MOBILITY_CM2_VS, W_EV_PER_ION_PAIR
from pulsed_ion_chamber.stopping_power import (
    E_MeV_u_to_LET_keV_um,
    calc_track_radius_cm,
    dose_rate_to_fluence_rate,
)


def _von_neumann_dt(ion_diff, grid_spacing_cm, ion_mobility, Efield_V_cm):
    """Largest time step dt (starting from 1 s and shrinking) that satisfies the
    von Neumann stability criterion for the explicit Lax-Wendroff scheme
    (Deghan, 2004), for full 3D diffusion (sx = sy = sz) and drift along z only.
    """
    # The second Deghan criterion, cx^2*cy^2*cz^2 <= 8*sx*sy*sz, is trivially
    # satisfied here since cx = cy = 0 (drift is along z only).
    dt = 1.0
    while True:
        dt /= 1.01
        s = ion_diff * dt / grid_spacing_cm**2  # sx = sy = sz
        cz = ion_mobility * Efield_V_cm * dt / grid_spacing_cm
        if 6 * s + cz**2 <= 1:
            return dt


@dataclass
class SimulationConfig:
    # --- beam / chamber physics ---
    E_MeV_u: float = 150.0
    particle: str = "proton"
    voltage_V: float = 200.0
    electrode_gap_cm: float = 0.2

    # --- pulsed-beam timing ---
    pulse_duration_s: float = 540e-6
    repetition_rate_hz: float = 50.0
    dose_rate_Gy_s: float = 60.0  # time-averaged dose rate
    n_pulses: int = 1
    n_clearance_separation_times: float = 2.0
    # Optional: the accelerator's RF frequency (e.g. a cyclotron's ~10-100 MHz
    # extraction RF), purely for the diagnostic check in __post_init__/summary()
    # below -- tracks within a pulse are NOT placed at explicit RF-bucket
    # times, see the note there for why.
    rf_frequency_hz: Optional[float] = None

    # --- grid (reduced/representative sub-volume, not a full chamber) ---
    # These defaults are deliberately coarse (sampled_radius_cm is about one
    # track radius, and grid_size_um is a coarse voxel size) so that
    # SimulationConfig() runs in well under a minute on a laptop -- a fast
    # correctness check, not a dosimetrically converged result. See
    # pulsed_ion_chamber.benchmark and README.md for the cost of refining
    # these towards a converged, publication-quality grid.
    grid_size_um: float = 40.0
    sampled_radius_cm: float = 0.002
    buffer_radius: int = 4
    no_z_electrode: int = 3
    max_voxels: float = 1e8

    seed: Optional[int] = None

    def __post_init__(self):
        self.unit_length_cm = self.grid_size_um * 1e-4
        self.LET_keV_um = E_MeV_u_to_LET_keV_um(self.E_MeV_u, self.particle)
        self.track_radius_cm = calc_track_radius_cm(self.LET_keV_um)
        self.Efield_V_cm = self.voltage_V / self.electrode_gap_cm
        self.area_cm2 = pi * self.sampled_radius_cm**2

        # --- grid layout ---
        self.no_xy = int(2 * self.sampled_radius_cm / self.unit_length_cm) + 2 * self.buffer_radius
        self.no_z = int(self.electrode_gap_cm / self.unit_length_cm)
        self.no_z_with_buffer = 2 * self.no_z_electrode + self.no_z
        if self.no_xy * self.no_xy * self.no_z > self.max_voxels:
            raise ValueError(
                f"Grid too large: {self.no_xy}x{self.no_xy}x{self.no_z} "
                f"({self.no_xy * self.no_xy * self.no_z:.3g} voxels). "
                "Increase grid_size_um or reduce sampled_radius_cm."
            )
        self.mid_xy = self.no_xy // 2
        self.outer_radius = self.no_xy / 2.0
        self.inner_radius = self.outer_radius - self.buffer_radius
        if self.inner_radius <= 0:
            # pulses.sample_xy_inside_cylinder rejection-samples uniformly in
            # [0, no_xy]^2 until it lands inside inner_radius; if that disk
            # has zero or negative radius the loop never (or almost never)
            # accepts and hangs. Catch it here instead of at runtime.
            raise ValueError(
                f"inner_radius = {self.inner_radius:.3g} <= 0 (no_xy={self.no_xy}, "
                f"buffer_radius={self.buffer_radius}): sampled_radius_cm is too small "
                "relative to buffer_radius/grid_size_um. Increase sampled_radius_cm, "
                "decrease grid_size_um, or decrease buffer_radius."
            )
        # precomputed once so hot loops can compare squared distances and
        # avoid a sqrt() per voxel/track (see solver_numba.py)
        self.inner_radius_sq = self.inner_radius**2

        # --- time step (von Neumann stability) and drift/pulse timing ---
        self.dt = _von_neumann_dt(
            ION_DIFFUSION_CM2_S, self.unit_length_cm, ION_MOBILITY_CM2_VS, self.Efield_V_cm
        )

        # Real proton beams (cyclotron or synchrocyclotron) arrive as a train
        # of RF-bucket micro-pulses within pulse_duration_s, not a smooth
        # rate -- but at every accelerator relevant here, the RF period is
        # far shorter than dt (itself already at its stability-limited
        # maximum for the chosen grid), so individual RF buckets can't be
        # resolved: many of them fall inside the same simulation time step
        # regardless of how track arrival times are distributed within it.
        # Averaging over the RF microstructure (pulses.py spreads tracks
        # pseudo-uniformly across the whole pulse) is therefore the correct
        # simplification, not an approximation of convenience. This check
        # makes that assumption explicit instead of silent.
        self.rf_cycles_per_time_step = self.dt * self.rf_frequency_hz if self.rf_frequency_hz else None
        if self.rf_cycles_per_time_step is not None and self.rf_cycles_per_time_step < 1.0:
            warnings.warn(
                f"dt ({self.dt:.3g} s) is shorter than one RF period "
                f"(1/{self.rf_frequency_hz:.4g} Hz = {1.0 / self.rf_frequency_hz:.3g} s): "
                "RF bucket structure may not be safely averaged over at this "
                "resolution -- consider whether explicit RF-bucket timing is "
                "needed for this configuration.",
                stacklevel=2,
            )

        self.separation_time_steps = int(
            self.electrode_gap_cm / (2.0 * ION_MOBILITY_CM2_VS * self.Efield_V_cm * self.dt)
        )
        self.clearance_time_steps = int(round(self.n_clearance_separation_times * self.separation_time_steps))

        self.pulse_time_bins = np.arange(0.0, self.pulse_duration_s + self.dt, self.dt)
        self.pulse_time_steps = len(self.pulse_time_bins) - 1

        self.pulse_period_steps = (
            int(round(1.0 / self.repetition_rate_hz / self.dt)) if self.repetition_rate_hz > 0 else self.pulse_time_steps
        )
        if self.n_pulses > 1 and self.pulse_period_steps < self.pulse_time_steps:
            raise ValueError(
                "Pulse period is shorter than the pulse itself at this dt "
                f"({self.pulse_period_steps} < {self.pulse_time_steps} steps); "
                "reduce pulse_duration_s or repetition_rate_hz."
            )

        self.total_time_steps = (
            (self.n_pulses - 1) * self.pulse_period_steps + self.pulse_time_steps + self.clearance_time_steps
        )

        # --- Gaussian track structure ---
        LET_eV_cm = self.LET_keV_um * 1e7
        self.N0 = LET_eV_cm / W_EV_PER_ION_PAIR
        self.Gaussian_factor = self.N0 / (pi * self.track_radius_cm**2)

        # --- number of tracks injected per pulse, from the average dose rate ---
        dose_per_pulse_Gy = self.dose_rate_Gy_s / self.repetition_rate_hz
        instantaneous_dose_rate_Gy_s = dose_per_pulse_Gy / self.pulse_duration_s
        fluence_rate_inst_cm2_s = dose_rate_to_fluence_rate(
            instantaneous_dose_rate_Gy_s, self.E_MeV_u, self.particle
        )
        self.number_of_tracks_per_pulse = max(
            1, int(round(fluence_rate_inst_cm2_s * self.pulse_duration_s * self.area_cm2))
        )

    def summary(self) -> str:
        rf_line = ""
        if self.rf_frequency_hz:
            rf_period_s = 1.0 / self.rf_frequency_hz
            rf_line = (
                f"\nRF microstructure     : {self.rf_frequency_hz / 1e6:.4g} MHz "
                f"(period {rf_period_s * 1e9:.3g} ns) -> {self.rf_cycles_per_time_step:.3g} "
                "RF cycles per time step (averaged over, not individually resolved)"
            )
        return (
            f"Particle              : {self.particle} @ {self.E_MeV_u:.1f} MeV/u "
            f"(LET = {self.LET_keV_um:.3g} keV/um, track radius b = {self.track_radius_cm * 1e4:.3g} um)\n"
            f"Chamber               : gap = {self.electrode_gap_cm} cm, V = {self.voltage_V} V "
            f"(E = {self.Efield_V_cm:.4g} V/cm)\n"
            f"Sampled sub-volume    : radius = {self.sampled_radius_cm * 1e4:.3g} um, "
            f"area = {self.area_cm2:.3g} cm^2\n"
            f"Grid                  : {self.no_xy} x {self.no_xy} x {self.no_z_with_buffer} voxels "
            f"({self.unit_length_cm * 1e4:.3g} um/voxel)\n"
            f"Time step dt          : {self.dt:.3e} s\n"
            f"Separation time       : {self.separation_time_steps} steps "
            f"({self.separation_time_steps * self.dt * 1e6:.3g} us)\n"
            f"Pulse                 : {self.pulse_duration_s * 1e6:.1f} us "
            f"({self.pulse_time_steps} steps), {self.number_of_tracks_per_pulse} tracks, "
            f"{self.repetition_rate_hz} Hz, {self.n_pulses} pulse(s)"
            f"{rf_line}\n"
            f"Total simulated time  : {self.total_time_steps * self.dt * 1e6:.3g} us "
            f"({self.total_time_steps} steps)"
        )
