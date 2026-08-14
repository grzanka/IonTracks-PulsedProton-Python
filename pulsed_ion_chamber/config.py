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
from math import exp, pi, sqrt
from typing import Optional

import numpy as np

from pulsed_ion_chamber.constants import (
    AIR_DENSITY_KG_M3,
    ION_DIFFUSION_CM2_S,
    ION_MOBILITY_CM2_VS,
    W_EV_PER_ION_PAIR,
)
from pulsed_ion_chamber.stopping_power import (
    E_MeV_u_to_LET_keV_um,
    calc_track_radius_cm,
    dose_rate_to_fluence_rate,
)

LATERAL_BOUNDARIES = ("absorbing", "reflecting")
SCORING_REGIONS = ("track_disc", "full_grid")


def _von_neumann_dt(ion_diff, grid_spacing_cm, ion_mobility, Efield_V_cm):
    """Largest time step dt (starting from 1 s and shrinking) that satisfies the
    von Neumann stability criterion for the explicit Lax-Wendroff scheme
    (Deghan, 2004), for full 3D diffusion (sx = sy = sz) and drift along z only.

    ion_diff and ion_mobility may each be a scalar or a sequence; when a
    sequence is given the returned dt satisfies the criterion for *every*
    (diffusion, mobility) pair, i.e. for whichever carrier species is the
    binding one. With the two Kanai species that is the negative ion, which is
    both faster (mu = 2.10 vs 1.36) and more diffusive (D = 4.35e-2 vs
    2.82e-2), so it sets dt on its own.
    """
    # The second Deghan criterion, cx^2*cy^2*cz^2 <= 8*sx*sy*sz, is trivially
    # satisfied here since cx = cy = 0 (drift is along z only).
    diffs = np.atleast_1d(ion_diff)
    mobilities = np.atleast_1d(ion_mobility)
    if len(diffs) != len(mobilities):
        raise ValueError(
            f"ion_diff and ion_mobility must describe the same species: got "
            f"{len(diffs)} diffusion coefficient(s) and {len(mobilities)} mobility/ies."
        )
    dt = 1.0
    while True:
        dt /= 1.01
        s = diffs * dt / grid_spacing_cm**2  # sx = sy = sz
        cz = mobilities * Efield_V_cm * dt / grid_spacing_cm
        if np.all(6 * s + cz**2 <= 1):
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

    # --- gas / carrier physics (None = use the module-level constants) ---
    # Leaving all four of these None reproduces the original IonTracks-Cython
    # model exactly: one averaged mobility and one averaged diffusion
    # coefficient shared by both carriers. Setting them (see
    # constants.ION_MOBILITY_POSITIVE_CM2_VS and friends) resolves the two
    # Kanai species separately. See docs/PHYSICS.md sec. 8.
    mu_positive_cm2_Vs: Optional[float] = None
    mu_negative_cm2_Vs: Optional[float] = None
    D_positive_cm2_s: Optional[float] = None
    D_negative_cm2_s: Optional[float] = None
    W_eV: float = W_EV_PER_ION_PAIR
    air_density_kg_m3: float = AIR_DENSITY_KG_M3

    # --- track placement vs. track count ---
    # Fraction of sampled_radius_cm that tracks are actually drawn within,
    # mirroring IonTracks-FEniCSx's track_source.chamber_fill_fraction: the
    # track *count* still comes from the full sampled_radius_cm disc (which is
    # also the scored disc), so a value below 1 concentrates the same charge
    # into a smaller area -- 1/f^2 times the nominal areal density. 1.0 (the
    # default) is the self-consistent IonTracks-Cython behaviour.
    chamber_fill_fraction: float = 1.0

    # --- lateral boundary condition on the outer voxel ring ---
    # "absorbing": the ring stays clamped at zero, so charge reaching it is
    #   lost (original IonTracks-Cython behaviour -- the ring is simply never
    #   updated by the Lax-Wendroff sweep).
    # "reflecting": zero-gradient (zero-flux) mirror, matching the natural
    #   boundary condition IonTracks-FEniCSx leaves on the cylinder wall. This
    #   is the better model of a sub-volume sampled from the *interior* of a
    #   large, uniformly irradiated chamber, where neighbouring gas returns as
    #   much charge as it takes. See docs/PHYSICS.md sec. 10.
    lateral_boundary: str = "absorbing"

    # --- what counts towards injected charge and recombination ---
    # "track_disc": only voxels inside sampled_radius_cm, i.e. the disc the
    #   tracks were drawn in (original IonTracks-Cython behaviour). Charge that
    #   diffuses out of that disc was counted as injected but its later
    #   recombination is not, so the scored column leaks -- which is why k_s
    #   creeps upward as sampled_radius_cm grows.
    # "full_grid": every voxel in the gap, matching IonTracks-FEniCSx, which
    #   integrates both the injected density and alpha*n+*n- over its whole
    #   mesh while confining tracks to a smaller disc. Combined with
    #   lateral_boundary="reflecting" this closes the books on the sampled
    #   volume: nothing enters or leaves sideways, so the edge systematic
    #   disappears and a much smaller column converges.
    scoring_region: str = "track_disc"

    # --- how far a track's Gaussian is deposited before it is truncated ---
    # In standard deviations. The radial dose distribution is written
    # exp(-r^2/b^2), which is a Gaussian of standard deviation sigma = b/sqrt(2),
    # so a cutoff of `n` sigma sits at r = n*b/sqrt(2) and discards a fraction
    # exp(-n^2/2) of each track's charge. At the default 10 sigma that is
    # 1.9e-22 -- six orders of magnitude below float64 epsilon, so the
    # truncation is exact in the only sense that matters numerically.
    # None disables truncation and deposits over the whole grid.
    track_cutoff_sigmas: Optional[float] = 10.0

    seed: Optional[int] = None

    def __post_init__(self):
        self.unit_length_cm = self.grid_size_um * 1e-4
        self.LET_keV_um = E_MeV_u_to_LET_keV_um(self.E_MeV_u, self.particle)
        self.track_radius_cm = calc_track_radius_cm(self.LET_keV_um)
        self.Efield_V_cm = self.voltage_V / self.electrode_gap_cm
        self.area_cm2 = pi * self.sampled_radius_cm**2

        self._resolve_carrier_constants()

        if self.lateral_boundary not in LATERAL_BOUNDARIES:
            raise ValueError(
                f"lateral_boundary must be one of {LATERAL_BOUNDARIES}, got {self.lateral_boundary!r}."
            )
        if self.scoring_region not in SCORING_REGIONS:
            raise ValueError(f"scoring_region must be one of {SCORING_REGIONS}, got {self.scoring_region!r}.")
        if isinstance(self.chamber_fill_fraction, bool) or not 0 < self.chamber_fill_fraction <= 1:
            raise ValueError(
                f"chamber_fill_fraction must be a number in (0, 1], got {self.chamber_fill_fraction!r}."
            )

        # --- grid layout ---
        # Round rather than truncate: int() silently shrinks the sampled disc
        # (0.0084 cm at 10 um/voxel truncates 16.8 -> 16, an 80 um radius)
        # while area_cm2 above still uses the requested 84 um, which would put
        # ~10% more charge into the grid than the track count implies.
        self.no_xy = int(round(2 * self.sampled_radius_cm / self.unit_length_cm)) + 2 * self.buffer_radius
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

        # Tracks are drawn inside this radius; inner_radius above remains the
        # scored (and track-counting) disc. Equal unless chamber_fill_fraction
        # was lowered.
        self.sampling_radius = self.inner_radius * self.chamber_fill_fraction
        self.sampling_radius_sq = self.sampling_radius**2

        # What the solvers actually compare (i-mid)^2 + (j-mid)^2 against when
        # deciding whether a voxel is scored. Made big enough to admit every
        # voxel -- corners included -- for "full_grid", so the kernels need no
        # extra branch: the radius test simply always passes.
        self.scoring_radius_sq = (
            self.inner_radius_sq if self.scoring_region == "track_disc" else 2.0 * self.no_xy**2
        )

        # --- time step (von Neumann stability) and drift/pulse timing ---
        self.dt = _von_neumann_dt(
            (self.D_positive, self.D_negative),
            self.unit_length_cm,
            (self.mu_positive, self.mu_negative),
            self.Efield_V_cm,
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

        # Half-gap transit of the *slowest* carrier, so the clearance period
        # below is long enough for whichever species lags. With the averaged
        # constants both species are equally fast and this is unchanged; with
        # the two Kanai species it is set by the positive ion (mu = 1.36),
        # which needs 98 us to cross a 2 mm gap against the negative ion's
        # 63.5 us. IonTracks-FEniCSx sizes its collection tail the same way
        # (utils/calculate_simulation_params.determine_simulation_time_one_track
        # takes d / min(v+, v-)), with a 1.3 safety coefficient on top --
        # reproduce that with n_clearance_separation_times = 2.6.
        self.slowest_mobility_cm2_Vs = min(self.mu_positive, self.mu_negative)
        self.separation_time_steps = int(
            self.electrode_gap_cm / (2.0 * self.slowest_mobility_cm2_Vs * self.Efield_V_cm * self.dt)
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
        self.N0 = LET_eV_cm / self.W_eV
        self.Gaussian_factor = self.N0 / (pi * self.track_radius_cm**2)

        # --- deposition stencil ---
        self.track_sigma_cm = self.track_radius_cm / sqrt(2.0)
        if self.track_cutoff_sigmas is None:
            # No truncation: a box this wide always spans the whole grid, since
            # no grid point can be further than no_xy voxels from any track.
            self.track_cutoff_voxels = float(self.no_xy)
            self.track_cutoff_cm = self.track_cutoff_voxels * self.unit_length_cm
            self.truncated_charge_fraction = 0.0
        else:
            if self.track_cutoff_sigmas <= 0:
                raise ValueError(
                    f"track_cutoff_sigmas must be positive (or None for no truncation), "
                    f"got {self.track_cutoff_sigmas!r}."
                )
            self.track_cutoff_cm = self.track_cutoff_sigmas * self.track_sigma_cm
            self.track_cutoff_voxels = self.track_cutoff_cm / self.unit_length_cm
            # Fraction of a track's charge discarded: for the 2D Gaussian
            # exp(-r^2/b^2), the integral beyond radius R is exp(-R^2/b^2).
            self.truncated_charge_fraction = exp(-((self.track_cutoff_cm / self.track_radius_cm) ** 2))

        # --- number of tracks injected per pulse, from the average dose rate ---
        dose_per_pulse_Gy = self.dose_rate_Gy_s / self.repetition_rate_hz
        instantaneous_dose_rate_Gy_s = dose_per_pulse_Gy / self.pulse_duration_s
        fluence_rate_inst_cm2_s = dose_rate_to_fluence_rate(
            instantaneous_dose_rate_Gy_s,
            self.E_MeV_u,
            self.particle,
            air_density_kg_m3=self.air_density_kg_m3,
        )
        self.number_of_tracks_per_pulse = max(
            1, int(round(fluence_rate_inst_cm2_s * self.pulse_duration_s * self.area_cm2))
        )

    def _resolve_carrier_constants(self):
        """Fill in per-species transport constants, defaulting to the single
        averaged pair the original IonTracks-Cython solvers use for both."""
        self.mu_positive = (
            ION_MOBILITY_CM2_VS if self.mu_positive_cm2_Vs is None else self.mu_positive_cm2_Vs
        )
        self.mu_negative = (
            ION_MOBILITY_CM2_VS if self.mu_negative_cm2_Vs is None else self.mu_negative_cm2_Vs
        )
        self.D_positive = ION_DIFFUSION_CM2_S if self.D_positive_cm2_s is None else self.D_positive_cm2_s
        self.D_negative = ION_DIFFUSION_CM2_S if self.D_negative_cm2_s is None else self.D_negative_cm2_s
        for name, value in (
            ("mu_positive_cm2_Vs", self.mu_positive),
            ("mu_negative_cm2_Vs", self.mu_negative),
            ("D_positive_cm2_s", self.D_positive),
            ("D_negative_cm2_s", self.D_negative),
        ):
            if value <= 0:
                # Both mobilities are magnitudes here: the drift *directions*
                # are hard-coded opposite in the solvers' z-stencils, so a
                # negative value (as IonTracks-FEniCSx's configs write
                # mu_negative) would flip the sign twice and send both species
                # to the same electrode.
                raise ValueError(f"{name} must be a positive magnitude, got {value!r}.")
        self.two_carrier_species = not (
            self.mu_positive == self.mu_negative and self.D_positive == self.D_negative
        )

    def scheme_coefficients(self):
        """Lax-Wendroff stencil weights, per carrier species.

        Returns (positive, negative), each a 4-tuple
        ``(lateral, z_minus, z_plus, centre)`` giving the weight on the six
        face neighbours and on the voxel itself. The two species differ only
        through their own s = D dt / h^2 and c = mu E dt / h, and drift in
        opposite directions along z -- which is why the negative tuple has its
        z_minus/z_plus swapped relative to how the positive one is applied.
        """

        def weights(diffusion, mobility):
            s = diffusion * self.dt / self.unit_length_cm**2
            c = mobility * self.Efield_V_cm * self.dt / self.unit_length_cm
            return (s, s + c * (c + 1.0) / 2.0, s + c * (c - 1.0) / 2.0, 1.0 - c * c - 6.0 * s)

        return weights(self.D_positive, self.mu_positive), weights(self.D_negative, self.mu_negative)

    def summary(self) -> str:
        rf_line = ""
        if self.rf_frequency_hz:
            rf_period_s = 1.0 / self.rf_frequency_hz
            rf_line = (
                f"\nRF microstructure     : {self.rf_frequency_hz / 1e6:.4g} MHz "
                f"(period {rf_period_s * 1e9:.3g} ns) -> {self.rf_cycles_per_time_step:.3g} "
                "RF cycles per time step (averaged over, not individually resolved)"
            )
        if self.two_carrier_species:
            carrier_line = (
                f"Carriers              : two species -- "
                f"mu+ = {self.mu_positive:.3g}, mu- = {self.mu_negative:.3g} cm^2/(V s); "
                f"D+ = {self.D_positive:.3g}, D- = {self.D_negative:.3g} cm^2/s\n"
            )
        else:
            carrier_line = (
                f"Carriers              : single averaged species -- "
                f"mu = {self.mu_positive:.3g} cm^2/(V s), D = {self.D_positive:.3g} cm^2/s\n"
            )
        fill_note = (
            ""
            if self.chamber_fill_fraction == 1.0
            else (
                f", tracks confined to {self.chamber_fill_fraction:.3g} x radius "
                f"({self.sampling_radius * self.unit_length_cm * 1e4:.3g} um -> "
                f"{1.0 / self.chamber_fill_fraction**2:.3g}x nominal areal density)"
            )
        )
        if self.track_cutoff_sigmas is None:
            cutoff_note = "no truncation (whole grid per track)"
        else:
            cutoff_note = (
                f"{self.track_cutoff_sigmas:.3g} sigma = "
                f"{self.track_cutoff_cm * 1e4:.3g} um = {self.track_cutoff_voxels:.1f} voxels "
                f"(discards {self.truncated_charge_fraction:.1e} of each track)"
            )
        return (
            f"Particle              : {self.particle} @ {self.E_MeV_u:.1f} MeV/u "
            f"(LET = {self.LET_keV_um:.3g} keV/um, track radius b = {self.track_radius_cm * 1e4:.3g} um)\n"
            f"Chamber               : gap = {self.electrode_gap_cm} cm, V = {self.voltage_V} V "
            f"(E = {self.Efield_V_cm:.4g} V/cm)\n"
            f"{carrier_line}"
            f"Gas / W               : rho_air = {self.air_density_kg_m3:.4g} kg/m^3, W = {self.W_eV:.4g} eV/ion pair\n"
            f"Sampled sub-volume    : radius = {self.sampled_radius_cm * 1e4:.3g} um, "
            f"area = {self.area_cm2:.3g} cm^2{fill_note}\n"
            f"Lateral boundary      : {self.lateral_boundary}, scoring over {self.scoring_region}\n"
            f"Deposition stencil    : {cutoff_note}\n"
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
