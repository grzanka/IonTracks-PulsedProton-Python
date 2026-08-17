"""Tabulated radial dose distributions, and the voxel stencil built from one.

The Gaussian track model in :mod:`~pulsed_ion_chamber.stopping_power`
(`exp(-r^2/b^2)`, with `b` from the Rossomme fit) is a two-parameter stand-in
for a real track: it has the right total charge and roughly the right width for
protons and light ions, and nothing else. This module replaces it with a
measured or computed radial dose distribution -- e.g. libamtrack's Cucinotta
RDD -- read from a table of `(r, D(r))`.

Why that matters here rather than being a refinement: a real RDD is not a
bump, it is a `1/r^2` penumbra spanning many decades. Energy is then deposited
*uniformly per decade of radius*, so no simulated column contains all of it,
and the core is orders of magnitude denser than any Gaussian of the same total
charge. Both facts change `k_s`, and in opposite directions. See
`examples/fe90_air/README.md`.

Two design choices are worth stating up front.

**The stencil is area-averaged, not point-sampled.** Each voxel gets the
integral of the RDD over its own footprint, not `D()` evaluated at its centre.
On a `1/r^2` profile the two differ without limit as the grid coarsens, and
only the first conserves the LET. :func:`build_track_stencil` gets this by
construction: it distributes each radial bin's *energy* onto voxels, so
whatever the grid, the deposited total is the tabulated total minus exactly
what fell outside the array.

**The stencil is built for one fixed track position.** A tabulated RDD is not
separable and has no natural truncation radius, so neither trick that makes
Gaussian deposition cheap (see :mod:`~pulsed_ion_chamber.solver_numba`) is
available; the honest implementation is a full-grid array built once. That is
the right trade for a single deliberate ion on the axis, which is what this is
for, and the wrong one for a dose-rate-driven pulse of millions of tracks.
:class:`~pulsed_ion_chamber.config.SimulationConfig` enforces the distinction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

__all__ = [
    "RadialDoseDistribution",
    "build_track_stencil",
    "TrackStencil",
    "chamber_ks",
]

FloatArray = npt.NDArray[np.float64]

# 1 J expressed in keV, and the elementary conversions the integrals below need.
KEV_PER_JOULE = 6.241509074e15


@dataclass
class RadialDoseDistribution:
    """A tabulated `D(r)`, plus the cumulative energy integral over it.

    ``r_cm`` is strictly increasing; ``dose_Gy`` is the dose at that radius.
    ``density_g_cm3`` is the gas the dose was computed for -- the radial
    *energy* integral scales linearly with it, so it has to travel with the
    table rather than be re-supplied at use time.
    """

    r_cm: FloatArray
    dose_Gy: FloatArray
    density_g_cm3: float
    source: str = "<array>"

    def __post_init__(self):
        self.r_cm = np.asarray(self.r_cm, dtype=float)
        self.dose_Gy = np.asarray(self.dose_Gy, dtype=float)
        if self.r_cm.ndim != 1 or self.r_cm.shape != self.dose_Gy.shape:
            raise ValueError(
                f"r_cm and dose_Gy must be 1-D and the same length, got "
                f"{self.r_cm.shape} and {self.dose_Gy.shape}."
            )
        if len(self.r_cm) < 2:
            raise ValueError(f"a radial dose distribution needs at least 2 points, got {len(self.r_cm)}.")
        if np.any(np.diff(self.r_cm) <= 0):
            raise ValueError("r_cm must be strictly increasing.")
        if np.any(self.r_cm <= 0) or np.any(self.dose_Gy < 0):
            raise ValueError("r_cm must be positive and dose_Gy non-negative.")

        # Energy per unit track length inside radius r:  2*pi*rho * int_0^r D r' dr'.
        # In J/cm with rho in g/cm^3 -> kg/cm^3, then keV/cm.
        rho_kg_cm3 = self.density_g_cm3 * 1e-3
        integrand = 2.0 * np.pi * self.r_cm * self.dose_Gy * rho_kg_cm3  # J/cm per cm of radius
        shells = np.diff(self.r_cm) * 0.5 * (integrand[1:] + integrand[:-1])
        # Nothing is added for r < r_cm[0]: the table's own range is taken to
        # hold the whole LET.
        #
        # That is a measurement, not an assumption. Integrating the Cucinotta
        # Fe-90 table over its tabulated range alone gives 0.5546 keV/um
        # against 0.5611 keV/um from libamtrack's independent Bethe
        # stopping-power table -- 1.2 % low, which is the trapezoid rule on 143
        # points per decade. Extrapolating D flat inside r_min instead adds
        # another 36 %, overshooting the Bethe LET by a third; and the log-log
        # slope at r_min is already -3.5, so a flat core is not what the model
        # does there anyway. Whatever the model's true sub-nanometre behaviour,
        # the energy it carries is already accounted for by the normalisation.
        self.cumulative_keV_per_cm: FloatArray = (
            np.concatenate([[0.0], np.cumsum(shells)]) * KEV_PER_JOULE
        )
        self.LET_keV_cm: float = float(self.cumulative_keV_per_cm[-1])

    @classmethod
    def from_csv(cls, path, density_g_cm3: float, r_unit: str = "m") -> "RadialDoseDistribution":
        """Read a two-column `(radius, dose)` table.

        Comment lines starting with ``#`` are skipped, as is a quoted header
        row; that is the shape libamtrack writes. ``r_unit`` is ``"m"`` or
        ``"cm"`` -- libamtrack emits metres, this code works in centimetres,
        and getting that wrong is a factor of 100 in the track radius that
        nothing downstream would flag.
        """
        if r_unit not in ("m", "cm"):
            raise ValueError(f"r_unit must be 'm' or 'cm', got {r_unit!r}.")
        path = Path(path)
        rows = []
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace('"', "").split(",")
                if len(parts) < 2:
                    continue
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue  # the header row
        if not rows:
            raise ValueError(f"no numeric (radius, dose) rows found in {path}.")
        data = np.array(rows, dtype=float)
        scale = 100.0 if r_unit == "m" else 1.0
        return cls(
            r_cm=data[:, 0] * scale,
            dose_Gy=data[:, 1],
            density_g_cm3=density_g_cm3,
            source=str(path),
        )

    @property
    def LET_keV_um(self) -> float:
        """The LET the table integrates to. Worth checking against an
        independent stopping-power table -- agreement confirms the RDD is
        normalised and that `r_unit`/density were right."""
        return self.LET_keV_cm * 1e-4

    def energy_within_keV_per_cm(self, r_cm) -> FloatArray:
        """Energy per unit track length inside radius ``r_cm``, interpolated
        in log-log where the table is smooth and clamped outside it."""
        r = np.atleast_1d(np.asarray(r_cm, dtype=float))
        out = np.interp(r, self.r_cm, self.cumulative_keV_per_cm,
                        left=0.0, right=self.LET_keV_cm)
        return out if np.ndim(r_cm) else out[0]

    def fraction_within(self, r_cm) -> float:
        """Share of the track's energy inside radius ``r_cm``."""
        return float(self.energy_within_keV_per_cm(r_cm)) / self.LET_keV_cm


@dataclass
class TrackStencil:
    """One track's deposition, as carrier density per voxel column.

    ``density_cm3`` is what gets added to every gap layer of the corresponding
    voxel column, in cm^-3 -- the same quantity the Gaussian kernel computes
    per voxel, so the two deposition paths are interchangeable downstream.
    """

    density_cm3: FloatArray  # (no_xy, no_xy)
    deposited_keV_per_cm: float  # what landed on the grid
    total_keV_per_cm: float  # what the table holds, in and out of the grid
    centre_xy: tuple  # grid coordinates the stencil was built around

    @property
    def in_domain_fraction(self) -> float:
        """Share of the track's energy the grid actually contains. The rest is
        created in the chamber but never simulated -- see :func:`chamber_ks`."""
        return self.deposited_keV_per_cm / self.total_keV_per_cm


def build_track_stencil(
    rdd: RadialDoseDistribution,
    unit_length_cm: float,
    no_xy: int,
    centre_xy: tuple,
    W_eV: float,
    samples_per_voxel: int = 8,
    max_samples_per_bin: int = 1 << 16,
) -> TrackStencil:
    """Area-average ``rdd`` onto a ``no_xy x no_xy`` grid centred on ``centre_xy``.

    Method: walk the table's own radial bins. Each bin `[r_k, r_k+1]` carries a
    known energy per unit track length, `dE_k`, from the trapezoid rule on
    `2*pi*rho*D(r)*r`. That energy is spread over `M_k` points evenly placed
    around the circle of radius `sqrt(r_k * r_k+1)` and accumulated into
    whichever voxel each point lands in.

    This is exactly conservative by construction -- the deposited total is
    `sum(dE_k)` minus whatever fell off the grid -- which is the property that
    matters. A `1/r^2` profile point-sampled at voxel centres instead loses
    charge at a rate that depends on the spacing, so `k_s` would drift with `h`
    for a purely numerical reason on top of the physical one.

    `M_k` is chosen to put ``samples_per_voxel`` points on each voxel the ring
    crosses, so the far field is smooth; bins well inside one voxel collapse to
    a single sample, which is correct rather than approximate -- all of that
    energy really does belong to the centre voxel. A per-bin golden-angle
    rotation keeps successive rings from landing on the same spokes of the
    square grid.
    """
    if no_xy < 1:
        raise ValueError(f"no_xy must be positive, got {no_xy}.")
    if unit_length_cm <= 0:
        raise ValueError(f"unit_length_cm must be positive, got {unit_length_cm}.")
    if samples_per_voxel < 1:
        raise ValueError(f"samples_per_voxel must be at least 1, got {samples_per_voxel}.")
    if W_eV <= 0:
        raise ValueError(f"W_eV must be positive, got {W_eV}.")

    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    energy_keV_per_cm = np.zeros((no_xy, no_xy), dtype=float)

    r = rdd.r_cm
    cum = rdd.cumulative_keV_per_cm
    # Energy in each tabulated bin, plus the flat inner disc as bin "-1" which
    # always lands on the centre voxel.
    bin_energy = np.diff(cum)
    bin_radius_cm = np.sqrt(r[:-1] * r[1:])

    # Nothing beyond the far corner of the array can land on it.
    corner_cm = np.hypot(max(cx, no_xy - cx), max(cy, no_xy - cy)) * unit_length_cm
    keep = bin_radius_cm <= corner_cm
    bin_energy = bin_energy[keep]
    bin_radius_voxels = bin_radius_cm[keep] / unit_length_cm

    golden = np.pi * (3.0 - np.sqrt(5.0))  # rotate each ring, avoid spokes
    for k, (energy, radius_vox) in enumerate(zip(bin_energy, bin_radius_voxels)):
        if energy <= 0.0:
            continue
        n_samples = int(np.ceil(2.0 * np.pi * radius_vox * samples_per_voxel))
        n_samples = min(max(n_samples, 1), max_samples_per_bin)
        theta = (np.arange(n_samples) + 0.5) * (2.0 * np.pi / n_samples) + k * golden
        xs = np.rint(cx + radius_vox * np.cos(theta)).astype(np.int64)
        ys = np.rint(cy + radius_vox * np.sin(theta)).astype(np.int64)
        inside = (xs >= 0) & (xs < no_xy) & (ys >= 0) & (ys < no_xy)
        if not inside.any():
            continue
        flat = xs[inside] * no_xy + ys[inside]
        counts = np.bincount(flat, minlength=no_xy * no_xy)
        energy_keV_per_cm += counts.reshape(no_xy, no_xy) * (energy / n_samples)

    # keV/cm of track in a voxel column -> ion pairs per cm -> pairs per cm^3.
    voxel_area_cm2 = unit_length_cm**2
    density = energy_keV_per_cm * 1e3 / W_eV / voxel_area_cm2

    return TrackStencil(
        density_cm3=density,
        deposited_keV_per_cm=float(energy_keV_per_cm.sum()),
        total_keV_per_cm=rdd.LET_keV_cm,
        centre_xy=(cx, cy),
    )


def chamber_ks(ks_in_domain: float, in_domain_fraction: float) -> float:
    """Rescale a simulated `k_s` to the whole chamber.

    The simulated column holds only ``in_domain_fraction`` of the track's
    charge. The rest is created in the gas further out, at densities orders of
    magnitude below anything that recombines measurably, and is collected in
    full. It still belongs in the denominator of the collection efficiency, so

        f_chamber = f_domain * F + (1 - F)

    Leaving it out reports the loss as a fraction of the simulated charge
    instead of the created charge, which for a `1/r^2` penumbra truncated at a
    fraction of a millimetre is a 20-40 % overstatement of `k_s - 1`.
    """
    if not 0 < in_domain_fraction <= 1:
        raise ValueError(f"in_domain_fraction must be in (0, 1], got {in_domain_fraction!r}.")
    if ks_in_domain <= 0:
        raise ValueError(f"ks_in_domain must be positive, got {ks_in_domain!r}.")
    f_domain = 1.0 / ks_in_domain
    return 1.0 / (f_domain * in_domain_fraction + (1.0 - in_domain_fraction))
