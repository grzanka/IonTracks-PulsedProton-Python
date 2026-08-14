"""Diagnostic plots for a completed run.

Four views, each answering a different question:

* injection rate      -- did the beam arrive when the pulse says it should?
* carrier evolution   -- do the clouds fill, then clear, within the run?
* recombination rate  -- when is charge actually being lost?
* track cross-section -- is the beam spot the shape and density intended?

Axes are auto-scaled to a readable unit (µs, and a power-of-ten multiple of
ion pairs) rather than left in raw SI, because these are read by eye.
"""

from pathlib import Path
from typing import Union

import numpy as np

from pulsed_ion_chamber.output import collected_charge_table, track_density_per_cm2

__all__ = ["save_diagnostic_plots"]

_SI_PREFIXES = ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""))


def _scale(values) -> tuple:
    """Return (divisor, unit_prefix) putting the peak of `values` in [1, 1000).

    The prefix already carries its trailing space when non-empty, so callers
    can interpolate it directly without leaving a gap at 1x scale.
    """
    peak = float(np.max(np.abs(values))) if len(values) else 0.0
    for divisor, prefix in _SI_PREFIXES:
        if peak >= divisor:
            return divisor, (prefix + " " if prefix else "")
    return 1.0, ""


def save_diagnostic_plots(result, directory: Union[str, Path], title: str = "") -> list:
    """Write the four diagnostic figures; return the paths written."""
    import matplotlib

    matplotlib.use("Agg")  # no display needed, and none may exist on a cluster
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = result.config
    table = collected_charge_table(result)
    t_us = table["time"] * 1e6
    dt_us = config.dt * 1e6
    pulse_end_us = config.pulse_duration_s * 1e6
    suffix = f" — {title}" if title else ""
    written = []

    def finish(fig, ax, name, mark_pulse=True):
        if mark_pulse:
            ax.axvline(pulse_end_us, color="0.4", ls="--", lw=1, label="pulse ends")
            ax.legend(frameon=False)
        ax.set_xlabel("time [µs]")
        ax.margins(x=0.01)
        fig.tight_layout()
        path = directory / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # --- 1. injection rate -------------------------------------------------
    # Per-step counts divided by dt gives a rate; expressed per microsecond so
    # the number matches the time axis rather than being 1e6 times larger.
    rate_per_us = table["injected_positive"] / dt_us
    div, prefix = _scale(rate_per_us)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_us, rate_per_us / div, lw=0.8, color="tab:blue")
    ax.set_ylabel(f"injection rate [{prefix}ion pairs / µs]")
    ax.set_title(f"Injection rate{suffix}")
    finish(fig, ax, "injection_rate.png")

    # --- 2. carrier evolution ---------------------------------------------
    div, prefix = _scale(table["n_positive"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_us, table["n_positive"] / div, lw=1.2, label="positive")
    ax.plot(t_us, table["n_negative"] / div, lw=1.2, ls="--", label="negative")
    ax.set_ylabel(f"carriers present [{prefix}ion pairs]")
    ax.set_title(f"Charge-carrier evolution{suffix}")
    finish(fig, ax, "carrier_evolution.png")

    # --- 3. recombination rate --------------------------------------------
    div, prefix = _scale(table["recombination"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_us, table["recombination"] / div, lw=0.8, color="tab:red")
    ax.set_ylabel(f"recombination [{prefix}ion pairs / step]")
    ax.set_title(f"Recombination rate ({config.dt * 1e9:.1f} ns per step){suffix}")
    finish(fig, ax, "recombination_rate.png")

    # --- 4. track areal density cross-section ------------------------------
    density = track_density_per_cm2(result)
    div, prefix = _scale(density.ravel())
    half_um = 0.5 * config.no_xy * config.unit_length_cm * 1e4
    extent = (-half_um, half_um, -half_um, half_um)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), layout="constrained")
    image = ax.imshow(density.T / div, origin="lower", extent=extent, cmap="magma")
    circles = [(config.inner_radius, ":", "scored radius")]
    if config.chamber_fill_fraction != 1.0:
        # Otherwise the two circles coincide and the legend implies a
        # distinction the run does not have.
        circles.insert(0, (config.sampling_radius, "-", "sampling radius"))
    from matplotlib.lines import Line2D

    handles = []
    for radius_voxels, style, label in circles:
        radius_um = radius_voxels * config.unit_length_cm * 1e4
        ax.add_patch(plt.Circle((0, 0), radius_um, fill=False, ls=style, lw=1.4, color="cyan"))
        # Proxy handles: a Circle patch would render in the legend as a filled
        # box, which says nothing about which line style is which.
        handles.append(Line2D([], [], color="cyan", ls=style, lw=1.4, label=label))
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title(f"Track areal density{suffix}", fontsize=11)
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8, labelcolor="white")
    fig.colorbar(image, ax=ax, label=f"tracks [{prefix}cm⁻²]")
    path = directory / "track_density_cross_section.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    return written
