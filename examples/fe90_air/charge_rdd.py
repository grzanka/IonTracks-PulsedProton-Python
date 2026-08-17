#!/usr/bin/env python3
"""Build a *charge* RDD from a *dose* RDD, and measure what changes.

The reproducible companion to docs/RDD-CHARGE-VS-DOSE.md. Every number quoted
there comes from running this.

The tabulated Cucinotta RDD is a dose distribution, and about a third of it is
an excitation core that creates no ion pair (Cucinotta et al. 1996 p.256, p.263;
verified against the tabulation in that document sec. 2). Converting it with
`n = rho D / W` therefore manufactures charge at the density peak. This rebuilds
the profile as the charge distribution it should have been:

  1. Delete the excitation core -- continue the verified `1/r^2` penumbra
     inward instead.
  2. Redistribute, do not discard, the delta-ray energy inside `r0` into a
     uniform disc: thermalisation smears the *charge*, it does not delete it.
  3. Convert with `W_delta = W * (delta-ray LET / total LET)`, so the total pair
     count is unchanged and only the radial distribution moves.

`r0` is the physical charge-cloud radius in the gas -- sub-excitation electron
thermalisation, micrometres in air at 1 atm, NOT the nm-scale adiabatic impact
parameter. It is the one number here not taken from the paper or the
tabulation, and `--scan-r0` is how much it matters.

Run:  python examples/fe90_air/charge_rdd.py [--ladder] [--scan-r0] [--r0-um R]
                                             [--threads N]
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_fe90 import TWO_SPECIES  # noqa: E402

from pulsed_ion_chamber.config import SimulationConfig  # noqa: E402
from pulsed_ion_chamber.rdd import KEV_PER_JOULE, RadialDoseDistribution, chamber_ks  # noqa: E402
from pulsed_ion_chamber.solver_numba import run_simulation_numba  # noqa: E402
from pulsed_ion_chamber.solver_numba_parallel import run_simulation_numba_parallel  # noqa: E402

HERE = Path(__file__).parent
DOSE_CSV = HERE / "data" / "rdd_cucinotta_fe90_air.csv"
W_EV = 34.2
RHO_G_CM3 = 1.225e-3
# Where the tabulation is verified to be a pure 1/r^2 penumbra (doc sec. 2).
PENUMBRA_LO_CM, PENUMBRA_HI_CM = 1e-5, 1e-2


def build_charge_rdd(r0_um: float, dose_csv=DOSE_CSV):
    """Return ``(charge_rdd, W_delta, dose_rdd)``."""
    dose = RadialDoseDistribution.from_csv(dose_csv, density_g_cm3=RHO_G_CM3)
    r = dose.r_cm.copy()

    # The 1/r^2 plateau constant K, from r^2 D over the verified penumbra.
    band = (r > PENUMBRA_LO_CM) & (r < PENUMBRA_HI_CM)
    K = float(np.median((r**2 * dose.dose_Gy)[band]))

    # 1. delete the excitation core: continue the penumbra inward.
    D = np.where(r < PENUMBRA_LO_CM, K / r**2, dose.dose_Gy)
    delta_only = RadialDoseDistribution(r_cm=r, dose_Gy=D, density_g_cm3=RHO_G_CM3)

    # 2. flatten inside r0, conserving the energy that was there.
    #    E [keV/cm] = rho [kg/cm^3] * D [Gy] * pi r0^2 [cm^2] * keV_per_J
    r0 = r0_um * 1e-4
    energy_inside = float(delta_only.energy_within_keV_per_cm(r0))
    d_flat = energy_inside / (KEV_PER_JOULE * np.pi * r0**2 * RHO_G_CM3 * 1e-3)
    D = np.where(r < r0, d_flat, D)

    charge = RadialDoseDistribution(
        r_cm=r, dose_Gy=D, density_g_cm3=RHO_G_CM3,
        source=f"charge-RDD from {Path(dose_csv).name}, r0 = {r0_um} um",
    )
    # 3. same pair count, corrected distribution.
    W_delta = W_EV * charge.LET_keV_cm / dose.LET_keV_cm
    return charge, W_delta, dose


def _write_table(rdd, path):
    """SimulationConfig takes a CSV path, so materialise the rebuilt profile."""
    with open(path, "w") as handle:
        handle.write('"r_m_x","r_m_y"\n')
        for r_cm, dose_Gy in zip(rdd.r_cm, rdd.dose_Gy):
            handle.write(f"{r_cm * 1e-2},{dose_Gy}\n")
    return path


def run(r0_um, h_um, threads, tmpdir):
    charge, W_delta, _ = build_charge_rdd(r0_um)
    path = _write_table(charge, Path(tmpdir) / f"charge_rdd_r0_{r0_um}.csv")
    config = SimulationConfig(
        E_MeV_u=90.0, particle="iron", voltage_V=200.0, electrode_gap_cm=0.2,
        pulse_duration_s=1e-7, repetition_rate_hz=50.0, dose_rate_Gy_s=1e-12,
        n_pulses=1, n_tracks=1, track_placement="axis",
        rdd_csv=str(path), W_eV=W_delta,
        grid_size_um=h_um, sampled_radius_cm=0.012,
        buffer_radius=max(2, int(round(20.0 / h_um))), no_z_electrode=3,
        lateral_boundary="absorbing", scoring_region="full_grid",
        max_voxels=1e10, memory_budget_fraction=None, seed=1, **TWO_SPECIES,
    )
    started = perf_counter()
    if threads > 1:
        result = run_simulation_numba_parallel(config, progress=False, num_threads=threads)
    else:
        result = run_simulation_numba(config, progress=False)
    return config, result, chamber_ks(result.ks, config.in_domain_let_fraction), perf_counter() - started


def main(args):
    import tempfile

    charge, W_delta, dose = build_charge_rdd(args.r0_um)
    share = 100 * charge.LET_keV_cm / dose.LET_keV_cm
    print(f"dose-RDD   : {dose.LET_keV_um:.4f} keV/um")
    print(f"charge-RDD : {charge.LET_keV_um:.4f} keV/um  ({share:.1f} % of it)")
    print(f"W_delta    : {W_delta:.2f} eV   (W = {W_EV})")
    print("             Cucinotta p.263 puts the delta-ray share at 55-70 %; the")
    print("             excitation core removed here carries 33.3 % of the LET.")
    print(f"pairs/cm   : {dose.LET_keV_cm * 1e3 / W_EV:.4g} either way, by construction\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.scan_r0:
            print(f"r0 sensitivity at h = {args.h_um} um:")
            print(f"{'r0[um]':>7} {'n0[cm^-3]':>11} {'ks(domain)':>11} {'ks(chamber)':>12} {'wall':>7}")
            for r0_um in (1.0, 2.0, 5.0, 10.0, 20.0):
                c, res, ksc, el = run(r0_um, args.h_um, args.threads, tmpdir)
                print(f"{r0_um:>7.1f} {c.track_stencil.density_cm3.max():>11.3e} "
                      f"{res.ks:>11.6f} {ksc:>12.6f} {el:>6.1f}s")
            print("\nr0 sets the value; h does not. That is the point -- the uncertainty")
            print("now lives in a measurable property of air, not in the discretisation.")

        if args.ladder:
            rungs = [10.0, 5.0, 2.5] + ([1.25] if args.fine else [])
            print(f"\nconvergence ladder at r0 = {args.r0_um} um:")
            print(f"{'h[um]':>6} {'n0[cm^-3]':>11} {'ks(domain)':>11} {'ks(chamber)':>12} {'wall':>8}")
            values = []
            for h_um in rungs:
                c, res, ksc, el = run(args.r0_um, h_um, args.threads, tmpdir)
                values.append(ksc)
                note = "  <- h > r0, core unresolved" if h_um > args.r0_um else ""
                print(f"{h_um:>6} {c.track_stencil.density_cm3.max():>11.3e} "
                      f"{res.ks:>11.6f} {ksc:>12.6f} {el:>7.1f}s{note}")
            print(f"increments {np.round(np.diff(values), 6)}")
            print("the dose-RDD ladder was +0.0645, +0.0513, +0.0432 and still climbing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--r0-um", type=float, default=5.0, help="charge-cloud radius (default 5)")
    parser.add_argument("--h-um", type=float, default=2.5, help="voxel size for --scan-r0 (default 2.5)")
    parser.add_argument("--ladder", action="store_true", help="run the h = 10, 5, 2.5 um ladder")
    parser.add_argument("--fine", action="store_true", help="add the 1.25 um rung (~13 min on 2 cores)")
    parser.add_argument("--scan-r0", action="store_true", help="vary r0 at fixed h")
    parser.add_argument("--threads", type=int, default=2)
    main(parser.parse_args())
