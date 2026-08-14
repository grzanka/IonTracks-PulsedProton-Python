"""Run state that lives outside the JIT kernels.

Three things both backends need, kept here so they cannot drift apart:

* :class:`Result` -- everything a finished run hands back.
* :class:`Diagnostics` -- the per-time-step record, accumulated as the run
  proceeds.
* :func:`apply_lateral_boundary` -- the once-per-step boundary update.

None of it is on the hot path (the boundary update touches four planes out of
a whole grid), so it is plain NumPy and plain Python: readable beats fast here.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["Result", "Diagnostics", "apply_lateral_boundary"]


@dataclass
class Result:
    """Everything a finished run hands back.

    The time-series arrays all have one entry per time step. Densities are in
    cm^-3 and the summed quantities are *density sums over voxels*, not
    absolute counts -- the voxel volume cancels in the collection efficiency,
    so the solver never multiplies it back in. :mod:`pulsed_ion_chamber.output`
    does that conversion when writing physical numbers to disk.
    """

    config: object  # the SimulationConfig this run was built from
    time_s: np.ndarray  # end-of-step time, seconds
    f_t: np.ndarray  # collection efficiency so far: (injected - recombined) / injected
    ks: float  # recombination correction factor 1/f after full clearance
    positive_array: np.ndarray  # final density snapshot, shape (no_xy, no_xy, no_z_with_buffer)
    negative_array: np.ndarray

    # --- per-time-step diagnostics, all summed over the scored region ---
    n_positive: np.ndarray  # positive carriers present at the end of each step
    n_negative: np.ndarray  # negative carriers present
    injected_positive: np.ndarray  # carriers created during the step
    injected_negative: np.ndarray  # identical to injected_positive by construction
    recombination: np.ndarray  # carrier pairs lost during the step
    track_density_xy: np.ndarray  # (no_xy, no_xy) count of track centres per voxel column


class Diagnostics:
    """Accumulates the per-time-step record while a run proceeds.

    Deliberately *not* a jitted structure: the kernels return scalars and this
    stores them, which keeps the numba signatures simple and means the two
    backends record identically by construction rather than by convention.
    """

    def __init__(self, config):
        steps = config.total_time_steps
        self.injected = np.zeros(steps)
        self.recombined = np.zeros(steps)
        self.n_positive = np.zeros(steps)
        self.n_negative = np.zeros(steps)
        self.track_density_xy = np.zeros((config.no_xy, config.no_xy))

    def count_track(self, x: float, y: float) -> None:
        """Record one track centre. `int()` floors onto the voxel it landed in."""
        self.track_density_xy[int(x), int(y)] += 1.0

    def count_tracks(self, xs: np.ndarray, ys: np.ndarray) -> None:
        """Record a whole time step's track centres at once.

        `np.add.at` rather than fancy-index `+=` because several tracks in the
        same step routinely land in the same voxel, and plain fancy indexing
        would keep only the last of them instead of summing.
        """
        np.add.at(self.track_density_xy, (xs.astype(np.int64), ys.astype(np.int64)), 1.0)

    def record(self, step, injected, recombined, total_positive, total_negative) -> None:
        self.injected[step] = injected
        self.recombined[step] = recombined
        self.n_positive[step] = total_positive
        self.n_negative[step] = total_negative

    def build_result(self, config, time_s, f_t, ks, positive_array, negative_array) -> Result:
        return Result(
            config=config,
            time_s=time_s,
            f_t=f_t,
            ks=ks,
            positive_array=positive_array,
            negative_array=negative_array,
            n_positive=self.n_positive,
            n_negative=self.n_negative,
            # Every track liberates one positive and one negative carrier, so
            # the two injection columns are equal by construction. Both are
            # kept so the written record matches the column layout other
            # IonTracks codes emit.
            injected_positive=self.injected,
            injected_negative=self.injected.copy(),
            recombination=self.recombined,
            track_density_xy=self.track_density_xy,
        )


def apply_lateral_boundary(array: np.ndarray, mode: str) -> None:
    """Enforce the chamber-wall boundary condition in place, once per step.

    The Lax-Wendroff sweep only writes indices ``1 .. no_xy-2``, so the outer
    ring of the array is whatever was last put there. That is the hook this
    function uses.

    ``"absorbing"`` is a deliberate no-op: the ring keeps its value and charge
    that reaches it leaves the simulation. Note the side effect -- track
    deposition *does* write to the ring, so Gaussian tails accumulate there and
    are then frozen, never drifting, diffusing or recombining, while still
    feeding their inward neighbour every step. On a wide column that is
    harmless (the ring is far from anything scored); at ``buffer_radius=1`` it
    corrupts the answer by 9 %.

    ``"reflecting"`` mirrors each adjacent interior plane outwards, giving a
    zero-gradient (zero-flux) wall, and incidentally overwrites that stranded
    charge every step. It is the physically right condition for a column
    sampled from the *interior* of a large, uniformly irradiated chamber: the
    gas just outside the column is statistically identical to the gas inside,
    so it returns as much charge as it takes.

    The z ends are untouched in both modes. Charge that drifts past the
    electrode buffer has been collected, and should leave.
    """
    if mode != "reflecting":
        return
    # Order matters only at the four corner columns, which end up mirroring the
    # j-neighbour of the already-mirrored i-plane. They carry negligible
    # density and are outside every scored region, so either order is fine.
    array[0, :, :] = array[1, :, :]
    array[-1, :, :] = array[-2, :, :]
    array[:, 0, :] = array[:, 1, :]
    array[:, -1, :] = array[:, -2, :]
