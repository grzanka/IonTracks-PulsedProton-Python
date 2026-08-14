"""Host resource discovery and guards.

Two failure modes this module exists to prevent.

**Memory.** The carrier arrays scale as `no_xy**2 * no_z_with_buffer`, so a
column a few millimetres across is gigabytes. Allocating more than the machine
has does not fail cleanly: on Linux it either drives the box into swap for
minutes at a time or gets the process killed by the OOM killer, in both cases
after the run has already started and with no useful diagnostic. Checking the
size up front turns that into an immediate, actionable error.

**Cores.** `numba.set_num_threads()` raises if asked for more threads than
Numba was configured with, and is silently useless if the process is
cpuset-restricted to fewer CPUs than the machine has (a common Slurm
situation -- see docs/PERFORMANCE.md). Clamping to what the process can
actually run on keeps a thread-count request from either crashing or lying.

Nothing here is a hard dependency: if a platform does not expose a figure, the
corresponding check is skipped rather than guessed.
"""

import os
import warnings
from typing import Optional

__all__ = [
    "available_cores",
    "available_memory_bytes",
    "check_memory_budget",
    "clamp_thread_count",
    "format_bytes",
    "total_memory_bytes",
]


def format_bytes(n: float) -> str:
    """Human-readable byte count, e.g. '1.80 GiB'."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024.0
    return f"{n:.2f} TiB"  # unreachable, keeps type checkers happy


def total_memory_bytes() -> Optional[int]:
    """Total physical RAM, or None if the platform does not report it."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def available_memory_bytes() -> Optional[int]:
    """RAM available to a new allocation without swapping, or None if unknown.

    Prefers Linux's ``MemAvailable``, which is the kernel's own estimate and
    correctly counts reclaimable page cache as available -- ``MemFree`` alone
    badly understates it on any machine that has been up for a while. Falls
    back to ``SC_AVPHYS_PAGES``, which does not count reclaimable cache and so
    is conservative.
    """
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def available_cores() -> int:
    """Number of CPUs this process may actually run on.

    ``os.sched_getaffinity`` reflects cpuset/taskset restrictions, which
    ``os.cpu_count()`` does not: under a Slurm allocation the two routinely
    disagree, and it is the affinity mask that decides whether extra threads
    have anywhere to run.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def check_memory_budget(
    required_bytes: float,
    budget_fraction: float = 0.8,
    what: str = "This simulation",
    hint: str = "",
) -> None:
    """Raise MemoryError if `required_bytes` exceeds a fraction of free RAM.

    Skipped silently when the platform does not report available memory. The
    default budget leaves a fifth of free RAM for everything else -- NumPy
    temporaries, the interpreter, and whatever else the user is running.
    """
    if budget_fraction is None:
        return
    if not 0 < budget_fraction <= 1:
        raise ValueError(f"budget_fraction must be in (0, 1], got {budget_fraction!r}.")

    available = available_memory_bytes()
    if available is None:
        return

    budget = available * budget_fraction
    if required_bytes <= budget:
        return

    total = total_memory_bytes()
    total_note = f" of {format_bytes(total)} total" if total else ""
    message = (
        f"{what} needs about {format_bytes(required_bytes)}, but only "
        f"{format_bytes(available)}{total_note} is available and the budget is "
        f"{budget_fraction:.0%} of that ({format_bytes(budget)})."
    )
    if hint:
        message += f" {hint}"
    message += (
        " Raise memory_budget_fraction (or set it to None to disable this check)"
        " if you are sure -- the usual outcome of getting this wrong is the OOM"
        " killer, not a slow run."
    )
    raise MemoryError(message)


def clamp_thread_count(requested: int) -> int:
    """Reduce a thread request to what the process can actually use.

    Bounded by the CPU affinity mask and by Numba's own configured maximum;
    warns whenever it has to reduce, so a benchmark cannot silently report a
    thread count that never existed.
    """
    if requested < 1:
        raise ValueError(f"num_threads must be at least 1, got {requested!r}.")

    limit = available_cores()
    reason = "the process CPU affinity mask"
    try:
        import numba

        if numba.config.NUMBA_NUM_THREADS < limit:
            limit = numba.config.NUMBA_NUM_THREADS
            reason = "NUMBA_NUM_THREADS"
    except (ImportError, AttributeError):  # pragma: no cover - numba is a hard dep
        pass

    if requested > limit:
        warnings.warn(
            f"num_threads={requested} exceeds {reason} ({limit}); using {limit} instead. "
            "Under Slurm this usually means the step was launched without "
            "--cpu-bind=none -- see docs/PERFORMANCE.md.",
            stacklevel=3,
        )
        return limit
    return requested
