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
    "memory_report",
    "total_memory_bytes",
]


def format_bytes(n: float) -> str:
    """Human-readable byte count, e.g. '1.80 GiB'."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024.0
    return f"{n:.2f} TiB"  # unreachable, keeps type checkers happy


def _cgroup_limit_bytes() -> Optional[int]:
    """The cgroup memory ceiling this process actually runs under, or None.

    ``/proc/meminfo`` describes the *machine*, not the container, and under a
    batch scheduler those are wildly different numbers: a Slurm job on a Helios
    GH200 node that asked for ``--mem=12G`` sees 858 GiB of node RAM in
    ``MemAvailable`` and is killed at 12 GiB by the cgroup. Every guard in this
    module is there to turn "OOM-killed twenty minutes in" into "MemoryError
    before allocating", and on a scheduled node the cgroup limit is the number
    that decides which of the two happens.

    Reads cgroup v2 (``memory.max``, the unified hierarchy Slurm uses on
    EL9) and falls back to v1 (``memory.limit_in_bytes``). Returns None when
    there is no cgroup, no limit (``max``), or the limit is the "unlimited"
    sentinel some v1 kernels report.
    """
    try:
        with open("/proc/self/cgroup") as handle:
            relative = None
            for line in handle:
                fields = line.rstrip("\n").split(":", 2)
                # v2 lines are "0::/path"; v1 memory lines are "N:memory:/path".
                if len(fields) == 3 and (fields[0] == "0" or "memory" in fields[1].split(",")):
                    relative = fields[2].lstrip("/")
                    break
    except OSError:
        return None
    if relative is None:
        return None

    candidates = [
        os.path.join("/sys/fs/cgroup", relative, "memory.max"),
        os.path.join("/sys/fs/cgroup/memory", relative, "memory.limit_in_bytes"),
    ]
    # Walk *up* the hierarchy too: Slurm sets the ceiling on the job scope and
    # runs the task in a leaf below it, where memory.max is often "max".
    limits = []
    for path in candidates:
        node = os.path.dirname(path)
        name = os.path.basename(path)
        while node.startswith("/sys/fs/cgroup"):
            try:
                with open(os.path.join(node, name)) as handle:
                    raw = handle.read().strip()
            except OSError:
                raw = ""
            if raw and raw != "max":
                try:
                    value = int(raw)
                except ValueError:
                    value = 0
                # v1 reports ~2^63 for "no limit"; anything at or above the
                # machine's RAM is not a real constraint either.
                if 0 < value < 2**62:
                    limits.append(value)
            node = os.path.dirname(node)
    return min(limits) if limits else None


def total_memory_bytes() -> Optional[int]:
    """Total RAM this process may use, or None if the platform does not report it.

    The cgroup limit wins over the machine's physical RAM when it is smaller --
    see :func:`_cgroup_limit_bytes`.
    """
    try:
        physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        physical = None
    limit = _cgroup_limit_bytes()
    if physical is None:
        return limit
    if limit is None:
        return physical
    return min(physical, limit)


def available_memory_bytes() -> Optional[int]:
    """RAM available to a new allocation without swapping, or None if unknown.

    Prefers Linux's ``MemAvailable``, which is the kernel's own estimate and
    correctly counts reclaimable page cache as available -- ``MemFree`` alone
    badly understates it on any machine that has been up for a while. Falls
    back to ``SC_AVPHYS_PAGES``, which does not count reclaimable cache and so
    is conservative.

    Clamped by whatever the cgroup still allows (limit minus current usage), so
    that under Slurm this reports the job's headroom rather than the node's.
    """
    node_available = None
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    node_available = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        pass
    if node_available is None:
        try:
            node_available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            node_available = None

    limit = _cgroup_limit_bytes()
    if limit is None:
        return node_available
    headroom = max(0, limit - _cgroup_usage_bytes())
    return headroom if node_available is None else min(node_available, headroom)


def _cgroup_usage_bytes() -> int:
    """Bytes currently charged to this process's cgroup, 0 if unreadable."""
    try:
        with open("/proc/self/cgroup") as handle:
            relative = None
            for line in handle:
                fields = line.rstrip("\n").split(":", 2)
                if len(fields) == 3 and (fields[0] == "0" or "memory" in fields[1].split(",")):
                    relative = fields[2].lstrip("/")
                    break
    except OSError:
        return 0
    if relative is None:
        return 0
    for name, root in (("memory.current", "/sys/fs/cgroup"), ("memory.usage_in_bytes", "/sys/fs/cgroup/memory")):
        try:
            with open(os.path.join(root, relative, name)) as handle:
                return int(handle.read().strip())
        except (OSError, ValueError):
            continue
    return 0


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


def memory_report(
    required_bytes: float,
    budget_fraction: Optional[float] = 0.8,
) -> str:
    """Human-readable "would this fit?" comparison, for sizing a run before
    starting it (a memory dry run).

    Uses the same numbers and the same arithmetic as `check_memory_budget`, so
    a `--dry-run` flag built on this and the guard that actually runs at
    `SimulationConfig` construction time can never disagree. `required_bytes`
    is normally `config.estimated_memory_bytes`; `budget_fraction` is
    normally `config.memory_budget_fraction`.

    Peak RSS during a real run tends to land a little above
    `required_bytes` -- 12 % on the full-electrode grid (docs/BENCHMARKS-LAPTOP.md
    sec. 3) -- because this only counts the carrier arrays, the arrival-time
    draw and the 2D scratch, not the interpreter, Numba's runtime or NumPy's
    transient temporaries. The default `budget_fraction=0.8` leaves headroom
    for that margin as well as for everything else running on the machine.
    """
    total = total_memory_bytes()
    available = available_memory_bytes()

    lines = [f"Estimated peak allocation : {format_bytes(required_bytes)}"]
    lines.append(f"Total RAM on this machine : {format_bytes(total) if total is not None else 'unknown'}")

    if available is None:
        lines.append("Currently available RAM   : unknown on this platform -- budget check skipped")
        return "\n".join(lines)

    lines.append(f"Currently available RAM   : {format_bytes(available)}")

    if budget_fraction is None:
        lines.append("Budget check              : disabled (memory_budget_fraction=None)")
        return "\n".join(lines)

    budget = available * budget_fraction
    fits = required_bytes <= budget
    lines.append(f"Budget ({budget_fraction:.0%} of available)    : {format_bytes(budget)}")
    lines.append(
        "Fits within budget        : "
        + ("yes" if fits else "NO -- this run would raise MemoryError before starting")
    )
    return "\n".join(lines)


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
