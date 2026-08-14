#!/usr/bin/env python3
"""Detect a hybrid CPU's core types, and hand out CPU lists to pin threads to.

Why a benchmark on a laptop needs this
--------------------------------------
On a server every core is the same core, so "4 threads" is an unambiguous
description of a run. On an Intel hybrid part it is not. A Core Ultra 5 225U
has three kinds of core:

    P    performance cores, highest clock, two SMT threads each
    E    efficiency cores, lower clock, no SMT, in a cluster sharing L2
    LP-E low-power efficiency cores on the SoC tile, lowest clock

"4 threads" could mean two P-cores plus two E-cores, or four E-cores, and those
differ by roughly a factor of two. Which one you get is up to the kernel's
scheduler and the machine's power state, so an unpinned laptop benchmark is not
reproducible even against itself. This module finds the groups so the benchmark
can pin to them deliberately and *report* which it used.

Detection, in order of preference
---------------------------------
1. ``/sys/devices/system/cpu/types/*/cpulist`` -- the kernel's own hybrid
   classification (``intel_core_*`` vs ``intel_atom_*``). Authoritative when
   present.
2. Maximum frequency clustering from ``lscpu -p``. Works on older kernels: the
   three core types have distinct ``cpuinfo_max_freq`` values.

E and LP-E both report as ``intel_atom``, so they are separated by max
frequency within that class -- the LP-E cluster clocks lower.

SMT siblings come from ``topology/thread_siblings_list``. The benchmark uses one
logical CPU per *physical* core by default: on a memory-bound kernel the second
SMT thread of a core adds contention rather than throughput, and mixing "cores"
and "threads" in one scaling curve is how a benchmark ends up reporting a
mysterious plateau at the P-core count.

Usage
-----
    python -m profiling.laptop_scaling.topology                 # human summary
    python -m profiling.laptop_scaling.topology --json
    python -m profiling.laptop_scaling.topology --cpulist perf 4
    python -m profiling.laptop_scaling.topology --cpulist econ 8
    python -m profiling.laptop_scaling.topology --cpulist smt 2
"""

import argparse
import glob
import json
import os
import subprocess
import sys


def _read(path: str) -> str | None:
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


def _parse_cpulist(text: str) -> list[int]:
    """Expand a Linux cpulist ('0-3,8,10-11') into integers."""
    cpus: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _online_cpus() -> list[int]:
    text = _read("/sys/devices/system/cpu/online")
    return _parse_cpulist(text) if text else sorted(os.sched_getaffinity(0))


def _max_freq_khz(cpu: int) -> int | None:
    for name in ("cpuinfo_max_freq", "scaling_max_freq"):
        value = _read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/{name}")
        if value:
            return int(value)
    return None


def _siblings(cpu: int) -> list[int]:
    for name in ("thread_siblings_list", "core_cpus_list"):
        text = _read(f"/sys/devices/system/cpu/cpu{cpu}/topology/{name}")
        if text:
            return _parse_cpulist(text)
    return [cpu]


def _kernel_core_types() -> dict[int, str]:
    """CPU -> 'intel_core' / 'intel_atom', from the kernel's hybrid listing."""
    types: dict[int, str] = {}
    for path in glob.glob("/sys/devices/system/cpu/types/*/cpulist"):
        # .../types/intel_atom_0/cpulist -> 'intel_atom'
        name = os.path.basename(os.path.dirname(path))
        family = name.rsplit("_", 1)[0] if name[-1].isdigit() else name
        text = _read(path)
        if text:
            for cpu in _parse_cpulist(text):
                types[cpu] = family
    return types


def _lscpu_max_mhz() -> dict[int, float]:
    """Fallback classification input: CPU -> max MHz, via lscpu."""
    try:
        out = subprocess.run(
            ["lscpu", "-p=CPU,MAXMHZ"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    result: dict[int, float] = {}
    for line in out.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cpu_str, _, mhz_str = line.partition(",")
        try:
            result[int(cpu_str)] = float(mhz_str.replace(",", "."))
        except ValueError:
            continue
    return result


def detect() -> dict:
    """Classify every online CPU into P / E / LP-E groups.

    Returns a dict with, per group, the physical cores (one representative
    logical CPU each) and the full logical CPU list.
    """
    cpus = _online_cpus()
    kernel_types = _kernel_core_types()
    lscpu_mhz = _lscpu_max_mhz()

    freq = {}
    for cpu in cpus:
        khz = _max_freq_khz(cpu)
        freq[cpu] = khz / 1000.0 if khz else lscpu_mhz.get(cpu)

    # One entry per physical core, keyed by its lowest-numbered logical CPU.
    cores: dict[int, dict] = {}
    for cpu in cpus:
        siblings = [s for s in _siblings(cpu) if s in cpus] or [cpu]
        key = min(siblings)
        cores.setdefault(key, {"cpus": sorted(siblings), "mhz": freq.get(cpu)})

    # Is this machine hybrid at all? Three independent signals, any of which is
    # conclusive. Asking first matters: without it a uniform server CPU (no SMT,
    # one frequency, no kernel type files) would fall through to the "no SMT
    # sibling => efficiency core" heuristic and be reported as 192 E-cores.
    families = set(kernel_types.values())
    freqs = {f for f in freq.values() if f}
    sibling_counts = {len(entry["cpus"]) for entry in cores.values()}
    hybrid = (
        {"intel_core", "intel_atom"} <= families
        or (len(freqs) > 1 and max(freqs) - min(freqs) > 100)
        or len(sibling_counts) > 1
    )

    # Class: the kernel's word if we have it, else "P if it has an SMT sibling",
    # which is true of every Intel hybrid part to date.
    def core_class(entry: dict) -> str:
        if not hybrid:
            return "CPU"
        primary = entry["cpus"][0]
        family = kernel_types.get(primary)
        if family == "intel_core":
            return "P"
        if family == "intel_atom":
            return "E"
        return "P" if len(entry["cpus"]) > 1 else "E"

    for entry in cores.values():
        entry["class"] = core_class(entry)

    # Split E from LP-E by max frequency: they are distinct clusters, and the
    # low-power one on the SoC tile always clocks lower. Only meaningful if the
    # frequencies actually differ, so a non-hybrid Atom part stays one group.
    e_freqs = sorted({e["mhz"] for e in cores.values() if e["class"] == "E" and e["mhz"]})
    if len(e_freqs) > 1 and (e_freqs[-1] - e_freqs[0]) > 100:
        for entry in cores.values():
            if entry["class"] == "E" and entry["mhz"] == e_freqs[0]:
                entry["class"] = "LP-E"

    groups: dict[str, list[dict]] = {"P": [], "E": [], "LP-E": [], "CPU": []}
    for key in sorted(cores):
        groups[cores[key]["class"]].append({"primary": key, **cores[key]})

    return {
        "hybrid": hybrid,
        "n_logical": len(cpus),
        "n_physical": len(cores),
        "groups": {
            name: {
                "n_cores": len(entries),
                "physical_cpus": [e["primary"] for e in entries],
                "logical_cpus": sorted(c for e in entries for c in e["cpus"]),
                "max_mhz": entries[0]["mhz"] if entries else None,
            }
            for name, entries in groups.items()
            if entries
        },
    }


def cpulist(topology: dict, ladder: str, n_threads: int) -> list[int]:
    """CPUs to pin ``n_threads`` threads to, for a named ladder.

    ``perf`` -- fastest cores first: every P core, then E, then LP-E, one
    thread per physical core. This is the ladder that answers "how fast can
    this laptop run it on N cores".

    ``econ`` -- efficiency cores only. Same thread counts on the small cores,
    which is what tells you how much of the P-core advantage is clock and how
    much is memory bandwidth the whole package shares anyway.

    ``smt`` -- both SMT threads of as few P cores as possible. A deliberate
    worst case: on a bandwidth-bound kernel two siblings of one core should be
    worth barely more than one.
    """
    groups = topology["groups"]

    def physical(name: str) -> list[int]:
        return groups.get(name, {}).get("physical_cpus", [])

    if ladder == "perf":
        # "CPU" is the uniform-machine group; on such a machine every ladder
        # that makes sense is the same ladder, and econ/smt simply fall through
        # to whatever is there.
        order = physical("P") + physical("E") + physical("LP-E") + physical("CPU")
    elif ladder == "econ":
        order = physical("E") + physical("LP-E") or physical("CPU")
    elif ladder == "smt":
        order = groups.get("P", {}).get("logical_cpus") or [
            c for e in groups.get("CPU", {}).get("logical_cpus", []) for c in [e]
        ]
        # Only meaningful where a core has more than one thread.
        if len(order) <= len(physical("P") + physical("CPU")):
            order = []
    else:
        raise ValueError(f"unknown ladder {ladder!r}")

    if n_threads > len(order):
        raise SystemExit(
            f"ladder {ladder!r} has only {len(order)} CPUs, {n_threads} requested"
        )
    return order[:n_threads]


def describe(topology: dict, cpus: list[int]) -> str:
    """'2P + 2E' -- what a pinned CPU list actually consists of."""
    counts: dict[str, int] = {}
    for name, group in topology["groups"].items():
        overlap = len(set(cpus) & set(group["logical_cpus"]))
        if overlap:
            counts[name] = overlap
    return " + ".join(f"{n}{name}" for name, n in counts.items()) or "?"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable dump")
    parser.add_argument(
        "--cpulist",
        nargs=2,
        metavar=("LADDER", "N"),
        help="print a comma-separated CPU list for taskset: perf|econ|smt N",
    )
    parser.add_argument("--describe", metavar="CPUS", help="describe a CPU list, e.g. 0,2,4,5")
    args = parser.parse_args()

    if sys.platform != "linux":
        print(f"this needs Linux sysfs (running on {sys.platform})", file=sys.stderr)
        return 2

    topology = detect()

    if args.cpulist:
        ladder, n = args.cpulist
        print(",".join(str(c) for c in cpulist(topology, ladder, int(n))))
        return 0

    if args.describe:
        print(describe(topology, _parse_cpulist(args.describe)))
        return 0

    if args.json:
        print(json.dumps(topology, indent=2))
        return 0

    print(f"{topology['n_physical']} physical cores, {topology['n_logical']} logical"
          f"{'  (hybrid)' if topology['hybrid'] else ''}")
    for name, group in topology["groups"].items():
        mhz = f"{group['max_mhz']:.0f} MHz" if group["max_mhz"] else "max MHz unknown"
        print(f"  {name:<5} {group['n_cores']:>2} cores  {mhz:>14}   "
              f"logical CPUs {group['logical_cpus']}")
    print()
    for ladder in ("perf", "econ", "smt"):
        try:
            available = len(cpulist(topology, ladder, 1)) and cpulist(
                topology, ladder, min(8, topology["n_physical"])
            )
        except SystemExit:
            available = None
        print(f"  ladder {ladder:<5} -> {available if available else 'not available'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
