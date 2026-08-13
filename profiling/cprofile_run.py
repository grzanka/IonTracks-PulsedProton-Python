"""cProfile a single run at a given thread count.

Raw wall-clock timing (thread_sweep.csv) hides *where* time goes, since the
two hot kernels are single opaque calls into compiled code from cProfile's
point of view. What cProfile *can* show is the Python-level driver
overhead -- track-schedule building, per-time-step Python loop, and the
per-call dispatch cost of invoking the njit kernels ~2,000-4,500 times over
the run -- which is exactly the "per-launch fixed cost" the README's fork-
join theory is about.
"""

import argparse
import cProfile
import pstats
from pathlib import Path

from profiling.common import run_once


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()
    metrics = run_once(args.threads)
    profiler.disable()

    pstats_path = args.out_prefix.with_suffix(".pstats")
    profiler.dump_stats(pstats_path)

    txt_path = args.out_prefix.with_suffix(".txt")
    with open(txt_path, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats("cumulative")
        stats.print_stats(40)

    print(f"threads={args.threads} wall_s={metrics['wall_s']:.2f} -> {pstats_path}, {txt_path}")


if __name__ == "__main__":
    main()
