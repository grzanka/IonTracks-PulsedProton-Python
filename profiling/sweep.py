"""CLI: run one (thread_count, threading_layer) point and append its
metrics as one CSV row. Invoked once per Slurm step by run_sweep.sh so
every point gets its own process -- and hence its own real CPU affinity
mask via --cpus-per-task/--cpu-bind=none (see README.md's cpuset-gotcha
note: a shared interactive shell's own process stays stuck at 1 CPU
regardless of how many the job holds)."""

import argparse
import csv
import os
from pathlib import Path

import numba

from profiling.common import run_once


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    metrics = run_once(args.threads, seed=args.seed)
    metrics["threading_layer"] = numba.threading_layer()
    metrics["env_threading_layer"] = os.environ.get("NUMBA_THREADING_LAYER", "default")
    metrics["affinity_cpus"] = len(os.sched_getaffinity(0))
    metrics["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "")

    write_header = not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    print(
        f"threads={args.threads} layer={metrics['threading_layer']} affinity={metrics['affinity_cpus']} "
        f"wall_s={metrics['wall_s']:.2f} ks={metrics['ks']:.6f} "
        f"nvcsw={metrics['voluntary_ctx_switches']} nivcsw={metrics['involuntary_ctx_switches']}"
    )


if __name__ == "__main__":
    main()
