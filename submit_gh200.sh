#!/usr/bin/env bash
# Submit the Grace Hopper (GH200) study to Helios. Run this on a Helios *access*
# node -- compute nodes cannot submit, and Slurm's refusal there is a bare
# "Access/permission denied":
#
#     ./submit_gh200.sh
#
# The headline job it submits is the full Markus electrode at **2 um voxels**:
# a 2656 x 2656 x 1010 grid, 7.05 G voxels, **210 GiB of carrier arrays**. That
# does not fit in the GH200's 96 GiB of HBM and does not fit in any GPU sold,
# so it runs on unified memory -- the arrays spill into the Grace CPU's
# LPDDR5X and the Hopper die reaches them across the coherent NVLink-C2C link.
# Which is why the job asks for a **whole node**: it is buying the 478 GiB of
# host memory, not the other three GPUs.
#
# Also submitted, unless --only is given: the resolution ladder (10 -> 1 um),
# the memory-mode comparison, the oversubscription ladder and the block-size
# sweep, each a short job of its own.
#
# Options, all optional:
#   ./submit_gh200.sh --only headline      just the 2 um full electrode
#   ./submit_gh200.sh --only ladders       just the four short benchmark ladders
#   ./submit_gh200.sh --steps 2000         steps for the headline run (default 500;
#                                          0 = run all 13019 and get a real k_s)
#   ./submit_gh200.sh --time 12:00:00      wall time for the headline job (default 08:00:00)
#   ./submit_gh200.sh --grid-um 1.0        refine the headline further (1 um is 56 G
#                                          voxels / 1.6 TiB and will NOT fit -- 1.5 um
#                                          is ~500 GiB and is the edge of the node)
#   ./submit_gh200.sh --memory host        allocator for the headline: auto (default),
#                                          managed or host
#   ./submit_gh200.sh --account plgXXX-gpu-gh200
#   ./submit_gh200.sh --shared             do not take a whole node (fits only the
#                                          ladders; the headline needs the node's RAM)
#   ./submit_gh200.sh --dry-run            print what would be submitted, submit nothing
#
# Results land in profiling/data/gh200/ as one JSON per run, job logs under
# logs/. See docs/BENCHMARKS-HELIOS-GH200.md for what the numbers mean.

set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"

ACCOUNT="${ACCOUNT:-plgccbmc15-gpu-gh200}"
PARTITION="${PARTITION:-plgrid-gpu-gh200}"
VENV="${VENV:-.venv-gh200}"
ONLY="all"
STEPS=500
HEADLINE_TIME="08:00:00"
LADDER_TIME="01:00:00"
GRID_UM="2.0"
MEMORY="auto"
EXCLUSIVE=1
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --only)      ONLY="$2";          shift 2 ;;
    --steps)     STEPS="$2";         shift 2 ;;
    --time)      HEADLINE_TIME="$2"; shift 2 ;;
    --grid-um)   GRID_UM="$2";       shift 2 ;;
    --memory)    MEMORY="$2";        shift 2 ;;
    --account)   ACCOUNT="$2";       shift 2 ;;
    --partition) PARTITION="$2";     shift 2 ;;
    --venv)      VENV="$2";          shift 2 ;;
    --shared)    EXCLUSIVE=0;        shift ;;
    --dry-run)   DRY_RUN=1;          shift ;;
    -h|--help)   sed -n '2,36p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

case "$ONLY" in all|headline|ladders) ;; *)
  echo "--only must be all, headline or ladders (got $ONLY)" >&2; exit 2 ;;
esac

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on a Helios access node, not a compute node." >&2
  exit 1
fi
if [ ! -d "$REPO/$VENV" ]; then
  echo "No $VENV in $REPO. Create it first -- docs/BENCHMARKS-HELIOS-GH200.md sec. 2." >&2
  exit 1
fi

mkdir -p "$REPO/logs" "$REPO/profiling/data/gh200"

# One GH200 node is 288 cores, 489600 MB and 4 GPUs, i.e. 4 Grace Hopper
# superchips. A single-GPU job's fair share is 72 cores / ~120 GB, and that is
# what --shared asks for. The default asks for the whole node with --mem=0
# (all of it) because the point of the headline run is the 478 GiB of LPDDR5X
# that the one GPU we use can address over C2C.
if [ "$EXCLUSIVE" = 1 ]; then
  SHAPE=(--exclusive --mem=0 --cpus-per-task=72 --gres=gpu:1)
else
  SHAPE=(--mem=120G --cpus-per-task=72 --gres=gpu:1)
fi

submit() {  # submit <name> <time> <command...>
  local name="$1" walltime="$2"; shift 2
  local args=(
    --account="$ACCOUNT" --partition="$PARTITION" "${SHAPE[@]}"
    --time="$walltime" --job-name="$name"
    --output="$REPO/logs/${name}-%j.out" --error="$REPO/logs/${name}-%j.err"
  )
  if [ "$DRY_RUN" = 1 ]; then
    printf 'sbatch %s <<script>>\n    %s\n\n' "${args[*]}" "$*"
    return
  fi
  # A here-doc script rather than --wrap, and `#!/bin/bash -l` rather than a
  # plain shebang: `module` is a shell function that Lmod installs from
  # /etc/profile.d, so a non-login job shell may not have it at all. --wrap
  # happens to work when submitted from a shell that already has the function
  # exported, which is exactly the kind of dependency that fails on the day
  # someone submits from cron.
  sbatch "${args[@]}" <<SCRIPT
#!/bin/bash -l
set -euo pipefail
module load Python/3.11.5
module load CUDA/12.9.1
cd "$REPO"
source "$VENV/bin/activate"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "host memory limit: \$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo unknown)"
$*
SCRIPT
}

# The headline: the full Markus electrode (sampled_radius_cm=0.265) at --grid-um.
# --max-steps keeps it to a measured per-step cost; --steps 0 runs the whole
# pulse train and yields a real k_s, which at 2 um is ~13,000 steps.
HEADLINE="python profiling/bench_gh200.py --sizes 0.265@$GRID_UM \
--memory $MEMORY --max-steps $STEPS \
--json profiling/data/gh200/full_electrode_${GRID_UM}um.json"

if [ "$ONLY" != "ladders" ]; then
  if [ "$EXCLUSIVE" != 1 ]; then
    echo "warning: --shared caps host memory at 120G; the ${GRID_UM} um full electrode needs more." >&2
  fi
  submit "gh200-full-${GRID_UM}um" "$HEADLINE_TIME" "$HEADLINE"
fi

if [ "$ONLY" != "headline" ]; then
  for ladder in resolution memory oversubscribe blocks; do
    submit "gh200-$ladder" "$LADDER_TIME" \
      "python profiling/bench_gh200.py --ladder $ladder --max-steps 200 \
--json profiling/data/gh200/${ladder}.json"
  done
fi

if [ "$DRY_RUN" = 1 ]; then
  echo
  echo "(dry run: nothing was submitted)"
  exit 0
fi

echo
echo "Watch the queue with:"
echo "  squeue -u \$USER"
echo
echo "Read the results with:"
echo "  cat logs/gh200-*.out"
