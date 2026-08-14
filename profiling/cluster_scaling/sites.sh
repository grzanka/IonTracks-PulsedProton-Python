#!/usr/bin/env bash
# Per-cluster settings for the scaling study. Sourced by submit_all.sh and by
# scaling_job.sbatch; not run on its own.
#
# Everything that differs between clusters lives here, so the job script and the
# submitter stay identical everywhere and a new machine is a new `case` branch
# rather than a new copy of the study.
#
#   site_configure <site>   sets SITE_* variables for that site
#   site_detect             guesses the site from the environment
#
# The variables it sets:
#   SITE_NAME            human-readable
#   SITE_MODULES         the `module load` line, or "" if none is needed
#   SITE_ACCOUNT         default Slurm account
#   SITE_PARTITION       default Slurm partition
#   SITE_THREADS         default thread ladder, chosen for that machine's topology
#   SITE_EXCLUSIVE       1 to request whole nodes by default
#   SITE_CORES_PER_NODE  for sanity-checking a thread request
#   SITE_WALLTIME_*      walltime for the 1-, 2- and default cases

site_detect() {
  # SLURM_CLUSTER_NAME is set on both clusters and is the least ambiguous
  # signal; the hostname prefix is the fallback for a login shell that has no
  # Slurm environment yet.
  local name="${SLURM_CLUSTER_NAME:-}"
  if [ -z "$name" ]; then
    case "$(hostname -f 2>/dev/null || hostname)" in
      *helios*) name=helios ;;
      *ares*)   name=ares ;;
    esac
  fi
  echo "${name:-unknown}"
}

site_configure() {
  case "$1" in

    helios)
      SITE_NAME="Cyfronet Helios"
      # 2 x AMD EPYC 9654: 192 cores, SMT off, 8 NUMA domains of 24 contiguous
      # cores, 768 MiB L3, ~900 GB/s.
      SITE_MODULES="GCCcore/13.3.0 Python/3.12.3"
      SITE_ACCOUNT="${ACCOUNT:-plgccbmc15-cpu}"
      SITE_PARTITION="${PARTITION:-plgrid}"
      # Powers of two to 128. 192 is never worth asking for (docs/HELIOS.md §4)
      # and 128 is already past the optimum, kept only to show the turn.
      SITE_THREADS="${THREAD_COUNTS:-1 2 4 8 16 32 64 128}"
      SITE_EXCLUSIVE="${EXCLUSIVE:-0}"
      SITE_CORES_PER_NODE=192
      SITE_WALLTIME_1="00:30:00"
      SITE_WALLTIME_2="00:20:00"
      SITE_WALLTIME_DEFAULT="00:15:00"
      ;;

    ares)
      SITE_NAME="Cyfronet Ares"
      # 2 x Intel Xeon Platinum 8268 (Cascade Lake): 48 cores, HT off,
      # 35.75 MiB L3 per socket, ~200 GB/s, 192 GB RAM.
      #
      # Sub-NUMA Clustering is ON: four NUMA domains of twelve cores, and the
      # CPU numbering is *interleaved* rather than contiguous --
      #     node 0: 0 1 2 3 7 8 12 13 14 18 19 20
      #     node 1: 4 5 6 9 10 11 15 16 17 21 22 23
      # so consecutive CPU ids are not in the same domain. Distances are 10
      # local, 11 to the sibling domain on the same socket, 21 across sockets.
      # This is why the ladder below is 12/24/48 rather than 16/32: those are
      # the domain, socket and node boundaries, and they are where the curve
      # should bend.
      #
      # Same Python and GCCcore as Helios, but Ares names its modules in lower
      # case and rolls the toolchain into one name rather than two. Confirmed by
      # `module spider python` on an Ares login node.
      #
      # Worth having landed on the same toolchain: both clusters end up with
      # Python 3.12.3, numba 0.67.0 and numpy 2.5.2, so a Helios-vs-Ares
      # comparison is a hardware comparison and not a software one.
      SITE_MODULES="${SITE_MODULES_OVERRIDE:-python/3.12.3-gcccore-13.3.0}"
      SITE_ACCOUNT="${ACCOUNT:-plgccbmc15-cpu}"
      SITE_PARTITION="${PARTITION:-plgrid}"
      SITE_THREADS="${THREAD_COUNTS:-1 2 4 8 12 24 48}"
      # Exclusive by default here, unlike Helios. A node is 48 cores rather than
      # 192, so a co-tenant takes a much larger share of a much smaller
      # bandwidth pool -- and booking the whole node costs a quarter as much.
      # See the README's "Should whole nodes be reserved?".
      SITE_EXCLUSIVE="${EXCLUSIVE:-1}"
      SITE_CORES_PER_NODE=48
      # Ares cores clock higher than Helios cores (3.9 GHz turbo vs ~3.5) and
      # get more bandwidth each, so single-core should be *faster* than Helios's
      # 572 s / 713 s. These are Helios's walltimes kept as-is: generous here,
      # and cheap to be wrong about in this direction.
      SITE_WALLTIME_1="00:30:00"
      SITE_WALLTIME_2="00:20:00"
      SITE_WALLTIME_DEFAULT="00:15:00"
      ;;

    *)
      return 1
      ;;
  esac
  return 0
}

site_walltime() {
  case "$1" in
    1) echo "$SITE_WALLTIME_1" ;;
    2) echo "$SITE_WALLTIME_2" ;;
    *) echo "$SITE_WALLTIME_DEFAULT" ;;
  esac
}
