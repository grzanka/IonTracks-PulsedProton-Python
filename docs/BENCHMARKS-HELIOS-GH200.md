# Running on Helios GH200 (Grace Hopper)

Helios has a second kind of node: `plgrid-gpu-gh200`, four NVIDIA **GH200
120GB** superchips per node. This page is what the CUDA backend does there, how
fine a grid it can hold, and what it costs to run a grid larger than the GPU.
For the A100 numbers and the CPU crossover see [GPU.md](GPU.md), for the CPU
partition [HELIOS.md](HELIOS.md), for the physics [PHYSICS.md](PHYSICS.md).

**The one-line answer:** the full 1 µm-voxel column — **2.0 G voxels, 60.6 GiB
of carrier arrays** — runs in HBM at **108 ms/step**, which is a resolution
this code could not reach on any machine previously benchmarked here. Ask for
`--mem` explicitly: the default 8-core allocation gives the job 12 GiB of host
RAM against the GPU's 95 GiB, and it is the host figure that decides whether
unified memory can take you past the GPU at all.

---

## 1. The machine

Measured on a `plgrid-gpu-gh200` worker node:

| | |
|---|---|
| node | 4 × **GH200 120GB**: 288 Neoverse-V2 cores, 478 GiB LPDDR5X, 4 GPUs |
| per GPU | **Hopper H100**, sm_90, 132 SMs, **95 GiB HBM3** (~4 TB/s), 60 MiB L2 |
| per superchip | 72 Grace cores, ~120 GiB LPDDR5X |
| CPU↔GPU | **NVLink-C2C**, ~450 GB/s each way, hardware cache-coherent |
| NUMA | 36 domains; the GPU's HBM appears as its own CPU-less node |
| driver / CUDA | driver 595.71, CUDA 13.2; toolkit module `CUDA/12.9.1` |
| stack | Python 3.11.5, CuPy 14.1.1 (`cupy-cuda13x`), Numba 0.67 |

Three device attributes decide what the rest of this page is about, and all
three are 1 here: `ConcurrentManagedAccess`,
`DirectManagedMemAccessFromHost`, and `PageableMemoryAccessUsesHostPageTables`.
The last one is the definition of an ATS machine — the GPU walks the *host's*
page tables, so host memory is not a staging area the way it is across PCIe,
it is slower memory in the same coherent address space.

## 2. Setup

Repeat the two `module load` lines in every new shell; the venv does not
remember them, and `module load` must not be piped (a pipe runs it in a
subshell and the environment is silently lost).

```bash
module load Python/3.11.5
```

```bash
module load CUDA/12.9.1
```

```bash
python -m venv .venv-gh200
```

```bash
source .venv-gh200/bin/activate
```

```bash
pip install -e ".[dev]"
```

```bash
pip install cupy-cuda13x  # aarch64 wheel, bundles its own CUDA runtime
```

```bash
pytest tests/test_cuda_backend.py -q  # 17 tests, ~4 s
```

`cupy-cuda13x` matches the 13.2 driver; the `CUDA/12.9.1` module is only what
Numba's CUDA target compiles against (it finds `libnvvm` through `CUDA_HOME`).
The x86 `venv/` in this repo will not work on aarch64 — build a separate one.

## 3. Ask for the memory

A GH200 node is 288 cores / 489600 MB / 4 GPUs, so one GPU's fair share is
**72 cores and ~120 GB**. The partition's `DefMemPerCPU` is 1536 MB, so a
request that names only cores gets a fraction of that:

```bash
srun -A <grant>-gpu-gh200 -p plgrid-gpu-gh200 --gres=gpu:1 --cpus-per-task=72 --mem=120G --time=8:00:00 --pty bash
```

For grids past the GPU's own 95 GiB, take the whole node — the point is its
478 GiB of LPDDR5X, not the other three GPUs:

```bash
srun -A <grant>-gpu-gh200 -p plgrid-gpu-gh200 --exclusive --gres=gpu:1 --cpus-per-task=288 --mem=0 --time=8:00:00 --pty bash
```

This matters more than it looks. The memory guards read the job's **cgroup**
limit, not `/proc/meminfo`: in a default 8-core allocation they report 11.88
GiB where the node has 736 GiB free, which is the number that is actually true
— exceed it and Slurm's OOM killer ends the run. An oversubscription ladder in
such a job declines to run at all rather than pretending:

```
note: this job's host headroom (10.40 GiB) cannot hold a grid past the
GPU's 94.44 GiB. Ask sbatch for more memory.
```

One more consequence of a small `--mem` on a big GPU: the final density
snapshot is two host arrays the size of the grid, so a 60 GiB run finishes and
*then* dies copying its answer home. Pass `return_fields=False` when only the
time series and `k_s` are wanted; `bench_gh200.py` always does.

## 4. Resolution — the 1 µm column

The AIC-144 Markus 2 mm case at a fixed 0.05 cm sampled radius, refining the
voxel size, all in device memory, 200 steps each:

| voxels | grid | carrier arrays | steps (full run) | ms/step | Gvoxel/s | eff. GB/s |
|---|---|---|---|---|---|---|
| 2 M | 106² × 210 | 72 MiB | 2 194 | 2.56 | 0.88 | 28 |
| 17 M | 206² × 410 | 531 MiB | 4 580 | 3.18 | 5.34 | 171 |
| 259 M | 506² × 1010 | 7.71 GiB | 13 019 | 15.4 | 16.6 | 532 |
| **2 034 M** | **1006² × 2010** | **60.6 GiB** | 31 564 | **108.1** | **18.7** | **599** |

`f` after 200 steps is identical across the memory modes at each size (§5), so
the physics does not move with the allocator.

**The 1 µm grid fits, and it runs at essentially full speed.** From 259 M to
2 034 M voxels — 7.9× the work — the step cost rises 7.0×, so the throughput
still *improves* slightly (16.6 → 18.7 Gvoxel/s). There is no cliff at 60 GiB
because there is nothing to fall off: it is all HBM.

Read the first two rows as the launch-latency floor, not as performance. At
2 M voxels the sweep is 0.9 Gvoxel/s because the grid cannot fill 132 SMs;
this is the same "small grids belong on the CPU" region as GPU.md §4, and it
starts to end around 17 M voxels.

The effective bandwidth column counts only compulsory traffic (2 carrier reads
+ 2 writes = 32 B per interior voxel) and tops out near 600 GB/s against HBM3's
~4 TB/s. The gap is not idleness — the sweep also loads six neighbours per
voxel, and only the two z-neighbours are unit-stride, so the real DRAM traffic
is a multiple of the compulsory figure. It does say where the remaining
head-room is: shared-memory tiling of the transverse stencil (GPU.md §9).

### What the refinement buys

Resolution is only worth paying for if the answer moves. The same three tiers
run *to completion* — a real `k_s`, not a per-step cost:

| voxel size | steps | wall | `k_s` | recombined | shift vs 10 µm |
|---|---|---|---|---|---|
| 10 µm | 2 194 | 3.5 s | 1.1087504 | 9.81 % | — |
| 5 µm | 4 580 | 11.4 s | 1.1081197 | 9.76 % | −0.057 % |
| **2 µm** | **13 019** | **183 s** | **1.1078208** | **9.73 %** | −0.084 % |

**`k_s` is already converged to better than 0.1 % at 10 µm here**, and the two
refinements move it by 6.3e-4 and then 3.0e-4 — close to first order in `h`, so
linear extrapolation puts the `h → 0` limit near 1.10762, about 0.02 % below
the 2 µm value. The physics case for 1 µm on *this* geometry is therefore weak,
and this table is the honest answer to "how fine should I go": not this fine.

What the capability is for is the cases where that does not hold — steeper
gradients, higher fields, smaller track radii, FLASH dose rates — where the
convergence tail is longer and, until now, untestable. The discretisation error
is now a measured, extrapolable quantity rather than an assumption.

Note the per-step costs here are lower than §4's — 14.0 ms against 15.4 at
2 µm, 1.6 against 2.6 at 10 µm. That is a property of `--max-steps`, not noise:
the pulse arrives at the start of the run, so a 200-step slice is
disproportionately deposition, and truncated timings **over**-estimate the
steady-state step cost by up to 60 % on the small grids. Use them to compare
rows with each other, not as a prediction of a full run's wall clock.

```bash
python profiling/bench_gh200.py --sizes 0.05@10,0.05@5,0.05@2 --max-steps 0
```

## 5. Where to put the arrays

The same 259 M-voxel grid — one that comfortably fits in HBM — through every
allocator and placement policy, 200 steps:

| `memory` | `advise` | ms/step | vs device | eff. GB/s | `f` after 200 steps |
|---|---|---|---|---|---|
| device | — | **15.33** | 1.00× | 535 | 0.996148 |
| managed | device | **15.62** | 1.02× | 525 | 0.996148 |
| managed | none | 98.77 | 6.4× | 83 | 0.996148 |
| host | — | 96.70 | 6.3× | 85 | 0.996148 |
| managed | host | 273.67 | 17.9× | 30 | 0.996148 |

**`f` agrees to every digit printed in all five.** The allocator is invisible
to the physics, which is the precondition for any of this being useful.

Four things this table settles:

**Managed memory is free when the grid fits** — 2 % over `cudaMalloc`, which is
inside the run-to-run spread. That is why `memory="auto"` is the default: a run
that fits pays nothing for the ability to spill, and a run that does not fit
now runs instead of raising `MemoryError`. The 2 % is bought by
`advise="device"`, which prefers HBM and prefetches; without it (`advise="none"`)
the same run is **6.4× slower**, because every page is migrated on its first
fault.

**Streaming from host memory costs 6.3×, not 60×.** `memory="host"` puts all
7.7 GiB in Grace's LPDDR5X and never moves it; the GPU reads it over C2C every
step. On a PCIe machine that experiment is not worth running. Here it is a
factor of six, on a link that is one seventh of HBM's bandwidth — the sweep's
cache reuse absorbs the rest.

**`advise="host"` on managed memory is the worst of both** — 17.9×, and worse
than plain host memory doing ostensibly the same thing. Pinning the preferred
location to the host does not switch migration *off*; it keeps the fault
machinery running and gives it nothing to work with. If the arrays belong in
host memory, use `memory="host"` and skip the machinery.

**No policy here is a good idea on a grid that fits.** All three slow paths are
declining HBM that was offered. They exist for the case in §7.

## 6. Block size

The sweep at 128, 256, 512 and 1024 threads per block (259 M voxels, 200 steps):

| threads/block | ms/step | Gvoxel/s |
|---|---|---|
| 128 | 20.62 | 12.4 |
| **256** | **15.40** | **16.6** |
| 512 | 17.38 | 14.7 |
| 1024 | — | `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES` |

**256 is the optimum on Hopper, as it was on the A100**, so the default is
unchanged — but it is now a measurement rather than an inherited constant, and
`PULSED_ION_CHAMBER_CUDA_TPB` re-measures it on the next architecture. 1024
does not launch at all: the sweep's register footprint plus 3 × 1024 × 8 B of
shared reduction arrays exceeds what an SM will give a single block.

## 7. Past the GPU: migration does not work here

The reach of unified memory, on a whole node (478 GiB host + 95 GiB HBM ≈ 570
GiB addressable):

| target | grid | carrier arrays | fits in |
|---|---|---|---|
| full electrode @ 5 µm | 1066² × 410 | 13.9 GiB | HBM (156 s on an A100, GPU.md §5) |
| **full electrode @ 2 µm** | **2656² × 1010** | **212 GiB** | unified memory only |
| full electrode @ 1.5 µm | 3539² × 1343 | 501 GiB | the edge of a whole node |
| full electrode @ 1 µm | 5306² × 2010 | 1.65 TiB | nothing on this machine |

So 1 µm is reachable per unit volume (§4) but not across the whole electrode:
that column is 1.65 TiB, about 3× the node. The 2 µm full electrode is the run
this hardware makes newly possible — 7.12 G voxels, 212 GiB, 2.2× the GPU's own
memory. It runs. It also, on managed memory, runs appallingly:

| | 2 µm full electrode, managed memory, `advise="device"` |
|---|---|
| grid | 2656² × 1010, 7.12 G voxels, 212.34 GiB |
| ms/step | **32 143** — thirty-two *seconds* a step |
| throughput | 0.22 Gvoxel/s, 7 GB/s effective |
| 500 of 13 019 steps | **4 h 28 min** (job 20774789; a full run would be ~116 h) |
| `f` at step 500 | 0.980600 — the physics is fine, only the speed is not |

**84× slower than the same GPU's HBM rate**, and 13× worse than even §5's
host-memory ratio would predict. This is the regime §5 explicitly refused to
extrapolate into, and the refusal was right: a fitting grid and an overflowing
one are not the same problem.

The mechanism is not subtle in hindsight. Page migration pays when there is a
resident working set that gets reused. **This sweep has none** — it streams all
four carrier arrays end to end every single step, so 212 GiB moves through 95
GiB of HBM per step no matter what, and `advise="device"` spends the whole step
prefetching pages into HBM while evicting the pages the next warps were about
to want. 7 GB/s is roughly 1/64 of the C2C link: almost all of the traffic is
migration overhead rather than the data the kernel asked for.

**So `memory="auto"` no longer chooses managed memory.** Past HBM, on an ATS
machine, it chooses `memory="host"`: the arrays stay in LPDDR5X and the GPU
reads them over C2C with no migration machinery in the loop, which is a cost
that is flat by construction and measured at 6.3× in §5. Asking for
`memory="managed"` on an oversubscribed grid now warns and points here.

**Not yet measured:** that host-memory path *on this 212 GiB grid*. §5's 6.3×
was measured on a grid that fits, and the whole lesson of this section is that
the two regimes differ — so treat ~2.4 s/step as the hypothesis to test, not a
result. `./submit_gh200.sh --only headline --steps 20 --memory host` is the
experiment, and 20 steps is deliberate: at the managed rate that is 11 minutes,
and there is no point spending another 4½ hours to learn the same thing twice.

## 8. Reproducing this page

From a Helios **access** node (compute nodes cannot submit; Slurm's refusal
there is a bare "Access/permission denied"):

```bash
./submit_gh200.sh  # the 2 um full electrode plus the four ladders
```

```bash
squeue -u $USER
```

Or interactively, inside an allocation from §3. The §4 table:

```bash
python profiling/bench_gh200.py --ladder resolution --max-steps 200
```

The §5 table:

```bash
python profiling/bench_gh200.py --ladder memory --max-steps 200
```

The §6 table:

```bash
python profiling/bench_gh200.py --ladder blocks --max-steps 200
```

The §7 case, sized to whatever memory the job holds:

```bash
python profiling/bench_gh200.py --ladder oversubscribe --max-steps 100
```

Drop `--max-steps` (or pass `0`) to run each case to completion and get a real
`k_s` instead of a per-step cost — at 1 µm that is 31 564 steps, about an hour.
