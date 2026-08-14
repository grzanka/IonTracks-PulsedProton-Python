# Algorithm

How the solver is actually implemented: data layout, the two hot loops, what
"batching" means and why it is exact, and where the parallelism is.

For the physics these operations represent see [PHYSICS.md](PHYSICS.md); for
what it all costs, [PERFORMANCE.md](PERFORMANCE.md).

---

## 1. The main loop

```
build the arrival schedule:  schedule[step] = how many tracks arrive at that step
allocate n+, n-, n+_next, n-_next

for step in range(total_time_steps):
    for each track arriving at this step:
        sample (x, y) inside the sampling disc
        deposit its Gaussian into n+ and n-, accumulating injected charge
    advance n+, n- one time step, accumulating recombined charge
    copy the interior of the _next arrays back over n+, n-
    apply the lateral boundary condition
    f_t[step] = (injected - recombined) / injected
```

Everything expensive is in the two inner operations: **track deposition** (§4)
and the **Lax-Wendroff update** (§6). Both are `O(grid)` per invocation; the
difference is that the update runs once per step while deposition runs once per
*track*, and there are far more tracks than steps.

---

## 2. Grid layout

One array per carrier, shape `(no_xy, no_xy, no_z_with_buffer)`, C-contiguous —
so **`k` (the z index, along the field) is the fastest-varying axis**. Every
loop in the code is written `k`-innermost so that memory is walked
sequentially.

```
        x,y (no_xy)                              z (no_z_with_buffer)
  +---------------------------+          +-------------------------------+
  |  buffer ring              |          | no_z_electrode  buffer layers |
  |  +---------------------+  |          +-------------------------------+
  |  |  scored disc        |  |          | no_z gap layers  <- tracks    |
  |  |  +---------------+  |  |          |                     deposited |
  |  |  | sampling disc |  |  |          |                     here      |
  |  |  +---------------+  |  |          +-------------------------------+
  |  +---------------------+  |          | no_z_electrode  buffer layers |
  +---------------------------+          +-------------------------------+
```

Three transverse radii, all in voxel units from the centre `mid_xy`:

| radius | meaning |
|---|---|
| `sampling_radius` | tracks are placed inside this (`inner_radius × chamber_fill_fraction`) |
| `inner_radius` | `= sampled_radius_cm / unit_length_cm`; the scored disc |
| `outer_radius = no_xy/2` | the array edge; `buffer_radius` voxels further out |

Comparisons against these are done on **squared** distances
(`scoring_radius_sq`) so no `sqrt` is called per voxel. `scoring_region` selects
whether `scoring_radius_sq` is the disc or a value large enough to admit every
voxel including the corners — that way "score everything" needs no extra branch
in the kernels, just a different constant.

Longitudinally, tracks are deposited only into the `no_z` gap layers.
`no_z_electrode` buffer layers at each end give drifting charge somewhere to go
in the step it leaves the gap, so nothing wraps or clips.

---

## 3. Track scheduling

**How many.** `number_of_tracks_per_pulse` comes from the dose (PHYSICS §5).

**When.** Arrival times are the normalised cumulative sum of uniform deviates:
draw `N` uniforms, take the running sum, divide by the last element, scale to
`pulse_duration_s`. This gives an increasing sequence spanning the window with
roughly uniform density and no rejection loop. Histogramming it against the
per-step time bins gives `schedule[step]`, the number of tracks to insert at
each step. The rescaling is done in place because at tens of millions of tracks
each full-size temporary is hundreds of megabytes, and this runs before the
carrier arrays exist, so it sets the run's peak memory (§10).

**Where.** `(x, y)` is rejection-sampled uniformly in the square `[0, no_xy]²`
and rejected unless it lands inside `sampling_radius`. Rejection rather than
sampling `(r, θ)` directly, because uniform-in-`r` would pile tracks onto the
axis; the accept rate is `π/4` at worst, so the loop is cheap.

---

## 4. Track deposition

The physical statement (PHYSICS §3): a track is a line parallel to `z` carrying
a 2D Gaussian cross-section, `n(r) = N₀/(πb²)·exp(−r²/b²)`. Three properties of
that expression are exploited, and all three are exact identities, not
approximations.

**(a) There is no `k` dependence.** The same 2D profile is written into every
gap layer. A naive implementation recomputes `exp(...)` inside the `k` loop,
`no_z` times per grid point; hoisting it out removes that redundancy entirely.
What remains inside the `k` loop is a single `+=` of an already-computed value.

**(b) The 2D Gaussian is separable.**

```
exp(−((i−x)² + (j−y)²)·h²/b²) = exp(−(i−x)²·h²/b²) · exp(−(j−y)²·h²/b²)
```

because `e^(a+b) = e^a·e^b`. So the profile over a `w×w` stencil needs `2w`
calls to `exp` and `w²` multiplications, rather than `w²` calls to `exp`. On the
stencil sizes used this is roughly a tenfold saving on the deposition, since
`exp` costs far more than a multiply.

**(c) The Gaussian is negligible outside a few `b`.** Deposition is restricted
to the square bounding box of `track_cutoff_voxels` around `(x, y)`, clipped to
the grid (PHYSICS §4). This is what makes the per-track cost independent of how
wide the grid is.

Per track, then: `2w` exponentials, `w²` multiplications, and `w²·no_z`
additions into each of the two carrier arrays.

---

## 5. Batching — what it means and why it is exact

### The problem

Deposition is `O(w²·no_z)` per track and there are a lot of tracks. But look at
what those `no_z` writes are: for a given `(i, j)`, **every** track that arrives
in the same time step writes its own value into the **same** column of `no_z`
voxels. If 13,000 tracks arrive in one step, that column is traversed 13,000
times, each time adding one number.

### The fix

A **batch is all the tracks arriving in a single time step.** Instead of
depositing them one at a time, sum their 2D profiles into a single scratch
array `total_density[no_xy, no_xy]` first, and only then broadcast that summed
profile down the `z` columns **once**.

```
unbatched   for each track t:  for each (i,j) in stencil(t):  for k in gap:  n[i,j,k] += g_t(i,j)
batched     for each track t:  for each (i,j) in stencil(t):                 total[i,j] += g_t(i,j)
            for each (i,j):                                   for k in gap:  n[i,j,k] += total[i,j]
```

Cost per time step, with `m` tracks in the batch:

| | insertion cost |
|---|---|
| unbatched | `m · w² · no_z` |
| batched | `m · w²  +  no_xy² · no_z` |

The `no_z` factor moves off the per-track term and onto a single per-step term.

### Why it is exact

Deposition is pure accumulation, and addition is associative:

```
Σ_t broadcast(g_t)  =  broadcast(Σ_t g_t)
```

Both sides add exactly the same set of numbers into each voxel. The only
difference is the *order* of the floating-point additions, which can change the
last bits — which is why the cross-backend tests compare to a relative tolerance
of 1e-9 rather than demanding bit equality.

### When it pays

The win scales with `m = tracks per step ≈ number_of_tracks_per_pulse /
pulse_time_steps`:

| scenario | `m` | `no_xy` | unbatched / batched insertion |
|---|---|---|---|
| AIC-144 `archive` tier | ≈ 13 | 22 | ~12× |
| full 5.3 mm electrode | ≈ 13 900 | 536 | ~32× |

At low dose rates `m` falls towards 1 and batching stops helping — with one
track per step the two expressions differ only by the `no_xy²·no_z` broadcast
term, which is then pure overhead. That is why both backends exist:
`solver_numba` (unbatched) is the simpler baseline, `solver_numba_parallel`
(batched) is the one to use when tracks per step is large.

There is a second, subtler benefit. Under Numba's threading, batching cuts the
number of parallel-region launches for deposition from one per *track* to one
per *step* — on the full-electrode run, from 24.6 million to 1,775. Each launch
costs a fork/join barrier regardless of how little work it contains, so this
matters as much as the arithmetic saving.

### The row index

Batching creates its own problem. Phase 1 wants to be parallel over grid rows
(so that concurrent writes to `total_density` never touch the same row), but a
row does not know which tracks reach it. Testing every track against every row
is `O(no_xy · m)` — on a 536-row grid with 13,900 tracks per step, that scan is
comparable to the real work and grows with the grid, undoing the point of the
stencil.

So `_build_row_index` first buckets tracks by the rows their stencil covers, in
CSR form: `offsets[i] .. offsets[i+1]` indexes into `track_ids`, listing exactly
the tracks that touch row `i`. Building it costs `O(m · w)` — the same order as
the work it feeds — and phase 1 then examines no track for a row it cannot
reach.

---

## 6. The Lax-Wendroff update

One step of the drift-diffusion equation, second order in space and time, on a
7-point stencil (the voxel and its six face neighbours). Writing
`s = D·dt/h²` and `c = µ·E·dt/h` for one species:

```
n_new[i,j,k] =  (s + c(c+1)/2) · n[i,j,k−1]      # upwind neighbour
              + (s + c(c−1)/2) · n[i,j,k+1]      # downwind neighbour
              + s · (n[i±1,j,k] + n[i,j±1,k])    # transverse: diffusion only
              + (1 − c² − 6s) · n[i,j,k]         # centre
              − α·dt·n₊[i,j,k]·n₋[i,j,k]         # recombination sink
```

Four points about the implementation:

- **The transverse terms carry no `c`** because drift is along `z` only. That
  also makes the second Deghan stability criterion trivially satisfied, so only
  `6s + c² ≤ 1` has to be solved for `dt`.
- **The two species differ only through their own `s` and `c`.**
  `config.scheme_coefficients()` returns a `(lateral, z_minus, z_plus, centre)`
  weight tuple for each, and the negative carrier's `z_minus`/`z_plus` are
  applied to the opposite neighbours, which is what makes it drift the other
  way. With one averaged species the two tuples are identical.
- **The recombination sink uses the previous step's densities for both
  carriers**, so the update is fully explicit — no iteration, and `n₊` and `n₋`
  can be advanced independently within a step.
- **Reads come from the current arrays, writes go to the `_next` arrays**, then
  the interior is copied back. Separate arrays are what make every voxel update
  independent of every other, which is what makes the loop trivially parallel.

The loop runs `i, j ∈ [1, no_xy−2]` and `k ∈ [1, no_z_with_buffer−2]`, so the
outermost shell is never written — see §7.

---

## 7. Boundaries

- **`z` ends** are never written by the sweep, stay at zero, and so absorb:
  charge that drifts into the last buffer layer leaves the simulation. Nothing
  further is needed because the electrode is where charge is supposed to go.
- **`x,y` ring** is likewise never written by the sweep. With
  `lateral_boundary="absorbing"` that is the whole story. With `"reflecting"`,
  `apply_lateral_boundary` copies each adjacent interior plane outwards after
  every step, giving a zero-gradient (zero-flux) wall.

The reflecting mode also cleans up an artifact of the absorbing one: the sweep
never updates the ring, but *deposition does write to it*, so track tails
accumulate there and then sit frozen — never drifting, diffusing or
recombining, while continuing to feed their inward neighbour every step.
Mirroring the interior overwrites that.

---

## 8. Scoring

Two scalars accumulate over the run:

- `no_initialised` — the deposited density summed over scored voxels, added by
  the deposition routine as it writes.
- `no_recombined` — `α·dt·n₊·n₋` summed over scored voxels in the gap, added by
  the update as it computes each voxel.

Both are sums of *densities on the same voxel set*, so the voxel volume cancels
in the ratio and never needs to appear. `f_t[step] = (no_initialised −
no_recombined) / no_initialised`, and `k_s = 1/f_t[-1]`.

Accumulating inside the kernels rather than reducing over the arrays afterwards
avoids a second full pass over the grid every step.

---

## 9. The two backends

Same physics, same RNG stream, same answer to 1e-9 — enforced by
`tests/test_backends_agree.py`.

| | `solver_numba.py` | `solver_numba_parallel.py` |
|---|---|---|
| threading | single-threaded `@njit` | `@njit(parallel=True)` |
| deposition | one track at a time; each broadcast down the gap | **batched** per step: sum into 2D, broadcast once, with a row index |
| extra state | none | 2D scratch array, per-step Gaussian factors, row index |
| best when | few tracks per time step | dense pulses, large grids |

Neither is "the reference" in a correctness sense — they check each other. The
independent physics check is `tests/test_single_track_vs_jaffe.py`, which
compares the single-track limit against analytic Jaffe theory.

## 10. Memory

| allocation | size | lifetime |
|---|---|---|
| four carrier arrays | `4 · no_xy² · no_z_with_buffer · 8 B` | whole run |
| arrival-time draw | `2 · n_tracks · 8 B` | before the run starts |
| `total_density` scratch | `no_xy² · 8 B` | whole run (batched backend) |
| per-step Gaussian factors | `2 · m · w · 8 B` | one step |

The first two are what matter, and they peak at different times — the schedule
is built and discarded before the carrier arrays are touched — so the estimate
is the larger of the two, not the sum. It counts allocations this code makes,
not the interpreter, Numba's runtime or NumPy's transient temporaries: on the
full-electrode run the estimate was 1.80 GiB against a measured peak RSS of
2.02 GiB, 12 % higher. The default 0.8 budget absorbs that margin. `SimulationConfig` computes this as
`estimated_memory_bytes` and refuses to build a config that would exceed
`memory_budget_fraction` (default 0.8) of available RAM, because the
alternative is discovering the problem via the OOM killer twenty minutes into a
run. Set it to `None` to opt out.

Thread requests are clamped the same way: `clamp_thread_count` reduces
`num_threads` to the process's CPU affinity mask and Numba's configured
maximum, warning when it does — so a benchmark cannot report a thread count
that never existed.

---

## 11. Complexity summary

With `T` time steps, `N` tracks per pulse, `w` stencil width, `X = no_xy`,
`Z = no_z`:

| | cost |
|---|---|
| deposition, unbatched | `N · w² · Z` |
| deposition, batched | `N · w²  +  T · X² · Z` |
| Lax-Wendroff sweep | `T · X² · Z` |
| memory | `X² · Z` |

Because `N ∝ dose · radius²` and `X ∝ radius`, the batched total scales as
`radius²` — the stencil is what removes the `radius⁴` term that unbatched,
untruncated deposition would otherwise contribute.

---

## 12. Where the parallelism is, and why it is race-free

Parallel regions are *within* a time step; the step loop itself is inherently
sequential, since step `n` reads what step `n−1` wrote.

- **Phase 1 (accumulate)** is `prange` over rows `i`. Iteration `i` writes only
  `total_density[i, :]`. Rows are disjoint, so no two iterations ever touch the
  same element — no locks, no atomics.
- **Phase 2 (broadcast)** and the **Lax-Wendroff sweep** are `prange` over the
  flattened `(i, j)` index. Each iteration writes only the `[i, j, :]` slice of
  the output arrays, and the sweep reads exclusively from the *current* arrays
  while writing exclusively to the *next* ones — so reads can never observe a
  partially-written neighbour.
- The scalar accumulators of §8 are ordinary `+=` reductions, which Numba
  recognises and privatises per thread. Their result can differ in the last few
  ULPs from the serial order, which is again why the tests use a relative
  tolerance.

Flattening `i·j` rather than parallelising `i` alone matters: `i` alone offers
only `no_xy` work items, which caps useful parallelism at a few dozen on the
grids used here.

**More threads is usually not the answer.** On small and medium grids each
parallel region does too little work to amortise its fork/join barrier, and
measured wall time gets *worse* with more threads. See
[PERFORMANCE.md](PERFORMANCE.md) §6 — the short version is that independent
single-threaded replicas are the right way to use many cores, and threading
only pays on grids large enough to be memory-bandwidth-bound.
