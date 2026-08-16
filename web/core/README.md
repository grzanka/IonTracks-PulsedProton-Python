# `pulsed-ion-chamber-wasm`

Rust port of the two hot loops in `pulsed_ion_chamber` (the drift-diffusion-
recombination solver), compiled to WebAssembly via `wasm-bindgen`/`wasm-pack`
for the browser prototype in `web/`. Deposition is batched (`solver.rs`'s
module docs) rather than a direct port of the simpler per-track backend this
crate started with -- unbatched deposition re-walks the whole electrode gap
for every track, which stopped being cheap once wider radii were allowed. See
[issue #6](https://github.com/grzanka/IonTracks-PulsedProton-Python/issues/6)
for the feasibility writeup this implements the first milestone of.

## What this narrows relative to the Python package

This is a deliberately scoped-down port, not a 1:1 translation -- see
`src/lib.rs`'s module docs for the full list. In short:

- **Single averaged carrier species only** (`config.py`'s default). The
  two-Kanai-species option (`docs/PHYSICS.md` sec. 8) isn't ported.
- `lateral_boundary` is always `"reflecting"`, `scoring_region` always
  `"track_disc"`, `n_pulses` always `1`, `particle` always `"proton"`.
- `buffer_radius`, `no_z_electrode`, `track_cutoff_sigmas`,
  `repetition_rate_hz` and `n_clearance_separation_times` are fixed
  constants -- cost-model levers, not physics intuition (see
  `docs/PERFORMANCE.md` in the repo root), hidden so the UI's inputs can't
  accidentally leave the browser's time/RAM budget through those knobs.
  `grid_size_um` _is_ exposed (default 10 µm, matching the library default).
- Hard, independent safety ceilings (`config::MAX_*`) bound memory (a real
  byte count), CPU time (`MAX_TOTAL_VOXEL_STEPS` -- `total_time_steps *
voxels`, not raw grid size; see that constant's doc comment for why and
  for the measured ~8.2 ns/voxel-step this crate's cost model is calibrated
  from), track count, and step count -- regardless of which input parameter
  drove a config there. Defense in depth, matching `SimulationConfig`'s own
  "refuse rather than OOM twenty minutes in" philosophy. The Worker also
  offers an on-device empirical measurement (run the real backend for a few
  seconds, extrapolate) alongside the instant analytical guess -- the wasm
  analogue of `benchmark.estimate_full_runtime_empirical`.
- The track-density cross-section (`solver::Simulation::track_density_xy`, a
  full `no_xy * no_xy` field) is accumulated but read only once, at run
  completion -- not streamed every ~150 ms like the four scalar time series
  (`n+`, `n-`, `injected`, `recombination`) the browser's live plots need.

## Validation

`cargo test` runs two kinds of check:

1. **Closed-form, RNG-independent**: `theory::tests::jaffe_ks_matches_python_reference`
   checks this crate's `jaffe_ks`/`Ei` implementation against a value computed
   directly from the repo's Python `theory.jaffe_ks` (mpmath, 50 digits).
   `solver::tests::single_track_matches_jaffe_theory` then reproduces
   `tests/test_single_track_vs_jaffe.py`'s exact scenario (single track, three
   seeds) and checks the _simulated_ recombination against that analytic
   reference, at the same 30%-relative tolerance the Python test uses --
   this crate's `sampler` module seeds a different PRNG than NumPy's (see its
   module docs), so exact reproduction isn't the goal; physical correctness
   is.
2. **Cross-checks against documented numbers**: `config::tests` checks that
   this crate's grid sizing and track counts match the tier table in
   `docs/BENCHMARKS-LAPTOP.md` for the `dev`/`archive` radii (within a few
   percent -- this crate fixes `air_density_kg_m3` at the library default
   rather than the archived AIC-144 campaign's 20°C value, a known, small,
   deliberate difference).

## Building

```bash
rustup target add wasm32-unknown-unknown   # once
cargo install wasm-pack                     # once, or use the installer script below
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh -s -- -y

cargo test                                  # native, fast
cargo clippy --all-targets -- -D warnings
cargo fmt --check

wasm-pack build --target web --release --out-dir ../src/lib/wasm-core/pkg
```

The last command is also `pnpm run build:wasm` from `web/` -- that's what
`web/package.json`'s `predev`/`prebuild` hooks call, so `pnpm dev` / `pnpm build`
from `web/` do this automatically.
