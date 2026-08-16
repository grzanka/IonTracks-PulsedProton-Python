# pulsed-ion-chamber-web (prototype)

**Adjust a pulsed proton beam's energy, dose rate, voltage and pulse
duration, and watch an ionisation-chamber recombination simulation run --
live, entirely in your browser.** No install, no server, no data leaves your
machine: the physics (Rust, compiled to WebAssembly) and the plots all run
client-side.

This is the first prototype from
[issue #6](https://github.com/grzanka/IonTracks-PulsedProton-Python/issues/6),
which lays out why a browser port is feasible only for a deliberately capped
sub-volume, and what it narrows relative to the full
[`pulsed_ion_chamber`](../README.md) Python package. `web/core/README.md`
has the exact list.

## How it works

The UI is two phases, on purpose -- so nothing runs until you've seen what it
costs:

1. **Configure.** Seven number-with-units fields (energy, voltage, electrode
   gap, dose rate, pulse duration, sampled column radius, grid spacing) --
   type a bare number in the field's selected unit, or a number with an
   inline unit suffix ("100 keV"), and switch units on the fly with the
   button group next to each field (following
   [APTG/dedx_web](https://github.com/APTG/dedx_web)'s calculator). Every
   change instantly recomputes the grid size, track count, time-step count
   and estimated peak memory (no allocation yet -- this is the wasm analogue
   of the Python package's `--dry-run`) and checks it against this
   prototype's hard browser-safety limits -- the run's own random seed isn't
   exposed; a fresh one is drawn each time you click "Run simulation", since
   the statistical scatter between runs of the same config is real physics
   (`docs/PHYSICS.md` sec. 14), not something to hide behind a fixed default.
2. **Run.** Only once you click "Run simulation" does anything execute. It
   runs in a Web Worker (so the page stays responsive and Cancel actually
   works), streaming the injection-rate, carrier-evolution and
   recombination-rate plots back every ~150 ms of wall time as the run
   progresses. The track-density cross-section -- a full 2D field, not a
   scalar time series -- renders once, at completion.
3. **Extrapolate (optional).** `k_s` never fully converges within this
   prototype's capped column radius; once a run finishes, "Extrapolate to an
   infinite column" runs one more (short) simulation at a second radius and
   fits `k_∞` from the two, per `docs/PHYSICS.md` sec. 14's `1/r` law -- so
   the radius cap becomes a documented feature to correct for, not a silent
   limitation.

## Local development

```bash
pnpm install
pnpm dev          # rebuilds the wasm package first (predev hook), then vite dev
```

```bash
pnpm run build    # rebuilds wasm, then the static site -> build/
pnpm run preview  # serve that build locally
```

```bash
pnpm run check          # svelte-check + tsc
pnpm run lint           # eslint
pnpm run format:check   # prettier
```

The Rust crate in `core/` needs its own toolchain the first time -- see
`core/README.md`.

## Deployment

`.github/workflows/deploy.yml` publishes `web/` to this repo's GitHub Pages
site on every push to `master` that passes CI (`.github/workflows/ci.yml`).
Nothing server-side: it's a static build, `BASE_PATH`-adjusted for the
project-pages sub-path.
