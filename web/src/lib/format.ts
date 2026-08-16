// SI-prefix helpers, ported from pulsed_ion_chamber/plots.py's `_scale`:
// pick the power-of-ten divisor that puts a series' peak in [1, 1000).

const SI_PREFIXES: [number, string][] = [
  [1e12, "T"],
  [1e9, "G"],
  [1e6, "M"],
  [1e3, "k"],
  [1, ""],
];

/** The divisor/prefix pair that puts `peak` in [1, 1000). Split out from
 * `siScale` so a caller tracking a running peak incrementally (as
 * +page.svelte does for its live plots -- see issue #19 W7) never has to
 * rescan its full series just to pick a unit. */
export function scaleForPeak(peak: number): { divisor: number; prefix: string } {
  const abs = Math.abs(peak);
  for (const [divisor, prefix] of SI_PREFIXES) {
    if (abs >= divisor) return { divisor, prefix: prefix ? `${prefix} ` : "" };
  }
  return { divisor: 1, prefix: "" };
}

export function siScale(values: readonly number[]): { divisor: number; prefix: string } {
  let peak = 0;
  for (const v of values) {
    const abs = Math.abs(v);
    if (abs > peak) peak = abs;
  }
  return scaleForPeak(peak);
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes.toFixed(0)} B`;
}

export function formatCount(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

export function formatSeconds(seconds: number): string {
  if (seconds < 1) return "< 1 s";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds - minutes * 60);
  return `${minutes} min ${remainder}s`;
}
