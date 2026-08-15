// SI-prefix helpers, ported from pulsed_ion_chamber/plots.py's `_scale`:
// pick the power-of-ten divisor that puts a series' peak in [1, 1000).

const SI_PREFIXES: [number, string][] = [
  [1e12, "T"],
  [1e9, "G"],
  [1e6, "M"],
  [1e3, "k"],
  [1, ""],
];

export function siScale(values: readonly number[]): { divisor: number; prefix: string } {
  let peak = 0;
  for (const v of values) {
    const abs = Math.abs(v);
    if (abs > peak) peak = abs;
  }
  for (const [divisor, prefix] of SI_PREFIXES) {
    if (peak >= divisor) return { divisor, prefix: prefix ? `${prefix} ` : "" };
  }
  return { divisor: 1, prefix: "" };
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
