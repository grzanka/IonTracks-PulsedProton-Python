// Text-input-with-units parsing, in the style of APTG/dedx_web's calculator
// (energy-parser.ts / available-units.ts): a value field accepts either a
// bare number (interpreted in the field's currently selected unit) or a
// number with an inline unit suffix, e.g. "100 keV" -- the same regex shape,
// extended to allow the micro sign (µ) for µm/µs.
//
// Each UnitDef converts to a fixed *base* unit (documented per table below),
// which is exactly what pulsed_ion_chamber_wasm's estimate()/WasmSimulation
// API expects -- unit handling stays entirely in this TS layer, the same
// separation dedx_web keeps between its unit-aware UI and the unit-agnostic
// WASM core.

export interface UnitDef {
  symbol: string;
  toBase: number;
}

export interface ParsedValue {
  value: number;
  unit: UnitDef | null; // null: no unit suffix was typed, caller's currently-selected unit applies
}
export interface ParseError {
  error: string;
}
export interface EmptyInput {
  empty: true;
}
export type ParseResult = ParsedValue | ParseError | EmptyInput;

const NUMBER_UNIT_RE = /^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Zµ/]+)?$/;

export function parseValueWithUnit(raw: string, units: readonly UnitDef[]): ParseResult {
  const trimmed = raw.trim();
  if (trimmed === "") return { empty: true };

  const match = trimmed.match(NUMBER_UNIT_RE);
  if (!match) return { error: "not a number" };

  const numberStr = match[1];
  const unitStr = match[2];
  const value = numberStr === undefined ? NaN : parseFloat(numberStr);
  if (Number.isNaN(value)) return { error: "not a number" };
  if (value <= 0) return { error: "must be positive" };
  if (!unitStr) return { value, unit: null };

  // Case-sensitive on purpose -- "Gy/s" vs "gy/s" ambiguity is exactly the
  // kind of silent mis-scale dedx_web's parser refuses to guess at.
  const unit = units.find((u) => u.symbol === unitStr);
  if (!unit) {
    const known = units.map((u) => u.symbol).join(", ");
    return { error: `unknown unit "${unitStr}" (expected one of ${known})` };
  }
  return { value, unit };
}

export function toBase(value: number, unit: UnitDef): number {
  return value * unit.toBase;
}

export function fromBase(value: number, unit: UnitDef): number {
  return value / unit.toBase;
}

/** Trim float noise for display without inventing false precision. */
export function formatForInput(value: number): string {
  if (!Number.isFinite(value)) return "";
  return Number(value.toPrecision(6)).toString();
}

// --- Unit tables -------------------------------------------------------
// Each table's base unit is what SimParams (wasm-core/loader.ts) expects.

export const ENERGY_UNITS: UnitDef[] = [
  // base: MeV
  { symbol: "keV", toBase: 1e-3 },
  { symbol: "MeV", toBase: 1 },
  { symbol: "GeV", toBase: 1e3 },
];

export const VOLTAGE_UNITS: UnitDef[] = [
  // base: V
  { symbol: "V", toBase: 1 },
  { symbol: "kV", toBase: 1e3 },
];

export const GAP_LENGTH_UNITS: UnitDef[] = [
  // base: cm
  { symbol: "mm", toBase: 0.1 },
  { symbol: "cm", toBase: 1 },
];

export const RADIUS_LENGTH_UNITS: UnitDef[] = [
  // base: cm
  { symbol: "µm", toBase: 1e-4 },
  { symbol: "mm", toBase: 0.1 },
  { symbol: "cm", toBase: 1 },
];

export const DOSE_RATE_UNITS: UnitDef[] = [
  // base: Gy/s
  { symbol: "Gy/s", toBase: 1 },
  { symbol: "Gy/min", toBase: 1 / 60 },
];

export const PULSE_TIME_UNITS: UnitDef[] = [
  // base: s
  { symbol: "µs", toBase: 1e-6 },
  { symbol: "ms", toBase: 1e-3 },
];

/**
 * PTW 23343 Markus chamber, full collecting electrode: r = 2.65 mm
 * (docs/PHYSICS.md sec. 7, docs/BENCHMARKS-LAPTOP.md's "full_electrode"
 * tier). This prototype's sampled column is deliberately a small fraction of
 * that -- see issue #6 sec. 1 -- so the UI reports the percentage rather
 * than leaving the reader to work out how small "0.08 mm" actually is.
 */
export const MARKUS_FULL_RADIUS_CM = 0.265;
