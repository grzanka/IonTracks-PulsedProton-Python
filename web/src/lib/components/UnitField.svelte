<script lang="ts">
  // A number-with-units input in the style of APTG/dedx_web's calculator:
  // type a bare number (interpreted in the selected unit) or a number with
  // an inline unit suffix ("100 keV"), and/or click a unit button to switch
  // units without retyping the number. See $lib/units.ts for the parser.
  import { untrack } from "svelte";
  import { fromBase, formatForInput, parseValueWithUnit, toBase, type UnitDef } from "$lib/units";

  interface Props {
    label: string;
    units: UnitDef[];
    value: number; // bindable, always in the table's base unit
    defaultUnitSymbol?: string;
    min?: number; // base unit
    max?: number; // base unit
    hint?: string;
    // Bindable: true whenever the displayed text failed to parse or is out
    // of range, so a rejected edit leaves `value` at its last-valid number
    // (see handleInput) while still being visible to the parent -- without
    // this, +page.svelte has no way to know the field it is about to run
    // with doesn't match what's on screen (issue #19 W3).
    invalid?: boolean;
  }

  let {
    label,
    units,
    value = $bindable(),
    defaultUnitSymbol,
    min,
    max,
    hint,
    // The rule can't see that this default is read externally, at mount, by
    // whichever parent binds `invalid`; it isn't a dead store.
    // eslint-disable-next-line no-useless-assignment
    invalid = $bindable(false),
  }: Props = $props();

  function findUnit(symbol: string | undefined): UnitDef {
    const found = symbol ? units.find((u) => u.symbol === symbol) : undefined;
    if (found) return found;
    const fallback = units[0];
    if (!fallback) throw new Error("UnitField requires a non-empty units list");
    return fallback;
  }

  // Seeded once from props at mount, then locally owned -- untrack() makes
  // that "once" explicit rather than an accidental non-reactive read.
  let selected = $state(untrack(() => findUnit(defaultUnitSymbol)));
  let raw = $state(untrack(() => formatForInput(fromBase(value, selected))));
  let error = $state("");

  function rangeMessage(unit: UnitDef): string {
    if (min !== undefined && max !== undefined) {
      return `must be between ${formatForInput(fromBase(min, unit))} and ${formatForInput(fromBase(max, unit))} ${unit.symbol}`;
    }
    if (min !== undefined)
      return `must be at least ${formatForInput(fromBase(min, unit))} ${unit.symbol}`;
    if (max !== undefined)
      return `must be at most ${formatForInput(fromBase(max, unit))} ${unit.symbol}`;
    return "";
  }

  function handleInput(event: Event): void {
    raw = (event.target as HTMLInputElement).value;
    const result = parseValueWithUnit(raw, units);
    if ("empty" in result) {
      error = "required";
      invalid = true;
      return;
    }
    if ("error" in result) {
      error = result.error;
      invalid = true;
      return;
    }
    const unit = result.unit ?? selected;
    if (result.unit) selected = result.unit; // typing "100 keV" adopts keV as the active unit
    const base = toBase(result.value, unit);
    if ((min !== undefined && base < min) || (max !== undefined && base > max)) {
      error = rangeMessage(unit);
      invalid = true;
      return;
    }
    error = "";
    invalid = false;
    value = base;
  }

  function selectUnit(unit: UnitDef): void {
    if (unit.symbol === selected.symbol) return;
    selected = unit;
    // Reformat the last *valid* value in the newly selected unit -- if the
    // field currently holds invalid text, switching units doesn't try to
    // reinterpret it, same as dedx_web leaves an invalid row's text alone.
    if (!error) raw = formatForInput(fromBase(value, selected));
  }

  function focusIndex(index: number, group: HTMLElement): void {
    const buttons = group.querySelectorAll<HTMLButtonElement>('[role="radio"]');
    buttons[index]?.focus();
  }

  function handleGroupKeydown(event: KeyboardEvent): void {
    const currentIndex = units.findIndex((u) => u.symbol === selected.symbol);
    const group = event.currentTarget as HTMLElement;
    let targetIndex: number | undefined;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      targetIndex = (currentIndex + 1) % units.length;
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      targetIndex = (currentIndex - 1 + units.length) % units.length;
    }
    if (targetIndex === undefined) return;
    const target = units[targetIndex];
    if (!target) return;
    event.preventDefault();
    selectUnit(target);
    focusIndex(targetIndex, group);
  }
</script>

<div class="unit-field">
  <span class="field-label">{label}</span>
  <div class="input-row">
    <input
      type="text"
      inputmode="decimal"
      class:invalid={!!error}
      value={raw}
      oninput={handleInput}
      aria-label={label}
      aria-invalid={!!error}
    />
    {#if units.length > 1}
      <div
        class="unit-group"
        role="radiogroup"
        tabindex="-1"
        aria-label={`Unit for ${label}`}
        onkeydown={handleGroupKeydown}
      >
        {#each units as unit (unit.symbol)}
          <button
            type="button"
            role="radio"
            aria-checked={unit.symbol === selected.symbol}
            tabindex={unit.symbol === selected.symbol ? 0 : -1}
            class:selected={unit.symbol === selected.symbol}
            onclick={() => selectUnit(unit)}
          >
            {unit.symbol}
          </button>
        {/each}
      </div>
    {:else}
      <span class="unit-fixed">{selected.symbol}</span>
    {/if}
  </div>
  {#if error}
    <span class="field-error">{error}</span>
  {:else if hint}
    <span class="field-hint">{hint}</span>
  {/if}
</div>

<style>
  .unit-field {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }

  .field-label {
    font-size: 0.85rem;
    color: #334155;
  }

  .input-row {
    display: flex;
    gap: 0.4rem;
    align-items: center;
  }

  input[type="text"] {
    width: 6rem;
    padding: 0.4rem 0.5rem;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    font-size: 0.9rem;
    font-variant-numeric: tabular-nums;
  }

  input[type="text"]:focus {
    outline: 2px solid #2563eb;
    outline-offset: 1px;
  }

  input[type="text"].invalid {
    border-color: #dc2626;
    background: #fef2f2;
  }

  .unit-group {
    display: inline-flex;
    gap: 2px;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 2px;
  }

  .unit-group button {
    border: 1px solid transparent;
    background: transparent;
    border-radius: 4px;
    padding: 0.3rem 0.5rem;
    font-size: 0.78rem;
    color: #64748b;
    cursor: pointer;
    line-height: 1;
  }

  .unit-group button.selected {
    background: #eff6ff;
    border-color: #2563eb;
    color: #1d4ed8;
    font-weight: 600;
  }

  .unit-fixed {
    font-size: 0.8rem;
    color: #64748b;
    padding: 0.3rem 0.4rem;
  }

  .field-error {
    font-size: 0.78rem;
    color: #b91c1c;
  }

  .field-hint {
    font-size: 0.78rem;
    color: #64748b;
  }
</style>
