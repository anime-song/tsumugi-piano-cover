// 生成の設定の並びと範囲 (意味は piano_cover/generate.py の引数の説明と同じ。文言は i18n.ts)
import type { Params } from "./api";
import type { useT } from "./i18n";

type Dict = ReturnType<typeof useT>;

export type NumberKey = Exclude<keyof Params, "arrangement" | "seconds" | "channel" | "style">;

export type SliderDef = { key: NumberKey; min: number; max: number; step: number };

export type GroupId = "source" | "expression" | "arrangement" | "sampling";

export type Group = { id: GroupId; sliders: SliderDef[] };

export const GROUPS: Group[] = [
  {
    id: "source",
    sliders: [
      { key: "source_cfg", min: 1, max: 4, step: 0.05 },
      { key: "onset_bias", min: 0, max: 8, step: 0.5 },
    ],
  },
  {
    id: "expression",
    sliders: [
      { key: "dynamics", min: 0, max: 4, step: 0.1 },
      { key: "density", min: 0, max: 3, step: 0.1 },
    ],
  },
  {
    id: "arrangement",
    sliders: [
      { key: "fill", min: -2, max: 4, step: 0.1 },
      { key: "above", min: -2, max: 3, step: 0.1 },
      { key: "span", min: -2, max: 3, step: 0.1 },
    ],
  },
  {
    id: "sampling",
    sliders: [
      { key: "temperature", min: 0.5, max: 1.5, step: 0.05 },
      { key: "top_p", min: 0.5, max: 1, step: 0.01 },
    ],
  },
];

export const CHANNEL_CFG: SliderDef = { key: "channel_cfg", min: 1, max: 5, step: 0.1 };

export const ALL_SLIDERS: SliderDef[] = [...GROUPS.flatMap((g) => g.sliders), CHANNEL_CFG];

export const LENGTHS: (number | null)[] = [null, 30, 60, 90];

export function formatValue(value: number, step: number): string {
  const digits = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
  return value.toFixed(digits);
}

/** 演奏者の欄の表示 (好みを使ったテイクは好みの名前。消した好みなら styleName が null を返す) */
export function performerText(params: Params, t: Dict, styleName: (id: string) => string | null): string {
  if (params.style) return t.chipStyle(styleName(params.style) ?? t.deletedStyle);
  return params.channel ? `#${params.channel}` : t.notSpecified;
}

/** 既定値から変えた項目 (テイクの一覧に出す) */
export function diffChips(params: Params, defaults: Params, t: Dict, styleName: (id: string) => string | null): string[] {
  const chips: string[] = [];
  if (params.style) chips.push(t.chipStyle(styleName(params.style) ?? t.deletedStyle));
  else if (params.channel) chips.push(t.chipPerformer(params.channel));
  for (const def of ALL_SLIDERS) {
    if (def.key === "channel_cfg" && !params.channel && !params.style) continue;
    if (!params.arrangement && ["fill", "above", "span"].includes(def.key)) continue;
    const value = params[def.key];
    if (Math.abs(value - defaults[def.key]) > 1e-9) chips.push(`${t.params[def.key].short} ${formatValue(value, def.step)}`);
  }
  if (!params.arrangement) chips.push(t.chipNoArrangement);
  if (params.seconds != null) chips.push(t.chipSeconds(params.seconds));
  return chips;
}

export function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
