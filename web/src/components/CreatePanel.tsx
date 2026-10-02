// 左の「作る」欄。設定はブラウザに覚えておき (曲をまたいで共通)、テイクの「設定を使う」で戻せる
import { useState } from "react";
import type { Config, ContinueFrom, Params, Take } from "../api";
import { useT } from "../i18n";
import { CHANNEL_CFG, GROUPS, LENGTHS, formatTime, formatValue, type SliderDef } from "../params";
import { Icon } from "./Icon";

// よく使う演奏者の番号 (テンプレート)。ブラウザに覚え、最初はこの 2 人
const PERFORMERS_KEY = "cover-studio:performers";
const DEFAULT_PERFORMERS = [227, 1088];

function loadPerformers(): number[] {
  try {
    const saved = JSON.parse(localStorage.getItem(PERFORMERS_KEY) ?? "null");
    if (Array.isArray(saved) && saved.every((n) => Number.isInteger(n) && n > 0)) return saved;
  } catch {
    // 覚えておけない環境では最初の 2 人
  }
  return DEFAULT_PERFORMERS;
}

function savePerformers(list: number[]) {
  try {
    localStorage.setItem(PERFORMERS_KEY, JSON.stringify(list));
  } catch {
    // 覚えておけなくても選べる
  }
}

export type Draft = {
  params: Params;
  model: string | null;
  count: number;
  seedLocked: boolean;
  seed: number | null;
};

type Props = {
  config: Config;
  draft: Draft;
  onChange: (draft: Draft) => void;
  continueFrom: ContinueFrom | null;
  continueTake: Take | null;
  onClearContinue: () => void;
  onGenerate: () => void;
  generating: boolean;
  disabled: string | null; // 生成できない理由
};

export function CreatePanel(props: Props) {
  const { config, draft, onChange, disabled } = props;
  const t = useT();
  const [performers, setPerformersState] = useState(loadPerformers);
  const setPerformers = (list: number[]) => {
    setPerformersState(list);
    savePerformers(list);
  };
  const defaults = config.defaults;
  const params = draft.params;
  const set = (patch: Partial<Params>) => onChange({ ...draft, params: { ...params, ...patch } });
  const model = config.models.find((m) => m.id === draft.model) ?? config.models[0];
  const maxChannel = model?.num_channels ? model.num_channels - 1 : undefined;
  const changed = Object.keys(defaults).some(
    (k) => JSON.stringify(params[k as keyof Params]) !== JSON.stringify(defaults[k as keyof Params]),
  );

  const slider = (def: SliderDef, disabledSlider = false) => {
    const value = params[def.key];
    const isDefault = Math.abs(value - defaults[def.key]) < 1e-9;
    const text = t.params[def.key];
    return (
      <div className={`slider ${disabledSlider ? "disabled" : ""}`} key={def.key}>
        <div className="slider-head">
          <label htmlFor={`p-${def.key}`}>
            {text.label}
            {!isDefault && (
              <button
                className="reset-dot"
                title={t.resetTo(formatValue(defaults[def.key], def.step))}
                onClick={() => set({ [def.key]: defaults[def.key] } as Partial<Params>)}
              />
            )}
          </label>
          <span className={`slider-value mono ${isDefault ? "" : "changed"}`}>{formatValue(value, def.step)}</span>
        </div>
        <input
          id={`p-${def.key}`}
          type="range"
          min={def.min}
          max={def.max}
          step={def.step}
          value={value}
          disabled={disabledSlider}
          onChange={(e) => set({ [def.key]: Number(e.target.value) } as Partial<Params>)}
          onDoubleClick={() => set({ [def.key]: defaults[def.key] } as Partial<Params>)}
          style={{ "--fill": `${((value - def.min) / (def.max - def.min)) * 100}%` } as React.CSSProperties}
        />
        <div className="slider-help">{text.help}</div>
      </div>
    );
  };

  return (
    <aside className="create">
      <div className="create-scroll">
        <div className="panel-title">
          <span>
            <Icon name="sparkle" /> {t.createTitle}
          </span>
          {changed && (
            <button className="link" onClick={() => onChange({ ...draft, params: { ...defaults } })}>
              {t.resetAll}
            </button>
          )}
        </div>

        {props.continueFrom && (
          <div className="continue-card">
            <Icon name="scissors" />
            <div>
              <b>
                {t.continueUntil(
                  props.continueTake ? takeLabel(props.continueTake, t) : props.continueFrom.take,
                  formatTime(props.continueFrom.seconds),
                )}
              </b>
              {t.continueRest}
            </div>
            <button className="icon-btn" onClick={props.onClearContinue} title={t.cancel}>
              <Icon name="close" size={16} />
            </button>
          </div>
        )}

        <div className="field">
          <label htmlFor="model">{t.model}</label>
          <select id="model" value={model?.id ?? ""} onChange={(e) => onChange({ ...draft, model: e.target.value })}>
            {(["export", "checkpoint", "hub"] as const).map((kind) => {
              const items = config.models.filter((m) => m.kind === kind);
              if (!items.length) return null;
              return (
                <optgroup label={t.modelKinds[kind]} key={kind}>
                  {items.map((m) => (
                    <option value={m.id} key={m.id}>
                      {m.label}
                      {config.loaded === m.id ? t.loaded : ""}
                    </option>
                  ))}
                </optgroup>
              );
            })}
          </select>
        </div>

        <div className="field">
          <label>{t.length}</label>
          <div className="segmented">
            {LENGTHS.map((value) => (
              <button
                key={String(value)}
                className={params.seconds === value ? "on" : ""}
                onClick={() => set({ seconds: value })}
              >
                {value == null ? t.fullSong : t.chipSeconds(value)}
              </button>
            ))}
          </div>
          <div className="slider-help">{t.lengthHelp}</div>
        </div>

        {GROUPS.map((group) => {
          const text = t.groups[group.id];
          return (
            <section className="group" key={group.id}>
              <div className="group-title">
                {text.title}
                {group.id === "arrangement" && (
                  <label className="switch" title={t.arrangementSwitch}>
                    <input
                      type="checkbox"
                      checked={params.arrangement}
                      onChange={(e) => set({ arrangement: e.target.checked })}
                    />
                    <span />
                  </label>
                )}
              </div>
              {text.note && <div className="group-note">{text.note}</div>}
              {group.sliders.map((def) => slider(def, group.id === "arrangement" && !params.arrangement))}
            </section>
          );
        })}

        <section className="group">
          <div className="group-title">{t.performer}</div>
          <div className="presets">
            <button className={`preset ${params.channel === 0 ? "on" : ""}`} onClick={() => set({ channel: 0 })}>
              {t.notSpecified}
            </button>
            {performers.map((n) => (
              <span key={n} className={`preset ${params.channel === n ? "on" : ""}`}>
                <button onClick={() => set({ channel: n })}>#{n}</button>
                <button
                  className="preset-remove"
                  title={t.removePerformer(n)}
                  aria-label={t.removePerformer(n)}
                  onClick={() => setPerformers(performers.filter((x) => x !== n))}
                >
                  <Icon name="close" size={11} />
                </button>
              </span>
            ))}
            {params.channel > 0 && !performers.includes(params.channel) && (
              <button
                className="preset add"
                onClick={() => setPerformers([...performers, params.channel])}
                title={t.addPerformerHelp}
              >
                <Icon name="plus" size={12} /> {t.addPerformer(params.channel)}
              </button>
            )}
          </div>
          <div className="field inline">
            <label htmlFor="channel">{t.performerNumber}</label>
            <input
              id="channel"
              type="number"
              min={0}
              max={maxChannel}
              value={params.channel}
              onChange={(e) => set({ channel: Math.max(0, Math.floor(Number(e.target.value) || 0)) })}
            />
            <span className="muted small">{t.performerHelp(maxChannel)}</span>
          </div>
          {slider(CHANNEL_CFG, !params.channel)}
        </section>
      </div>

      <div className="create-footer">
        <div className="footer-row">
          <div className="segmented small" title={t.countHelp}>
            {[1, 2, 3, 4].map((n) => (
              <button key={n} className={draft.count === n ? "on" : ""} onClick={() => onChange({ ...draft, count: n })}>
                ×{n}
              </button>
            ))}
          </div>
          <label className="seed" title={t.seedHelp}>
            <input
              type="checkbox"
              checked={draft.seedLocked}
              onChange={(e) =>
                onChange({
                  ...draft,
                  seedLocked: e.target.checked,
                  seed: draft.seed ?? Math.floor(Math.random() * 2 ** 31),
                })
              }
            />
            {t.seed}
            <input
              className="mono"
              type="number"
              value={draft.seed ?? ""}
              placeholder={t.random}
              disabled={!draft.seedLocked}
              onChange={(e) => onChange({ ...draft, seed: e.target.value === "" ? null : Number(e.target.value) })}
            />
          </label>
        </div>
        <button className="generate" disabled={!!disabled || props.generating} onClick={props.onGenerate}>
          <Icon name="sparkle" size={18} />
          {props.continueFrom ? t.regenerateRest : t.generate}
          <span className="count">×{draft.count}</span>
        </button>
        {disabled && <div className="footer-note">{disabled}</div>}
      </div>
    </aside>
  );
}

export function takeLabel(take: Take, t: ReturnType<typeof useT>): string {
  return take.name || t.take(take.number);
}
