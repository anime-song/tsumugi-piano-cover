// テイクの一覧 (新しい順) と、選んだテイクの詳細
import { useEffect, useState } from "react";
import type { Config, Params, Take } from "../api";
import { progressText, useT } from "../i18n";
import { ALL_SLIDERS, diffChips, formatTime, formatValue } from "../params";
import { takeLabel } from "./CreatePanel";
import { Icon } from "./Icon";

type CardProps = {
  take: Take;
  defaults: Params;
  selected: boolean;
  playing: boolean;
  continueLabel: string | null;
  onSelect: () => void;
  onPlay: () => void;
  onFavorite: () => void;
  onReuse: () => void;
  onCancel: () => void;
  onDelete: () => void;
};

/** seed から決まる色の組 (テイクの見分け用) */
export function artStyle(take: Take): React.CSSProperties {
  const h = (take.seed * 137 + take.number * 53) % 360;
  return {
    background: `linear-gradient(135deg, hsl(${h} 85% 62%), hsl(${(h + 55) % 360} 80% 45%))`,
  };
}

export function TakeCard(props: CardProps) {
  const { take } = props;
  const t = useT();
  const active = take.state === "queued" || take.state === "running";
  const chips = diffChips(take.params, props.defaults, t);
  return (
    <div
      className={`take ${props.selected ? "selected" : ""} ${take.state}`}
      onClick={props.onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && props.onSelect()}
    >
      <button
        className="take-art"
        style={artStyle(take)}
        disabled={take.state !== "done"}
        onClick={(e) => {
          e.stopPropagation();
          props.onPlay();
        }}
        title={take.state === "done" ? t.play : undefined}
      >
        {take.state === "done" ? (
          <Icon name={props.playing ? "pause" : "play"} size={20} />
        ) : active ? (
          <span className="spinner" />
        ) : (
          <Icon name="close" size={18} />
        )}
      </button>
      <div className="take-body">
        <div className="take-head">
          <span className="take-name">{takeLabel(take, t)}</span>
          {take.name && <span className="muted small">#{take.number}</span>}
          {take.count > 1 && (
            <span className="muted small" title={t.batchHelp}>
              {take.batch_index + 1}/{take.count}
            </span>
          )}
        </div>
        <div className="chips">
          {props.continueLabel && <span className="chip continue">✂ {props.continueLabel}</span>}
          {chips.length ? (
            chips.map((c) => (
              <span className="chip" key={c}>
                {c}
              </span>
            ))
          ) : (
            <span className="chip plain">{t.defaultSettings}</span>
          )}
        </div>
        {active ? (
          <div className="take-progress">
            <div className="bar">
              <div style={{ width: `${Math.round((take.progress ?? 0) * 100)}%` }} />
            </div>
            <span className="muted small">
              {take.state === "queued" ? t.queued : (progressText(t, take.message) ?? t.generating)}
            </span>
          </div>
        ) : take.state === "done" ? (
          <div className="take-meta muted small">
            {formatTime(take.duration ?? 0)} · {t.notes(take.notes)} · seed {take.seed} · {shortModel(take.model.label)}
          </div>
        ) : (
          <div className="take-meta error small">{take.state === "cancelled" ? t.stopped : take.error}</div>
        )}
      </div>
      <div className="take-actions" onClick={(e) => e.stopPropagation()}>
        {active ? (
          <button className="icon-btn" onClick={props.onCancel} title={t.stop}>
            <Icon name="stop" size={16} />
          </button>
        ) : (
          <>
            <button
              className={`icon-btn star ${take.favorite ? "on" : ""}`}
              onClick={props.onFavorite}
              title={t.favorite}
            >
              <Icon name="star" size={17} filled={take.favorite} />
            </button>
            <button className="icon-btn" onClick={props.onReuse} title={t.reuseHelp}>
              <Icon name="reuse" size={17} />
            </button>
            <button className="icon-btn danger" onClick={props.onDelete} title={t.delete}>
              <Icon name="trash" size={17} />
            </button>
          </>
        )}
      </div>
    </div>
  );
}

export function shortModel(label: string): string {
  return label.replace(/\/best\.pt/, "").replace(/ \(Hugging Face\)$/, "");
}

type DetailProps = {
  take: Take;
  config: Config;
  position: () => number;
  midiUrl: string;
  onRename: (name: string) => void;
  onMemo: (memo: string) => void;
  onFavorite: () => void;
  onReuse: () => void;
  onContinue: (seconds: number) => void;
  onWav: () => Promise<void>;
  onDelete: () => void;
};

export function TakeDetail(props: DetailProps) {
  const { take, config } = props;
  const t = useT();
  const [name, setName] = useState(take.name);
  const [memo, setMemo] = useState(take.memo);
  const [rendering, setRendering] = useState(false);
  useEffect(() => {
    setName(take.name);
    setMemo(take.memo);
  }, [take.id, take.name, take.memo]);
  const defaults = config.defaults;
  const done = take.state === "done";

  return (
    <div className="detail">
      <div className="detail-head">
        <div className="detail-art" style={artStyle(take)} />
        <div className="detail-title">
          <input
            className="name-input"
            value={name}
            placeholder={t.take(take.number)}
            onChange={(e) => setName(e.target.value)}
            onBlur={() => name !== take.name && props.onRename(name)}
            onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
          />
          <div className="muted small">
            {new Date(take.created).toLocaleString(t.lang)} · {shortModel(take.model.label)}
          </div>
        </div>
        <button className={`icon-btn star ${take.favorite ? "on" : ""}`} onClick={props.onFavorite} title={t.favorite}>
          <Icon name="star" size={20} filled={take.favorite} />
        </button>
      </div>

      <div className="detail-actions">
        <button className="btn" onClick={props.onReuse}>
          <Icon name="reuse" size={16} /> {t.reuse}
        </button>
        <button
          className="btn"
          disabled={!done}
          onClick={() => props.onContinue(props.position())}
          title={t.continueHereHelp}
        >
          <Icon name="scissors" size={16} /> {t.continueHere}
        </button>
        <a className={`btn ${done ? "" : "disabled"}`} href={done ? props.midiUrl : undefined} download>
          <Icon name="download" size={16} /> MIDI
        </a>
        <button
          className="btn"
          disabled={!done || rendering}
          onClick={async () => {
            setRendering(true);
            try {
              await props.onWav();
            } finally {
              setRendering(false);
            }
          }}
          title={t.wavHelp}
        >
          <Icon name="download" size={16} /> {rendering ? t.wavRendering : t.wav}
        </button>
        <button className="btn danger" onClick={props.onDelete} title={t.delete}>
          <Icon name="trash" size={16} />
        </button>
      </div>

      <table className="params">
        <tbody>
          <tr>
            <th>{t.seed}</th>
            <td className="mono">{take.seed}</td>
          </tr>
          <tr>
            <th>{t.length}</th>
            <td>{take.params.seconds == null ? t.fullLength : t.chipSeconds(take.params.seconds)}</td>
          </tr>
          <tr className={take.params.channel ? "changed" : ""}>
            <th>{t.performer}</th>
            <td>{take.params.channel ? `#${take.params.channel}` : t.notSpecified}</td>
          </tr>
          {ALL_SLIDERS.map((def) => {
            const value = take.params[def.key];
            const changed = Math.abs(value - defaults[def.key]) > 1e-9;
            const off =
              (def.key === "channel_cfg" && !take.params.channel) ||
              (!take.params.arrangement && ["fill", "above", "span"].includes(def.key));
            return (
              <tr key={def.key} className={`${changed ? "changed" : ""} ${off ? "off" : ""}`}>
                <th>{t.params[def.key].label}</th>
                <td className="mono">
                  {formatValue(value, def.step)}
                  {changed && <span className="muted"> {t.defaultValue(formatValue(defaults[def.key], def.step))}</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <label className="memo">
        <span className="muted small">{t.memo}</span>
        <textarea
          value={memo}
          placeholder={t.memoPlaceholder}
          onChange={(e) => setMemo(e.target.value)}
          onBlur={() => memo !== take.memo && props.onMemo(memo)}
          rows={3}
        />
      </label>
    </div>
  );
}
