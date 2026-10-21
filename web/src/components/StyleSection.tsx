// 「演奏者」の欄の中の、お気に入りから学習した好み (演奏者の埋め込みを 1 本だけ学習したもの)
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, type ModelEntry, type Style } from "../api";
import { progressText, useT } from "../i18n";
import { Icon } from "./Icon";

const active = (s: Style) => s.state === "queued" || s.state === "running";

export function useStyles() {
  return useQuery({
    queryKey: ["styles"],
    queryFn: api.styles,
    refetchInterval: (q) => (q.state.data?.styles.some(active) ? 1000 : false),
  });
}

/** テイクの一覧・詳細に出す好みの名前 (消した好みなら null) */
export function useStyleName(): (id: string) => string | null {
  const styles = useStyles().data?.styles ?? [];
  return (id) => styles.find((s) => s.id === id)?.name ?? null;
}

type Props = {
  model: ModelEntry | undefined;
  selected: string | null;
  onSelect: (id: string | null) => void;
};

export function StyleSection({ model, selected, onSelect }: Props) {
  const t = useT();
  const qc = useQueryClient();
  const data = useStyles().data;
  const [error, setError] = useState<string | null>(null);
  if (!data || !model) return null;

  const styles = data.styles.filter((s) => s.model.id === model.id);
  const favorites = data.favorites.filter((f) => f.audio);
  const songs = new Set(favorites.map((f) => f.project)).size;
  const training = data.styles.some(active);
  const selectedStyle = selected ? data.styles.find((s) => s.id === selected) : undefined;
  const unusable = selected != null && !(selectedStyle?.state === "done" && selectedStyle.model.id === model.id);

  const run = async (fn: () => Promise<unknown>) => {
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
    void qc.invalidateQueries({ queryKey: ["styles"] });
  };
  const rename = (s: Style) => {
    const name = prompt(t.styleName, s.name);
    if (name && name.trim() && name.trim() !== s.name) void run(() => api.renameStyle(s.id, name.trim()));
  };
  const remove = (s: Style) => {
    if (!confirm(t.confirmDeleteStyle(s.name))) return;
    if (selected === s.id) onSelect(null);
    void run(() => api.deleteStyle(s.id));
  };

  return (
    <div className="styles">
      <div className="styles-title">
        <Icon name="star" size={13} /> {t.styleTitle}
      </div>
      <div className="group-note">{t.styleNote}</div>
      {styles.some((s) => s.state === "done") && (
        <div className="presets">
          {styles
            .filter((s) => s.state === "done")
            .map((s) => (
              <span
                key={s.id}
                className={`preset ${selected === s.id ? "on" : ""}`}
                title={t.styleInfo(s.used ?? 0, s.loss_before ?? 0, s.loss_after ?? 0)}
              >
                <button onClick={() => onSelect(s.id)} onDoubleClick={() => rename(s)}>
                  {s.name}
                </button>
                <button
                  className="preset-remove"
                  title={t.deleteStyle(s.name)}
                  aria-label={t.deleteStyle(s.name)}
                  onClick={() => remove(s)}
                >
                  <Icon name="close" size={11} />
                </button>
              </span>
            ))}
        </div>
      )}
      {unusable && (
        <div className="style-row error small">
          {t.styleUnusable}
          <button className="link" onClick={() => onSelect(null)}>
            {t.cancel}
          </button>
        </div>
      )}
      {styles
        .filter((s) => s.state !== "done")
        .map((s) =>
          active(s) ? (
            <div className="style-row take-progress" key={s.id}>
              <div className="bar">
                <div style={{ width: `${Math.round((s.progress ?? 0) * 100)}%` }} />
              </div>
              <span className="muted small">
                {s.state === "queued" ? t.styleQueued : (progressText(t, s.message) ?? t.styleTraining)}
              </span>
              {s.job && (
                <button className="icon-btn" onClick={() => void run(() => api.cancelJob(s.job!))} title={t.stop}>
                  <Icon name="stop" size={14} />
                </button>
              )}
            </div>
          ) : (
            <div className="style-row small" key={s.id}>
              <span className={s.state === "error" ? "error" : "muted"}>
                {s.name}: {s.state === "error" ? (s.error ?? t.styleFailed) : t.styleCancelled}
              </span>
              <button className="icon-btn" onClick={() => remove(s)} title={t.deleteStyle(s.name)}>
                <Icon name="trash" size={14} />
              </button>
            </div>
          ),
        )}
      <button
        className="btn small"
        disabled={!favorites.length || training}
        title={t.learnStyleHelp}
        onClick={() => void run(() => api.createStyle({ name: "", model: model.id }))}
      >
        <Icon name="sparkle" size={14} /> {favorites.length ? t.learnStyle(favorites.length, songs) : t.noStyleFavorites}
      </button>
      {error && <div className="error small">{error}</div>}
    </div>
  );
}
