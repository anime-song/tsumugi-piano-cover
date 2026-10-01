// 曲の一覧と、新しい曲の追加 (音源だけなら tsumugi で採譜、MIDI も一緒なら採譜を飛ばす)
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api, type Config, type ProjectSummary } from "../api";
import { Icon } from "../components/Icon";
import { useT } from "../i18n";

const AUDIO = /\.(mp3|wav|flac|m4a|ogg|opus|aac)$/i;
const MIDI = /\.(mid|midi)$/i;

export function Library({ config, navigate }: { config: Config; navigate: (to: string) => void }) {
  const qc = useQueryClient();
  const t = useT();
  const projects = useQuery({
    queryKey: ["projects"],
    queryFn: api.projects,
    refetchInterval: (q) => (q.state.data?.some((p) => p.busy) ? 2000 : false),
  });
  const [audio, setAudio] = useState<File | null>(null);
  const [midi, setMidi] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const take = (files: FileList | File[]) => {
    for (const f of Array.from(files)) {
      if (AUDIO.test(f.name)) {
        setAudio(f);
        if (!title) setTitle(f.name.replace(/\.[^.]+$/, ""));
      } else if (MIDI.test(f.name)) {
        setMidi(f);
        if (!title && !audio) setTitle(f.name.replace(/\.[^.]+$/, ""));
      } else setError(t.unsupportedFile(f.name));
    }
  };

  const create = async () => {
    if (!audio && !midi) return;
    setBusy(true);
    setError(null);
    try {
      const form = new FormData();
      if (audio) form.append("audio", audio);
      if (midi) form.append("midi", midi);
      form.append("title", title);
      const detail = await api.createProject(form);
      await qc.invalidateQueries({ queryKey: ["projects"] });
      navigate(`/p/${encodeURIComponent(detail.id)}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (p: ProjectSummary) => {
    if (!confirm(t.confirmDeleteSong(p.title, p.takes))) return;
    try {
      await api.deleteProject(p.id);
      await qc.invalidateQueries({ queryKey: ["projects"] });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="library">
      <section
        className={`drop ${over ? "over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          take(e.dataTransfer.files);
        }}
      >
        <div className="drop-main" onClick={() => input.current?.click()}>
          <div className="drop-icon">
            <Icon name="plus" size={26} />
          </div>
          <div>
            <b>{t.newSong}</b>
            <div className="muted small">
              {t.dropHelp}
              {!config.tsumugi && t.noTsumugi}
            </div>
          </div>
          <input
            ref={input}
            type="file"
            multiple
            hidden
            accept=".mp3,.wav,.flac,.m4a,.ogg,.opus,.aac,.mid,.midi"
            onChange={(e) => {
              if (e.target.files) take(e.target.files);
              e.target.value = "";
            }}
          />
        </div>
        {(audio || midi) && (
          <div className="drop-form">
            <div className="files">
              {audio && (
                <span className="chip">
                  <Icon name="music" size={13} /> {audio.name}
                  <button onClick={() => setAudio(null)} aria-label={t.remove}>
                    <Icon name="close" size={12} />
                  </button>
                </span>
              )}
              {midi && (
                <span className="chip">
                  <Icon name="layers" size={13} /> {midi.name}
                  <button onClick={() => setMidi(null)} aria-label={t.remove}>
                    <Icon name="close" size={12} />
                  </button>
                </span>
              )}
            </div>
            <input className="title-input" value={title} onChange={(e) => setTitle(e.target.value)} placeholder={t.songTitle} />
            <button className="btn primary" disabled={busy} onClick={create}>
              {busy ? t.uploading : midi ? t.create : t.createAndTranscribe}
            </button>
          </div>
        )}
      </section>
      {error && (
        <div className="alert" onClick={() => setError(null)}>
          {error}
        </div>
      )}

      <div className="grid">
        {projects.data?.map((p) => (
          <a
            key={p.id}
            className="song-card"
            href={`/p/${encodeURIComponent(p.id)}`}
            onClick={(e) => {
              e.preventDefault();
              navigate(`/p/${encodeURIComponent(p.id)}`);
            }}
          >
            <div className="song-art" style={songArt(p.title)}>
              <Icon name="music" size={28} />
            </div>
            <div className="song-body">
              <div className="song-title">{p.title}</div>
              <div className="muted small">{status(p, t)}</div>
            </div>
            <button
              className="icon-btn danger"
              title={t.delete}
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                void remove(p);
              }}
            >
              <Icon name="trash" size={16} />
            </button>
          </a>
        ))}
        {projects.data?.length === 0 && <div className="muted">{t.noSongs}</div>}
      </div>
    </div>
  );
}

function status(p: ProjectSummary, t: ReturnType<typeof useT>): string {
  const state = p.transcribe?.state;
  if (!p.has_source) {
    if (state === "queued") return t.statusQueued;
    if (state === "running") return t.statusRunning;
    if (state === "error") return t.statusFailed;
    return t.statusNoSource;
  }
  const parts = [t.takesCount(p.takes)];
  if (p.favorites) parts.push(`★ ${p.favorites}`);
  if (p.busy) parts.push(t.generatingNow);
  return parts.join(" · ");
}

function songArt(title: string): React.CSSProperties {
  let h = 0;
  for (const c of title) h = (h * 31 + c.charCodeAt(0)) % 360;
  return { background: `linear-gradient(135deg, hsl(${h} 70% 55%), hsl(${(h + 40) % 360} 75% 35%))` };
}
