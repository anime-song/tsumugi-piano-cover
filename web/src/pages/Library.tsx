// 曲の一覧と、新しい曲の追加
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, type Config, type ProjectSummary } from "../api";
import { Icon } from "../components/Icon";
import { NewSongForm, songArt, songStatus } from "../components/NewSong";
import { useT } from "../i18n";

export function Library({ config, navigate }: { config: Config; navigate: (to: string) => void }) {
  const qc = useQueryClient();
  const t = useT();
  const projects = useQuery({
    queryKey: ["projects"],
    queryFn: api.projects,
    refetchInterval: (q) => (q.state.data?.some((p) => p.busy) ? 2000 : false),
  });
  const [error, setError] = useState<string | null>(null);

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
      <NewSongForm config={config} onCreated={(pid) => navigate(`/p/${encodeURIComponent(pid)}`)} />
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
              <div className="muted small">{songStatus(p, t)}</div>
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
