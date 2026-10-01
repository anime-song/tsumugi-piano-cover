// 上の帯の曲名。押すと曲の一覧が開き、そこから別の曲へ移ったり新しい曲を足したりできる
import { useQuery } from "@tanstack/react-query";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { api } from "../api";
import { useT } from "../i18n";
import { Icon } from "./Icon";
import { songArt, songStatus } from "./NewSong";

type Props = {
  pid: string;
  onSelect: (pid: string) => void;
  onNew: () => void;
  onAll: () => void;
};

export function SongSwitcher({ pid, onSelect, onNew, onAll }: Props) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const root = useRef<HTMLDivElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.projects });
  const current = projects.data?.find((p) => p.id === pid);

  // メニューは幅 340px あるので、狭い画面ではそのままだと右にはみ出す。開いたときに
  // 画面内へ収まる左位置を計算して入れる (幅の上限は styles.css の max-width)
  useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const el = menu.current;
      const box = root.current?.getBoundingClientRect();
      const anchor = button.current?.getBoundingClientRect();
      if (!el || !box || !anchor) return;
      const vw = document.documentElement.clientWidth;
      const margin = 8;
      const left = Math.min(Math.max(margin, anchor.left), Math.max(margin, vw - el.offsetWidth - margin));
      el.style.left = `${Math.round(left - box.left)}px`;
    };
    place();
    window.addEventListener("resize", place);
    return () => window.removeEventListener("resize", place);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    void projects.refetch();
    const onDown = (e: MouseEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const q = query.trim().toLowerCase();
  const shown = (projects.data ?? []).filter((p) => !q || p.title.toLowerCase().includes(q));

  return (
    <div className="switcher" ref={root}>
      <button ref={button} className={`switcher-btn ${open ? "open" : ""}`} onClick={() => setOpen(!open)} title={t.switchSong}>
        <span className="switcher-art" style={songArt(current?.title ?? pid)} />
        <span className="switcher-title">{current?.title ?? pid}</span>
        <Icon name="chevron" size={14} />
      </button>
      {open && (
        <div className="switcher-menu" ref={menu}>
          <button
            className="switcher-new"
            onClick={() => {
              setOpen(false);
              onNew();
            }}
          >
            <span className="switcher-plus">
              <Icon name="plus" size={14} />
            </span>
            {t.newSong}
          </button>
          {(projects.data?.length ?? 0) > 6 && (
            <input
              className="switcher-search"
              autoFocus
              value={query}
              placeholder={t.searchSongs}
              onChange={(e) => setQuery(e.target.value)}
            />
          )}
          <div className="switcher-list">
            {shown.map((p) => (
              <button
                key={p.id}
                className={`switcher-item ${p.id === pid ? "current" : ""}`}
                onClick={() => {
                  setOpen(false);
                  if (p.id !== pid) onSelect(p.id);
                }}
              >
                <span className="switcher-art" style={songArt(p.title)} />
                <span className="switcher-item-body">
                  <span className="switcher-item-title">{p.title}</span>
                  <span className="muted small">{songStatus(p, t)}</span>
                </span>
              </button>
            ))}
            {shown.length === 0 && <div className="muted small switcher-empty">{t.noMatch}</div>}
          </div>
          <button
            className="switcher-all link"
            onClick={() => {
              setOpen(false);
              onAll();
            }}
          >
            {t.allSongs}
          </button>
        </div>
      )}
    </div>
  );
}
