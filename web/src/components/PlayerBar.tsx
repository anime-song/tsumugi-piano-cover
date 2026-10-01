// 画面の下に固定する再生バー。シークバーには原曲の波形とカバーの音の多さを重ねて描く
import { useEffect, useMemo, useRef, useSyncExternalStore } from "react";
import { useT } from "../i18n";
import { formatTime } from "../params";
import type { Player } from "../player";
import { Icon } from "./Icon";

type Props = { player: Player; title: string; subtitle: string };

export function usePlayerState(player: Player) {
  useSyncExternalStore(player.subscribe, player.getVersion);
  return player;
}

export function PlayerBar({ player, title, subtitle }: Props) {
  usePlayerState(player);
  const t = useT();
  const timeRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    let raf = 0;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      if (timeRef.current) timeRef.current.textContent = formatTime(player.position());
    };
    loop();
    return () => cancelAnimationFrame(raf);
  }, [player]);

  const pianoNote =
    player.pianoState === "loading"
      ? t.pianoLoading(Math.round(player.pianoProgress * 100))
      : player.pianoState === "fallback"
        ? t.pianoFallback
        : null;

  return (
    <footer className="player">
      <div className="player-info">
        <div className="player-title">{title}</div>
        <div className="player-sub">{pianoNote ?? subtitle}</div>
      </div>
      <div className="player-main">
        <div className="player-controls">
          <button className="icon-btn" onClick={() => player.seek(player.position() - 5)} title={t.back5}>
            <Icon name="back" />
          </button>
          <button className="play-btn" onClick={() => player.toggle()} title={t.playPause}>
            <Icon name={player.playing ? "pause" : "play"} />
          </button>
          <button className="icon-btn" onClick={() => player.seek(player.position() + 5)} title={t.forward5}>
            <Icon name="forward" />
          </button>
          <button
            className={`icon-btn ${player.loopOn ? "on" : ""}`}
            onClick={() => player.toggleLoop()}
            disabled={!player.loop}
            title={t.loopHelp}
          >
            <Icon name="loop" />
          </button>
        </div>
        <div className="player-seek">
          <span className="mono" ref={timeRef}>
            0:00
          </span>
          <Overview player={player} />
          <span className="mono">{formatTime(player.duration)}</span>
        </div>
      </div>
      <div className="player-mix">
        <span className={player.original ? "" : "muted"}>{t.original}</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={player.mix}
          onChange={(e) => player.setMix(Number(e.target.value))}
          onDoubleClick={() => player.setMix(0.5)}
          aria-label={t.mixLabel}
          title={t.mixHelp}
        />
        <span>{t.cover}</span>
      </div>
    </footer>
  );
}

function Overview({ player }: { player: Player }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragging = useRef(false);
  usePlayerState(player);

  // 原曲の波形のピーク (横 600 区間)
  const peaks = useMemo(() => {
    const buffer = player.original;
    if (!buffer) return null;
    const data = buffer.getChannelData(0);
    const bins = 600;
    const out = new Float32Array(bins);
    const step = Math.max(1, Math.floor(data.length / bins));
    for (let i = 0; i < bins; i++) {
      let m = 0;
      const start = i * step;
      for (let j = start; j < Math.min(start + step, data.length); j += 16) m = Math.max(m, Math.abs(data[j]));
      out[i] = m;
    }
    return out;
  }, [player.original]);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * dpr)) canvas.width = Math.round(w * dpr);
      if (canvas.height !== Math.round(h * dpr)) canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const style = getComputedStyle(canvas);
      const duration = player.duration || 1;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = style.getPropertyValue("--seek-bg");
      ctx.fillRect(0, 0, w, h);
      if (peaks && player.original) {
        ctx.fillStyle = style.getPropertyValue("--seek-wave");
        const scale = player.original.duration / duration;
        for (let i = 0; i < peaks.length; i++) {
          const x = (i / peaks.length) * w * scale;
          const ph = Math.max(1, peaks[i] * h * 0.9);
          ctx.fillRect(x, (h - ph) / 2, Math.max(1, (w * scale) / peaks.length - 0.5), ph);
        }
      }
      // カバーの音の多さ (1 秒ごと)
      const notes = player.notes;
      if (notes.length) {
        const bins = Math.max(1, Math.ceil(duration));
        const counts = new Uint16Array(bins);
        for (const n of notes) counts[Math.min(bins - 1, Math.floor(n[0]))]++;
        let max = 1;
        for (const c of counts) max = Math.max(max, c);
        ctx.fillStyle = style.getPropertyValue("--accent");
        ctx.globalAlpha = 0.75;
        for (let i = 0; i < bins; i++) {
          if (!counts[i]) continue;
          const ch = (counts[i] / max) * (h * 0.45);
          ctx.fillRect((i / duration) * w, h - ch, Math.max(1, w / duration - 0.5), ch);
        }
        ctx.globalAlpha = 1;
      }
      if (player.loop) {
        ctx.fillStyle = style.getPropertyValue("--loop");
        const [a, b] = player.loop;
        ctx.fillRect((a / duration) * w, 0, ((b - a) / duration) * w, h);
      }
      const x = (player.position() / duration) * w;
      ctx.fillStyle = style.getPropertyValue("--playhead");
      ctx.fillRect(x - 1, 0, 2, h);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [player, peaks]);

  const seekAt = (clientX: number) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    player.seek(((clientX - rect.left) / rect.width) * player.duration);
  };

  return (
    <canvas
      ref={canvasRef}
      className="overview"
      onPointerDown={(e) => {
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
        dragging.current = true;
        seekAt(e.clientX);
      }}
      onPointerMove={(e) => dragging.current && seekAt(e.clientX)}
      onPointerUp={() => (dragging.current = false)}
    />
  );
}
