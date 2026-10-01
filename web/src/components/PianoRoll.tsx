// カバーの音 (と、薄く原曲の MIDI) をピアノロールで見せる。再生位置に合わせて流れる。
// クリックでその位置へ、ドラッグでループ区間、ホイールで横に動かす (Ctrl / ⌘ + ホイールで拡大縮小)。
import { useEffect, useRef, useState } from "react";
import type { CoverView, SourceView } from "../api";
import { useT } from "../i18n";
import { formatTime } from "../params";
import type { Player } from "../player";

type Props = {
  player: Player;
  cover: CoverView | null;
  source: SourceView | null;
  showSource: boolean;
  continueAt: number | null;
  height?: number;
};

const CHORD_H = 18;
const PEDAL_H = 10;
export const CANVAS_FONT = `system-ui, "Segoe UI", "Hiragino Sans", "Yu Gothic UI", "Noto Sans JP", sans-serif`;
const GROUP_COLORS = ["--src-melody", "--src-bass", "--src-keys", "--src-guitar", "--src-other"];

export function PianoRoll({ player, cover, source, showSource, continueAt, height = 300 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const view = useRef({ start: 0, seconds: 16 });
  const drag = useRef<{ x: number; t: number; moved: boolean } | null>(null);
  const t = useT();
  const [zoom, setZoom] = useState(16);
  const dataRef = useRef({ cover, source, showSource, continueAt });
  dataRef.current = { cover, source, showSource, continueAt };

  useEffect(() => {
    view.current.seconds = zoom;
  }, [zoom]);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;
    let colors: Record<string, string> = {};
    const readColors = () => {
      const style = getComputedStyle(canvas);
      const names = [
        "--roll-bg",
        "--roll-black-key",
        "--roll-grid",
        "--roll-bar",
        "--roll-text",
        "--accent",
        "--accent-2",
        "--playhead",
        "--loop",
        "--continue",
        ...GROUP_COLORS,
      ];
      colors = Object.fromEntries(names.map((n) => [n, style.getPropertyValue(n).trim()]));
    };
    readColors();
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", readColors);

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const { cover, source, showSource, continueAt } = dataRef.current;
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const pos = player.position();
      const v = view.current;
      // 再生中は再生位置が右端の 85% を越えたら次のページへ送る
      if (player.playing && (pos > v.start + v.seconds * 0.85 || pos < v.start)) {
        v.start = Math.max(0, pos - v.seconds * 0.1);
      }
      const t0 = v.start;
      const t1 = v.start + v.seconds;
      const xOf = (t: number) => ((t - t0) / v.seconds) * w;

      // 音域: 見えている音に合わせる (最低 2 オクターブ)
      let lo = 108;
      let hi = 21;
      const scan = (notes: number[][], pitchIndex: number) => {
        for (const n of notes) {
          if (n[pitchIndex] < lo) lo = n[pitchIndex];
          if (n[pitchIndex] > hi) hi = n[pitchIndex];
        }
      };
      if (cover) scan(cover.notes, 2);
      if (source && (showSource || !cover)) scan(source.notes, 2);
      if (hi < lo) {
        lo = 48;
        hi = 84;
      }
      lo = Math.max(21, lo - 2);
      hi = Math.min(108, hi + 2);
      if (hi - lo < 24) {
        const mid = (hi + lo) / 2;
        lo = Math.floor(mid - 12);
        hi = Math.ceil(mid + 12);
      }
      const top = CHORD_H;
      const bottom = h - PEDAL_H;
      const rowH = (bottom - top) / (hi - lo + 1);
      const yOf = (pitch: number) => top + (hi - pitch) * rowH;

      ctx.fillStyle = colors["--roll-bg"];
      ctx.fillRect(0, 0, w, h);
      // 黒鍵の段
      ctx.fillStyle = colors["--roll-black-key"];
      for (let p = lo; p <= hi; p++) {
        if ([1, 3, 6, 8, 10].includes(p % 12)) ctx.fillRect(0, yOf(p), w, rowH);
      }
      // 拍と小節
      if (source) {
        ctx.fillStyle = colors["--roll-grid"];
        for (const b of source.beats) if (b >= t0 && b <= t1) ctx.fillRect(Math.round(xOf(b)), top, 1, bottom - top);
        ctx.fillStyle = colors["--roll-bar"];
        for (const b of source.downbeats)
          if (b >= t0 && b <= t1) ctx.fillRect(Math.round(xOf(b)), top, 1, bottom - top);
      } else {
        ctx.fillStyle = colors["--roll-grid"];
        for (let s = Math.ceil(t0); s <= t1; s++) ctx.fillRect(Math.round(xOf(s)), top, 1, bottom - top);
      }
      // ループ区間
      if (player.loop) {
        const [a, b] = player.loop;
        ctx.globalAlpha = player.loopOn ? 1 : 0.4;
        ctx.fillStyle = colors["--loop"];
        ctx.fillRect(xOf(a), 0, xOf(b) - xOf(a), h);
        ctx.globalAlpha = 1;
      }
      // 原曲の MIDI (薄く)
      if (source && (showSource || !cover)) {
        ctx.globalAlpha = cover ? 0.45 : 0.85;
        for (const n of source.notes) {
          if (n[0] > t1) break;
          if (n[0] + n[1] < t0) continue;
          ctx.fillStyle = colors[GROUP_COLORS[n[4]] ?? "--src-other"];
          ctx.fillRect(xOf(n[0]), yOf(n[2]) + 1, Math.max(2, (n[1] / v.seconds) * w), Math.max(1, rowH - 2));
        }
        ctx.globalAlpha = 1;
      }
      // カバー: 鳴り終わり (ペダル) を薄く、押している長さを濃く
      if (cover) {
        for (const n of cover.notes) {
          if (n[0] > t1) break;
          if (n[4] < t0) continue;
          const x = xOf(n[0]);
          const y = yOf(n[2]);
          ctx.globalAlpha = 0.18;
          ctx.fillStyle = colors["--accent"];
          ctx.fillRect(x, y + 1, Math.max(2, ((n[4] - n[0]) / v.seconds) * w), Math.max(1, rowH - 2));
          ctx.globalAlpha = 0.35 + 0.65 * (n[3] / 127);
          const grad = ctx.createLinearGradient(x, 0, x + 40, 0);
          grad.addColorStop(0, colors["--accent"]);
          grad.addColorStop(1, colors["--accent-2"]);
          ctx.fillStyle = grad;
          ctx.fillRect(x, y, Math.max(2.5, (n[1] / v.seconds) * w), Math.max(2, rowH));
        }
        ctx.globalAlpha = 1;
        // ペダル
        ctx.fillStyle = colors["--accent"];
        ctx.globalAlpha = 0.5;
        for (const [a, b] of cover.pedals) {
          if (a > t1 || b < t0) continue;
          ctx.fillRect(xOf(a), h - PEDAL_H + 3, Math.max(1, xOf(b) - xOf(a) - 1), PEDAL_H - 5);
        }
        ctx.globalAlpha = 1;
      }
      // コード名
      ctx.fillStyle = colors["--roll-bg"];
      ctx.fillRect(0, 0, w, CHORD_H);
      if (source) {
        ctx.font = `11px ${CANVAS_FONT}`;
        ctx.fillStyle = colors["--roll-text"];
        ctx.textBaseline = "middle";
        let lastX = -Infinity;
        for (const [t, name] of source.chords) {
          if (t > t1) break;
          const x = xOf(t);
          if (x < -40 || x - lastX < 28) continue;
          ctx.fillText(name, x + 2, CHORD_H / 2);
          lastX = x;
        }
      }
      // 続きを作り直す位置
      if (continueAt != null && continueAt >= t0 && continueAt <= t1) {
        ctx.fillStyle = colors["--continue"];
        ctx.fillRect(xOf(continueAt) - 1, 0, 2, h);
      }
      // 再生位置
      const px = xOf(pos);
      if (px >= 0 && px <= w) {
        ctx.fillStyle = colors["--playhead"];
        ctx.fillRect(px - 1, 0, 2, h);
      }
      // 時刻の目盛り
      ctx.font = `10px ${CANVAS_FONT}`;
      ctx.fillStyle = colors["--roll-text"];
      ctx.textBaseline = "bottom";
      const every = v.seconds > 40 ? 10 : v.seconds > 16 ? 5 : 2;
      for (let s = Math.ceil(t0 / every) * every; s <= t1; s += every) {
        ctx.fillText(formatTime(s), xOf(s) + 3, bottom - 2);
      }
    };
    draw();
    return () => {
      cancelAnimationFrame(raf);
      media.removeEventListener("change", readColors);
    };
  }, [player]);

  // Ctrl + ホイールでページごと拡大しないよう、passive でない listener で受ける
  useEffect(() => {
    const canvas = canvasRef.current!;
    const onWheel = (e: WheelEvent) => {
      const v = view.current;
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const t = timeAt(e.clientX);
        const next = Math.min(120, Math.max(4, v.seconds * Math.exp(e.deltaY * 0.002)));
        v.start = Math.max(0, t - ((t - v.start) * next) / v.seconds);
        setZoom(next);
      } else {
        const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        if (delta) e.preventDefault();
        v.start = Math.max(0, v.start + (delta / 800) * v.seconds);
      }
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, []);

  function timeAt(clientX: number) {
    const rect = canvasRef.current!.getBoundingClientRect();
    return view.current.start + ((clientX - rect.left) / rect.width) * view.current.seconds;
  }

  return (
    <div className="roll">
      <canvas
        ref={canvasRef}
        style={{ height }}
        onPointerDown={(e) => {
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
          drag.current = { x: e.clientX, t: timeAt(e.clientX), moved: false };
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (!d) return;
          if (Math.abs(e.clientX - d.x) > 4) d.moved = true;
          if (d.moved) {
            const t = timeAt(e.clientX);
            player.setLoop([Math.max(0, Math.min(d.t, t)), Math.max(d.t, t)]);
          }
        }}
        onPointerUp={(e) => {
          const d = drag.current;
          drag.current = null;
          if (!d) return;
          if (!d.moved) {
            player.seek(timeAt(e.clientX));
          } else if (player.loop && player.loop[1] - player.loop[0] < 0.5) {
            player.setLoop(null);
          } else if (player.loop) {
            player.seek(player.loop[0]);
          }
        }}
      />
      <div className="roll-zoom">
        <span>{t.view}</span>
        <input
          type="range"
          min={4}
          max={120}
          step={1}
          value={zoom}
          onChange={(e) => setZoom(Number(e.target.value))}
          aria-label={t.viewSeconds}
        />
        <span className="mono">{Math.round(zoom)} {t.sec}</span>
      </div>
    </div>
  );
}
