// 採譜中に確定したノートを、曲全体のピアノロールに順に描く。ステムは 1 つずつ左から右へ採譜されていく。
// 原曲の音源を再生していれば再生位置も出し、クリックでそこへ移る
import { useEffect, useRef, useState } from "react";
import { ApiError, api, type LiveProgress, type LiveStem } from "../api";
import { useT } from "../i18n";
import type { Player } from "../player";
import { CANVAS_FONT } from "./PianoRoll";

// [start, end, pitch, final]
type LiveNote = [number, number, number, boolean];

const STEM_COLORS: Record<string, string> = {
  vocals: "--src-melody",
  bass: "--src-bass",
  piano: "--src-keys",
  guitar: "--src-guitar",
  other: "--src-other",
};
const stemColor = (stem: string) => STEM_COLORS[stem] ?? "--src-other";

export function LiveTranscription({ pid, player, active }: { pid: string; player: Player; active: boolean }) {
  const t = useT();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const notes = useRef(new Map<string, { stem: string; note: LiveNote }>());
  const seq = useRef(0);
  const [stems, setStems] = useState<LiveStem[]>([]);
  const [stage, setStage] = useState<string | null>(null);
  const [progress, setProgress] = useState<LiveProgress | null>(null);
  const [unsupported, setUnsupported] = useState(false);
  const stemsRef = useRef<LiveStem[]>([]);
  stemsRef.current = stems;

  // 1 秒ごとに、前回の続きのノートだけを取りに行く
  useEffect(() => {
    let stopped = false;
    let timer = 0;
    const poll = async () => {
      try {
        const live = await api.transcribeLive(pid, seq.current);
        if (stopped) return;
        for (const [id, stem, start, end, pitch, final] of live.events) {
          notes.current.set(id, { stem, note: [start, end, pitch, !!final] });
        }
        seq.current = live.seq;
        setStems(live.stems);
        setStage(live.stage);
        setProgress(live.progress);
        if (!live.active && !active) return;
      } catch (e) {
        // 古いサーバ (この API がない) なら出さない。それ以外は一時的なものとして次で取り直す
        if (e instanceof ApiError && e.status === 404) {
          setUnsupported(true);
          return;
        }
      }
      if (!stopped) timer = window.setTimeout(poll, 1000);
    };
    void poll();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [pid, active]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
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
      const color = (name: string) => style.getPropertyValue(name).trim();

      const stemList = stemsRef.current;
      const duration = Math.max(
        1,
        player.original?.duration ?? 0,
        ...stemList.map((s) => s.duration),
      );
      const lo = 24;
      const hi = 100;
      const top = 4;
      const bottom = h - 16;
      const rowH = (bottom - top) / (hi - lo + 1);
      const xOf = (s: number) => (s / duration) * w;

      ctx.fillStyle = color("--roll-bg");
      ctx.fillRect(0, 0, w, h);
      // 30 秒ごとの目盛り
      ctx.fillStyle = color("--roll-grid");
      ctx.font = `10px ${CANVAS_FONT}`;
      for (let s = 30; s < duration; s += 30) ctx.fillRect(Math.round(xOf(s)), top, 1, bottom - top);
      ctx.fillStyle = color("--roll-text");
      ctx.textBaseline = "bottom";
      for (let s = 0; s < duration; s += 60) ctx.fillText(`${Math.floor(s / 60)}:00`, xOf(s) + 3, h - 2);

      const palette = new Map(Object.values(STEM_COLORS).map((name) => [name, color(name)]));
      for (const { stem, note } of notes.current.values()) {
        const [start, end, pitch, final] = note;
        if (pitch < lo || pitch > hi) continue;
        ctx.globalAlpha = final ? 0.9 : 0.5;
        ctx.fillStyle = palette.get(stemColor(stem))!;
        ctx.fillRect(xOf(start), top + (hi - pitch) * rowH, Math.max(1.5, xOf(end) - xOf(start)), Math.max(1.5, rowH));
      }
      ctx.globalAlpha = 1;

      // 採譜中のステムの、どこまで終わったか
      const current = stemList.find((s) => !s.done);
      if (current && active) {
        const x = xOf(current.pos);
        ctx.fillStyle = color(stemColor(current.stem));
        ctx.globalAlpha = 0.12;
        ctx.fillRect(x, 0, w - x, h);
        ctx.globalAlpha = 1;
        ctx.fillRect(x - 1, 0, 2, h);
      }
      if (player.original) {
        ctx.fillStyle = color("--playhead");
        ctx.fillRect(xOf(player.position()) - 1, 0, 2, h);
      }
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [player, active, unsupported]);

  if (unsupported) return null;
  const stageNames = t.transcribeStages as Record<string, string>;
  const stemNames = t.stems as Record<string, string>;
  // 分離の進み具合は、採譜が終わったあと (失敗・中止も含む) に残らないようにする。
  // jobs.py 側は最後の状態をそのまま返すので、出すのは実行中だけ
  const showProgress = active && progress !== null && progress.total > 0;

  return (
    <div className="live">
      <div className="live-stages">
        {(["separate", "transcribe", "refine", "velocity", "beat"] as const).map((s, i, all) => {
          const index = stage ? all.indexOf(stage as (typeof all)[number]) : -1;
          const state = !active && stage ? "done" : i < index ? "done" : i === index ? "now" : "";
          return (
            <span key={s} className={`live-stage ${state}`}>
              {stageNames[s]}
              {/* 今の段階に進み具合があれば、塊の数も出す (ステム分離が長いので) */}
              {i === index && showProgress && (
                <span className="live-stage-count">
                  {progress.done}/{progress.total}
                </span>
              )}
            </span>
          );
        })}
      </div>
      {/* 高さは常に確保する (出たり消えたりで下のピアノロールが動かないように) */}
      <div
        className={`live-bar ${showProgress ? "" : "hidden"}`}
        role="progressbar"
        aria-hidden={!showProgress}
        aria-valuemin={0}
        aria-valuemax={showProgress ? progress.total : 1}
        aria-valuenow={showProgress ? progress.done : 0}
      >
        <i
          style={{
            width: showProgress ? `${Math.min(100, Math.round((progress.done / progress.total) * 100))}%` : "0%",
          }}
        />
      </div>
      <canvas
        ref={canvasRef}
        className="live-roll"
        onPointerDown={(e) => {
          if (!player.original) return;
          const rect = canvasRef.current!.getBoundingClientRect();
          const duration = Math.max(player.original.duration, ...stems.map((s) => s.duration));
          player.seek(((e.clientX - rect.left) / rect.width) * duration);
        }}
      />
      <div className="live-stems">
        {stems.map((s) => (
          <span key={s.stem} className={`live-stem ${s.done ? "done" : ""}`}>
            <i style={{ background: `var(${stemColor(s.stem)})` }} />
            {stemNames[s.stem] ?? s.stem}
            <span className="mono">
              {s.done ? "✓" : `${Math.round((s.pos / Math.max(1, s.duration)) * 100)}%`}
            </span>
          </span>
        ))}
        <span className="muted small live-note">{t.liveNote}</span>
      </div>
    </div>
  );
}
