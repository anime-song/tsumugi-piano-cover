// 原曲の音源とカバー (ピアノの音源で鳴らす) を同じ時計で再生する。
// カバーは原曲の時間軸の上に生成しているので、同じ秒で並べればそのまま重なる。
// テイクを切り替えても再生位置はそのままなので、同じ所を聴き比べられる。
import { CacheStorage, SplendidGrandPiano, renderOffline, type Smplr } from "smplr";
import type { CoverNote } from "./api";

const LOOKAHEAD = 0.3; // 何秒先までの音を予約するか
const TICK_MS = 25;

export type PianoState = "loading" | "ready" | "fallback";

export class Player {
  readonly ctx: AudioContext;
  private originalGain: GainNode;
  private coverGain: GainNode;
  private piano: Smplr | null = null;
  pianoState: PianoState = "loading";
  pianoProgress = 0;

  original: AudioBuffer | null = null;
  originalUrl: string | null = null;
  originalLoading = false;
  private originalSource: AudioBufferSourceNode | null = null;

  notes: CoverNote[] = [];
  private notesEnd = 0;
  private nextIndex = 0;
  private fallbackVoices = new Set<OscillatorNode>();

  playing = false;
  private anchorCtx = 0;
  private anchorPos = 0;
  private pausedPos = 0;
  private timer: number | null = null;

  loop: [number, number] | null = null;
  loopOn = false;
  mix = 0.5; // 0 = 原曲だけ、1 = カバーだけ
  private extraDuration = 0;

  private listeners = new Set<() => void>();
  version = 0;

  constructor() {
    this.ctx = new AudioContext({ latencyHint: "interactive" });
    this.originalGain = this.ctx.createGain();
    this.coverGain = this.ctx.createGain();
    this.originalGain.connect(this.ctx.destination);
    this.coverGain.connect(this.ctx.destination);
    this.setMix(this.mix);
    this.loadPiano();
  }

  private loadPiano() {
    try {
      const piano = SplendidGrandPiano(this.ctx, {
        destination: this.coverGain,
        storage: new CacheStorage("cover-studio-piano"),
        onLoadProgress: (p) => {
          this.pianoProgress = p.total ? p.loaded / p.total : 0;
          this.emit();
        },
      });
      this.piano = piano;
      piano.ready.then(
        () => {
          this.pianoState = "ready";
          this.emit();
        },
        (e) => this.useFallback(e),
      );
    } catch (e) {
      this.useFallback(e);
    }
  }

  private useFallback(reason: unknown) {
    console.warn("ピアノ音源を読めませんでした", reason);
    // ピアノの音源 (ネットから取る) が読めなければ、簡単な音で鳴らす
    this.piano = null;
    this.pianoState = "fallback";
    this.emit();
  }

  // ------------------------------------------------------------ 購読
  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };
  getVersion = () => this.version;
  private emit() {
    this.version++;
    for (const l of this.listeners) l();
  }

  // ------------------------------------------------------------ 中身
  async setOriginal(url: string | null) {
    if (url === this.originalUrl) return;
    const wasPlaying = this.playing;
    const pos = this.position();
    if (wasPlaying) this.pause();
    this.originalUrl = url;
    this.original = null;
    this.emit();
    if (!url) return;
    this.originalLoading = true;
    this.emit();
    try {
      const res = await fetch(url);
      const buffer = await this.ctx.decodeAudioData(await res.arrayBuffer());
      if (this.originalUrl === url) this.original = buffer;
    } catch (e) {
      console.warn("音源を読めませんでした", e);
    } finally {
      if (this.originalUrl === url) this.originalLoading = false;
      this.emit();
    }
    if (wasPlaying && this.originalUrl === url) this.play(pos);
  }

  setNotes(notes: CoverNote[]) {
    if (notes === this.notes) return;
    const sorted = [...notes].sort((a, b) => a[0] - b[0]);
    this.notesEnd = sorted.reduce((m, n) => Math.max(m, n[4]), 0);
    if (this.playing) {
      const pos = this.position();
      this.stopSound();
      this.notes = sorted;
      this.start(pos);
    } else {
      this.notes = sorted;
    }
    this.emit();
  }

  /** 原曲もカバーもないときに、ピアノロールで見せている長さ (原曲の MIDI など) */
  setExtraDuration(seconds: number) {
    this.extraDuration = seconds;
    this.emit();
  }

  get duration(): number {
    return Math.max(this.original?.duration ?? 0, this.notesEnd, this.extraDuration);
  }

  position(): number {
    if (!this.playing) return this.pausedPos;
    return Math.max(0, this.anchorPos + (this.ctx.currentTime - this.anchorCtx));
  }

  // ------------------------------------------------------------ 操作
  play(from?: number) {
    if (this.playing) this.stopSound();
    void this.ctx.resume();
    let pos = from ?? this.pausedPos;
    if (pos >= this.duration - 0.05) pos = this.loopOn && this.loop ? this.loop[0] : 0;
    this.start(pos);
    this.emit();
  }

  pause() {
    if (!this.playing) return;
    this.pausedPos = this.position();
    this.stopSound();
    this.playing = false;
    this.emit();
  }

  toggle() {
    if (this.playing) this.pause();
    else this.play();
  }

  seek(pos: number) {
    pos = Math.max(0, Math.min(pos, this.duration));
    if (this.playing) {
      this.stopSound();
      this.start(pos);
    } else {
      this.pausedPos = pos;
    }
    this.emit();
  }

  setMix(mix: number) {
    this.mix = mix;
    const t = this.ctx.currentTime;
    this.originalGain.gain.setTargetAtTime(Math.min(1, 2 * (1 - mix)), t, 0.02);
    this.coverGain.gain.setTargetAtTime(Math.min(1, 2 * mix), t, 0.02);
    this.emit();
  }

  setLoop(loop: [number, number] | null) {
    this.loop = loop;
    this.loopOn = loop != null;
    this.emit();
  }

  toggleLoop() {
    if (!this.loop) return;
    this.loopOn = !this.loopOn;
    this.emit();
  }

  // ------------------------------------------------------------ 予約
  private start(pos: number) {
    this.playing = true;
    this.anchorCtx = this.ctx.currentTime + 0.05;
    this.anchorPos = pos;
    if (this.original && pos < this.original.duration) {
      const source = this.ctx.createBufferSource();
      source.buffer = this.original;
      source.connect(this.originalGain);
      source.start(this.anchorCtx, pos);
      this.originalSource = source;
    }
    this.nextIndex = lowerBound(this.notes, pos);
    this.timer = window.setInterval(this.tick, TICK_MS);
    this.tick();
  }

  private stopSound() {
    if (this.timer != null) window.clearInterval(this.timer);
    this.timer = null;
    if (this.originalSource) {
      try {
        this.originalSource.stop();
      } catch {
        // まだ始まっていない
      }
      this.originalSource.disconnect();
      this.originalSource = null;
    }
    this.piano?.stop();
    for (const osc of this.fallbackVoices) {
      try {
        osc.stop();
      } catch {
        // すでに止まっている
      }
    }
    this.fallbackVoices.clear();
  }

  private tick = () => {
    const pos = this.position();
    if (this.loopOn && this.loop && pos >= this.loop[1]) {
      this.stopSound();
      this.start(this.loop[0]);
      return;
    }
    if (pos >= this.duration + 0.3) {
      this.stopSound();
      this.playing = false;
      this.pausedPos = 0;
      this.emit();
      return;
    }
    const horizon = pos + LOOKAHEAD;
    const loopEnd = this.loopOn && this.loop ? this.loop[1] : Infinity;
    const now = this.ctx.currentTime;
    while (this.nextIndex < this.notes.length && this.notes[this.nextIndex][0] < horizon) {
      const note = this.notes[this.nextIndex++];
      if (note[0] >= loopEnd) continue;
      const when = this.anchorCtx + (note[0] - this.anchorPos);
      if (when < now - 0.03) continue;
      this.playNote(note, Math.max(when, now));
    }
  };

  private playNote(note: CoverNote, when: number) {
    const duration = Math.max(0.05, note[4] - note[0]);
    if (this.piano && this.pianoState === "ready") {
      this.piano.start({ note: note[2], velocity: note[3], time: when, duration });
      return;
    }
    if (this.pianoState === "loading") return;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = "triangle";
    osc.frequency.value = 440 * Math.pow(2, (note[2] - 69) / 12);
    const peak = 0.12 * Math.pow(note[3] / 127, 1.5);
    gain.gain.setValueAtTime(0, when);
    gain.gain.linearRampToValueAtTime(peak, when + 0.005);
    gain.gain.setTargetAtTime(0, when + 0.005, 0.6);
    gain.gain.setTargetAtTime(0, when + duration, 0.08);
    osc.connect(gain).connect(this.coverGain);
    osc.start(when);
    osc.stop(when + duration + 0.5);
    this.fallbackVoices.add(osc);
    osc.onended = () => this.fallbackVoices.delete(osc);
  }
}

function lowerBound(notes: CoverNote[], time: number): number {
  let lo = 0;
  let hi = notes.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (notes[mid][0] < time) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

let instance: Player | null = null;
export function getPlayer(): Player {
  instance ??= new Player();
  return instance;
}

/** カバーをピアノの音源で WAV に書き出す (ブラウザの中で、実時間より速く作る) */
export async function downloadCoverWav(notes: CoverNote[], filename: string) {
  const end = notes.reduce((m, n) => Math.max(m, n[4]), 0);
  const result = await renderOffline(
    async (ctx) => {
      const piano = SplendidGrandPiano(ctx, { storage: new CacheStorage("cover-studio-piano") });
      await piano.ready;
      for (const n of notes) {
        piano.start({ note: n[2], velocity: n[3], time: n[0], duration: Math.max(0.05, n[4] - n[0]) });
      }
    },
    { duration: end + 2, sampleRate: 44100 },
  );
  result.downloadWav16(filename);
}
