"""拍の上の演奏 -> 楽譜: V2 のカバー (拍のトークンで、原曲の拍の上に生成したもの) を楽譜にするための入力の形。

V2 のカバーは、小節の区切りも、各音が小節の何拍目のどこにあるか (1 拍 12 刻み) も分かっている。そこで演奏の時刻を
「拍の番号 x BEAT_SECONDS」の仮の時刻 (拍の時刻) に置き換えて、今の演奏 -> 楽譜モデル (piano_perf2score.model) に入れる。
1 拍がいつも同じ長さになるので、テンポの探索 (倍・半分の取り違え) がいらず、小節の開始時刻も与えられる
(生成では MTIME を与えた小節線から決めて、モデルには予測させない。generate.transcribe_on_beats)。

学習の組 (BeatSynthWindowDataset):
    楽譜の窓 -> 合成演奏 (render.COVER_RENDER、実際のカバーの弾き方に合わせたもの) -> 楽譜の正しい拍で BeatTokenizer.quantize
    (V2 と同じ丸め) -> 拍の時刻の演奏 -> make_item
V2 の拍は SheetSage2-Refined の拍なので、楽譜の拍と階層が違うことがある (6/8 の 8 分を拍にする、4/4 の 8 分を拍にするなど)。
学習では LEVEL_PROBS の割合で、拍を楽譜の拍の 2 分割・3 分割・2 つまとめにした格子で丸める。

使う楽譜は select_songs で選ぶ (beat_songs.npz): 低品質の楽譜 (piano_score.quality.low_quality) を外し、音域
(音高の 5〜95%) が MIN_PITCH_SPAN 以上の、カバーに近い曲だけにする (local/probe_score_input_gap.py の測定)。
拍子より長い小節を含む窓は使わない。

カバーには手の情報がないので、メロディが中央の C あたりまで下がると左手の段に書いてしまう。そこで演奏の行にメロディの
印を付ける (モデルの melody_rows)。学習では楽譜の上の段の一番上の音、生成では原曲の SheetSage2-Refined のメロディと
同じ位置 (MELODY_TICKS 刻み以内)・同じ音名の音のうち一番高いもの (melody_from_annotation)。生成で印が外れても困らないよう、
学習では MELODY_DROP_SONG の窓で印をすべて消し、残りも MELODY_DROP_NOTE の音で消す。
音の長さ (V2 のカバーはペダルで響きを残して鍵盤を早く離す) をそのまま音価に写さないよう、学習では drop_durations の割合の
窓で長さの列を消す。生成では長さを消して入れる (generate.transcribe_on_beats)。

    python -m piano_perf2score.beat_input --cache-dir data/piano_score/synth     # beat_songs.npz を作る
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from piano_ar.beat_tokenizer import BeatGrid, BeatTokenizer
from piano_ar.tokenizer import KIND_NOTE, KIND_PEDAL_OFF, KIND_PEDAL_ON
from piano_score.data import ScoreCache
from piano_score.quality import low_quality
from piano_score.tokenizer import ScoreTokenizer

from .data import FRAMES_PER_SECOND, UNMARKED_TEMPO, PerformanceVocab, SynthWindowDataset, make_item
from .render import COVER_RENDER, Performance, _score_notes, beat_positions, measure_qpm, render

BEAT_SECONDS = 0.48  # 拍の時刻での 1 拍 (1 刻み = 4 フレーム)
MIN_PITCH_SPAN = 39  # カバーの音域の下位 25% (local/probe_score_input_gap.py)
# 丸める格子の拍: 楽譜の拍 1 つを何拍にするか。0.5 は 2 拍をまとめて 1 拍 (小節の拍の数が偶数のときだけ)
LEVEL_PROBS = {1: 0.7, 2: 0.15, 0.5: 0.1, 3: 0.05}
SONGS_FILE = "beat_songs.npz"
MELODY_TICKS = 2  # メロディの音と演奏の音を同じ位置とみなすずれ (刻み)
MELODY_DROP_SONG = 0.3
MELODY_DROP_NOTE = 0.1


def on_beats(
    events: np.ndarray, grid: BeatGrid, measure_frames: np.ndarray, tokenizer: BeatTokenizer
) -> tuple[Performance, float]:
    """イベント配列 (フレーム、piano_ar のイベントの形) を拍の刻みに丸めて、拍の時刻の演奏にする。
    (演奏 (最初の音 = 0、measure_starts に各小節の開始), 原点 (丸めた拍の時刻での最初の音)) を返す"""
    q = tokenizer.quantize(events, grid)
    tpb = tokenizer.ticks_per_beat
    seconds = q.tick / tpb * BEAT_SECONDS
    is_note = q.kind == KIND_NOTE
    duration = tokenizer.duration_values[q.duration_bin[is_note]] / tpb * BEAT_SECONDS
    notes = np.stack(
        [seconds[is_note], seconds[is_note] + duration, q.pitch[is_note], q.velocity[is_note]], axis=1
    ).astype(np.float64)
    is_pedal = (q.kind == KIND_PEDAL_ON) | (q.kind == KIND_PEDAL_OFF)
    pedal = np.stack([seconds[is_pedal], (q.kind[is_pedal] == KIND_PEDAL_ON).astype(np.float64)], axis=1)
    starts = grid.beat_position(np.asarray(measure_frames, dtype=np.float64)) * BEAT_SECONDS
    origin = float(notes[:, 0].min()) if len(notes) else 0.0
    notes[:, :2] -= origin
    pedal = pedal.reshape(-1, 2)
    pedal[:, 0] -= origin
    return Performance(notes, pedal, starts - origin), origin


def mark_melody(performance: Performance, times: np.ndarray, pitches: np.ndarray, octave_free: bool) -> np.ndarray:
    """拍の時刻の演奏の音のうち、メロディの音 (拍の時刻 times・音高 pitches) と同じ位置 (MELODY_TICKS 刻み以内) の音に
    印を付ける。octave_free なら音名だけで比べ、同じ位置の候補のうち一番高い音にする"""
    notes = performance.notes
    marked = np.zeros(len(notes), dtype=bool)
    if not len(notes) or not len(times):
        return marked
    tolerance = MELODY_TICKS / BeatTokenizer().ticks_per_beat * BEAT_SECONDS + 1e-6
    order = np.argsort(notes[:, 0])
    onsets = notes[order, 0]
    for t, p in zip(times.tolist(), pitches.tolist()):
        lo, hi = np.searchsorted(onsets, t - tolerance), np.searchsorted(onsets, t + tolerance, side="right")
        candidates = order[lo:hi]
        if octave_free:
            candidates = [i for i in candidates if int(notes[i, 2]) % 12 == int(p) % 12]
        else:
            candidates = [i for i in candidates if int(notes[i, 2]) == int(p)]
        if candidates:
            marked[max(candidates, key=lambda i: (notes[i, 2], -abs(notes[i, 0] - t)))] = True
    return marked


def melody_from_annotation(performance: Performance, annotation: dict, grid: BeatGrid, origin: float) -> np.ndarray:
    """原曲の SheetSage2-Refined のメロディ (秒) から、拍の時刻の演奏の音にメロディの印を付ける"""
    melody = annotation.get("melody") or []
    if not melody:
        return np.zeros(len(performance.notes), dtype=bool)
    frames = np.array([float(m["start"]) for m in melody]) * FRAMES_PER_SECOND
    times = grid.beat_position(frames) * BEAT_SECONDS - origin
    return mark_melody(performance, times, np.array([int(m["pitch"]) for m in melody]), octave_free=True)


def performance_events(performance: Performance) -> np.ndarray:
    """演奏 (秒) を piano_ar のイベント配列 (フレーム、[N, 5] = onset, kind, pitch, duration, velocity) にする"""
    fr = FRAMES_PER_SECOND
    notes = performance.notes
    rows = []
    if len(notes):
        onset = np.round(notes[:, 0] * fr).astype(np.int64)
        duration = np.maximum(1, np.round((notes[:, 1] - notes[:, 0]) * fr)).astype(np.int64)
        velocity = np.clip(np.round(notes[:, 3]), 1, 127).astype(np.int64)
        rows.append(
            np.stack([onset, np.full_like(onset, KIND_NOTE), notes[:, 2].astype(np.int64), duration, velocity], 1)
        )
    # ペダルは踏む・離すが交互になるように並べ、踏んだままなら最後に離す (BeatTokenizer.quantize の前提)
    down = False
    pedal_rows = []
    for time, value in performance.pedal.tolist():
        if bool(value) != down:
            down = bool(value)
            pedal_rows.append([round(time * fr), KIND_PEDAL_ON if down else KIND_PEDAL_OFF, 0, 0, 0])
    if down:
        end = max(pedal_rows[-1][0], round(float(notes[:, 1].max()) * fr) if len(notes) else 0)
        pedal_rows.append([end + 1, KIND_PEDAL_OFF, 0, 0, 0])
    if pedal_rows:
        rows.append(np.asarray(pedal_rows, dtype=np.int64))
    if not rows:
        return np.zeros((0, 5), dtype=np.int64)
    table = np.concatenate(rows)
    # 同じフレームの離す・踏むの順を崩さないよう、時刻だけで安定に並べる
    return table[np.argsort(table[:, 0], kind="stable")]


def level_grid(beat_times: np.ndarray, beat_measures: np.ndarray, level: float) -> np.ndarray | None:
    """楽譜の拍の時刻を、level (楽譜の拍 1 つを何拍にするか) の格子にする。できなければ None"""
    if level == 1:
        return beat_times
    if level in (2, 3):
        n = int(level)
        parts = [beat_times[:-1] + (beat_times[1:] - beat_times[:-1]) * k / n for k in range(n)]
        return np.sort(np.concatenate([*parts, beat_times[-1:]]))
    # 2 拍まとめ: 各小節の拍の数が偶数のときだけ、小節の頭から 1 つおきに取る
    keep = []
    for m in np.unique(beat_measures):
        index = np.flatnonzero(beat_measures == m)
        if len(index) % 2:
            return None
        keep += index[::2].tolist()
    return beat_times[np.asarray(keep, dtype=np.int64)]


class BeatSynthWindowDataset(SynthWindowDataset):
    """楽譜の窓と、それを描き出して拍の刻みに丸めた演奏 (拍の時刻) の組。窓の取り方は SynthWindowDataset と同じ"""

    def __init__(
        self,
        cache: ScoreCache,
        *,
        level_probs: dict | None = None,
        melody: bool = False,
        drop_durations: float = 0.0,
        **kwargs,
    ) -> None:
        kwargs.setdefault("render_config", COVER_RENDER)
        super().__init__(cache, **kwargs)
        self.beat_tokenizer = BeatTokenizer()
        self.level_probs = LEVEL_PROBS if level_probs is None else level_probs
        self.melody = melody
        self.vocab = PerformanceVocab(melody=melody)
        # 長さの列を消す窓の割合。検証は曲ごとに固定 (曲の番号の偶奇で、消す・消さないを半分ずつ)
        self.drop_durations = drop_durations

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        song = int(self.songs[index])
        num_measures = self.cache.num_measures(song)
        W = self.window_measures
        rng = np.random.default_rng(None if self.train else song)
        for attempt in range(8):
            song_start = not self.train or (attempt == 0 and random.random() < self.song_start_prob)
            start = 0 if song_start else random.randint(0, max(0, num_measures - W // 2))
            stop = min(num_measures, start + W)
            first = max(0, start - self.context_measures)
            last = min(num_measures, stop + 1)  # 窓の後の 1 小節も演奏に入れる
            bodies = self.cache.measure_tokens(song, first, last)
            measures, _ = self.tokenizer.decode([[self.mtime_first, *b.tolist()] for b in bodies])
            # 拍子より長い小節 (壊れた MusicXML) を含む窓は使わない。検証は曲の冒頭に固定なのでそのまま
            if not self.train or all(m.length <= m.nominal_length for m in measures):
                break
        if self.augment is not None and self.augment.transpose > 0:
            shift = random.randint(-self.augment.transpose, self.augment.transpose)
            if shift:
                bodies = self._transpose(bodies, shift)
                measures, _ = self.tokenizer.decode([[self.mtime_first, *b.tolist()] for b in bodies])
        qpm = measure_qpm(measures, self.cache.song_seconds(song)[first : last + 1])
        if np.all(np.abs(qpm - 120.0) < 0.5) and not any(self.is_metronome[b].any() for b in bodies):
            median, sigma, low, high = UNMARKED_TEMPO
            qpm = np.full_like(qpm, float(np.clip(median * math.exp(rng.normal(0.0, sigma)), low, high)))
        performance = render(measures, rng, self.render_config, qpm)

        level = 1
        if self.train:
            level = random.choices(list(self.level_probs), weights=list(self.level_probs.values()))[0]
        beats = level_grid(performance.beat_times, performance.beat_measures, level)
        if beats is None or len(beats) < 4:
            beats = performance.beat_times
        fr = FRAMES_PER_SECOND
        events = performance_events(performance)
        last_frame = float((events[:, 0] + events[:, 3]).max()) if len(events) else 1.0
        measure_frames = performance.measure_starts * fr
        first_frame = min(0.0, float(events[:, 0].min()) if len(events) else 0.0)
        max_bar_beats = self.beat_tokenizer.config.max_bar_beats
        grid = BeatGrid.from_frames(beats * fr, measure_frames, first_frame, last_frame, max_bar_beats)
        if grid is None and beats is not performance.beat_times:
            grid = BeatGrid.from_frames(
                performance.beat_times * fr, measure_frames, first_frame, last_frame, max_bar_beats
            )
        if grid is None:  # 拍が 4 つ未満の窓 (とても短い曲)
            if not self.train:
                raise ValueError(f"拍の格子を作れない曲: {self.cache.ids[song]}")
            return self[random.randrange(len(self))]
        on_grid, origin = on_beats(events, grid, measure_frames, self.beat_tokenizer)
        if self.melody and (not self.train or random.random() >= MELODY_DROP_SONG):
            on_grid.melody = self._score_melody(measures, performance, grid, origin, on_grid)
        if self.train:
            drop = random.random() < self.drop_durations
        else:
            drop = self.drop_durations > 0 and song % 2 == 0
        lengths = np.array([float(m.length) for m in measures])
        return make_item(
            self.tokenizer, self.vocab, bodies, lengths, on_grid, start=start, stop=stop, first=first,
            window_measures=W, max_rows=self.max_rows, train=self.train, noise=None, exact_starts=True,
            drop_durations=drop,
        )  # fmt: skip

    def _score_melody(self, measures, performance: Performance, grid: BeatGrid, origin: float, on_grid: Performance):
        """楽譜の上の段の一番上の音を、拍の時刻の演奏の音に写してメロディの印にする"""
        offsets = np.concatenate([[0.0], np.cumsum([float(m.length) for m in measures])])
        positions, _ = beat_positions(measures, offsets)
        if len(positions) < 2:
            return np.zeros(len(on_grid.notes), dtype=bool)
        top = [n for n in _score_notes(measures, offsets) if n.top and n.group.staff == 1]
        if not top:
            return np.zeros(len(on_grid.notes), dtype=bool)
        seconds = np.interp([n.on for n in top], positions, performance.beat_times)
        times = grid.beat_position(seconds * FRAMES_PER_SECOND) * BEAT_SECONDS - origin
        marked = mark_melody(on_grid, times, np.array([n.pitch for n in top]), octave_free=False)
        if self.train:
            marked &= np.array([random.random() >= MELODY_DROP_NOTE for _ in range(len(marked))], dtype=bool)
        return marked


def song_pitch_spans(cache: ScoreCache) -> np.ndarray:
    """曲ごとの音域 (楽譜の音高の 95% - 5%、半音)"""
    tok = ScoreTokenizer(cache.tokenizer_config)
    tokenizer_pitch = np.full(len(tok.kinds), -1, dtype=np.int64)
    for token, (kind, value) in enumerate(zip(tok.kinds, tok.values)):
        if kind == "pitch":
            tokenizer_pitch[token] = int(value[0])
    tokens = np.load(cache.cache_dir / "tokens.npy", mmap_mode="r")
    spans = np.zeros(len(cache.ids), dtype=np.float64)
    for song in range(len(cache.ids)):
        lo = cache.token_offsets[cache.measure_offsets[song]]
        hi = cache.token_offsets[cache.measure_offsets[song + 1]]
        pitches = tokenizer_pitch[np.asarray(tokens[lo:hi], dtype=np.int64)]
        pitches = pitches[pitches >= 0]
        if len(pitches):
            p5, p95 = np.percentile(pitches, [5, 95])
            spans[song] = p95 - p5
    return spans


def select_songs(
    cache: ScoreCache, quality_path: str | Path, min_span: float = MIN_PITCH_SPAN
) -> dict[str, np.ndarray]:
    """学習に使う曲の番号。low_quality でなく、音域が min_span 以上の曲"""
    quality = json.loads(Path(quality_path).read_text(encoding="utf-8"))
    spans = song_pitch_spans(cache)
    ok = np.array([str(i) in quality and not low_quality(quality[str(i)]) for i in cache.ids]) & (spans >= min_span)
    return {"songs": np.flatnonzero(ok), "spans": spans}


def load_songs(cache: ScoreCache, split: str) -> np.ndarray:
    data = np.load(cache.cache_dir / SONGS_FILE)
    songs = data["songs"]
    return songs[cache.is_val[songs]] if split == "val" else songs[~cache.is_val[songs]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default="data/piano_score/synth")
    parser.add_argument("--quality", default="data/piano_score/quality.json")
    parser.add_argument("--min-pitch-span", type=float, default=MIN_PITCH_SPAN)
    args = parser.parse_args()
    cache = ScoreCache(args.cache_dir)
    selected = select_songs(cache, args.quality, args.min_pitch_span)
    np.savez(cache.cache_dir / SONGS_FILE, **selected, min_span=args.min_pitch_span)
    songs = selected["songs"]
    measures = int(sum(cache.num_measures(int(s)) for s in songs))
    print(
        f"{len(songs)} / {len(cache.ids)} 曲 (検証 {int(cache.is_val[songs].sum())})、{measures} 小節を "
        f"{cache.cache_dir / SONGS_FILE} に書いた"
    )


if __name__ == "__main__":
    main()
