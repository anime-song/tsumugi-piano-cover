"""拍のトークン: 1 パッチ = 1 小節で、小節の中の位置と音の長さを拍の刻み (BeatTokenizerConfig.ticks_per_beat) で表す。

イベント配列は PianoTokenizer と同じ int32 [N, 5] (onset, kind, pitch, duration, velocity、時間はフレーム) のまま持ち、
曲の拍 (BeatGrid) と一緒にトークン化する。拍は SheetSage2-Refined の注釈 (piano_ar.sheetsage) から作る。

1 パッチ (1 小節) のトークン列の文法は PianoTokenizer と同じ形:
    event   := [POS] (PITCH DUR VEL | PEDAL_ON | PEDAL_OFF)
    patch   := event* (EOP | EOS)
POS は小節の頭からの刻み (0 〜 小節の拍数 × ticks_per_beat - 1)。直前のイベントと同じなら省略する。
同じ位置の中は PEDAL_OFF -> PEDAL_ON -> ノート (pitch 昇順)。

小節の拍数とテンポはトークンにしない。生成では原曲の拍に置いて弾くので、どちらも分かっている。Global の条件として渡す
(tokenize_window の bar_beats / tempo)。刻みからのずれ (マイクロタイミング) は持たず、戻すときは刻みを拍の時刻に置く。

時刻の丸め方:
    打鍵・離鍵・ペダルを、拍の番号 (小数) × ticks_per_beat を四捨五入した刻みにする (同じ写像なので前後は保たれる)。
    DUR は、離鍵がペダルのどの区間にあるかを元の時刻と変えないビンのうち、丸めた離鍵に一番近いものを選ぶ
    (PianoTokenizer.pedal_safe_duration_bins と同じ考え方。丸めた刻みで同じ位置に重なると前後が崩れるため)。
    同じ刻み・同じ鍵盤の音は長い方だけを残す。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BeatTokenizerConfig
from .tokenizer import (
    DURATION,
    EOP,
    EOS,
    KIND,
    KIND_NOTE,
    KIND_PEDAL_OFF,
    KIND_PEDAL_ON,
    NUM_SPECIAL_TOKENS,
    ONSET,
    PAD,
    PEDAL_OFF,
    PEDAL_ON,
    PITCH,
    TOKEN_GROUPS,
    VELOCITY,
    sort_events,
    trim_overlapping_notes,
)


def _duration_values(config: BeatTokenizerConfig) -> np.ndarray:
    exact = np.arange(1, config.exact_duration_ticks + 1)
    geometric = np.geomspace(
        config.exact_duration_ticks + 1,
        config.max_duration_ticks,
        config.num_duration_bins - config.exact_duration_ticks,
    )
    values = np.concatenate([exact, np.round(geometric)]).astype(np.int64)
    for i in range(1, len(values)):
        values[i] = max(values[i], values[i - 1] + 1)
    return values


@dataclass
class BeatGrid:
    """曲の拍 (フレーム、小数) と小節の区切り。拍はイベントの前後を覆うように、端の拍の間隔で延ばしてある。

    beats [B]: 拍の時刻。bar_starts [P + 1]: 各小節の最初の拍の番号と、最後の小節の終わり (最後の拍の番号)"""

    beats: np.ndarray
    bar_starts: np.ndarray

    @property
    def num_bars(self) -> int:
        return len(self.bar_starts) - 1

    @property
    def bar_beats(self) -> np.ndarray:
        return np.diff(self.bar_starts)

    def beat_position(self, frames: np.ndarray) -> np.ndarray:
        """フレームを拍の番号 (小数) にする。両端の外は端の拍の間隔で延ばす"""
        b = self.beats
        index = np.arange(len(b), dtype=np.float64)
        knots_t = np.concatenate([[b[0] - (b[1] - b[0]) * 1e4], b, [b[-1] + (b[-1] - b[-2]) * 1e4]])
        knots_i = np.concatenate([[-1e4], index, [index[-1] + 1e4]])
        return np.interp(np.asarray(frames, dtype=np.float64), knots_t, knots_i)

    def frames_at(self, beat_position: np.ndarray) -> np.ndarray:
        """拍の番号 (小数) をフレームにする (beat_position の逆)"""
        b = self.beats
        index = np.arange(len(b), dtype=np.float64)
        knots_i = np.concatenate([[-1e4], index, [index[-1] + 1e4]])
        knots_t = np.concatenate([[b[0] - (b[1] - b[0]) * 1e4], b, [b[-1] + (b[-1] - b[-2]) * 1e4]])
        return np.interp(np.asarray(beat_position, dtype=np.float64), knots_i, knots_t)

    def tempo(self) -> np.ndarray:
        """小節ごとの BPM (その小節の拍の長さの平均から)。フレームは 10ms 単位でなくてもよいよう、frame_rate は呼ぶ側で掛ける"""
        start, end = self.bar_starts[:-1], self.bar_starts[1:]
        return (end - start) / np.maximum(self.beats[end] - self.beats[start], 1e-6)

    @classmethod
    def build(
        cls,
        beat_seconds: np.ndarray,
        downbeat_seconds: np.ndarray,
        frame_rate: int,
        first_frame: float,
        last_frame: float,
        max_bar_beats: int,
    ) -> BeatGrid | None:
        """拍と小節の頭 (秒) から作る。first_frame〜last_frame (イベントの範囲) を覆うまで、端の拍の間隔で拍と小節を延ばす。
        拍が 4 つ未満なら None"""
        beats = np.asarray(beat_seconds, dtype=np.float64) * frame_rate
        if len(beats) < 4 or np.any(np.diff(beats) <= 0):
            return None
        index = np.round(np.interp(np.asarray(downbeat_seconds) * frame_rate, beats, np.arange(len(beats)))).astype(int)
        downs = sorted(set(index.tolist()) & set(range(len(beats))))
        if not downs:
            downs = [0]
        # 前に延ばす
        first_period = beats[1] - beats[0]
        lead = max(0, int(np.ceil((beats[0] - first_frame) / first_period)) + 1)
        beats = np.concatenate([beats[0] - first_period * np.arange(lead, 0, -1), beats])
        downs = [d + lead for d in downs]
        # 後ろに延ばす (最後の拍が最後のイベントより後になるまで、さらに 1 拍)
        last_period = beats[-1] - beats[-2]
        tail = max(0, int(np.ceil((last_frame - beats[-1]) / last_period)) + 1)
        beats = np.concatenate([beats, beats[-1] + last_period * np.arange(1, tail + 1)])

        # 小節の頭: 注釈の小節の頭。その前と後ろは、隣の小節と同じ拍数で区切る
        lengths = np.diff(downs)
        first_meter = int(lengths[0]) if len(lengths) else 4
        last_meter = int(lengths[-1]) if len(lengths) else 4
        starts = {0, *range(downs[0] - first_meter, 0, -first_meter), *downs}
        end = len(beats) - 1
        starts |= set(range(downs[-1] + last_meter, end, last_meter))
        starts = sorted(s for s in starts if s < end) + [end]
        # 長すぎる小節は分ける
        bounded = [starts[0]]
        for s in starts[1:]:
            while s - bounded[-1] > max_bar_beats:
                bounded.append(bounded[-1] + max_bar_beats)
            bounded.append(s)
        return cls(beats, np.asarray(bounded, dtype=np.int64))


@dataclass
class QuantizedSong:
    """拍の刻みに丸めたイベント。tick は曲の最初の拍からの刻み、duration はノートの刻み数 (ビンの値)"""

    tick: np.ndarray
    kind: np.ndarray
    pitch: np.ndarray
    duration_bin: np.ndarray
    velocity: np.ndarray
    end_tick: int  # 曲の終わり (最後のノートの離鍵の刻み)


class BeatTokenizer:
    def __init__(self, config: BeatTokenizerConfig | None = None) -> None:
        self.config = config or BeatTokenizerConfig()
        c = self.config
        self.ticks_per_beat = c.ticks_per_beat
        self.num_pitches = c.pitch_max - c.pitch_min + 1
        self.duration_values = _duration_values(c)
        self._duration_bounds = np.sqrt(self.duration_values[:-1] * self.duration_values[1:])

        # PianoARModel が使う名前 (time_offset = 位置のトークンの始まり) は PianoTokenizer と同じにする
        self.time_offset = NUM_SPECIAL_TOKENS
        self.pitch_offset = self.time_offset + c.max_bar_ticks
        self.duration_offset = self.pitch_offset + self.num_pitches
        self.velocity_offset = self.duration_offset + len(self.duration_values)
        self.vocab_size = self.velocity_offset + c.num_velocity_bins

        group = np.full(self.vocab_size, -1, dtype=np.int64)
        group[[EOP, EOS]] = TOKEN_GROUPS.index("end")
        group[[PEDAL_ON, PEDAL_OFF]] = TOKEN_GROUPS.index("pedal")
        group[self.time_offset : self.pitch_offset] = TOKEN_GROUPS.index("time")
        group[self.pitch_offset : self.duration_offset] = TOKEN_GROUPS.index("pitch")
        group[self.duration_offset : self.velocity_offset] = TOKEN_GROUPS.index("duration")
        group[self.velocity_offset :] = TOKEN_GROUPS.index("velocity")
        self.token_group = group
        self.group_names = TOKEN_GROUPS

    # ------------------------------------------------------------------
    def duration_bin(self, ticks: np.ndarray | int) -> np.ndarray:
        return np.searchsorted(self._duration_bounds, ticks)

    def velocity_bin(self, velocity: np.ndarray | int) -> np.ndarray:
        return np.clip(
            np.asarray(velocity) * self.config.num_velocity_bins // 128, 0, self.config.num_velocity_bins - 1
        )

    def velocity_value(self, bin_index: int) -> int:
        return int(np.clip(round((bin_index + 0.5) * 128 / self.config.num_velocity_bins), 1, 127))

    def tempo_bin(self, bpm: np.ndarray) -> np.ndarray:
        """BPM を 1〜num_tempo_bins の段階にする (0 は「指定なし」に空けておく)"""
        c = self.config
        position = np.log(np.clip(bpm, c.tempo_min, c.tempo_max) / c.tempo_min) / np.log(c.tempo_max / c.tempo_min)
        return 1 + np.minimum((position * c.num_tempo_bins).astype(np.int64), c.num_tempo_bins - 1)

    # ------------------------------------------------------------------
    def quantize(self, events: np.ndarray, grid: BeatGrid) -> QuantizedSong:
        tpb = self.ticks_per_beat
        events = events.astype(np.int64)
        notes = events[events[:, KIND] == KIND_NOTE]
        pedals = events[events[:, KIND] != KIND_NOTE]

        # ペダル: 丸めて、つぶれた区間 (on == off) は 1 刻みにする。踏み替え (off のすぐ後に on) が同じ刻みに丸まっても
        # つなげない (つなげると踏み替えで音を切ったことが消え、前の音が次のペダルの終わりまで伸びる)。
        # 前の off を次の on の刻みまで縮め、同じ刻みの PEDAL_OFF -> PEDAL_ON として残す
        bounds_frames = pedals[:, ONSET].astype(np.float64)  # on, off, on, off, ... (midi_to_events が交互を保証)
        pedal_ticks = np.round(grid.beat_position(bounds_frames) * tpb).astype(np.int64)
        merged: list[list[int]] = []
        for on, off in pedal_ticks.reshape(-1, 2):
            if merged:
                on = max(int(on), merged[-1][0] + 1)
                merged[-1][1] = min(merged[-1][1], on)
            merged.append([int(on), int(max(off, on + 1))])
        pedal_bounds = np.asarray(merged, dtype=np.int64).reshape(-1)

        onset = np.round(grid.beat_position(notes[:, ONSET]) * tpb).astype(np.int64)
        release_frames = (notes[:, ONSET] + notes[:, DURATION]).astype(np.float64)
        release = np.maximum(onset + 1, np.round(grid.beat_position(release_frames) * tpb).astype(np.int64))
        bins = self.duration_bin(release - onset)
        if len(pedal_bounds) and len(notes):
            # 離鍵が元の時刻でペダルのどの区間にあったか (同じ時刻のペダルより前とみなす)。偶数 2k は k 番目のペダルの前、
            # 奇数 2k+1 は k 番目のペダルの中。ペダルの数は丸めても変わらないので、番号はそのまま使える
            target = np.searchsorted(bounds_frames, release_frames, side="left")
            rounded = np.searchsorted(pedal_bounds, onset + self.duration_values[bins], side="left")
            for i in np.flatnonzero(rounded != target):
                s = target[i]
                lo = pedal_bounds[s - 1] if s > 0 else -np.inf
                hi = pedal_bounds[s] if s < len(pedal_bounds) else np.inf
                candidates = onset[i] + self.duration_values
                ok = np.flatnonzero((candidates > lo) & (candidates <= hi))
                if len(ok):
                    bins[i] = ok[np.argmin(np.abs(candidates[ok] - release[i]))]

        rows = []
        if len(notes):
            note_rows = np.stack(
                [onset, np.full_like(onset, KIND_NOTE), notes[:, PITCH], bins, notes[:, VELOCITY]], axis=1
            )
            # 同じ刻み・同じ鍵盤は長い方だけ
            note_rows = note_rows[np.lexsort((-note_rows[:, 3], note_rows[:, 2], note_rows[:, 0]))]
            _, first = np.unique(note_rows[:, [0, 2]], axis=0, return_index=True)
            rows.append(note_rows[np.sort(first)])
        for on, off in np.asarray(merged, dtype=np.int64).reshape(-1, 2):
            rows.append(np.array([[on, KIND_PEDAL_ON, 0, 0, 0], [off, KIND_PEDAL_OFF, 0, 0, 0]], dtype=np.int64))
        if not rows:
            empty = np.zeros(0, dtype=np.int64)
            return QuantizedSong(empty, empty, empty, empty, empty, 0)
        table = np.concatenate(rows)
        table = table[np.lexsort((table[:, 2], table[:, 1], table[:, 0]))]
        is_note = table[:, 1] == KIND_NOTE
        ends = table[is_note, 0] + self.duration_values[table[is_note, 3]]
        end_tick = int(max(ends.max(initial=0), table[:, 0].max() + 1))
        return QuantizedSong(table[:, 0], table[:, 1], table[:, 2], table[:, 3], table[:, 4], end_tick)

    def tokenize_window(
        self, song: QuantizedSong, grid: BeatGrid, start_bar: int, num_bars: int, frame_rate: int | None = None
    ) -> dict[str, np.ndarray]:
        """start_bar から num_bars 小節をトークン化する。

        返り値:
            tokens [P, max_patch_tokens]、patch_valid [P] (曲の終わりより後の小節は False)
            pedal_state [P] (小節の頭でペダルが踏まれているか)、bar_beats [P]、tempo [P] (段階、0 は範囲外の小節)
        """
        c = self.config
        tpb = self.ticks_per_beat
        L = c.max_patch_tokens
        P = num_bars
        tokens = np.full((P, L), PAD, dtype=np.int64)
        patch_valid = np.zeros(P, dtype=bool)
        pedal_state = np.zeros(P, dtype=np.int64)
        bar_beats = np.zeros(P, dtype=np.int64)
        tempo = np.zeros(P, dtype=np.int64)

        bar_ticks = grid.bar_starts * tpb
        last_bar = int(np.searchsorted(bar_ticks, max(song.end_tick - 1, 0), side="right") - 1)
        last_bar = min(last_bar, grid.num_bars - 1)
        bars = np.arange(start_bar, start_bar + P)
        inside = (bars >= 0) & (bars <= last_bar)
        patch_valid[:] = inside
        bpm = grid.tempo() * (frame_rate or c.frame_rate) * 60
        bar_beats[inside] = grid.bar_beats[bars[inside]]
        tempo[inside] = self.tempo_bin(bpm[bars[inside]])

        pedal_on = song.tick[song.kind == KIND_PEDAL_ON]
        pedal_off = song.tick[song.kind == KIND_PEDAL_OFF]
        if len(pedal_on):
            starts = bar_ticks[bars[inside]]
            index = np.searchsorted(pedal_on, starts, side="left") - 1
            pedal_state[inside] = (index >= 0) & (pedal_off[np.maximum(index, 0)] >= starts)

        event_bar = np.searchsorted(bar_ticks, song.tick, side="right") - 1
        for p in np.flatnonzero(inside):
            bar = bars[p]
            lo, hi = np.searchsorted(event_bar, [bar, bar + 1])
            sequence: list[int] = []
            previous = -1
            for i in range(lo, hi):
                pos = int(song.tick[i] - bar_ticks[bar])
                event = [self.time_offset + pos] if pos != previous else []
                if song.kind[i] == KIND_NOTE:
                    event += [
                        self.pitch_offset + int(song.pitch[i]) - c.pitch_min,
                        self.duration_offset + int(song.duration_bin[i]),
                        self.velocity_offset + int(self.velocity_bin(song.velocity[i])),
                    ]
                else:
                    event.append(PEDAL_ON if song.kind[i] == KIND_PEDAL_ON else PEDAL_OFF)
                if len(sequence) + len(event) > L - 1:
                    break
                sequence += event
                previous = pos
            sequence.append(EOS if bar == last_bar else EOP)
            tokens[p, : len(sequence)] = sequence
        return {
            "tokens": tokens,
            "patch_valid": patch_valid,
            "pedal_state": pedal_state,
            "bar_beats": bar_beats,
            "tempo": tempo,
        }

    def patches_to_events(self, patches: list[list[int]], grid: BeatGrid, start_bar: int = 0) -> np.ndarray:
        """小節ごとのトークン列を、grid の拍の時刻に置いたイベント配列 (フレーム) に戻す"""
        c = self.config
        tpb = self.ticks_per_beat
        rows = []
        for p, sequence in enumerate(patches):
            bar_tick = int(grid.bar_starts[min(start_bar + p, grid.num_bars)]) * tpb
            pos = 0
            pending: list[int] = []
            for token in sequence:
                if token in (PAD, EOP, EOS):
                    break
                if self.time_offset <= token < self.pitch_offset:
                    pos = token - self.time_offset
                elif token in (PEDAL_ON, PEDAL_OFF):
                    kind = KIND_PEDAL_ON if token == PEDAL_ON else KIND_PEDAL_OFF
                    rows.append([bar_tick + pos, kind, 0, 0, 0])
                else:
                    pending.append(token)
                    if len(pending) == 3:
                        pitch = pending[0] - self.pitch_offset + c.pitch_min
                        duration = int(self.duration_values[pending[1] - self.duration_offset])
                        velocity = self.velocity_value(pending[2] - self.velocity_offset)
                        rows.append([bar_tick + pos, KIND_NOTE, pitch, duration, velocity])
                        pending = []
        if not rows:
            return np.zeros((0, 5), dtype=np.int32)
        table = np.asarray(rows, dtype=np.int64)
        onset = grid.frames_at(table[:, 0] / tpb)
        events = table.copy()
        events[:, ONSET] = np.round(onset)
        note = table[:, KIND] == KIND_NOTE
        # 離鍵も絶対の時刻で丸めてから長さにする (長さを丸めると、同じ刻みのペダルと 1 フレーム前後して関係が崩れる)
        release = np.round(grid.frames_at((table[note, 0] + table[note, DURATION]) / tpb))
        events[note, DURATION] = np.maximum(1, release - events[note, ONSET])
        return trim_overlapping_notes(sort_events(events)).astype(np.int32)
