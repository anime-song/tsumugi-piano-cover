"""MIDI <-> イベント配列 <-> パッチごとのトークン列の変換。

イベント配列は int32 の [N, 5] (onset, kind, pitch, duration, velocity) で、時間はすべてフレーム単位。
duration はビン化前のフレーム数のまま持つので、時間伸縮などの拡張をかけてからトークン化できる。

1 パッチ (既定 2 秒) のトークン列の文法:
    event   := [TIME] (PITCH DUR VEL | PEDAL_ON | PEDAL_OFF)
    patch   := event* (EOP | EOS)
TIME は直前のイベントと onset が同じなら省略する (和音は PITCH DUR VEL が続く)。パッチ先頭のイベントは必ず TIME を持つ。
同じ onset の中は PEDAL_OFF -> PEDAL_ON -> ノート (pitch 昇順) の順に並べる。EOS は曲の終わりを含むパッチの終端。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import TokenizerConfig

PAD, EOP, EOS, PEDAL_ON, PEDAL_OFF = 0, 1, 2, 3, 4
NUM_SPECIAL_TOKENS = 5

# イベント配列の列と kind。kind の値は同じ onset 内の並び順も兼ねる
ONSET, KIND, PITCH, DURATION, VELOCITY = range(5)
KIND_PEDAL_OFF, KIND_PEDAL_ON, KIND_NOTE = 0, 1, 2

# 損失をトークンの種類別に見るためのグループ
TOKEN_GROUPS = ("end", "pedal", "time", "pitch", "duration", "velocity")


def _duration_values(config: TokenizerConfig) -> np.ndarray:
    exact = np.arange(1, config.exact_duration_frames + 1)
    geometric = np.geomspace(
        config.exact_duration_frames + 1,
        config.max_duration_frames,
        config.num_duration_bins - config.exact_duration_frames,
    )
    values = np.concatenate([exact, np.round(geometric)]).astype(np.int64)
    for i in range(1, len(values)):
        values[i] = max(values[i], values[i - 1] + 1)
    return values


class PianoTokenizer:
    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        c = self.config
        self.patch_frames = c.patch_frames
        self.num_pitches = c.pitch_max - c.pitch_min + 1
        self.duration_values = _duration_values(c)
        # 対数軸で最も近いビンに丸めるための境界 (隣り合うビンの幾何平均)
        self._duration_bounds = np.sqrt(self.duration_values[:-1] * self.duration_values[1:])

        self.time_offset = NUM_SPECIAL_TOKENS
        self.pitch_offset = self.time_offset + self.patch_frames
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

    # ------------------------------------------------------------------
    # 値 <-> ビン
    # ------------------------------------------------------------------
    def duration_bin(self, frames: np.ndarray | int) -> np.ndarray:
        return np.searchsorted(self._duration_bounds, frames)

    def velocity_bin(self, velocity: np.ndarray | int) -> np.ndarray:
        return np.clip(
            np.asarray(velocity) * self.config.num_velocity_bins // 128, 0, self.config.num_velocity_bins - 1
        )

    def velocity_value(self, bin_index: int) -> int:
        return int(np.clip(round((bin_index + 0.5) * 128 / self.config.num_velocity_bins), 1, 127))

    # ------------------------------------------------------------------
    # MIDI -> イベント配列
    # ------------------------------------------------------------------
    def midi_to_events(self, path: str | Path, trim: bool = True) -> tuple[np.ndarray, int]:
        """MIDI を読み、イベント配列と曲の終端フレームを返す。trim=True なら先頭の無音を詰める"""
        from symusic import Score

        c = self.config
        score = Score.from_file(str(path)).to("second")
        notes = []
        pedals = []
        for track in score.tracks:
            if track.is_drum:
                continue
            notes.extend((n.time, n.duration, n.pitch, n.velocity) for n in track.notes)
            pedals.extend((p.time, p.time + p.duration) for p in track.pedals)

        rows = []
        if notes:
            arr = np.asarray(notes, dtype=np.float64)
            onset = np.round(arr[:, 0] * c.frame_rate).astype(np.int64)
            duration = np.maximum(1, np.round(arr[:, 1] * c.frame_rate)).astype(np.int64)
            pitch = arr[:, 2].astype(np.int64)
            velocity = np.clip(arr[:, 3], 1, 127).astype(np.int64)
            keep = (pitch >= c.pitch_min) & (pitch <= c.pitch_max)
            note_rows = np.stack([onset, np.full_like(onset, KIND_NOTE), pitch, duration, velocity], axis=1)[keep]
            # 同じ onset・同じ pitch の重複は長い方だけ残す (生成側は同一 onset 内で pitch 昇順を強制する)
            note_rows = note_rows[np.lexsort((-note_rows[:, DURATION], note_rows[:, PITCH], note_rows[:, ONSET]))]
            _, first = np.unique(note_rows[:, [ONSET, PITCH]], axis=0, return_index=True)
            rows.append(note_rows[np.sort(first)])

        # ペダルは区間に直してから、丸めで重なったものをつなげる。on と off が交互に並ぶことを保証する
        merged: list[list[int]] = []
        for start, end in sorted(pedals):
            on = int(round(start * c.frame_rate))
            off = max(on + 1, int(round(end * c.frame_rate)))
            if merged and on < merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], off)
            else:
                merged.append([on, off])
        for on, off in merged:
            rows.append(np.array([[on, KIND_PEDAL_ON, 0, 0, 0], [off, KIND_PEDAL_OFF, 0, 0, 0]], dtype=np.int64))

        if not rows:
            return np.zeros((0, 5), dtype=np.int32), 0
        events = np.concatenate(rows)
        if trim:
            events[:, ONSET] -= events[:, ONSET].min()
        events = sort_events(events)
        notes_mask = events[:, KIND] == KIND_NOTE
        end_frame = int(
            max((events[notes_mask, ONSET] + events[notes_mask, DURATION]).max(initial=0), events[:, ONSET].max() + 1)
        )
        return events.astype(np.int32), end_frame

    # ------------------------------------------------------------------
    # イベント配列 -> パッチごとのトークン
    # ------------------------------------------------------------------
    def event_tokens(self, row: np.ndarray, local_onset: int, with_time: bool) -> list[int]:
        tokens = [self.time_offset + local_onset] if with_time else []
        kind = row[KIND]
        if kind == KIND_NOTE:
            tokens += [
                self.pitch_offset + int(row[PITCH]) - self.config.pitch_min,
                self.duration_offset + int(self.duration_bin(row[DURATION])),
                self.velocity_offset + int(self.velocity_bin(row[VELOCITY])),
            ]
        else:
            tokens.append(PEDAL_ON if kind == KIND_PEDAL_ON else PEDAL_OFF)
        return tokens

    def tokenize_window(
        self,
        events: np.ndarray,
        end_frame: int,
        start_frame: int,
        num_patches: int,
    ) -> dict[str, np.ndarray]:
        """start_frame から num_patches 個のパッチをトークン化する。

        返り値:
            tokens: [P, max_patch_tokens] (PAD 埋め)
            patch_valid: [P] 曲の終端より後のパッチは False
            pedal_state: [P] パッチ開始時点でペダルが踏まれているか
        """
        F = self.patch_frames
        L = self.config.max_patch_tokens
        tokens = np.full((num_patches, L), PAD, dtype=np.int64)
        patch_valid = np.zeros(num_patches, dtype=bool)
        pedal_state = np.zeros(num_patches, dtype=np.int64)

        last_patch = (end_frame - 1 - start_frame) // F
        num_valid = int(min(num_patches, max(last_patch + 1, 0)))
        patch_valid[:num_valid] = True

        # パッチ開始時点のペダル状態 (on < 開始 <= off なら踏まれている)
        pedal_on = events[events[:, KIND] == KIND_PEDAL_ON, ONSET]
        pedal_off = events[events[:, KIND] == KIND_PEDAL_OFF, ONSET]
        if len(pedal_on):  # ペダルを一度も踏まない曲もある
            patch_starts = start_frame + np.arange(num_valid) * F
            index = np.searchsorted(pedal_on, patch_starts, side="left") - 1
            pedal_state[:num_valid] = (index >= 0) & (pedal_off[np.maximum(index, 0)] >= patch_starts)

        relative = events[:, ONSET].astype(np.int64) - start_frame
        selected = (relative >= 0) & (relative < num_valid * F)
        window_events = events[selected]
        relative = relative[selected]
        patch_index = relative // F
        bounds = np.searchsorted(patch_index, np.arange(num_valid + 1))

        for p in range(num_valid):
            sequence: list[int] = []
            previous_onset = -1
            for i in range(bounds[p], bounds[p + 1]):
                local_onset = int(relative[i] - p * F)
                event = self.event_tokens(window_events[i], local_onset, local_onset != previous_onset)
                if len(sequence) + len(event) > L - 1:
                    break
                sequence += event
                previous_onset = local_onset
            sequence.append(EOS if p == last_patch else EOP)
            tokens[p, : len(sequence)] = sequence
        return {"tokens": tokens, "patch_valid": patch_valid, "pedal_state": pedal_state}

    # ------------------------------------------------------------------
    # トークン -> イベント配列 -> MIDI
    # ------------------------------------------------------------------
    def patches_to_events(self, patches: list[list[int]]) -> np.ndarray:
        """生成したパッチごとのトークン列 (終端トークン込みでも可) をイベント配列に戻す"""
        rows = []
        for p, sequence in enumerate(patches):
            onset = 0
            pending: list[int] = []
            for token in sequence:
                if token in (PAD, EOP, EOS):
                    break
                if self.time_offset <= token < self.pitch_offset:
                    onset = token - self.time_offset
                elif token in (PEDAL_ON, PEDAL_OFF):
                    kind = KIND_PEDAL_ON if token == PEDAL_ON else KIND_PEDAL_OFF
                    rows.append([p * self.patch_frames + onset, kind, 0, 0, 0])
                else:
                    pending.append(token)
                    if len(pending) == 3:
                        pitch = pending[0] - self.pitch_offset + self.config.pitch_min
                        duration = int(self.duration_values[pending[1] - self.duration_offset])
                        velocity = self.velocity_value(pending[2] - self.velocity_offset)
                        rows.append([p * self.patch_frames + onset, KIND_NOTE, pitch, duration, velocity])
                        pending = []
        if not rows:
            return np.zeros((0, 5), dtype=np.int32)
        return trim_overlapping_notes(sort_events(np.asarray(rows, dtype=np.int64))).astype(np.int32)

    def events_to_midi(self, events: np.ndarray, path: str | Path) -> None:
        from symusic import ControlChange, Note, Score, Track

        fr = self.config.frame_rate
        score = Score(ttype="second")
        track = Track(name="Piano", program=0, ttype="second")
        for onset, kind, pitch, duration, velocity in events.tolist():
            if kind == KIND_NOTE:
                track.notes.append(Note(onset / fr, duration / fr, pitch, velocity, "second"))
            else:
                track.controls.append(ControlChange(onset / fr, 64, 127 if kind == KIND_PEDAL_ON else 0, "second"))
        score.tracks.append(track)
        score.dump_midi(str(path))


def sort_events(events: np.ndarray) -> np.ndarray:
    return events[np.lexsort((events[:, PITCH], events[:, KIND], events[:, ONSET]))]


def trim_overlapping_notes(events: np.ndarray) -> np.ndarray:
    """同じ鍵盤の音が次の音に重なっていたら、次の音を優先して前の音をその onset で切る。

    duration はビン化 (長い音ほど粗い) と予測の誤差で長めに出ることがあり、同じ鍵盤で次の音と重なると
    MIDI の再生では前の音の note off が次の音を止めてしまう。実際のピアノでも同じ鍵盤は重ねて鳴らせない。
    違う鍵盤どうしの重なり (和音やレガート) はそのまま残す。
    """
    events = events.copy()
    note_index = np.flatnonzero(events[:, KIND] == KIND_NOTE)
    notes = events[note_index]
    order = np.lexsort((notes[:, ONSET], notes[:, PITCH]))
    onset, pitch = notes[order, ONSET], notes[order, PITCH]
    end = onset + notes[order, DURATION]
    same_key_next = np.flatnonzero(pitch[:-1] == pitch[1:])
    overlap = same_key_next[end[same_key_next] > onset[same_key_next + 1]]
    target = note_index[order[overlap]]
    events[target, DURATION] = np.maximum(1, onset[overlap + 1] - onset[overlap])
    return events


class PatchGrammar:
    """生成時に 1 パッチ分のトークン列が文法に従うよう、次に出せるトークンを絞る"""

    def __init__(self, tokenizer: PianoTokenizer, pedal_down: bool) -> None:
        self.tok = tokenizer
        self.pedal_down = pedal_down
        self.length = 0
        self.stage = "event_start"  # event_start / after_time / after_pitch / after_duration
        self.onset = -1
        # 同じ onset 内の並び: 0 = まだ何もない, 1 = PEDAL_OFF, 2 = PEDAL_ON, 3 = ノート
        self.order = 0
        self.last_pitch = -1
        self.finished = False
        self.song_end = False

    def allowed(self) -> torch.Tensor:
        tok = self.tok
        mask = torch.zeros(tok.vocab_size, dtype=torch.bool)
        if self.finished:
            mask[PAD] = True
            return mask
        # 上限に達したら終端だけを許す
        if self.length >= tok.config.max_patch_tokens - 1 and self.stage == "event_start":
            mask[[EOP, EOS]] = True
            return mask
        if self.stage == "after_pitch":
            mask[tok.duration_offset : tok.velocity_offset] = True
        elif self.stage == "after_duration":
            mask[tok.velocity_offset :] = True
        elif self.stage == "after_time":
            self._allow_same_onset(mask, after_time=True)
        else:
            mask[[EOP, EOS]] = True
            # 残りトークン数で 1 イベントが収まる場合だけ新しいイベントを始められる
            if self.length + 5 <= tok.config.max_patch_tokens:
                mask[tok.time_offset + self.onset + 1 : tok.pitch_offset] = True
                if self.onset >= 0:
                    self._allow_same_onset(mask, after_time=False)
        return mask

    def _allow_same_onset(self, mask: torch.Tensor, after_time: bool) -> None:
        tok = self.tok
        order = 0 if after_time else self.order
        if order < 1 and self.pedal_down:
            mask[PEDAL_OFF] = True
        if order < 2 and not self.pedal_down:
            mask[PEDAL_ON] = True
        last_pitch = -1 if after_time else self.last_pitch
        first_pitch = max(0, last_pitch - tok.config.pitch_min + 1) if order == 3 else 0
        mask[tok.pitch_offset + first_pitch : tok.duration_offset] = True

    def update(self, token: int) -> None:
        tok = self.tok
        if self.finished:
            return
        self.length += 1
        if token in (EOP, EOS):
            self.finished = True
            self.song_end = token == EOS
        elif tok.time_offset <= token < tok.pitch_offset:
            self.onset = token - tok.time_offset
            self.order = 0
            self.last_pitch = -1
            self.stage = "after_time"
        elif token in (PEDAL_ON, PEDAL_OFF):
            self.pedal_down = token == PEDAL_ON
            self.order = 2 if token == PEDAL_ON else 1
            self.stage = "event_start"
        elif tok.pitch_offset <= token < tok.duration_offset:
            self.last_pitch = token - tok.pitch_offset + tok.config.pitch_min
            self.order = 3
            self.stage = "after_pitch"
        elif tok.duration_offset <= token < tok.velocity_offset:
            self.stage = "after_duration"
        else:
            self.stage = "event_start"
