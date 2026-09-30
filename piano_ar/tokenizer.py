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


class BatchGrammar:
    """生成時に 1 パッチ分のトークン列が文法に従うよう、次に出せるトークンを絞る (N 行をまとめて)。
    状態は行ごとのテンソルで GPU の上に置き、トークンごとに CPU と行き来しない。その場で書き換える (同じテンソルのまま)
    ので、CUDA Graph に取り込める。パッチの頭で reset する"""

    # stage: 0 = イベントの始め, 1 = TIME の後, 2 = PITCH の後, 3 = DUR の後
    EVENT_START, AFTER_TIME, AFTER_PITCH, AFTER_DURATION = range(4)

    def __init__(self, tokenizer: PianoTokenizer, rows: int, device: torch.device | str) -> None:
        self.tok = tokenizer
        self.pedal_down = torch.zeros(rows, dtype=torch.bool, device=device)
        self.finished = torch.zeros(rows, dtype=torch.bool, device=device)
        self.song_end = torch.zeros(rows, dtype=torch.bool, device=device)
        self.length = torch.zeros(rows, dtype=torch.long, device=device)
        self.stage = torch.zeros(rows, dtype=torch.long, device=device)
        self.onset = torch.zeros(rows, dtype=torch.long, device=device)
        # 同じ onset 内の並び: 0 = まだ何もない, 1 = PEDAL_OFF, 2 = PEDAL_ON, 3 = ノート
        self.order = torch.zeros(rows, dtype=torch.long, device=device)
        self.last_pitch = torch.zeros(rows, dtype=torch.long, device=device)
        self.vocab = torch.arange(tokenizer.vocab_size, device=device)[None]

    def reset(self, pedal_down: torch.Tensor, finished: torch.Tensor | None = None) -> None:
        """パッチの頭の状態にする。pedal_down [N] はパッチの頭のペダル、finished [N] の行は PAD だけを出す (曲が終わった行)"""
        self.pedal_down.copy_(pedal_down)
        if finished is None:
            self.finished.zero_()
        else:
            self.finished.copy_(finished)
        self.song_end.zero_()
        self.length.zero_()
        self.stage.fill_(self.EVENT_START)
        self.onset.fill_(-1)
        self.order.zero_()
        self.last_pitch.fill_(-1)

    def allowed(self) -> torch.Tensor:
        """次に出せるトークン [N, vocab] (bool)"""
        tok, v = self.tok, self.vocab
        max_tokens = tok.config.max_patch_tokens
        stage = self.stage[:, None]
        end = (v == EOP) | (v == EOS)
        # 上限に達したら終端だけを許す
        limit = (self.length >= max_tokens - 1)[:, None] & (stage == self.EVENT_START)
        after_pitch = (stage == self.AFTER_PITCH) & (v >= tok.duration_offset) & (v < tok.velocity_offset)
        after_duration = (stage == self.AFTER_DURATION) & (v >= tok.velocity_offset)
        after_time = (stage == self.AFTER_TIME) & self._same_onset(
            torch.zeros_like(self.order), torch.full_like(self.last_pitch, -1)
        )
        # 残りトークン数で 1 イベントが収まる場合だけ新しいイベントを始められる
        room = (self.length + 5 <= max_tokens)[:, None]
        new_time = room & (v >= tok.time_offset + self.onset[:, None] + 1) & (v < tok.pitch_offset)
        same = room & (self.onset >= 0)[:, None] & self._same_onset(self.order, self.last_pitch)
        event_start = (stage == self.EVENT_START) & (end | new_time | same)
        mask = torch.where(limit, end, after_pitch | after_duration | after_time | event_start)
        return torch.where(self.finished[:, None], v == PAD, mask)

    def _same_onset(self, order: torch.Tensor, last_pitch: torch.Tensor) -> torch.Tensor:
        tok, v = self.tok, self.vocab
        order, pedal = order[:, None], self.pedal_down[:, None]
        pedal_off = (order < 1) & pedal & (v == PEDAL_OFF)
        pedal_on = (order < 2) & ~pedal & (v == PEDAL_ON)
        first_pitch = torch.where(order == 3, (last_pitch[:, None] - tok.config.pitch_min + 1).clamp(min=0), 0)
        pitch = (v >= tok.pitch_offset + first_pitch) & (v < tok.duration_offset)
        return pedal_off | pedal_on | pitch

    def update(self, token: torch.Tensor) -> None:
        """token [N] を出したあとの状態にする (終わった行は変えない)"""
        tok = self.tok
        live = ~self.finished
        is_time = live & (token >= tok.time_offset) & (token < tok.pitch_offset)
        is_pedal = live & ((token == PEDAL_ON) | (token == PEDAL_OFF))
        is_pitch = live & (token >= tok.pitch_offset) & (token < tok.duration_offset)
        is_duration = live & (token >= tok.duration_offset) & (token < tok.velocity_offset)
        is_velocity = live & (token >= tok.velocity_offset)
        self.length += live.long()
        self.song_end |= live & (token == EOS)
        self.finished |= live & ((token == EOP) | (token == EOS))
        self.onset.copy_(torch.where(is_time, token - tok.time_offset, self.onset))
        self.pedal_down.copy_(torch.where(is_pedal, token == PEDAL_ON, self.pedal_down))
        order = torch.where(token == PEDAL_ON, 2, 1)
        self.order.copy_(torch.where(is_time, 0, torch.where(is_pedal, order, torch.where(is_pitch, 3, self.order))))
        pitch = token - tok.pitch_offset + tok.config.pitch_min
        self.last_pitch.copy_(torch.where(is_time, -1, torch.where(is_pitch, pitch, self.last_pitch)))
        stage = torch.where(is_time, self.AFTER_TIME, self.stage)
        stage = torch.where(is_pedal | is_velocity, self.EVENT_START, stage)
        stage = torch.where(is_pitch, self.AFTER_PITCH, stage)
        self.stage.copy_(torch.where(is_duration, self.AFTER_DURATION, stage))
