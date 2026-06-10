from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from symusic import Score

from src_v2.config import SourceChunkConfig


@dataclass(frozen=True)
class NoteEvent:
    start: float
    end: float
    pitch: int
    program: int
    is_drum: int
    track_role: int
    velocity: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


def _score_to_second(path: str | Path):
    return Score.from_file(str(path)).to("second")


def _infer_source_track_role(track_name: str | None) -> int:
    # トラック名から役割（メロディ/ハーモニー/ベースなど）を推定
    if not track_name:
        return 0
    normalized = track_name.strip().lower()
    if "melody" in normalized:
        return 1
    if "harmony" in normalized:
        return 2
    if "bass" in normalized:
        return 3
    return 0


def _sorted_source_events(path: str | Path, min_duration_seconds: float) -> list[NoteEvent]:
    # MIDIから音符イベントをロードして時間順にソート
    score = _score_to_second(path)
    events: list[NoteEvent] = []
    for track in score.tracks:
        program = 128 if track.is_drum else int(track.program)
        is_drum = int(track.is_drum)
        track_role = _infer_source_track_role(getattr(track, "name", None))
        for note in track.notes:
            start = float(note.time)
            end = float(note.time + note.duration)
            # 極端に短い音符は最小持続時間に補正
            if end - start < min_duration_seconds:
                end = start + min_duration_seconds
            events.append(
                NoteEvent(
                    start=start,
                    end=end,
                    pitch=int(note.pitch),
                    program=program,
                    is_drum=is_drum,
                    track_role=track_role,
                    velocity=int(note.velocity),
                )
            )

    events.sort(key=lambda event: (event.start, event.pitch, event.program))
    return events


def _sorted_target_events(
    path: str | Path, min_duration_seconds: float
) -> tuple[list[NoteEvent], list[tuple[float, float]]]:
    # ピアノカバーMIDIから、音高とペダルの区間情報をロード
    score = _score_to_second(path)
    events: list[NoteEvent] = []
    pedals: list[tuple[float, float]] = []
    for track in score.tracks:
        for note in track.notes:
            start = float(note.time)
            end = float(note.time + note.duration)
            if end - start < min_duration_seconds:
                end = start + min_duration_seconds
            events.append(
                NoteEvent(
                    start=start,
                    end=end,
                    pitch=int(note.pitch),
                    program=0,
                    is_drum=0,
                    track_role=0,
                    velocity=int(note.velocity),
                )
            )
        for pedal in track.pedals:
            pedals.append((float(pedal.time), float(pedal.time + pedal.duration)))

    events.sort(key=lambda event: (event.start, event.pitch))
    pedals.sort()
    return events, pedals


def _trim_events(events: list[NoteEvent]) -> tuple[list[NoteEvent], float]:
    # 曲の開始時間が0秒になるよう全体をシフト
    if not events:
        return [], 0.0
    trim = min(event.start for event in events)
    trimmed = [
        NoteEvent(
            start=event.start - trim,
            end=event.end - trim,
            pitch=event.pitch,
            program=event.program,
            is_drum=event.is_drum,
            track_role=event.track_role,
            velocity=event.velocity,
        )
        for event in events
    ]
    return trimmed, trim


def _trim_pedals(pedals: list[tuple[float, float]], trim: float) -> list[tuple[float, float]]:
    return [(start - trim, end - trim) for start, end in pedals]


def load_trimmed_source_events(path: str | Path, config: SourceChunkConfig) -> list[NoteEvent]:
    # 原曲の音符データをロードして先頭を0秒にシフト
    events = _sorted_source_events(path, min_duration_seconds=config.min_duration_seconds)
    trimmed, _ = _trim_events(events)
    return trimmed


def load_trimmed_target_events(
    path: str | Path,
    min_duration_seconds: float = 0.03,
) -> tuple[list[NoteEvent], list[tuple[float, float]]]:
    # カバーの音符とペダルデータをロードして先頭を0秒にシフト
    events, pedals = _sorted_target_events(path, min_duration_seconds=min_duration_seconds)
    trimmed_events, trim = _trim_events(events)
    trimmed_pedals = _trim_pedals(pedals, trim)
    return trimmed_events, trimmed_pedals


def chunk_source_events(
    events: list[NoteEvent],
    config: SourceChunkConfig,
) -> dict[str, torch.Tensor]:
    # 音符リストを時間窓（Chunk）ごとに分割してテンソルに変換
    if not events:
        empty = torch.zeros((0, config.max_notes_per_chunk, 4), dtype=torch.float32)
        empty_long = torch.zeros((0, config.max_notes_per_chunk), dtype=torch.long)
        empty_bool = torch.zeros((0, config.max_notes_per_chunk), dtype=torch.bool)
        return {
            "source_features": empty,
            "source_programs": empty_long,
            "source_drums": empty_long,
            "source_track_roles": empty_long,
            "source_note_mask": empty_bool,
            "source_chunk_times": torch.zeros((0,), dtype=torch.float32),
        }

    # 最終音符の終了位置からチャンク総数を算出
    last_end = max(event.end for event in events)
    num_chunks = max(1, int((max(0.0, last_end - config.window_seconds) / config.hop_seconds) + 1.0))
    if config.max_chunks_per_song is not None:
        num_chunks = min(num_chunks, config.max_chunks_per_song)

    feature_chunks = []
    program_chunks = []
    drum_chunks = []
    role_chunks = []
    mask_chunks = []
    chunk_times = []

    for chunk_index in range(num_chunks):
        chunk_start = chunk_index * config.hop_seconds
        chunk_end = chunk_start + config.window_seconds
        # 窓に重なる音符イベントを抽出
        overlapping = [event for event in events if event.start < chunk_end and event.end > chunk_start]
        # 時間、ピッチ、音色順でソートして最大数でクリップ
        overlapping.sort(key=lambda event: (max(event.start, chunk_start), event.pitch, event.program))
        overlapping = overlapping[: config.max_notes_per_chunk]

        features = torch.zeros((config.max_notes_per_chunk, 4), dtype=torch.float32)
        programs = torch.zeros((config.max_notes_per_chunk,), dtype=torch.long)
        drums = torch.zeros((config.max_notes_per_chunk,), dtype=torch.long)
        roles = torch.zeros((config.max_notes_per_chunk,), dtype=torch.long)
        mask = torch.zeros((config.max_notes_per_chunk,), dtype=torch.bool)

        for note_index, event in enumerate(overlapping):
            # 特徴量: 1. ピッチ, 2. 相対開始位置, 3. 対数持続時間, 4. 左境界超過フラグ
            features[note_index, 0] = event.pitch / 127.0
            features[note_index, 1] = (event.start - chunk_start) / config.window_seconds
            features[note_index, 2] = torch.log1p(torch.tensor(event.duration)).item()
            features[note_index, 3] = float(event.start < chunk_start)
            programs[note_index] = event.program
            drums[note_index] = event.is_drum
            roles[note_index] = event.track_role
            mask[note_index] = True

        feature_chunks.append(features)
        program_chunks.append(programs)
        drum_chunks.append(drums)
        role_chunks.append(roles)
        mask_chunks.append(mask)
        chunk_times.append(chunk_start)

    return {
        "source_features": torch.stack(feature_chunks, dim=0),
        "source_programs": torch.stack(program_chunks, dim=0),
        "source_drums": torch.stack(drum_chunks, dim=0),
        "source_track_roles": torch.stack(role_chunks, dim=0),
        "source_note_mask": torch.stack(mask_chunks, dim=0),
        "source_chunk_times": torch.tensor(chunk_times, dtype=torch.float32),
    }
