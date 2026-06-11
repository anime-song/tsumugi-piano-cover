from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from external.amt_model.models.interval_boundaries import PitchIntervalTargets
from symusic import ControlChange, Note, Score, Tempo, TimeSignature, Track

from src_v2.config import TargetRollConfig
from src_v2.data.midi import NoteEvent, load_trimmed_target_events


@dataclass(frozen=True)
class SegmentSong:
    segments: torch.Tensor
    segment_times: torch.Tensor
    segment_valid_lengths: torch.Tensor
    interval_targets: tuple[PitchIntervalTargets | None, ...]
    num_frames: int
    duration_seconds: float


def _last_note_or_pedal_end(notes: list[NoteEvent], pedals: list[tuple[float, float]]) -> float:
    last_note_end = max((note.end for note in notes), default=0.0)
    last_pedal_end = max((end for _, end in pedals), default=0.0)
    return max(last_note_end, last_pedal_end)


def _frame_index(time_seconds: float, frame_seconds: float) -> int:
    return max(0, int(round(time_seconds / frame_seconds)))


def compute_target_num_frames(
    notes: list[NoteEvent],
    pedals: list[tuple[float, float]],
    config: TargetRollConfig,
    max_time_seconds: float | None = None,
) -> int:
    last_end = _last_note_or_pedal_end(notes, pedals)
    if max_time_seconds is not None:
        last_end = min(last_end, max_time_seconds)
    return max(1, _frame_index(last_end, config.frame_seconds) + 1)


def compute_segment_start_frames(num_frames: int, config: TargetRollConfig) -> list[int]:
    frames_per_phrase = config.frames_per_phrase
    hop_frames = config.frames_per_hop
    if num_frames <= frames_per_phrase:
        start_frames = [0]
    else:
        start_frames = list(range(0, num_frames - frames_per_phrase + 1, hop_frames))
        # 最後のセグメントを最終フレームにアライン
        final_start = num_frames - frames_per_phrase
        if start_frames[-1] != final_start:
            start_frames.append(final_start)
    if config.max_phrases_per_song is not None:
        start_frames = start_frames[: config.max_phrases_per_song]
    return start_frames


def midi_to_target_roll(
    notes: list[NoteEvent],
    pedals: list[tuple[float, float]],
    config: TargetRollConfig,
    max_time_seconds: float | None = None,
    transpose_semitones: int = 0,
) -> torch.Tensor:
    num_frames = compute_target_num_frames(
        notes,
        pedals,
        config,
        max_time_seconds=max_time_seconds,
    )
    roll = torch.zeros((num_frames, config.feature_dim), dtype=torch.float32)

    pitch_count = config.pitch_count
    onset_slice = slice(0, pitch_count)
    sustain_slice = slice(pitch_count, pitch_count * 2)
    velocity_slice = slice(pitch_count * 2, pitch_count * 3)
    pedal_index = config.feature_dim - 1

    for note in notes:
        if max_time_seconds is not None and note.start > max_time_seconds:
            continue
        shifted_pitch = note.pitch + transpose_semitones
        pitch = int(max(config.pitch_min, min(shifted_pitch, config.pitch_max)) - config.pitch_min)
        start_frame = _frame_index(note.start, config.frame_seconds)
        end_time = note.end if max_time_seconds is None else min(note.end, max_time_seconds)
        end_frame = max(start_frame + 1, _frame_index(end_time, config.frame_seconds))
        start_frame = min(start_frame, num_frames - 1)
        end_frame = min(end_frame, num_frames)

        roll[start_frame, onset_slice.start + pitch] = 1.0
        roll[start_frame:end_frame, sustain_slice.start + pitch] = 1.0
        roll[start_frame:end_frame, velocity_slice.start + pitch] = max(note.velocity, 1) / 127.0

    for start, end in pedals:
        if max_time_seconds is not None and start > max_time_seconds:
            continue
        start_frame = _frame_index(start, config.frame_seconds)
        end_time = end if max_time_seconds is None else min(end, max_time_seconds)
        end_frame = max(start_frame + 1, _frame_index(end_time, config.frame_seconds))
        start_frame = min(start_frame, num_frames - 1)
        end_frame = min(end_frame, num_frames)
        roll[start_frame:end_frame, pedal_index] = 1.0

    return roll


def _extract_note_intervals_from_events(
    notes: list[NoteEvent],
    config: TargetRollConfig,
    num_frames: int,
    max_time_seconds: float | None = None,
    transpose_semitones: int = 0,
) -> list[list[tuple[int, int]]]:
    """NoteEventリストから直接、ピッチごとのフレーム区間を構築する。
    ロールをスキャンする O(pitch_count × num_frames) のループを完全に排除。"""
    pitch_count = config.pitch_count
    intervals: list[list[tuple[int, int]]] = [[] for _ in range(pitch_count)]

    for note in notes:
        if max_time_seconds is not None and note.start > max_time_seconds:
            continue
        shifted_pitch = note.pitch + transpose_semitones
        pitch_index = int(max(config.pitch_min, min(shifted_pitch, config.pitch_max)) - config.pitch_min)
        if pitch_index < 0 or pitch_index >= pitch_count:
            continue

        start_frame = _frame_index(note.start, config.frame_seconds)
        end_time = note.end if max_time_seconds is None else min(note.end, max_time_seconds)
        # end_frame は sustain の最終フレーム（inclusive）
        end_frame = max(start_frame, _frame_index(end_time, config.frame_seconds) - 1)
        start_frame = min(start_frame, num_frames - 1)
        end_frame = min(end_frame, num_frames - 1)

        intervals[pitch_index].append((start_frame, end_frame))

    # ピッチごとに開始フレーム順でソートし、重複する区間をマージ
    for pitch_index in range(pitch_count):
        raw = intervals[pitch_index]
        if not raw:
            continue
        raw.sort()
        merged: list[tuple[int, int]] = [raw[0]]
        for start, end in raw[1:]:
            prev_start, prev_end = merged[-1]
            # onset が同じフレーム → 後続ノートの onset で先行ノートが切れる
            if start == prev_start:
                merged[-1] = (prev_start, max(prev_end, end))
            elif start <= prev_end + 1:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        intervals[pitch_index] = merged

    return intervals


def _build_segment_interval_targets(
    notes: list[NoteEvent],
    num_frames: int,
    start_frames: list[int],
    valid_lengths: list[int],
    config: TargetRollConfig,
    max_time_seconds: float | None = None,
    transpose_semitones: int = 0,
) -> tuple[PitchIntervalTargets, ...]:
    full_intervals = _extract_note_intervals_from_events(
        notes, config, num_frames,
        max_time_seconds=max_time_seconds,
        transpose_semitones=transpose_semitones,
    )
    targets: list[PitchIntervalTargets] = []
    pitch_start_indices = [0] * config.pitch_count

    for start_frame, valid_length in zip(start_frames, valid_lengths, strict=True):
        end_frame_exclusive = start_frame + valid_length
        pitch_intervals: list[list[tuple[int, int]]] = [[] for _ in range(config.pitch_count)]
        has_onset: list[list[bool]] = [[] for _ in range(config.pitch_count)]
        has_offset: list[list[bool]] = [[] for _ in range(config.pitch_count)]
        onset_offsets: list[list[float]] = [[] for _ in range(config.pitch_count)]
        offset_offsets: list[list[float]] = [[] for _ in range(config.pitch_count)]

        for pitch_index, intervals in enumerate(full_intervals):
            idx = pitch_start_indices[pitch_index]
            while idx < len(intervals) and intervals[idx][1] < start_frame:
                idx += 1
            pitch_start_indices[pitch_index] = idx

            for j in range(idx, len(intervals)):
                interval_start, interval_end = intervals[j]
                if interval_start >= end_frame_exclusive:
                    break
                local_start = max(interval_start, start_frame) - start_frame
                local_end = min(interval_end, end_frame_exclusive - 1) - start_frame
                pitch_intervals[pitch_index].append((local_start, local_end))
                has_onset[pitch_index].append(interval_start >= start_frame)
                has_offset[pitch_index].append(interval_end < end_frame_exclusive or interval_end >= num_frames - 1)
                onset_offsets[pitch_index].append(0.0)
                offset_offsets[pitch_index].append(0.0)

        targets.append(
            PitchIntervalTargets(
                intervals=pitch_intervals,
                has_onset=has_onset,
                has_offset=has_offset,
                onset_offsets=onset_offsets,
                offset_offsets=offset_offsets,
            )
        )
    return tuple(targets)


def segment_grid_from_roll(
    roll: torch.Tensor,
    config: TargetRollConfig,
    skip_intervals: bool = False,
    notes: list[NoteEvent] | None = None,
    max_time_seconds: float | None = None,
    transpose_semitones: int = 0,
) -> SegmentSong:
    num_frames = int(roll.shape[0])

    start_frames = compute_segment_start_frames(num_frames, config)
    frames_per_phrase = config.frames_per_phrase

    segments = torch.zeros((len(start_frames), frames_per_phrase, config.feature_dim), dtype=torch.float32)
    valid_lengths: list[int] = []

    for segment_index, start_frame in enumerate(start_frames):
        end_frame = min(num_frames, start_frame + frames_per_phrase)
        window = roll[start_frame:end_frame]
        segments[segment_index, : window.shape[0]] = window
        valid_lengths.append(int(window.shape[0]))

    segment_times = torch.tensor(
        [start_frame * config.frame_seconds for start_frame in start_frames], dtype=torch.float32
    )
    segment_valid_lengths = torch.tensor(valid_lengths, dtype=torch.long)

    if skip_intervals or notes is None:
        interval_targets: tuple[PitchIntervalTargets | None, ...] = tuple([None] * len(start_frames))
    else:
        interval_targets = _build_segment_interval_targets(
            notes, num_frames, start_frames, valid_lengths, config,
            max_time_seconds=max_time_seconds,
            transpose_semitones=transpose_semitones,
        )

    duration_seconds = max(0.0, (num_frames - 1) * config.frame_seconds)
    return SegmentSong(
        segments=segments,
        segment_times=segment_times,
        segment_valid_lengths=segment_valid_lengths,
        interval_targets=interval_targets,
        num_frames=num_frames,
        duration_seconds=duration_seconds,
    )


def target_midi_to_segment_song(
    midi_path: str | Path,
    config: TargetRollConfig,
    max_time_seconds: float | None = None,
    transpose_semitones: int = 0,
    skip_intervals: bool = False,
) -> SegmentSong:
    # MIDIからピアノロールへの変換とセグメント分割
    notes, pedals = load_trimmed_target_events(midi_path, min_duration_seconds=config.min_duration_seconds)
    roll = midi_to_target_roll(
        notes,
        pedals,
        config,
        max_time_seconds=max_time_seconds,
        transpose_semitones=transpose_semitones,
    )
    return segment_grid_from_roll(
        roll, config,
        skip_intervals=skip_intervals,
        notes=notes,
        max_time_seconds=max_time_seconds,
        transpose_semitones=transpose_semitones,
    )


def segment_grid_from_duration(duration_seconds: float, config: TargetRollConfig) -> SegmentSong:
    duration_seconds = max(duration_seconds, config.frame_seconds)
    num_frames = max(1, _frame_index(duration_seconds, config.frame_seconds) + 1)
    empty_roll = torch.zeros((num_frames, config.feature_dim), dtype=torch.float32)
    return segment_grid_from_roll(empty_roll, config)


def _segment_center_weights(length: int, device: torch.device) -> torch.Tensor:
    # 三角窓による重み計算
    if length <= 1:
        return torch.ones((length,), dtype=torch.float32, device=device)
    positions = torch.linspace(-1.0, 1.0, steps=length, device=device)
    weights = 1.0 - positions.abs()
    return weights.clamp_min(1.0e-3)


def segments_to_roll(
    segment_rolls: torch.Tensor,
    segment_times: torch.Tensor,
    num_frames: int,
    config: TargetRollConfig,
) -> torch.Tensor:
    # セグメント群を結合して曲全体のピアノロールを再構成
    if segment_rolls.numel() == 0:
        return torch.zeros((num_frames, config.feature_dim), dtype=torch.float32)

    merged = torch.zeros((num_frames, config.feature_dim), dtype=torch.float32, device=segment_rolls.device)
    frame_weight_sum = torch.zeros((num_frames, 1), dtype=torch.float32, device=segment_rolls.device)
    velocity_sum = torch.zeros((num_frames, config.pitch_count), dtype=torch.float32, device=segment_rolls.device)
    velocity_weight = torch.zeros((num_frames, config.pitch_count), dtype=torch.float32, device=segment_rolls.device)

    pitch_count = config.pitch_count
    onset_slice = slice(0, pitch_count)
    sustain_slice = slice(pitch_count, pitch_count * 2)
    velocity_slice = slice(pitch_count * 2, pitch_count * 3)
    pedal_index = config.feature_dim - 1

    for segment_index in range(int(segment_rolls.shape[0])):
        start_frame = _frame_index(float(segment_times[segment_index].item()), config.frame_seconds)
        end_frame = min(num_frames, start_frame + config.frames_per_phrase)
        window = segment_rolls[segment_index, : end_frame - start_frame]

        frame_weights = _segment_center_weights(window.shape[0], window.device).unsqueeze(-1)

        merged[start_frame:end_frame, onset_slice] += window[:, onset_slice] * frame_weights
        merged[start_frame:end_frame, sustain_slice] += window[:, sustain_slice] * frame_weights
        merged[start_frame:end_frame, pedal_index : pedal_index + 1] += window[:, pedal_index : pedal_index + 1] * (
            frame_weights
        )
        frame_weight_sum[start_frame:end_frame] += frame_weights

        # velocity は発音中のフレームのみ加重平均
        note_weight = torch.maximum(window[:, onset_slice], window[:, sustain_slice])
        weighted_note_weight = note_weight * frame_weights
        velocity_sum[start_frame:end_frame] += window[:, velocity_slice] * weighted_note_weight
        velocity_weight[start_frame:end_frame] += weighted_note_weight

    merged[:, onset_slice] = merged[:, onset_slice] / frame_weight_sum.clamp_min(1.0e-6)
    merged[:, sustain_slice] = merged[:, sustain_slice] / frame_weight_sum.clamp_min(1.0e-6)
    merged[:, pedal_index : pedal_index + 1] = merged[:, pedal_index : pedal_index + 1] / frame_weight_sum.clamp_min(
        1.0e-6
    )
    merged[:, velocity_slice] = velocity_sum / velocity_weight.clamp_min(1.0e-6)
    return merged.cpu()


def roll_to_score(
    roll: torch.Tensor,
    config: TargetRollConfig,
    velocity_default: int = 96,
) -> Score:
    # ピアノロールからMIDI（Score）へ変換
    score = Score.from_tpq(480)
    track = Track()
    ticks_per_second = score.ticks_per_quarter * 2.0

    pitch_count = config.pitch_count
    onset = roll[:, :pitch_count]
    sustain = roll[:, pitch_count : pitch_count * 2]
    velocity = roll[:, pitch_count * 2 : pitch_count * 3]
    pedal = roll[:, -1]

    for pitch_index in range(pitch_count):
        active = False
        note_start = 0
        velocity_values: list[float] = []
        for frame_index in range(int(roll.shape[0])):
            onset_active = float(onset[frame_index, pitch_index].item()) >= config.onset_threshold
            sustain_active = float(sustain[frame_index, pitch_index].item()) >= config.sustain_threshold

            if onset_active:
                # 発音中の場合は既存の音符を登録して新規開始
                if active:
                    end_time = frame_index * config.frame_seconds
                    start_tick = int(round(note_start * ticks_per_second))
                    end_tick = int(round(end_time * ticks_per_second))
                    midi_velocity = int(round((sum(velocity_values) / max(1, len(velocity_values))) * 127.0))
                    track.notes.append(
                        Note(
                            start_tick,
                            max(1, end_tick - start_tick),
                            config.pitch_min + pitch_index,
                            max(1, min(127, midi_velocity or velocity_default)),
                        )
                    )
                active = True
                note_start = frame_index * config.frame_seconds
                velocity_values = []

            if active and sustain_active:
                velocity_values.append(float(velocity[frame_index, pitch_index].item()))

            # sustain終了または次の発音開始による音符の登録
            next_starts = False
            if frame_index + 1 < int(roll.shape[0]):
                next_starts = float(onset[frame_index + 1, pitch_index].item()) >= config.onset_threshold
            if active and (not sustain_active or next_starts or frame_index == int(roll.shape[0]) - 1):
                end_frame = frame_index + 1 if sustain_active else frame_index
                end_time = max(note_start + config.frame_seconds, end_frame * config.frame_seconds)
                start_tick = int(round(note_start * ticks_per_second))
                end_tick = int(round(end_time * ticks_per_second))
                midi_velocity = int(round((sum(velocity_values) / max(1, len(velocity_values))) * 127.0))
                track.notes.append(
                    Note(
                        start_tick,
                        max(1, end_tick - start_tick),
                        config.pitch_min + pitch_index,
                        max(1, min(127, midi_velocity or velocity_default)),
                    )
                )
                active = False
                velocity_values = []

    pedal_active = False
    pedal_start = 0.0
    for frame_index in range(int(roll.shape[0])):
        pedal_on = float(pedal[frame_index].item()) >= config.pedal_threshold
        if pedal_on and not pedal_active:
            pedal_active = True
            pedal_start = frame_index * config.frame_seconds
            start_tick = int(round(pedal_start * ticks_per_second))
            track.controls.append(ControlChange(start_tick, 64, 127))
        if pedal_active and (not pedal_on or frame_index == int(roll.shape[0]) - 1):
            end_frame = frame_index + 1 if pedal_on else frame_index
            end_time = max(pedal_start + config.frame_seconds, end_frame * config.frame_seconds)
            end_tick = int(round(end_time * ticks_per_second))
            track.controls.append(ControlChange(end_tick, 64, 0))
            pedal_active = False

    score.tracks.append(track)
    score.tempos.append(Tempo(0, 120))
    score.time_signatures.append(TimeSignature(0, 4, 4))
    return score


def write_roll_midi(roll: torch.Tensor, config: TargetRollConfig, output_path: str | Path) -> None:
    roll_to_score(roll, config).dump_midi(str(output_path))
