from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import nn

from external.amt_model.models.interval_boundaries import gather_interval_endpoint_features
from external.amt_model.models.semi_crf import decode_pitch_intervals
from src_v2.config import SegmentAutoencoderConfig, TargetRollConfig
from src_v2.data.phrase import segments_to_roll


class IntervalScorer(nn.Module):
    def __init__(self, input_dim: int, head_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")

        self.head_dim = int(head_dim)
        self.proj = nn.Linear(input_dim, self.head_dim * 2 + 1)
        self.query_scale = 1.0 / math.sqrt(float(self.head_dim))

    def forward(self, pitch_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pitch_features.dim() != 5:
            raise ValueError("pitch_features must have shape [B, S, F, P, D]")

        interval_proj = self.proj(pitch_features)
        interval_query, interval_key, interval_diag = torch.split(
            interval_proj,
            [self.head_dim, self.head_dim, 1],
            dim=-1,
        )
        return interval_query * self.query_scale, interval_key, interval_diag.squeeze(-1)


class IntervalBoundaryPredictor(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 2),
        )

    def forward(self, interval_features: torch.Tensor) -> torch.Tensor:
        if interval_features.dim() != 2:
            raise ValueError("interval_features must have shape [N, D]")
        return self.net(interval_features)


def predict_interval_boundary_logits(
    boundary_predictor: IntervalBoundaryPredictor | None,
    pitch_features: torch.Tensor,
    interval_batch: Sequence[Sequence[Sequence[tuple[int, int]]]],
) -> tuple[torch.Tensor | None, list[tuple[int, int, int, int, int]]]:
    if boundary_predictor is None:
        return None, []
    interval_features, entries = gather_interval_endpoint_features(pitch_features, interval_batch)
    if not entries:
        return pitch_features.new_zeros((0, 2)), []
    return boundary_predictor(interval_features), entries


def decode_interval_boundary_flags(
    boundary_logits: torch.Tensor | None,
    entries: Sequence[tuple[int, int, int, int, int]],
    *,
    batch_size: int,
    num_pitches: int,
) -> list[list[list[tuple[bool, bool]]]]:
    flags = [[[] for _ in range(num_pitches)] for _ in range(batch_size)]
    if boundary_logits is None or not entries:
        return flags

    boundary_presence = boundary_logits > 0.0
    for row_index, (batch_index, pitch_index, _, _, _) in enumerate(entries):
        flags[batch_index][pitch_index].append(
            (
                bool(boundary_presence[row_index, 0].item()),
                bool(boundary_presence[row_index, 1].item()),
            )
        )
    return flags


@dataclass
class DecodedIntervalNote:
    pitch_index: int
    start_frame: int
    end_frame: int
    velocity_sum: float
    velocity_count: int
    has_onset: bool
    has_offset: bool

    @property
    def velocity(self) -> float:
        if self.velocity_count <= 0:
            return 0.0
        return float(self.velocity_sum) / float(self.velocity_count)


class PitchIntervalStitcher:
    def __init__(
        self,
        *,
        pitch_count: int,
        total_frames: int,
        merge_gap_frames: int = 1,
        merge_onset_frames: int = 1,
    ) -> None:
        self.pitch_count = int(pitch_count)
        self.total_frames = int(total_frames)
        self.merge_gap_frames = int(merge_gap_frames)
        self.merge_onset_frames = int(merge_onset_frames)
        self.notes_by_pitch: dict[int, list[DecodedIntervalNote]] = {
            pitch_index: [] for pitch_index in range(self.pitch_count)
        }
        self.last_closed_global_frames = [0] * self.pitch_count

    def get_forced_start_positions(
        self,
        *,
        window_start_frame: int,
        valid_model_frames: int,
    ) -> list[int]:
        if valid_model_frames <= 0:
            return [0] * self.pitch_count
        return [
            max(0, min(last_closed_frame - int(window_start_frame), int(valid_model_frames) - 1))
            for last_closed_frame in self.last_closed_global_frames
        ]

    def consume_window(
        self,
        *,
        intervals_by_pitch: list[list[tuple[int, int]]],
        boundary_flags_by_pitch: list[list[tuple[bool, bool]]] | None,
        velocity: torch.Tensor,
        window_start_frame: int,
        valid_model_frames: int,
    ) -> None:
        for pitch_index, pitch_intervals in enumerate(intervals_by_pitch):
            pitch_notes = self.notes_by_pitch[pitch_index]
            pitch_boundary_flags = boundary_flags_by_pitch[pitch_index] if boundary_flags_by_pitch is not None else []
            local_last_closed_frame: int | None = None

            for interval_index, (begin_frame, end_frame) in enumerate(pitch_intervals):
                boundary_flag = (
                    pitch_boundary_flags[interval_index] if interval_index < len(pitch_boundary_flags) else None
                )
                note = self._build_interval_note(
                    pitch_index=pitch_index,
                    begin_frame=int(begin_frame),
                    end_frame=int(end_frame),
                    boundary_flag=boundary_flag,
                    velocity=velocity,
                    window_start_frame=int(window_start_frame),
                    valid_model_frames=int(valid_model_frames),
                )
                if note is None:
                    continue

                if pitch_notes and int(note.start_frame) < int(pitch_notes[-1].end_frame):
                    if note.has_onset:
                        pitch_notes[-1] = note
                    else:
                        self._merge_note_segments(pitch_notes[-1], note, overwrite_offset=True)
                    if note.has_offset:
                        local_last_closed_frame = int(end_frame)
                    continue

                if note.has_onset:
                    pitch_notes.append(note)
                if note.has_offset:
                    local_last_closed_frame = int(end_frame)

            if local_last_closed_frame is not None:
                self.last_closed_global_frames[pitch_index] = int(window_start_frame) + int(local_last_closed_frame)

    def finalize(self) -> list[DecodedIntervalNote]:
        for pitch_notes in self.notes_by_pitch.values():
            if pitch_notes:
                pitch_notes[-1].has_offset = True

        notes = [note for pitch_notes in self.notes_by_pitch.values() for note in pitch_notes if note.has_offset]
        return self._merge_nearby_notes(notes)

    def _build_interval_note(
        self,
        *,
        pitch_index: int,
        begin_frame: int,
        end_frame: int,
        boundary_flag: tuple[bool, bool] | None,
        velocity: torch.Tensor,
        window_start_frame: int,
        valid_model_frames: int,
    ) -> DecodedIntervalNote | None:
        if valid_model_frames <= 0:
            return None

        if boundary_flag is None:
            has_onset = bool(begin_frame > 0 or int(window_start_frame) <= 0)
            has_offset = bool(
                end_frame < valid_model_frames - 1
                or int(window_start_frame) + int(valid_model_frames) >= self.total_frames
            )
        else:
            has_onset, has_offset = boundary_flag

        local_start = max(0, int(begin_frame))
        local_end = min(int(end_frame), int(valid_model_frames) - 1)
        if local_end < local_start:
            return None

        start_frame = max(0, min(int(window_start_frame) + local_start, self.total_frames - 1))
        end_frame_exclusive = max(
            start_frame + 1,
            min(int(window_start_frame) + local_end + 1, self.total_frames),
        )
        if start_frame >= self.total_frames:
            return None

        velocity_window = velocity[local_start : local_end + 1, pitch_index]
        return DecodedIntervalNote(
            pitch_index=int(pitch_index),
            start_frame=int(start_frame),
            end_frame=int(end_frame_exclusive),
            velocity_sum=float(velocity_window.sum().item()),
            velocity_count=int(velocity_window.shape[0]),
            has_onset=bool(has_onset),
            has_offset=bool(has_offset),
        )

    @staticmethod
    def _merge_note_segments(
        target: DecodedIntervalNote,
        source: DecodedIntervalNote,
        *,
        overwrite_offset: bool,
    ) -> None:
        target.end_frame = max(int(target.end_frame), int(source.end_frame))
        target.velocity_sum += float(source.velocity_sum)
        target.velocity_count += int(source.velocity_count)
        target.has_onset = bool(target.has_onset or source.has_onset)
        if overwrite_offset:
            target.has_offset = bool(source.has_offset)
        else:
            target.has_offset = bool(target.has_offset or source.has_offset)

    def _merge_nearby_notes(self, notes: list[DecodedIntervalNote]) -> list[DecodedIntervalNote]:
        if not notes:
            return []

        ordered = sorted(
            notes,
            key=lambda note: (
                note.pitch_index,
                note.start_frame,
                note.end_frame,
            ),
        )
        merged: list[DecodedIntervalNote] = []
        current = replace(ordered[0])
        for note in ordered[1:]:
            can_merge_by_gap = (
                note.pitch_index == current.pitch_index
                and note.start_frame <= current.end_frame + self.merge_gap_frames
                and not note.has_onset
            )
            can_merge_by_onset = (
                note.pitch_index == current.pitch_index
                and abs(note.start_frame - current.start_frame) <= self.merge_onset_frames
            )

            if can_merge_by_gap:
                self._merge_note_segments(current, note, overwrite_offset=True)
                continue
            if can_merge_by_onset:
                self._merge_note_segments(current, note, overwrite_offset=False)
                continue
            merged.append(current)
            current = replace(note)
        merged.append(current)
        return sorted(
            merged,
            key=lambda note: (
                note.start_frame,
                note.pitch_index,
                note.end_frame,
            ),
        )


def reconstruct_semi_crf_segments_to_roll(
    outputs: dict[str, torch.Tensor],
    model_config: SegmentAutoencoderConfig,
    roll_config: TargetRollConfig,
    segment_times: torch.Tensor,
    segment_valid_lengths: torch.Tensor,
    num_frames: int,
    boundary_predictor: IntervalBoundaryPredictor | None,
) -> torch.Tensor:
    stitcher = PitchIntervalStitcher(
        pitch_count=roll_config.pitch_count,
        total_frames=int(num_frames),
    )

    for segment_index in range(int(segment_valid_lengths.shape[0])):
        valid_length = int(segment_valid_lengths[segment_index].item())
        if valid_length <= 0:
            continue
        start_frame = int(round(float(segment_times[segment_index].item()) / float(roll_config.frame_seconds)))
        forced_start_pos = stitcher.get_forced_start_positions(
            window_start_frame=start_frame,
            valid_model_frames=valid_length,
        )
        decoded_intervals = decode_pitch_intervals(
            outputs["interval_query"][segment_index : segment_index + 1, :valid_length],
            outputs["interval_key"][segment_index : segment_index + 1, :valid_length],
            outputs["interval_diag"][segment_index : segment_index + 1, :valid_length],
            [valid_length],
            length_scaling=model_config.semi_crf_length_scaling,
            length_penalty=model_config.semi_crf_length_penalty,
            note_bias=model_config.semi_crf_note_bias,
            track_batch_size=model_config.semi_crf_track_batch_size,
            forced_start_pos=[forced_start_pos],
        )[0]

        boundary_flags_by_pitch = None
        if boundary_predictor is not None:
            boundary_logits, boundary_entries = predict_interval_boundary_logits(
                boundary_predictor,
                outputs["pitch_features"][segment_index : segment_index + 1, :valid_length],
                [decoded_intervals],
            )
            boundary_flags_by_pitch = decode_interval_boundary_flags(
                boundary_logits,
                boundary_entries,
                batch_size=1,
                num_pitches=roll_config.pitch_count,
            )[0]

        stitcher.consume_window(
            intervals_by_pitch=decoded_intervals,
            boundary_flags_by_pitch=boundary_flags_by_pitch,
            velocity=outputs["velocity"][segment_index, :valid_length],
            window_start_frame=start_frame,
            valid_model_frames=valid_length,
        )

    recon_roll = torch.zeros((num_frames, roll_config.feature_dim), dtype=torch.float32)
    onset_slice = slice(0, roll_config.pitch_count)
    sustain_slice = slice(roll_config.pitch_count, roll_config.pitch_count * 2)
    velocity_slice = slice(roll_config.pitch_count * 2, roll_config.pitch_count * 3)

    for note in stitcher.finalize():
        recon_roll[note.start_frame, onset_slice.start + note.pitch_index] = 1.0
        recon_roll[note.start_frame : note.end_frame, sustain_slice.start + note.pitch_index] = 1.0
        recon_roll[note.start_frame : note.end_frame, velocity_slice.start + note.pitch_index] = note.velocity

    pedal_segment_rolls = torch.zeros(
        (
            int(segment_valid_lengths.shape[0]),
            int(outputs["pedal_logits"].shape[1]),
            roll_config.feature_dim,
        ),
        dtype=torch.float32,
        device=outputs["pedal_logits"].device,
    )
    pedal_segment_rolls[:, :, -1] = torch.sigmoid(outputs["pedal_logits"].squeeze(-1))
    pedal_roll = segments_to_roll(
        pedal_segment_rolls,
        segment_times,
        int(num_frames),
        roll_config,
    )
    recon_roll[:, -1] = pedal_roll[:, -1]
    return recon_roll
