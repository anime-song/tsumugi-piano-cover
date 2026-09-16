from __future__ import annotations

from tsumugi_piano_cover.data.phrase import (
    SegmentSong,
    compute_segment_start_frames,
    compute_target_num_frames,
    midi_to_target_roll,
    roll_to_score,
    segment_grid_from_duration,
    segment_grid_from_roll,
    segments_to_roll,
    target_midi_to_segment_song,
    write_roll_midi,
)

__all__ = [
    "SegmentSong",
    "compute_segment_start_frames",
    "compute_target_num_frames",
    "midi_to_target_roll",
    "roll_to_score",
    "segment_grid_from_duration",
    "segment_grid_from_roll",
    "segments_to_roll",
    "target_midi_to_segment_song",
    "write_roll_midi",
]
