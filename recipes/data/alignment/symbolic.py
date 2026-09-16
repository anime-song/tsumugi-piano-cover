from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from recipes.data.alignment.audio_sync import compute_target_to_source_audio_sync_path
from recipes.data.index import PairEntry
from tsumugi_piano_cover.config import AlignmentConfig
from tsumugi_piano_cover.data.audio import get_audio_duration_seconds
from tsumugi_piano_cover.data.midi import NoteEvent, load_trimmed_target_events_with_offset


@dataclass(frozen=True)
class SymbolicAlignmentResult:
    """Alignment result kept under the historical class name for cache compatibility."""

    target_to_source: torch.Tensor
    target_has_match: torch.Tensor
    matched_target_frames: torch.Tensor
    matched_source_frames: torch.Tensor
    target_num_frames: int
    source_num_frames: int
    target_duration_seconds: float
    source_duration_seconds: float
    average_match_cost: float
    gap_ratio: float
    max_absolute_offset_seconds: float
    alignment_kind: str = "audio_sync"
    target_time_knots_seconds: torch.Tensor | None = None
    source_time_knots_seconds: torch.Tensor | None = None


def resolve_alignment_cache_path(cache_dir: str | Path, entry: PairEntry) -> Path:
    return Path(cache_dir) / entry.original_id / f"{entry.piano_id}.pt"


def build_alignment_cache_payload(
    entry: PairEntry,
    config: AlignmentConfig,
    result: SymbolicAlignmentResult,
) -> dict[str, object]:
    return {
        # Version 3 uses source-audio time directly; older caches used source MIDI trimming.
        "version": 3,
        "song_name": entry.song_name,
        "original_id": entry.original_id,
        "piano_id": entry.piano_id,
        "source_audio_path": entry.source_audio_path,
        "target_midi_path": entry.target_midi_path,
        "target_audio_path": entry.target_audio_path,
        "alignment_config": asdict(config),
        "alignment_kind": result.alignment_kind,
        "target_to_source": result.target_to_source.cpu(),
        "target_has_match": result.target_has_match.cpu(),
        "matched_target_frames": result.matched_target_frames.cpu(),
        "matched_source_frames": result.matched_source_frames.cpu(),
        "target_time_knots_seconds": (
            None if result.target_time_knots_seconds is None else result.target_time_knots_seconds.cpu()
        ),
        "source_time_knots_seconds": (
            None if result.source_time_knots_seconds is None else result.source_time_knots_seconds.cpu()
        ),
        "summary": {
            "target_num_frames": result.target_num_frames,
            "source_num_frames": result.source_num_frames,
            "target_duration_seconds": result.target_duration_seconds,
            "source_duration_seconds": result.source_duration_seconds,
            "average_match_cost": result.average_match_cost,
            "gap_ratio": result.gap_ratio,
            "max_absolute_offset_seconds": result.max_absolute_offset_seconds,
        },
    }


def save_alignment_cache(
    cache_dir: str | Path,
    entry: PairEntry,
    config: AlignmentConfig,
    result: SymbolicAlignmentResult,
) -> Path:
    path = resolve_alignment_cache_path(cache_dir, entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_alignment_cache_payload(entry, config, result), path)
    return path


def load_alignment_cache(path: str | Path) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"alignment cache must be a dict payload: {path}")
    return payload


def compute_pair_audio_alignment(entry: PairEntry, config: AlignmentConfig) -> SymbolicAlignmentResult:
    """Align target MIDI time to source audio time using the two audio tracks."""
    if entry.source_audio_path is None:
        raise FileNotFoundError(f"source audio path not found for original_id={entry.original_id}")
    if entry.target_audio_path is None:
        raise FileNotFoundError(f"target audio path not found for piano_id={entry.piano_id}")

    target_events, _, target_trim_seconds = load_trimmed_target_events_with_offset(
        entry.target_midi_path,
        min_duration_seconds=config.frame_seconds * 0.5,
    )
    target_duration_seconds = _notes_duration_seconds(target_events)

    audio_sync_path = compute_target_to_source_audio_sync_path(
        target_audio_path=entry.target_audio_path,
        source_audio_path=entry.source_audio_path,
        sample_rate=config.audio_sample_rate,
        feature_rate=config.sync_feature_rate,
        step_weights=config.sync_step_weights,
        threshold_rec=config.sync_threshold_rec,
    )
    source_duration_seconds = get_audio_duration_seconds(entry.source_audio_path)
    target_knots_seconds, source_knots_seconds = _build_trimmed_midi_time_knots(
        target_audio_knots_seconds=audio_sync_path.target_time_knots_seconds,
        source_audio_knots_seconds=audio_sync_path.source_time_knots_seconds,
        target_trim_seconds=target_trim_seconds,
        target_duration_seconds=target_duration_seconds,
        source_duration_seconds=source_duration_seconds,
    )

    return _build_alignment_result_from_time_knots(
        target_knots_seconds=target_knots_seconds,
        source_knots_seconds=source_knots_seconds,
        target_duration_seconds=target_duration_seconds,
        source_duration_seconds=source_duration_seconds,
        frame_seconds=config.frame_seconds,
        alignment_kind="audio_sync",
    )


def _notes_duration_seconds(notes: list[NoteEvent]) -> float:
    if not notes:
        return 0.0
    return float(max(note.end for note in notes))


def _build_trimmed_midi_time_knots(
    target_audio_knots_seconds: np.ndarray,
    source_audio_knots_seconds: np.ndarray,
    target_trim_seconds: float,
    target_duration_seconds: float,
    source_duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Target MIDI is trimmed to its first note; source time remains absolute audio time.
    target_knots_seconds = np.asarray(target_audio_knots_seconds, dtype=np.float32) - float(target_trim_seconds)
    source_knots_seconds = np.asarray(source_audio_knots_seconds, dtype=np.float32)

    target_knots_seconds = np.clip(target_knots_seconds, 0.0, float(target_duration_seconds))
    source_knots_seconds = np.clip(source_knots_seconds, 0.0, float(source_duration_seconds))
    return _deduplicate_knot_path(
        target_knots_seconds=target_knots_seconds,
        source_knots_seconds=source_knots_seconds,
        target_duration_seconds=target_duration_seconds,
        source_duration_seconds=source_duration_seconds,
    )


def _deduplicate_knot_path(
    target_knots_seconds: np.ndarray,
    source_knots_seconds: np.ndarray,
    target_duration_seconds: float,
    source_duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    if target_knots_seconds.shape != source_knots_seconds.shape:
        raise ValueError("target/source knot shapes must match")

    pairs: list[tuple[float, float]] = []
    for target_time, source_time in zip(target_knots_seconds.tolist(), source_knots_seconds.tolist(), strict=False):
        if not np.isfinite(target_time) or not np.isfinite(source_time):
            continue
        if pairs and target_time < pairs[-1][0]:
            continue
        if pairs and abs(target_time - pairs[-1][0]) <= 1.0e-6:
            pairs[-1] = (target_time, source_time)
            continue
        pairs.append((target_time, source_time))

    if not pairs:
        pairs = [(0.0, 0.0)]

    if pairs[0][0] > 0.0:
        pairs.insert(0, (0.0, max(0.0, min(source_duration_seconds, pairs[0][1]))))
    else:
        pairs[0] = (0.0, max(0.0, min(source_duration_seconds, pairs[0][1])))

    if target_duration_seconds > pairs[-1][0]:
        pairs.append((float(target_duration_seconds), max(0.0, min(source_duration_seconds, pairs[-1][1]))))
    else:
        pairs[-1] = (
            float(target_duration_seconds),
            max(0.0, min(source_duration_seconds, pairs[-1][1])),
        )

    target_out = np.asarray([item[0] for item in pairs], dtype=np.float32)
    source_out = np.asarray([item[1] for item in pairs], dtype=np.float32)
    return target_out, np.maximum.accumulate(source_out)


def _build_alignment_result_from_time_knots(
    target_knots_seconds: np.ndarray,
    source_knots_seconds: np.ndarray,
    target_duration_seconds: float,
    source_duration_seconds: float,
    frame_seconds: float,
    alignment_kind: str,
) -> SymbolicAlignmentResult:
    target_num_frames = max(1, _frame_index(target_duration_seconds, frame_seconds) + 1)
    source_num_frames = max(1, _frame_index(source_duration_seconds, frame_seconds) + 1)
    target_frame_times = np.arange(target_num_frames, dtype=np.float32) * float(frame_seconds)
    mapped_source_times = np.interp(
        target_frame_times,
        target_knots_seconds,
        source_knots_seconds,
        left=float(source_knots_seconds[0]),
        right=float(source_knots_seconds[-1]),
    )
    target_to_source = np.clip(
        np.round(mapped_source_times / frame_seconds).astype(np.int64),
        0,
        max(0, source_num_frames - 1),
    )

    matched_pairs: list[tuple[int, int]] = []
    for target_time, source_time in zip(target_knots_seconds.tolist(), source_knots_seconds.tolist(), strict=False):
        target_frame = int(np.clip(round(target_time / frame_seconds), 0, target_num_frames - 1))
        source_frame = int(np.clip(round(source_time / frame_seconds), 0, source_num_frames - 1))
        if matched_pairs and matched_pairs[-1] == (target_frame, source_frame):
            continue
        matched_pairs.append((target_frame, source_frame))

    matched_target_frames = torch.tensor([pair[0] for pair in matched_pairs], dtype=torch.long)
    matched_source_frames = torch.tensor([pair[1] for pair in matched_pairs], dtype=torch.long)
    target_has_match = torch.zeros((target_num_frames,), dtype=torch.bool)
    if int(matched_target_frames.numel()) > 0:
        target_has_match[matched_target_frames] = True

    absolute_offsets = np.abs(source_knots_seconds - target_knots_seconds)
    average_match_cost = float(absolute_offsets.mean()) if int(absolute_offsets.size) > 0 else 0.0
    max_absolute_offset_seconds = float(absolute_offsets.max()) if int(absolute_offsets.size) > 0 else 0.0
    gap_ratio = 1.0 - float(target_has_match.float().mean().item())

    return SymbolicAlignmentResult(
        target_to_source=torch.from_numpy(target_to_source.copy()),
        target_has_match=target_has_match,
        matched_target_frames=matched_target_frames,
        matched_source_frames=matched_source_frames,
        target_num_frames=target_num_frames,
        source_num_frames=source_num_frames,
        target_duration_seconds=float(target_duration_seconds),
        source_duration_seconds=float(source_duration_seconds),
        average_match_cost=average_match_cost,
        gap_ratio=gap_ratio,
        max_absolute_offset_seconds=max_absolute_offset_seconds,
        alignment_kind=alignment_kind,
        target_time_knots_seconds=torch.from_numpy(target_knots_seconds.copy()),
        source_time_knots_seconds=torch.from_numpy(source_knots_seconds.copy()),
    )


def _frame_index(time_seconds: float, frame_seconds: float) -> int:
    return max(0, int(round(time_seconds / frame_seconds)))
