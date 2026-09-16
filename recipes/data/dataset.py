from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from recipes.data.index import PairEntry
from tsumugi_piano_cover.config import TargetRollConfig
from tsumugi_piano_cover.data.audio import load_audio
from tsumugi_piano_cover.data.midi import load_trimmed_target_events
from tsumugi_piano_cover.data.segment import (
    SegmentSong,
    compute_segment_start_frames,
    compute_target_num_frames,
    target_midi_to_segment_song,
)


@dataclass(frozen=True)
class AutoencoderSegmentSample:
    song_name: str
    piano_id: str
    segment: torch.Tensor
    segment_time: float
    segment_valid_length: int


@dataclass(frozen=True)
class DiffusionSongSample:
    song_name: str
    original_id: str
    piano_id: str
    performer_id: int
    source_audio: torch.Tensor
    source_audio_length: int
    alignment_source_times: torch.Tensor
    alignment_mask: torch.Tensor
    target_segments: torch.Tensor
    target_segment_times: torch.Tensor
    target_num_frames: int


def _load_alignment_source_times(
    alignment_path: Path | None,
    segment_times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # alignment は教師リズムを動かさず、target時刻から参照しやすいsource時刻だけを返す
    aligned_source_times = segment_times.clone()
    alignment_mask = torch.zeros_like(segment_times, dtype=torch.bool)
    if alignment_path is None or not alignment_path.is_file():
        return aligned_source_times, alignment_mask

    payload = torch.load(alignment_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported alignment payload: {alignment_path}")

    target_knots = payload.get("target_time_knots_seconds")
    source_knots = payload.get("source_time_knots_seconds")
    if not isinstance(target_knots, torch.Tensor) or not isinstance(source_knots, torch.Tensor):
        raise KeyError(f"alignment payload lacks time knots: {alignment_path}")
    if target_knots.ndim != 1 or source_knots.ndim != 1 or target_knots.numel() != source_knots.numel():
        raise ValueError(f"invalid alignment knot shapes: {alignment_path}")
    if int(target_knots.numel()) < 2:
        return aligned_source_times, alignment_mask

    target_knots = target_knots.float()
    source_knots = source_knots.float()
    clamped_times = segment_times.float().clamp(float(target_knots[0].item()), float(target_knots[-1].item()))
    right_indices = torch.searchsorted(target_knots, clamped_times, right=False).clamp(1, target_knots.numel() - 1)
    left_indices = right_indices - 1
    left_target = target_knots[left_indices]
    right_target = target_knots[right_indices]
    ratio = (clamped_times - left_target) / (right_target - left_target).clamp_min(1.0e-6)
    aligned_source_times = source_knots[left_indices] + ratio * (
        source_knots[right_indices] - source_knots[left_indices]
    )
    alignment_mask = (segment_times >= target_knots[0]) & (segment_times <= target_knots[-1])

    target_has_match = payload.get("target_has_match")
    alignment_config = payload.get("alignment_config", {})
    frame_seconds = float(alignment_config.get("frame_seconds", 0.0)) if isinstance(alignment_config, dict) else 0.0
    if isinstance(target_has_match, torch.Tensor) and target_has_match.numel() > 0 and frame_seconds > 0.0:
        frame_indices = torch.round(segment_times.float() / frame_seconds).long()
        frame_in_range = (frame_indices >= 0) & (frame_indices < target_has_match.numel())
        safe_indices = frame_indices.clamp(0, int(target_has_match.numel()) - 1)
        alignment_mask = alignment_mask & frame_in_range & target_has_match.bool()[safe_indices]

    return aligned_source_times, alignment_mask


class SegmentAutoencoderDataset(Dataset[AutoencoderSegmentSample]):
    def __init__(
        self,
        entries: list[PairEntry],
        target_roll_config: TargetRollConfig,
        augment_pitch_shift: bool = False,
        pitch_shift_min_semitones: int = 0,
        pitch_shift_max_semitones: int = 0,
        max_cached_songs: int | None = 16,
    ) -> None:
        self.target_roll_config = target_roll_config
        self.augment_pitch_shift = augment_pitch_shift
        self.pitch_shift_min_semitones = pitch_shift_min_semitones
        self.pitch_shift_max_semitones = pitch_shift_max_semitones
        self.max_cached_songs = max_cached_songs
        self._segment_song_cache: OrderedDict[str, SegmentSong] = OrderedDict()
        self.samples: list[tuple[PairEntry, int]] = []
        self.song_to_sample_indices: dict[str, list[int]] = {}

        # MIDIからセグメントインデックスのリストを構築
        for entry in entries:
            notes, pedals = load_trimmed_target_events(
                entry.target_midi_path,
                min_duration_seconds=self.target_roll_config.min_duration_seconds,
            )
            num_frames = compute_target_num_frames(notes, pedals, self.target_roll_config)
            num_segments = len(compute_segment_start_frames(num_frames, self.target_roll_config))
            for segment_index in range(num_segments):
                sample_index = len(self.samples)
                self.samples.append((entry, segment_index))
                self.song_to_sample_indices.setdefault(str(entry.target_midi_path), []).append(sample_index)

    def __len__(self) -> int:
        return len(self.samples)

    def _get_segment_song(self, entry: PairEntry) -> SegmentSong:
        # LRUキャッシュを介してピアノロールを取得
        cache_key = str(entry.target_midi_path)
        cached = self._segment_song_cache.get(cache_key)
        if cached is not None:
            self._segment_song_cache.move_to_end(cache_key)
            return cached

        segment_song = target_midi_to_segment_song(
            entry.target_midi_path,
            self.target_roll_config,
            skip_intervals=True,
        )
        self._segment_song_cache[cache_key] = segment_song
        if self.max_cached_songs is not None and len(self._segment_song_cache) > self.max_cached_songs:
            self._segment_song_cache.popitem(last=False)
        return segment_song

    def _sample_pitch_shift(self, segment: torch.Tensor) -> int:
        if not self.augment_pitch_shift:
            return 0
        requested_min = int(self.pitch_shift_min_semitones)
        requested_max = int(self.pitch_shift_max_semitones)
        if requested_min == 0 and requested_max == 0:
            return 0
        pitch_count = self.target_roll_config.pitch_count
        onset = segment[:, :pitch_count]
        sustain = segment[:, pitch_count : pitch_count * 2]

        # 鍵盤範囲（88鍵）に収まるようシフト幅を制限
        active_pitch_mask = ((onset >= 0.5) | (sustain >= 0.5)).any(dim=0)
        active_pitch_indices = active_pitch_mask.nonzero(as_tuple=False).flatten()
        if int(active_pitch_indices.numel()) == 0:
            return 0

        min_pitch_index = int(active_pitch_indices.min().item())
        max_pitch_index = int(active_pitch_indices.max().item())
        lower_bound = -min_pitch_index
        upper_bound = (pitch_count - 1) - max_pitch_index
        actual_min = max(requested_min, lower_bound)
        actual_max = min(requested_max, upper_bound)
        if actual_min > actual_max:
            return 0
        return random.randint(actual_min, actual_max)

    def _transpose_segment_tensors(
        self,
        segment: torch.Tensor,
        shift: int,
    ) -> torch.Tensor:
        # ピッチシフト（移調）の実行
        if shift == 0:
            return segment.clone()

        pitch_count = self.target_roll_config.pitch_count
        shifted_segment = torch.zeros_like(segment)

        if shift > 0:
            source_slice = slice(0, pitch_count - shift)
            target_slice = slice(shift, pitch_count)
        else:
            source_slice = slice(-shift, pitch_count)
            target_slice = slice(0, pitch_count + shift)

        shifted_segment[:, target_slice] = segment[:, source_slice]
        shifted_segment[:, pitch_count + target_slice.start : pitch_count + target_slice.stop] = segment[
            :, pitch_count + source_slice.start : pitch_count + source_slice.stop
        ]
        shifted_segment[:, pitch_count * 2 + target_slice.start : pitch_count * 2 + target_slice.stop] = segment[
            :, pitch_count * 2 + source_slice.start : pitch_count * 2 + source_slice.stop
        ]
        # ペダル情報は移調せずそのままコピー
        shifted_segment[:, -1] = segment[:, -1]
        return shifted_segment

    def __getitem__(self, index: int) -> AutoencoderSegmentSample:
        entry, segment_index = self.samples[index]
        segment_song = self._get_segment_song(entry)
        segment = segment_song.segments[segment_index]
        shift = self._sample_pitch_shift(segment)
        segment = self._transpose_segment_tensors(segment, shift)

        segment_time = float(segment_song.segment_times[segment_index].item())
        valid_length = int(segment_song.segment_valid_lengths[segment_index].item())

        return AutoencoderSegmentSample(
            song_name=entry.song_name,
            piano_id=entry.piano_id,
            segment=segment,
            segment_time=segment_time,
            segment_valid_length=valid_length,
        )


class SegmentSongBatchSampler(Sampler[list[int]]):
    # 同一曲のセグメントを同じバッチにまとめるサンプラー
    def __init__(
        self,
        dataset: SegmentAutoencoderDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        song_keys = list(self.dataset.song_to_sample_indices.keys())
        if self.shuffle:
            random.shuffle(song_keys)

        # キャッシュヒット率向上のため、曲のリストを一定サイズ(chunk)ごとに処理
        # キャッシュサイズ(max_cached_songs)の半分程度、指定がなければ8とする
        chunk_size = max(1, self.dataset.max_cached_songs // 2) if self.dataset.max_cached_songs else 8

        for i in range(0, len(song_keys), chunk_size):
            chunk_keys = song_keys[i : i + chunk_size]
            chunk_batches: list[list[int]] = []
            for song_key in chunk_keys:
                indices = list(self.dataset.song_to_sample_indices[song_key])
                if self.shuffle:
                    random.shuffle(indices)
                for offset in range(0, len(indices), self.batch_size):
                    batch = indices[offset : offset + self.batch_size]
                    if len(batch) < self.batch_size and self.drop_last:
                        continue
                    chunk_batches.append(batch)

            if self.shuffle:
                random.shuffle(chunk_batches)
            yield from chunk_batches

    def __len__(self) -> int:
        total = 0
        for indices in self.dataset.song_to_sample_indices.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


class SegmentDiffusionDataset(Dataset[DiffusionSongSample]):
    def __init__(
        self,
        entries: list[PairEntry],
        target_roll_config: TargetRollConfig,
        alignment_cache_dir: str | None = None,
        audio_sample_rate: int = 22050,
        audio_channels: int = 2,
    ) -> None:
        self.entries = entries
        self.target_roll_config = target_roll_config
        self.alignment_cache_dir = Path(alignment_cache_dir) if alignment_cache_dir else None
        self.audio_sample_rate = audio_sample_rate
        self.audio_channels = audio_channels

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> DiffusionSongSample:
        entry = self.entries[index]
        if entry.source_audio_path is None:
            raise FileNotFoundError(
                f"source audio was not found for original_id={entry.original_id}"
            )
        source_audio = load_audio(
            entry.source_audio_path,
            sample_rate=self.audio_sample_rate,
            num_channels=self.audio_channels,
        )
        source_audio_length = int(source_audio.shape[-1])

        # 原曲長に合わせたピアノカバーのセグメント化
        max_time_seconds = source_audio_length / self.audio_sample_rate
        segment_song = target_midi_to_segment_song(
            entry.target_midi_path,
            self.target_roll_config,
            max_time_seconds=max_time_seconds,
        )
        alignment_path = (
            self.alignment_cache_dir / entry.original_id / f"{entry.piano_id}.pt"
            if self.alignment_cache_dir is not None
            else None
        )
        alignment_source_times, alignment_mask = _load_alignment_source_times(
            alignment_path,
            segment_song.segment_times,
        )
        return DiffusionSongSample(
            song_name=entry.song_name,
            original_id=entry.original_id,
            piano_id=entry.piano_id,
            performer_id=entry.performer_id,
            source_audio=source_audio,
            source_audio_length=source_audio_length,
            alignment_source_times=alignment_source_times,
            alignment_mask=alignment_mask,
            target_segments=segment_song.segments,
            target_segment_times=segment_song.segment_times,
            target_num_frames=segment_song.num_frames,
        )


def collate_autoencoder_samples(
    samples: list[AutoencoderSegmentSample],
) -> dict[str, torch.Tensor | list[str]]:
    batch_size = len(samples)
    frames_per_segment = samples[0].segment.shape[0]
    feature_dim = samples[0].segment.shape[1]

    target_segments = torch.zeros((batch_size, 1, frames_per_segment, feature_dim), dtype=torch.float32)
    segment_mask = torch.ones((batch_size, 1), dtype=torch.bool)
    segment_times = torch.zeros((batch_size, 1), dtype=torch.float32)
    segment_valid_lengths = torch.zeros((batch_size,), dtype=torch.long)
    metadata: list[str] = []

    for batch_index, sample in enumerate(samples):
        target_segments[batch_index, 0] = sample.segment
        segment_times[batch_index, 0] = sample.segment_time
        segment_valid_lengths[batch_index] = sample.segment_valid_length
        metadata.append(sample.piano_id)

    return {
        "target_segments": target_segments,
        "segment_mask": segment_mask,
        "segment_times": segment_times,
        "segment_valid_lengths": segment_valid_lengths,
        "metadata": metadata,
    }


def collate_diffusion_samples(samples: list[DiffusionSongSample]) -> dict[str, torch.Tensor | list[tuple[str, str]]]:
    batch_size = len(samples)
    max_source_samples = max(sample.source_audio.shape[-1] for sample in samples)
    max_segments = max(sample.target_segments.shape[0] for sample in samples)
    frames_per_segment = samples[0].target_segments.shape[1]
    feature_dim = samples[0].target_segments.shape[2]

    source_audio = torch.zeros((batch_size, 2, max_source_samples), dtype=torch.float32)
    source_audio_lengths = torch.zeros((batch_size,), dtype=torch.long)
    alignment_source_times = torch.zeros((batch_size, max_segments), dtype=torch.float32)
    alignment_mask = torch.zeros((batch_size, max_segments), dtype=torch.bool)
    target_segments = torch.zeros((batch_size, max_segments, frames_per_segment, feature_dim), dtype=torch.float32)
    segment_mask = torch.zeros((batch_size, max_segments), dtype=torch.bool)
    segment_times = torch.zeros((batch_size, max_segments), dtype=torch.float32)
    target_num_frames = torch.zeros((batch_size,), dtype=torch.long)
    performer_ids = torch.zeros((batch_size,), dtype=torch.long)
    metadata: list[tuple[str, str]] = []

    for batch_index, sample in enumerate(samples):
        num_source_samples = sample.source_audio.shape[-1]
        source_audio[batch_index, :, :num_source_samples] = sample.source_audio
        source_audio_lengths[batch_index] = sample.source_audio_length

        num_segments = sample.target_segments.shape[0]
        target_segments[batch_index, :num_segments] = sample.target_segments
        alignment_source_times[batch_index, :num_segments] = sample.alignment_source_times
        alignment_mask[batch_index, :num_segments] = sample.alignment_mask
        segment_mask[batch_index, :num_segments] = True
        segment_times[batch_index, :num_segments] = sample.target_segment_times
        target_num_frames[batch_index] = sample.target_num_frames
        performer_ids[batch_index] = sample.performer_id
        metadata.append((sample.song_name, sample.piano_id))

    return {
        "source_audio": source_audio,
        "source_audio_lengths": source_audio_lengths,
        "alignment_source_times": alignment_source_times,
        "alignment_mask": alignment_mask,
        "target_segments": target_segments,
        "segment_mask": segment_mask,
        "segment_times": segment_times,
        "target_num_frames": target_num_frames,
        "performer_ids": performer_ids,
        "metadata": metadata,
    }
