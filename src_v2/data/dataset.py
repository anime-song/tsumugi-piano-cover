from __future__ import annotations

from collections import OrderedDict
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Sampler
from external.amt_model.models.interval_boundaries import PitchIntervalTargets

from src_v2.config import SourceChunkConfig, TargetRollConfig
from src_v2.data.index import PairEntry
from src_v2.data.midi import chunk_source_events, load_trimmed_source_events, load_trimmed_target_events
from src_v2.data.segment import (
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
    interval_targets: PitchIntervalTargets | None


@dataclass(frozen=True)
class DiffusionSongSample:
    song_name: str
    original_id: str
    piano_id: str
    performer_id: int
    source_features: torch.Tensor
    source_programs: torch.Tensor
    source_drums: torch.Tensor
    source_track_roles: torch.Tensor
    source_note_mask: torch.Tensor
    source_chunk_times: torch.Tensor
    target_segments: torch.Tensor
    target_segment_times: torch.Tensor
    target_num_frames: int


def _source_song_max_time(chunked: dict[str, torch.Tensor], config: SourceChunkConfig) -> float | None:
    if chunked["source_chunk_times"].numel() == 0:
        return None
    last_chunk_start = float(chunked["source_chunk_times"][-1].item())
    return last_chunk_start + config.window_seconds


class SegmentAutoencoderDataset(Dataset[AutoencoderSegmentSample]):
    def __init__(
        self,
        entries: list[PairEntry],
        target_roll_config: TargetRollConfig,
        decoder_mode: str = "semi-crf",
        augment_pitch_shift: bool = False,
        pitch_shift_min_semitones: int = 0,
        pitch_shift_max_semitones: int = 0,
        max_cached_songs: int | None = 16,
    ) -> None:
        self.target_roll_config = target_roll_config
        self.decoder_mode = decoder_mode
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
            skip_intervals=(self.decoder_mode == "frame"),
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

    def _extract_single_segment_interval_targets(
        self,
        full_intervals: tuple[list[tuple[int, int]], ...],
        start_frame: int,
        valid_length: int,
        num_frames: int,
        shift: int,
    ) -> PitchIntervalTargets:
        end_frame_exclusive = start_frame + valid_length
        config = self.target_roll_config
        pitch_count = config.pitch_count

        pitch_intervals: list[list[tuple[int, int]]] = [[] for _ in range(pitch_count)]
        has_onset: list[list[bool]] = [[] for _ in range(pitch_count)]
        has_offset: list[list[bool]] = [[] for _ in range(pitch_count)]
        onset_offsets: list[list[float]] = [[] for _ in range(pitch_count)]
        offset_offsets: list[list[float]] = [[] for _ in range(pitch_count)]

        for pitch_index, intervals in enumerate(full_intervals):
            shifted_pitch_index = pitch_index + shift
            if not (0 <= shifted_pitch_index < pitch_count):
                continue

            for interval_start, interval_end in intervals:
                if interval_end < start_frame:
                    continue
                if interval_start >= end_frame_exclusive:
                    break
                local_start = max(interval_start, start_frame) - start_frame
                local_end = min(interval_end, end_frame_exclusive - 1) - start_frame
                pitch_intervals[shifted_pitch_index].append((local_start, local_end))
                has_onset[shifted_pitch_index].append(interval_start >= start_frame)
                has_offset[shifted_pitch_index].append(interval_end < end_frame_exclusive or interval_end >= num_frames - 1)
                onset_offsets[shifted_pitch_index].append(0.0)
                offset_offsets[shifted_pitch_index].append(0.0)

        from external.amt_model.models.interval_boundaries import PitchIntervalTargets
        return PitchIntervalTargets(
            intervals=pitch_intervals,
            has_onset=has_onset,
            has_offset=has_offset,
            onset_offsets=onset_offsets,
            offset_offsets=offset_offsets,
        )

    def __getitem__(self, index: int) -> AutoencoderSegmentSample:
        entry, segment_index = self.samples[index]
        segment_song = self._get_segment_song(entry)
        segment = segment_song.segments[segment_index]
        shift = self._sample_pitch_shift(segment)
        segment = self._transpose_segment_tensors(segment, shift)

        segment_time = float(segment_song.segment_times[segment_index].item())
        valid_length = int(segment_song.segment_valid_lengths[segment_index].item())

        if segment_song.full_intervals is not None:
            # セグメント開始時間をフレームインデックスに変換して区間を切り出す
            start_frame = max(0, int(round(segment_time / self.target_roll_config.frame_seconds)))
            interval_targets = self._extract_single_segment_interval_targets(
                full_intervals=segment_song.full_intervals,
                start_frame=start_frame,
                valid_length=valid_length,
                num_frames=segment_song.num_frames,
                shift=shift,
            )
        else:
            interval_targets = None

        return AutoencoderSegmentSample(
            song_name=entry.song_name,
            piano_id=entry.piano_id,
            segment=segment,
            segment_time=segment_time,
            segment_valid_length=valid_length,
            interval_targets=interval_targets,
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
        source_chunk_config: SourceChunkConfig,
        target_roll_config: TargetRollConfig,
    ) -> None:
        self.entries = entries
        self.source_chunk_config = source_chunk_config
        self.target_roll_config = target_roll_config

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> DiffusionSongSample:
        entry = self.entries[index]
        source_events = load_trimmed_source_events(entry.source_midi_path, self.source_chunk_config)
        chunked = chunk_source_events(source_events, self.source_chunk_config)

        # 原曲長に合わせたピアノカバーのセグメント化
        max_time_seconds = _source_song_max_time(chunked, self.source_chunk_config)
        segment_song = target_midi_to_segment_song(
            entry.target_midi_path,
            self.target_roll_config,
            max_time_seconds=max_time_seconds,
        )
        return DiffusionSongSample(
            song_name=entry.song_name,
            original_id=entry.original_id,
            piano_id=entry.piano_id,
            performer_id=entry.performer_id,
            source_features=chunked["source_features"],
            source_programs=chunked["source_programs"],
            source_drums=chunked["source_drums"],
            source_track_roles=chunked["source_track_roles"],
            source_note_mask=chunked["source_note_mask"],
            source_chunk_times=chunked["source_chunk_times"],
            target_segments=segment_song.segments,
            target_segment_times=segment_song.segment_times,
            target_num_frames=segment_song.num_frames,
        )


def collate_autoencoder_samples(
    samples: list[AutoencoderSegmentSample],
) -> dict[str, torch.Tensor | list[str] | list[PitchIntervalTargets | None]]:
    batch_size = len(samples)
    frames_per_segment = samples[0].segment.shape[0]
    feature_dim = samples[0].segment.shape[1]

    target_segments = torch.zeros((batch_size, 1, frames_per_segment, feature_dim), dtype=torch.float32)
    segment_mask = torch.ones((batch_size, 1), dtype=torch.bool)
    segment_times = torch.zeros((batch_size, 1), dtype=torch.float32)
    segment_valid_lengths = torch.zeros((batch_size,), dtype=torch.long)
    metadata: list[str] = []
    interval_targets: list[PitchIntervalTargets | None] = []

    for batch_index, sample in enumerate(samples):
        target_segments[batch_index, 0] = sample.segment
        segment_times[batch_index, 0] = sample.segment_time
        segment_valid_lengths[batch_index] = sample.segment_valid_length
        metadata.append(sample.piano_id)
        interval_targets.append(sample.interval_targets)

    return {
        "target_segments": target_segments,
        "segment_mask": segment_mask,
        "segment_times": segment_times,
        "segment_valid_lengths": segment_valid_lengths,
        "interval_targets": interval_targets,
        "metadata": metadata,
    }


def collate_diffusion_samples(samples: list[DiffusionSongSample]) -> dict[str, torch.Tensor | list[tuple[str, str]]]:
    batch_size = len(samples)
    max_chunks = max(sample.source_features.shape[0] for sample in samples)
    max_notes = samples[0].source_features.shape[1]
    note_dim = samples[0].source_features.shape[2]
    max_segments = max(sample.target_segments.shape[0] for sample in samples)
    frames_per_segment = samples[0].target_segments.shape[1]
    feature_dim = samples[0].target_segments.shape[2]

    source_features = torch.zeros((batch_size, max_chunks, max_notes, note_dim), dtype=torch.float32)
    source_programs = torch.zeros((batch_size, max_chunks, max_notes), dtype=torch.long)
    source_drums = torch.zeros((batch_size, max_chunks, max_notes), dtype=torch.long)
    source_track_roles = torch.zeros((batch_size, max_chunks, max_notes), dtype=torch.long)
    source_note_mask = torch.zeros((batch_size, max_chunks, max_notes), dtype=torch.bool)
    source_chunk_mask = torch.zeros((batch_size, max_chunks), dtype=torch.bool)
    source_chunk_times = torch.zeros((batch_size, max_chunks), dtype=torch.float32)
    target_segments = torch.zeros((batch_size, max_segments, frames_per_segment, feature_dim), dtype=torch.float32)
    segment_mask = torch.zeros((batch_size, max_segments), dtype=torch.bool)
    segment_times = torch.zeros((batch_size, max_segments), dtype=torch.float32)
    target_num_frames = torch.zeros((batch_size,), dtype=torch.long)
    performer_ids = torch.zeros((batch_size,), dtype=torch.long)
    metadata: list[tuple[str, str]] = []

    for batch_index, sample in enumerate(samples):
        num_chunks = sample.source_features.shape[0]
        source_features[batch_index, :num_chunks] = sample.source_features
        source_programs[batch_index, :num_chunks] = sample.source_programs
        source_drums[batch_index, :num_chunks] = sample.source_drums
        source_track_roles[batch_index, :num_chunks] = sample.source_track_roles
        source_note_mask[batch_index, :num_chunks] = sample.source_note_mask
        source_chunk_mask[batch_index, :num_chunks] = True
        source_chunk_times[batch_index, :num_chunks] = sample.source_chunk_times

        num_segments = sample.target_segments.shape[0]
        target_segments[batch_index, :num_segments] = sample.target_segments
        segment_mask[batch_index, :num_segments] = True
        segment_times[batch_index, :num_segments] = sample.target_segment_times
        target_num_frames[batch_index] = sample.target_num_frames
        performer_ids[batch_index] = sample.performer_id
        metadata.append((sample.song_name, sample.piano_id))

    return {
        "source_features": source_features,
        "source_programs": source_programs,
        "source_drums": source_drums,
        "source_track_roles": source_track_roles,
        "source_note_mask": source_note_mask,
        "source_chunk_mask": source_chunk_mask,
        "source_chunk_times": source_chunk_times,
        "target_segments": target_segments,
        "segment_mask": segment_mask,
        "segment_times": segment_times,
        "target_num_frames": target_num_frames,
        "performer_ids": performer_ids,
        "metadata": metadata,
    }

