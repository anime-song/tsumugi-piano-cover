from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PairEntry:
    # 原曲（ソース）とピアノカバー（ターゲット）のペア情報
    song_name: str
    original_id: str
    piano_id: str
    target_midi_path: str
    performer_id: int
    source_audio_path: str | None = None
    target_audio_path: str | None = None


_AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


def _resolve_optional_audio_path(root: str | Path, *parts: str) -> str | None:
    # 拡張子違いを吸収しつつ、存在する音源ファイルを 1 つ探す
    base = Path(root).joinpath(*parts)
    for suffix in _AUDIO_SUFFIXES:
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return str(candidate)
    return None


def build_pair_index(
    dataset_json_path: str | Path,
    piano_midi_dir: str | Path,
    piano_to_performer_json: str | Path,
    original_audio_dir: str | Path | None = None,
    piano_audio_dir: str | Path | None = None,
) -> list[PairEntry]:
    # データセットメタデータと演奏者情報からペアリストを構築する
    dataset = json.loads(Path(dataset_json_path).read_text(encoding="utf-8"))
    performer_map = json.loads(Path(piano_to_performer_json).read_text(encoding="utf-8"))

    piano_root = Path(piano_midi_dir)
    entries: list[PairEntry] = []

    for song_name, meta in dataset.items():
        original_id = meta["original"]
        source_audio_path = (
            _resolve_optional_audio_path(original_audio_dir, original_id) if original_audio_dir is not None else None
        )

        for piano_id in meta["pianos"]:
            target_path = piano_root / original_id / f"{piano_id}.mid"
            performer_id = performer_map.get(piano_id)
            if not target_path.exists() or performer_id is None:
                continue
            target_audio_path = (
                _resolve_optional_audio_path(piano_audio_dir, original_id, piano_id)
                if piano_audio_dir is not None
                else None
            )

            entries.append(
                PairEntry(
                    song_name=song_name,
                    original_id=original_id,
                    piano_id=piano_id,
                    target_midi_path=str(target_path),
                    performer_id=int(performer_id),
                    source_audio_path=source_audio_path,
                    target_audio_path=target_audio_path,
                )
            )

    return entries


def split_pairs_by_song(
    entries: list[PairEntry],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[PairEntry]]:
    # データリークを防ぐため、曲名（song_name）単位で分割する
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    song_names = sorted({entry.song_name for entry in entries})
    rng = random.Random(seed)
    rng.shuffle(song_names)

    num_songs = len(song_names)
    train_cut = int(num_songs * train_fraction)
    val_cut = train_cut + int(num_songs * val_fraction)

    song_to_split: dict[str, str] = {}
    for idx, song_name in enumerate(song_names):
        if idx < train_cut:
            song_to_split[song_name] = "train"
        elif idx < val_cut:
            song_to_split[song_name] = "val"
        else:
            song_to_split[song_name] = "test"

    splits = {"train": [], "val": [], "test": []}
    for entry in entries:
        splits[song_to_split[entry.song_name]].append(entry)

    return splits
