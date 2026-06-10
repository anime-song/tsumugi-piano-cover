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
    source_midi_path: str
    piano_id: str
    target_midi_path: str
    performer_id: int


def build_pair_index(
    dataset_json_path: str | Path,
    original_midi_dir: str | Path,
    piano_midi_dir: str | Path,
    piano_to_performer_json: str | Path,
) -> list[PairEntry]:
    # データセットメタデータと演奏者情報からペアリストを構築する
    dataset = json.loads(Path(dataset_json_path).read_text(encoding="utf-8"))
    performer_map = json.loads(Path(piano_to_performer_json).read_text(encoding="utf-8"))

    original_root = Path(original_midi_dir)
    piano_root = Path(piano_midi_dir)
    entries: list[PairEntry] = []

    for song_name, meta in dataset.items():
        original_id = meta["original"]
        source_path = original_root / f"{original_id}.mid"
        if not source_path.exists():
            continue

        for piano_id in meta["pianos"]:
            target_path = piano_root / original_id / f"{piano_id}.mid"
            performer_id = performer_map.get(piano_id)
            if not target_path.exists() or performer_id is None:
                continue

            entries.append(
                PairEntry(
                    song_name=song_name,
                    original_id=original_id,
                    source_midi_path=str(source_path),
                    piano_id=piano_id,
                    target_midi_path=str(target_path),
                    performer_id=int(performer_id),
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
