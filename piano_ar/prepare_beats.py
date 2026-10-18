"""事前学習のキャッシュ (piano_ar.prepare の出力) に、曲ごとの拍 (SheetSage2-Refined の注釈) を足す。

    python -m piano_ar.prepare_beats --cache-dir data/piano_ar/pretraining_v2

注釈は piano_ar.sheetsage で MIDI を合成音にしてから取ったもの (YouTube の採譜は data/sheetsage/pretraining/<ID>.json、
MAESTRO は data/sheetsage/maestro/<年>/<名前>.json)。キャッシュのイベントは先頭の無音を詰めてあるので、MIDI を読み直して
詰めた量を求め、拍の時刻を同じ基準 (フレーム) にそろえる。

出力 (--cache-dir/beats.npz):
    offsets [S + 1]   曲ごとの拍の範囲 (注釈のない曲は空)
    beats   [N]       拍の時刻 (フレーム、float32、キャッシュのイベントと同じ基準)
    downbeat [N]      小節の頭か
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MANIFEST = Path("data/metadata/pretraining_manifest.csv")


def _song(task: tuple[str, str, str, int]) -> tuple[np.ndarray, np.ndarray] | None:
    from .tokenizer import ONSET, PianoTokenizer

    midi, annotation, maestro_root, frame_rate = task
    path = Path(annotation)
    if not path.exists() or not Path(midi).exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    events, _ = PianoTokenizer().midi_to_events(midi, trim=False)
    if len(events) == 0:
        return None
    shift = int(events[:, ONSET].min())
    beats = np.asarray(data["beats"], dtype=np.float64) * frame_rate - shift
    downbeats = np.asarray(data["downbeats"], dtype=np.float64) * frame_rate - shift
    flags = np.zeros(len(beats), dtype=bool)
    if len(downbeats) and len(beats):
        nearest = np.abs(beats[:, None] - downbeats[None]).argmin(0)
        flags[nearest] = True
    return beats.astype(np.float32), flags


def main() -> None:
    from .data import PretrainingCache

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default="data/piano_ar/pretraining_v2")
    parser.add_argument("--midi-dir", default="Dataset/pretraining_midi")
    parser.add_argument("--maestro-dir", default="Dataset/maestro-v3.0.0")
    parser.add_argument("--sheetsage-dir", default="data/sheetsage/pretraining")
    parser.add_argument("--maestro-sheetsage-dir", default="data/sheetsage/maestro")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cache = PretrainingCache(args.cache_dir)
    fr = cache.tokenizer_config.frame_rate
    with open(MANIFEST, encoding="utf-8") as f:
        midi_paths = {row["id"]: row["midi_path"] for row in csv.DictReader(f)}
    maestro = cache.source_names.index("maestro") if "maestro" in cache.source_names else -1
    tasks = []
    for song_id, source in zip(cache.video_ids, cache.sources):
        song_id = str(song_id)
        if source == maestro:
            midi = Path(midi_paths[song_id])
            relative = midi.relative_to(args.maestro_dir).with_suffix(".json")
            tasks.append((str(midi), str(Path(args.maestro_sheetsage_dir) / relative), args.maestro_dir, fr))
        else:
            tasks.append(
                (str(Path(args.midi_dir) / f"{song_id}.mid"), str(Path(args.sheetsage_dir) / f"{song_id}.json"), "", fr)
            )
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_song, tasks, chunksize=16))
    lengths = [0 if r is None else len(r[0]) for r in results]
    beats = [r[0] for r in results if r is not None]
    flags = [r[1] for r in results if r is not None]
    np.savez(
        Path(args.cache_dir) / "beats.npz",
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        beats=np.concatenate(beats) if beats else np.zeros(0, np.float32),
        downbeat=np.concatenate(flags) if flags else np.zeros(0, bool),
    )
    have = sum(r is not None for r in results)
    print(f"曲 {len(results)} のうち拍のある曲 {have} ({have / max(len(results), 1):.1%})")


if __name__ == "__main__":
    main()
