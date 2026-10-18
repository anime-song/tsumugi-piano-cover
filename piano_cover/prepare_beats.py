"""カバーのキャッシュに、拍のトークンの学習で使う拍を足した別のキャッシュを作る。

    python -m piano_cover.prepare_beats data/piano_cover_v5s data/piano_cover_v6b

カバーの拍は、カバーの音声の注釈 (data/sheetsage/cover)、音声がなければ MIDI の合成音の注釈 (data/sheetsage/pretraining)。
原曲の拍は原曲の音声の注釈 (data/sheetsage/original)。カバーの拍は piano_cover.beat_align で原曲の階層にそろえ、
原曲の拍の番号 (小数) に対応させる。ほかのファイル (カバーのイベント・原曲の行・時刻の対応など) は元のキャッシュのまま写す。

足すファイル:
    cover_beats.npz   offsets [C + 1]、times [N] (カバーの拍、キャッシュのイベントと同じ基準のフレーム)、downbeat [N]、
                      source_beat [N] (対応する原曲の拍の番号、小数)、ok [N] (原曲の拍に対応したか)
    source_beats.npz  offsets [S + 1]、beats [M] (原曲の拍、音声の絶対時刻のフレーム)、downbeat [M]
注釈のないカバー・原曲は空にする (学習では使わない)。
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .beat_align import ALIGN_STEP, align_beats


def _cover(task: tuple) -> dict | None:
    from piano_ar.tokenizer import ONSET, PianoTokenizer

    midi, cover_ann, source_ann, align, ok_steps, frame_rate = task
    if not Path(midi).exists() or not Path(source_ann).exists():
        return None
    for path in cover_ann:
        if Path(path).exists():
            cover = json.loads(Path(path).read_text(encoding="utf-8"))
            break
    else:
        return None
    source = json.loads(Path(source_ann).read_text(encoding="utf-8"))
    events, _ = PianoTokenizer().midi_to_events(midi, trim=False)
    if len(events) == 0:
        return None
    shift = int(events[:, ONSET].min())

    def to_source(seconds: np.ndarray) -> np.ndarray:
        return np.interp((seconds * frame_rate - shift) / ALIGN_STEP, np.arange(len(align)), align) / frame_rate

    def usable(seconds: np.ndarray) -> np.ndarray:
        step = (seconds * frame_rate - shift) / ALIGN_STEP
        inside = (step >= 0) & (step <= len(align) - 1)
        return inside & ok_steps[np.clip(np.round(step).astype(int), 0, len(align) - 1)]

    result = align_beats(
        np.asarray(cover["beats"]), np.asarray(cover["downbeats"]), np.asarray(source["beats"]),
        np.asarray(source["downbeats"]), to_source, usable,
    )  # fmt: skip
    if len(result.times) < 4:
        return None
    return {
        "times": (result.times * frame_rate - shift).astype(np.float32),
        "downbeat": result.downbeat.astype(bool),
        "source_beat": result.source_beat.astype(np.float32),
        "ok": result.ok.astype(bool),
    }


def main() -> None:
    from .data import CoverCache

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--cover-sheetsage", default="data/sheetsage/cover")
    parser.add_argument("--midi-sheetsage", default="data/sheetsage/pretraining")
    parser.add_argument("--source-sheetsage", default="data/sheetsage/original")
    parser.add_argument("--cover-midi", default="Dataset/pianos_midi")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cache = CoverCache(args.src)
    fr = cache.tokenizer_config.frame_rate
    tasks = []
    for cover in range(len(cache.video_ids)):
        original_id = str(cache.source_ids[cache.source_index[cover]])
        piano_id = str(cache.video_ids[cover])
        ok = cache.cover_align_ok(cover)
        align = cache.cover_align(cover).astype(np.float64)
        tasks.append(
            (
                str(Path(args.cover_midi) / original_id / f"{piano_id}.mid"),
                [
                    str(Path(args.cover_sheetsage) / original_id / f"{piano_id}.json"),
                    str(Path(args.midi_sheetsage) / f"{piano_id}.json"),
                ],
                str(Path(args.source_sheetsage) / f"{original_id}.json"),
                align,
                np.ones(len(align), bool) if ok is None else ok.astype(bool),
                fr,
            )
        )
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_cover, tasks, chunksize=8))

    args.dst.mkdir(parents=True, exist_ok=True)
    for path in args.src.iterdir():
        if path.is_file():
            shutil.copyfile(path, args.dst / path.name)
    lengths = [0 if r is None else len(r["times"]) for r in results]
    kept = [r for r in results if r is not None]

    def joined(key: str, dtype) -> np.ndarray:
        return np.concatenate([r[key] for r in kept]).astype(dtype) if kept else np.zeros(0, dtype)

    np.savez(
        args.dst / "cover_beats.npz",
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        times=joined("times", np.float32),
        downbeat=joined("downbeat", bool),
        source_beat=joined("source_beat", np.float32),
        ok=joined("ok", bool),
    )

    source_beats, source_down, source_lengths = [], [], []
    for original_id in cache.source_ids:
        path = Path(args.source_sheetsage) / f"{original_id}.json"
        if not path.exists():
            source_lengths.append(0)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        beats = np.asarray(data["beats"], dtype=np.float64) * fr
        flags = np.zeros(len(beats), dtype=bool)
        if len(data["downbeats"]) and len(beats):
            flags[np.abs(beats[:, None] - np.asarray(data["downbeats"])[None] * fr).argmin(0)] = True
        source_beats.append(beats.astype(np.float32))
        source_down.append(flags)
        source_lengths.append(len(beats))
    np.savez(
        args.dst / "source_beats.npz",
        offsets=np.concatenate([[0], np.cumsum(source_lengths)]).astype(np.int64),
        beats=np.concatenate(source_beats) if source_beats else np.zeros(0, np.float32),
        downbeat=np.concatenate(source_down) if source_down else np.zeros(0, bool),
    )
    meta = json.loads((args.dst / "meta.json").read_text(encoding="utf-8"))
    meta["beats"] = {"from": str(args.src), "covers_with_beats": len(kept)}
    (args.dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"カバー {len(results)} 本のうち拍のあるもの {len(kept)}、原曲 {len(cache.source_ids)} 曲のうち {len(source_beats)}"
    )


if __name__ == "__main__":
    main()
