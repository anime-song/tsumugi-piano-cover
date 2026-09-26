"""カバー学習用のキャッシュを作る (原曲の行・カバーのイベント・カバー -> 原曲の時刻の対応)。

    python -m piano_cover.prepare

入力:
    data/metadata/dataset.json                  曲ごとの原曲とカバーの動画 ID
    Dataset/original_midis_v2/merged/<原曲>.mid  tsumugi で採譜した原曲 (まだない曲は飛ばす)
    Dataset/pianos_midi/<原曲>/<カバー>.mid      Transkun で採譜したカバー
    data/alignments/<原曲>/<カバー>.pt           音声同士の DTW で求めたカバー -> 原曲の時刻の対応
    data/alignments/source_offsets.json         アラインメントの原曲側の時刻の基準 (下記)
    data/metadata/piano_to_performer.json        カバー -> チャンネル ID (事前学習と同じ番号)

出力 (--out-dir):
    source_rows.npy / sources.npz   原曲の行 (piano_cover.source) を連結したものと曲ごとの範囲
    cover_events.npy / covers.npz   カバーのイベント (先頭の無音を詰めたもの) と曲ごとの範囲・原曲の番号・分割
    align.npy                       カバーの ALIGN_STEP フレームごとの、対応する原曲のフレーム (平滑化済み, float32)
    meta.json

アラインメントのキャッシュは旧モデルが作ったもので、時刻はどちらも「そのときの MIDI の最初の音」が 0 になっている。
    カバー側: カバー MIDI (今と同じ Transkun のもの) の最初の音の onset
    原曲側  : 旧 tsumugi の原曲 MIDI (Dataset/original_midis/merged) の最初の音の onset -> source_offsets.json
新しい原曲 MIDI は音声の絶対時刻なので、原曲側に source_offsets の秒数を足して戻す。

DTW の経路をそのまま使うとガタつく (4 秒の平滑化との差が上位 5% で 0.14 秒) ので、
--smooth-seconds の移動平均をかける。カバーのリズムは伸縮せず、この対応は cross-attention の位置にだけ使う。
"""

from __future__ import annotations

import argparse
import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from piano_ar.config import TokenizerConfig
from piano_ar.tokenizer import KIND, KIND_NOTE, ONSET, PianoTokenizer

from .source import load_source

PAIR_DATASET = Path("data/metadata/dataset.json")
PIANO_TO_PERFORMER = Path("data/metadata/piano_to_performer.json")
# アラインメントを持つ間隔 (フレーム)。10 フレーム = 0.1 秒
ALIGN_STEP = 10


def split_of(original_id: str, val_percent: float, test_percent: float) -> int:
    """原曲の動画 ID のハッシュで 0 学習 / 1 検証 / 2 テストに分ける (同じ原曲のカバーは同じ側に入る)"""
    bucket = zlib.crc32(original_id.encode()) % 10000 / 100
    if bucket < val_percent:
        return 1
    if bucket < val_percent + test_percent:
        return 2
    return 0


def cover_alignment(
    alignment_path: Path,
    raw_note_start: float,
    source_offset: float,
    shift_frames: int,
    num_frames: int,
    frame_rate: int,
    smooth: float,
) -> np.ndarray:
    """カバー (先頭を shift_frames 詰めたもの) の ALIGN_STEP フレームごとに、原曲 (音声の絶対時刻) のフレームを返す。

    raw_note_start: カバー MIDI の最初の音の onset (秒)、source_offset: 旧原曲 MIDI の最初の音の onset (秒)
    """
    import torch

    payload = torch.load(alignment_path, map_location="cpu", weights_only=False)
    target = payload["target_time_knots_seconds"].numpy().astype(np.float64)
    source = payload["source_time_knots_seconds"].numpy().astype(np.float64)
    grid = np.arange(0, num_frames + ALIGN_STEP, ALIGN_STEP)
    seconds = (grid + shift_frames) / frame_rate - raw_note_start
    mapped = (np.interp(seconds, target, source) + source_offset) * frame_rate
    width = max(1, int(round(smooth * frame_rate / ALIGN_STEP)))
    if width > 1 and len(mapped) > 1:
        padded = np.pad(mapped, (width // 2, width - 1 - width // 2), mode="edge")
        mapped = np.convolve(padded, np.ones(width) / width, mode="valid")
    return mapped.astype(np.float32)


def _load_song(args: tuple) -> dict | str:
    original_id, covers, source_dir, cover_dir, alignment_dir, source_offset, config, smooth = args
    try:
        tokenizer = PianoTokenizer(config)
        rows, source_end = load_source(source_dir / f"{original_id}.mid", config.frame_rate)
        results = []
        for piano_id in covers:
            midi_path = cover_dir / original_id / f"{piano_id}.mid"
            alignment_path = alignment_dir / original_id / f"{piano_id}.pt"
            if not midi_path.exists() or not alignment_path.exists():
                continue
            events, end_frame = tokenizer.midi_to_events(midi_path, trim=False)
            notes = events[events[:, KIND] == KIND_NOTE]
            if len(notes) == 0:
                continue
            shift = int(events[:, ONSET].min())
            events[:, ONSET] -= shift
            end_frame -= shift
            align = cover_alignment(
                alignment_path,
                notes[:, ONSET].min() / config.frame_rate,
                source_offset,
                shift,
                end_frame,
                config.frame_rate,
                smooth,
            )
            results.append((piano_id, events, end_frame, align))
        return {"original_id": original_id, "rows": rows, "end_frame": source_end, "covers": results}
    except Exception as e:  # 壊れた MIDI などは飛ばす
        return f"{original_id}: {e!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", default="Dataset/original_midis_v2/merged")
    parser.add_argument("--cover-dir", default="Dataset/pianos_midi")
    parser.add_argument("--alignment-dir", default="data/alignments")
    parser.add_argument("--out-dir", default="data/piano_cover")
    parser.add_argument("--smooth-seconds", type=float, default=4.0, help="アラインメントの移動平均の幅")
    parser.add_argument("--val-percent", type=float, default=5.0)
    parser.add_argument("--test-percent", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config = TokenizerConfig()
    source_dir, cover_dir, alignment_dir = Path(args.source_dir), Path(args.cover_dir), Path(args.alignment_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(PAIR_DATASET.read_text(encoding="utf-8"))
    performer = json.loads(PIANO_TO_PERFORMER.read_text(encoding="utf-8"))
    source_offsets = json.loads((alignment_dir / "source_offsets.json").read_text(encoding="utf-8"))

    jobs = []
    missing_source = 0
    for entry in pairs.values():
        original_id = entry["original"]
        if not (source_dir / f"{original_id}.mid").exists() or original_id not in source_offsets:
            missing_source += 1
            continue
        jobs.append(
            (
                original_id,
                entry["pianos"],
                source_dir,
                cover_dir,
                alignment_dir,
                source_offsets[original_id],
                config,
                args.smooth_seconds,
            )
        )
    print(f"原曲 {len(jobs)} 曲 (原曲の MIDI がまだない曲 {missing_source})")

    source_rows, source_ids, source_ends, source_lengths = [], [], [], []
    cover_events, cover_ids, cover_ends, cover_lengths, cover_source, channels, splits = [], [], [], [], [], [], []
    aligns, align_lengths = [], []
    failed = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(_load_song, jobs, chunksize=4):
            if isinstance(result, str):
                failed += 1
                print(f"失敗 {result}")
                continue
            if not result["covers"]:
                continue
            source_index = len(source_ids)
            source_ids.append(result["original_id"])
            source_rows.append(result["rows"])
            source_lengths.append(len(result["rows"]))
            source_ends.append(result["end_frame"])
            split = split_of(result["original_id"], args.val_percent, args.test_percent)
            for piano_id, events, end_frame, align in result["covers"]:
                cover_ids.append(piano_id)
                cover_events.append(events)
                cover_lengths.append(len(events))
                cover_ends.append(end_frame)
                cover_source.append(source_index)
                channels.append(int(performer.get(piano_id, 0)))
                splits.append(split)
                aligns.append(align)
                align_lengths.append(len(align))

    def offsets(lengths: list[int]) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    np.save(out_dir / "source_rows.npy", np.concatenate(source_rows).astype(np.int32))
    np.savez(
        out_dir / "sources.npz",
        offsets=offsets(source_lengths),
        end_frames=np.asarray(source_ends, dtype=np.int64),
        video_ids=np.asarray(source_ids),
    )
    np.save(out_dir / "cover_events.npy", np.concatenate(cover_events).astype(np.int32))
    np.save(out_dir / "align.npy", np.concatenate(aligns).astype(np.float32))
    np.savez(
        out_dir / "covers.npz",
        offsets=offsets(cover_lengths),
        align_offsets=offsets(align_lengths),
        end_frames=np.asarray(cover_ends, dtype=np.int64),
        video_ids=np.asarray(cover_ids),
        source_index=np.asarray(cover_source, dtype=np.int64),
        channels=np.asarray(channels, dtype=np.int64),
        split=np.asarray(splits, dtype=np.int64),
    )
    meta = {"tokenizer": asdict(config), "align_step": ALIGN_STEP, "smooth_seconds": args.smooth_seconds}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    splits_array = np.asarray(splits)
    hours = sum(cover_ends) / config.frame_rate / 3600
    print(
        f"原曲 {len(source_ids)} 曲 / カバー {len(cover_ids)} 本 ({hours:.0f} 時間) / "
        f"学習 {(splits_array == 0).sum()} 検証 {(splits_array == 1).sum()} テスト {(splits_array == 2).sum()} / 失敗 {failed}"
    )
    print(f"出力: {out_dir}")


if __name__ == "__main__":
    main()
