"""採譜済み MIDI をまとめてイベント配列のキャッシュにする。

    python -m piano_ar.prepare --midi-dir Dataset/pretraining_midi --out-dir data/piano_ar/pretraining

出力:
    events.npy  全曲のイベント配列を連結した int32 [N, 5] (学習時は mmap で読む)
    songs.npz   曲ごとの範囲・終端フレーム・チャンネル ID・検証用フラグ
    meta.json   トークナイザー設定とチャンネル数
"""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import TokenizerConfig
from .tokenizer import PianoTokenizer

VIDEOS_CSV = Path("data/metadata/channel_videos_filtered.csv")
PERFORMER_INDEX = Path("data/metadata/performer_index.json")
CHANNEL_INDEX = Path("data/metadata/channel_index.json")
PAIR_DATASET = Path("data/metadata/dataset.json")


def update_channel_index(channel_ids: set[str]) -> dict[str, int]:
    """ペアデータの performer_index と同じ ID 空間で、足りないチャンネルを末尾に追加する。

    0 は「チャンネル指定なし」として空けておく (performer_index は 1 始まり)。
    既存の番号は変えないので、データを追加して作り直しても学習済みの埋め込みと対応が崩れない。
    """
    path = CHANNEL_INDEX if CHANNEL_INDEX.exists() else PERFORMER_INDEX
    index: dict[str, int] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    next_id = max(index.values(), default=0) + 1
    for channel_id in sorted(channel_ids - set(index)):
        index[channel_id] = next_id
        next_id += 1
    CHANNEL_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def _load(args: tuple[str, TokenizerConfig]) -> tuple[np.ndarray, int] | str:
    path, config = args
    try:
        return PianoTokenizer(config).midi_to_events(path)
    except Exception as e:  # 壊れた MIDI は飛ばす
        return repr(e)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--midi-dir", default="Dataset/pretraining_midi")
    parser.add_argument("--out-dir", default="data/piano_ar/pretraining")
    parser.add_argument(
        "--val-percent", type=float, default=1.0, help="検証に回す曲の割合 (video_id のハッシュで決める)"
    )
    parser.add_argument("--min-seconds", type=float, default=20.0, help="これより短い曲は使わない")
    parser.add_argument(
        "--keep-pair-videos",
        action="store_true",
        help="ペアデータ (dataset.json) に含まれる動画も使う。既定ではカバー学習の検証・テストへの漏れを防ぐため除く",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config = TokenizerConfig()
    tokenizer = PianoTokenizer(config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with VIDEOS_CSV.open(encoding="utf-8-sig") as f:
        video_to_channel = {row["video_id"]: row["channel_id"] for row in csv.DictReader(f)}
    excluded: set[str] = set()
    if not args.keep_pair_videos and PAIR_DATASET.exists():
        for entry in json.loads(PAIR_DATASET.read_text(encoding="utf-8")).values():
            excluded.add(entry.get("original"))
            excluded.update(entry.get("pianos", []))

    paths = []
    skipped_pair = 0
    for path in sorted(Path(args.midi_dir).glob("*.mid")):
        if path.stem in excluded:
            skipped_pair += 1
            continue
        paths.append(path)
    channel_index = update_channel_index({video_to_channel[p.stem] for p in paths if p.stem in video_to_channel})
    print(
        f"MIDI {len(paths)} 本 (ペアデータとの重複で除外 {skipped_pair} 本) / チャンネル ID 数 {max(channel_index.values()) + 1}"
    )

    events_list, video_ids, channels, end_frames, lengths = [], [], [], [], []
    failed = short = 0
    with ProcessPoolExecutor(args.workers) as pool:
        results = pool.map(_load, [(str(p), config) for p in paths], chunksize=32)
        for i, (path, result) in enumerate(zip(paths, results), 1):
            if isinstance(result, str):
                failed += 1
                print(f"失敗 {path.name}: {result}")
                continue
            events, end_frame = result
            if end_frame < args.min_seconds * config.frame_rate:
                short += 1
                continue
            events_list.append(events)
            lengths.append(len(events))
            video_ids.append(path.stem)
            channels.append(channel_index.get(video_to_channel.get(path.stem, ""), 0))
            end_frames.append(end_frame)
            if i % 2000 == 0:
                print(f"{i}/{len(paths)}")

    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.save(out_dir / "events.npy", np.concatenate(events_list).astype(np.int32))
    is_val = np.array([zlib.crc32(v.encode()) % 10000 < args.val_percent * 100 for v in video_ids])
    np.savez(
        out_dir / "songs.npz",
        offsets=offsets,
        end_frames=np.asarray(end_frames, dtype=np.int64),
        channels=np.asarray(channels, dtype=np.int64),
        video_ids=np.asarray(video_ids),
        is_val=is_val,
    )
    num_channels = max(channel_index.values()) + 1
    meta = {"tokenizer": asdict(config), "num_channels": num_channels, "channel_index": str(CHANNEL_INDEX)}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    hours = sum(end_frames) / config.frame_rate / 3600
    print(
        f"曲数 {len(video_ids)} (検証 {int(is_val.sum())}) / 合計 {hours:.0f} 時間 / 失敗 {failed} / 短すぎて除外 {short}"
    )
    print(f"チャンネル不明 {sum(c == 0 for c in channels)} 曲 / 語彙サイズ {tokenizer.vocab_size}")
    print(f"出力: {out_dir}")


if __name__ == "__main__":
    main()
