"""事前学習の曲 (local/build_manifest.py の一覧) の MIDI をまとめてイベント配列のキャッシュにする。

    python -m piano_ar.prepare

YouTube の曲は Transkun で採譜した Dataset/pretraining_midi/<ID>.mid、MAESTRO は配布 MIDI をそのまま読む。
まだ採譜していない曲は飛ばすので、採譜の途中でも作れる (増えたら作り直す)。

出力:
    events.npy  全曲のイベント配列を連結した int32 [N, 5] (学習時は mmap で読む)
    songs.npz   曲ごとの範囲・終端フレーム・チャンネル ID・演奏者 (-1 は不明)・データセット・検証用フラグ
    meta.json   トークナイザー設定・チャンネル数・データセット名
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

PERFORMER_INDEX = Path("data/metadata/performer_index.json")
CHANNEL_INDEX = Path("data/metadata/channel_index.json")
PAIR_DATASET = Path("data/metadata/dataset.json")
MANIFEST = Path("data/metadata/pretraining_manifest.csv")
# songs.npz の sources の番号 (local/build_manifest.py の source 列)
SOURCES = ("channels", "pijama", "pop2piano", "piast", "maestro")


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
    parser.add_argument("--manifest", default=str(MANIFEST), help="local/build_manifest.py が作る曲の一覧")
    parser.add_argument("--midi-dir", default="Dataset/pretraining_midi", help="YouTube の曲の採譜済み MIDI")
    parser.add_argument("--out-dir", default="data/piano_ar/pretraining")
    parser.add_argument("--val-percent", type=float, default=1.0, help="検証に回す曲の割合 (曲 ID のハッシュで決める)")
    parser.add_argument("--min-seconds", type=float, default=20.0, help="これより短い曲は使わない")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config = TokenizerConfig()
    tokenizer = PianoTokenizer(config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.manifest).open(encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    # マニフェストでも除いているが、カバー学習の検証・テストへの漏れは致命的なので念のためもう一度除く
    excluded: set[str] = set()
    if PAIR_DATASET.exists():
        for entry in json.loads(PAIR_DATASET.read_text(encoding="utf-8")).values():
            excluded.add(entry.get("original"))
            excluded.update(entry.get("pianos", []))

    rows, missing = [], 0
    for row in manifest:
        if row["id"] in excluded:
            continue
        path = Path(row["midi_path"]) if row["midi_path"] else Path(args.midi_dir) / f"{row['id']}.mid"
        if not path.exists():  # まだ採譜していない曲
            missing += 1
            continue
        rows.append({**row, "path": path})
    # チャンネル ID (UC...) の演奏者はチャンネル番号も振る (チャンネル条件で学習する場合と、以前のモデルとの互換のため)
    channel_index = update_channel_index({r["performer"] for r in rows if r["performer"].startswith("UC")})
    performer_index: dict[str, int] = {}
    for r in rows:
        if r["performer"]:
            performer_index.setdefault(r["performer"], len(performer_index))
    print(
        f"曲 {len(rows)} (まだ MIDI がない {missing}) / 演奏者 {len(performer_index)} / チャンネル ID 数 {max(channel_index.values()) + 1}"
    )

    events_list, song_ids, channels, performers, sources, end_frames, lengths = [], [], [], [], [], [], []
    failed = short = 0
    with ProcessPoolExecutor(args.workers) as pool:
        results = pool.map(_load, [(str(r["path"]), config) for r in rows], chunksize=32)
        for i, (row, result) in enumerate(zip(rows, results), 1):
            if isinstance(result, str):
                failed += 1
                print(f"失敗 {row['path'].name}: {result}")
                continue
            events, end_frame = result
            if end_frame < args.min_seconds * config.frame_rate:
                short += 1
                continue
            events_list.append(events)
            lengths.append(len(events))
            song_ids.append(row["id"])
            channels.append(channel_index.get(row["performer"], 0))
            performers.append(performer_index.get(row["performer"], -1))
            sources.append(SOURCES.index(row["source"]))
            end_frames.append(end_frame)
            if i % 5000 == 0:
                print(f"{i}/{len(rows)}")

    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.save(out_dir / "events.npy", np.concatenate(events_list).astype(np.int32))
    is_val = np.array([zlib.crc32(v.encode()) % 10000 < args.val_percent * 100 for v in song_ids])
    np.savez(
        out_dir / "songs.npz",
        offsets=offsets,
        end_frames=np.asarray(end_frames, dtype=np.int64),
        channels=np.asarray(channels, dtype=np.int64),
        performers=np.asarray(performers, dtype=np.int64),
        sources=np.asarray(sources, dtype=np.int64),
        video_ids=np.asarray(song_ids),
        is_val=is_val,
    )
    num_channels = max(channel_index.values()) + 1
    meta = {
        "tokenizer": asdict(config),
        "num_channels": num_channels,
        "channel_index": str(CHANNEL_INDEX),
        "sources": list(SOURCES),
        "num_performers": len(performer_index),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    frames = np.asarray(end_frames)
    source_array = np.asarray(sources)
    print(
        f"曲数 {len(song_ids)} (検証 {int(is_val.sum())}) / 合計 {frames.sum() / config.frame_rate / 3600:.0f} 時間 / 失敗 {failed} / 短すぎて除外 {short}"
    )
    for k, name in enumerate(SOURCES):
        part = source_array == k
        print(f"  {name:10s} {int(part.sum()):6d} 曲 {frames[part].sum() / config.frame_rate / 3600:6.0f} 時間")
    print(f"演奏者不明 {sum(p < 0 for p in performers)} 曲 / 語彙サイズ {tokenizer.vocab_size}")
    print(f"出力: {out_dir}")


if __name__ == "__main__":
    main()
