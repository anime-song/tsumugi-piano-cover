"""事前学習用の楽譜 (Dataset/piano_musicxml) をまとめて小節ごとのトークン列のキャッシュにする。

    python -m piano_score.prepare

反復記号は展開してから並べる。読めない楽譜 (段の構成が違う・対応していない記号など) は飛ばし、理由ごとの数を meta.json に残す。
MTIME (演奏上の小節の開始時刻) は学習時に決める (楽譜だけの事前学習ではテンポ記号から、合成演奏では描き出した時刻から)
ので、キャッシュには MTIME を除いたトークン列と、テンポ記号どおりの各小節の開始時刻を入れる。

出力:
    tokens.npy    全曲の全小節のトークン列 (MTIME を除く、終端トークン込み) を連結した int16
    measures.npz  token_offsets [M+1] (小節ごとの範囲) / seconds [M] (テンポ記号どおりの開始時刻、最初の音 = 0)
    songs.npz     measure_offsets [S+1] (曲ごとの小節の範囲) / ids / is_val
    titles.json   曲 ID -> 曲名 (ASAP / ATEPP の評価曲との重なりを調べる用。公開しない)
    meta.json     トークナイザー設定・語彙サイズ・曲数・読めなかった理由
"""

from __future__ import annotations

import argparse
import json
import os
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .musicxml import ScoreError, parse_xml, read_musicxml, score_title
from .tokenizer import ScoreTokenizer


def _load(path: str) -> tuple[list[np.ndarray], np.ndarray, str] | str:
    tokenizer = ScoreTokenizer()
    try:
        root = parse_xml(path)
        measures = read_musicxml(root)
        seconds = tokenizer.nominal_starts(measures)
        patches = tokenizer.encode(measures, seconds)
    except ScoreError as e:
        message = str(e)
        return " ".join(message.split(" ")[:2]) if message.startswith("語彙にない") else message.split(" ")[0]
    except Exception as e:  # 想定していない壊れ方
        return f"例外 {type(e).__name__}"
    # MTIME (各小節の先頭のトークン) は学習時に入れ直す
    return [np.asarray(p[1:], dtype=np.int16) for p in patches], seconds.astype(np.float32), score_title(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="Dataset/piano_musicxml")
    parser.add_argument("--out-dir", default="data/piano_score/pretraining")
    parser.add_argument("--val-percent", type=float, default=1.0, help="検証に回す曲の割合 (曲 ID のハッシュで決める)")
    parser.add_argument("--min-measures", type=int, default=4, help="これより小節の少ない曲は使わない")
    parser.add_argument("--limit", type=int, default=0, help="動作確認用に先頭から何曲だけ使うか (0 なら全部)")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    tokenizer = ScoreTokenizer()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(n for n in os.listdir(args.dir) if n.endswith(".musicxml"))
    if args.limit:
        names = names[: args.limit]
    print(f"楽譜 {len(names)} 曲")

    tokens_list, token_lengths, seconds_list, measure_counts, ids = [], [], [], [], []
    titles: dict[str, str] = {}
    errors: Counter = Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        results = pool.map(_load, [str(Path(args.dir) / n) for n in names], chunksize=64)
        for i, (name, result) in enumerate(zip(names, results), 1):
            if isinstance(result, str):
                errors[result] += 1
            else:
                patches, seconds, title = result
                if len(patches) < args.min_measures:
                    errors["小節が少なすぎる"] += 1
                else:
                    song_id = name.removesuffix(".musicxml")
                    tokens_list += patches
                    token_lengths += [len(p) for p in patches]
                    seconds_list.append(seconds)
                    measure_counts.append(len(patches))
                    ids.append(song_id)
                    titles[song_id] = title
            if i % 10000 == 0:
                print(f"{i}/{len(names)} 使える {len(ids)}", flush=True)

    token_offsets = np.concatenate([[0], np.cumsum(token_lengths)]).astype(np.int64)
    measure_offsets = np.concatenate([[0], np.cumsum(measure_counts)]).astype(np.int64)
    np.save(out_dir / "tokens.npy", np.concatenate(tokens_list))
    np.savez(out_dir / "measures.npz", token_offsets=token_offsets, seconds=np.concatenate(seconds_list))
    is_val = np.array([zlib.crc32(i.encode()) % 10000 < args.val_percent * 100 for i in ids])
    np.savez(out_dir / "songs.npz", measure_offsets=measure_offsets, ids=np.asarray(ids), is_val=is_val)
    (out_dir / "titles.json").write_text(json.dumps(titles, ensure_ascii=False, indent=0), encoding="utf-8")

    lengths = np.asarray(token_lengths)
    meta = {
        "tokenizer": asdict(tokenizer.config),
        "vocab_size": tokenizer.vocab_size,
        "songs": len(ids),
        "val_songs": int(is_val.sum()),
        "measures": int(len(lengths)),
        "tokens": int(lengths.sum()),
        "skipped": dict(errors.most_common()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    limit = tokenizer.config.max_patch_tokens - 1  # MTIME の分を空ける
    print(f"使える {len(ids)} / {len(names)} 曲 (検証 {int(is_val.sum())}) / 小節 {len(lengths)} / トークン {lengths.sum()}")
    print(f"1 小節のトークン数 50/99/99.9% = {np.percentile(lengths, [50, 99, 99.9]).round()} / 上限超え {(lengths > limit).sum()}")
    print(f"1 曲の小節数 50/99% = {np.percentile(measure_counts, [50, 99]).round()}")
    for reason, count in errors.most_common(12):
        print(f"  {count:6d} {reason}")
    print(f"出力: {out_dir}")


if __name__ == "__main__":
    main()
