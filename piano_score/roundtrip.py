"""MusicXML -> トークン -> MusicXML の往復を確かめ、読めなかった理由とトークン数の分布を集計する。

    python -m piano_score.roundtrip --num 2000 --export-dir outputs/score_roundtrip --export 5

各曲について
    1. 読み込み -> トークン -> 復元 -> トークン が一致するか (tokenizer の往復)
    2. 復元した楽譜を MusicXML に書き出して読み直し、トークンが一致するか (書き出しの往復)
を調べる。--export を付けると、書き出した MusicXML を目で確認できるよう保存する。
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .musicxml import ScoreError, read_musicxml, write_musicxml
from .tokenizer import ScoreTokenizer


def _first_difference(a: list[list[int]], b: list[list[int]], tokenizer: ScoreTokenizer) -> str:
    if len(a) != len(b):
        return f"小節数 {len(a)} != {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            j = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
            show = lambda s: " ".join(tokenizer.describe(t) for t in s[max(0, j - 4) : j + 4])  # noqa: E731
            return f"小節 {i} の {j} 番目: {show(x)} || {show(y)}"
    return ""


def check(args: tuple[str, str | None]) -> dict:
    path, export_path = args
    tokenizer = ScoreTokenizer()
    try:
        measures = read_musicxml(path)
        tokens = tokenizer.encode(measures)
    except ScoreError as e:
        message = str(e) if "語彙にない" in str(e) else str(e).split(" ")[0]
        return {"path": path, "error": message, "detail": repr(e)}
    except Exception as e:  # 想定していない壊れ方
        return {"path": path, "error": f"例外 {type(e).__name__}", "detail": repr(e)}
    result = {"path": path, "lengths": [len(t) for t in tokens], "measures": len(measures)}
    # 復元した楽譜にはテンポがないので、MTIME はトークンから復元した開始時刻を渡して比べる
    decoded, starts = tokenizer.decode(tokens)
    again = tokenizer.encode(decoded, starts)
    result["token_roundtrip"] = _first_difference(tokens, again, tokenizer)
    try:
        if export_path:
            out = export_path
        else:
            handle, out = tempfile.mkstemp(suffix=".musicxml")
            os.close(handle)
        write_musicxml(decoded, out)
        reread = tokenizer.encode(read_musicxml(out), starts)
        result["xml_roundtrip"] = _first_difference(tokens, reread, tokenizer)
        if not export_path:
            os.remove(out)
    except Exception as e:
        result["xml_roundtrip"] = f"例外 {e!r}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="Dataset/piano_musicxml")
    parser.add_argument("--num", type=int, default=2000, help="調べる曲数 (ランダムに選ぶ)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export-dir", default="outputs/score_roundtrip")
    parser.add_argument("--export", type=int, default=0, help="書き出した MusicXML を保存する曲数")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    names = sorted(os.listdir(args.dir))
    random.Random(args.seed).shuffle(names)
    names = names[: args.num]
    export_dir = Path(args.export_dir)
    jobs = [
        (str(Path(args.dir) / name), str(export_dir / name) if i < args.export else None)
        for i, name in enumerate(names)
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(check, jobs, chunksize=8))

    tokenizer = ScoreTokenizer()
    ok = [r for r in results if "error" not in r]
    errors = Counter(r["error"] for r in results if "error" in r)
    print(f"曲 {len(results)} / 使える {len(ok)} ({len(ok) / len(results):.1%}) / 語彙 {tokenizer.vocab_size}")
    for reason, count in errors.most_common(20):
        print(f"  {count:5d} {reason}")
    shown = set()
    for r in results:
        if "detail" in r and r["error"] not in shown:
            shown.add(r["error"])
            print(f"  例 {Path(r['path']).name}: {r['detail']}")

    for key in ("token_roundtrip", "xml_roundtrip"):
        failed = [r for r in ok if r[key]]
        print(f"{key}: 不一致 {len(failed)} / {len(ok)}")
        for r in failed[:5]:
            print(f"  {Path(r['path']).name}: {r[key]}")

    lengths = np.concatenate([r["lengths"] for r in ok])
    measures = np.array([r["measures"] for r in ok])
    totals = np.array([sum(r["lengths"]) for r in ok])
    print(
        f"1 小節のトークン数 50/90/99/99.9% = {np.percentile(lengths, [50, 90, 99, 99.9]).round(1)} / 最大 {lengths.max()}"
    )
    limit = tokenizer.config.max_patch_tokens
    print(f"  上限 {limit} を超える小節 {int((lengths > limit).sum())} ({(lengths > limit).mean():.3%})")
    print(f"1 曲の小節数 50/90/99% = {np.percentile(measures, [50, 90, 99]).round()} (反復展開後)")
    print(f"1 曲のトークン数 50/90/99% = {np.percentile(totals, [50, 90, 99]).round()}")


if __name__ == "__main__":
    main()
