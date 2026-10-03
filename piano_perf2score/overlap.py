"""ASAP / ATEPP の楽譜と事前学習の楽譜 (prepare.py のキャッシュ) の重なりを、音の並びで調べる。

    python -m piano_perf2score.overlap

曲名は表記の揺れが大きいので使わない。楽譜を「同時に鳴り始める音の音高クラスの集合」の列にして、
続く 5 つの集合を 1 つの n-gram (12 bit x 5 = 60 bit) にし、評価曲の n-gram のうち事前学習の各曲に含まれる割合
(containment) を求める。全音符が同じ音の繰り返しの n-gram は曲を区別しないので使わない。
反復の展開の有無・音部記号・符尾の向きなどの書き方の違いには影響されない。移調した楽譜は見つからない。

評価曲の楽譜は music21 で読む (piano_score.musicxml では読めない書き方の楽譜もあるため)。
事前学習の側はキャッシュのトークン列から、位置 (BEAT) のトークンごとに区切って音高のトークンを集める。

出力 (data/piano_score/eval_overlap.json、公開しない):
    評価曲ごとの楽譜のパス・n-gram の数・containment が --min-containment 以上の事前学習の曲 (ID・割合・検証かどうか・曲名)
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from piano_score.data import ScoreCache
from piano_score.tokenizer import MLEN, ScoreTokenizer

NGRAM = 5


def ngram_hashes(masks: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """音高クラスの集合 (12 bit) の列から、続く NGRAM 個をつないだ値 [N - NGRAM + 1] (uint64)。
    groups を渡すと、別の曲にまたがる n-gram と、全部同じ集合の n-gram を -1 (使わない) にする"""
    masks = masks.astype(np.uint64)
    n = len(masks) - NGRAM + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    value = np.zeros(n, dtype=np.uint64)
    for k in range(NGRAM):
        value = (value << np.uint64(12)) | masks[k : k + n]
    keep = np.zeros(n, dtype=bool)
    for k in range(1, NGRAM):
        keep |= masks[k : k + n] != masks[:n]
    if groups is not None:
        keep &= groups[:n] == groups[NGRAM - 1 :]
    value[~keep] = np.uint64(2**64 - 1)
    return value


def _score_masks(path: str) -> np.ndarray | str:
    """楽譜を、同時に鳴り始める音の音高クラスの集合の列にする (music21。全パートをまとめる)"""
    import music21

    try:
        score = music21.converter.parse(path)
    except Exception as e:  # 読めない楽譜
        return f"{type(e).__name__}"
    onsets: dict[float, int] = {}
    for n in score.flatten().notes:  # flatten した後の offset は曲の頭からの位置
        offset = round(float(n.offset), 4)
        for p in n.pitches:
            onsets[offset] = onsets.get(offset, 0) | (1 << (p.midi % 12))
    return np.array([onsets[k] for k in sorted(onsets)], dtype=np.int64)


def pretraining_masks(cache: ScoreCache, tokenizer: ScoreTokenizer) -> tuple[np.ndarray, np.ndarray]:
    """事前学習の全曲を音高クラスの集合の列にする。(集合 [E], 曲番号 [E])"""
    kinds = tokenizer.kinds
    pitch_class = np.zeros(tokenizer.vocab_size, dtype=np.int64)
    is_beat = np.zeros(tokenizer.vocab_size, dtype=bool)
    for i, kind in enumerate(kinds):
        if kind == "pitch":
            pitch_class[i] = 1 << (tokenizer.values[i][0] % 12)
        elif kind == "beat":
            is_beat[i] = True
    tokens = np.load(cache.cache_dir / "tokens.npy", mmap_mode="r")
    song_of_measure = np.repeat(np.arange(len(cache.ids)), np.diff(cache.measure_offsets))
    masks_list, songs_list = [], []
    step = 20_000_000
    for start in range(0, len(tokens), step):
        chunk = np.asarray(tokens[start : start + step + 1], dtype=np.int64)
        # 位置 (BEAT) のトークンで区切る。MLEN の後の BEAT は小節の長さなので区切りにしない
        boundary = is_beat[chunk]
        boundary[1:] &= chunk[:-1] != MLEN
        boundary = boundary[: min(step, len(chunk))]
        chunk = chunk[: len(boundary)]
        starts = np.flatnonzero(boundary)
        if not len(starts):
            continue
        masks = np.bitwise_or.reduceat(pitch_class[chunk], starts)
        measure = np.searchsorted(cache.token_offsets, start + starts, side="right") - 1
        keep = masks != 0  # 指示だけの位置
        masks_list.append(masks[keep])
        songs_list.append(song_of_measure[measure[keep]])
        print(f"事前学習のトークン {min(start + step, len(tokens)) / len(tokens):.0%}", flush=True)
    # 区切りがチャンクをまたぐと、またいだ位置の音がその前の位置に入らない (全体の数か所なので気にしない)
    return np.concatenate(masks_list), np.concatenate(songs_list)


def eval_scores() -> list[dict]:
    """評価曲の楽譜。ASAP は曲ごとの xml_score.musicxml、ATEPP はメタデータに楽譜のある曲"""
    items = []
    for path in sorted(Path("Dataset/asap").rglob("xml_score.musicxml")):
        items.append({"source": "asap", "path": str(path), "name": str(path.parent.relative_to("Dataset/asap"))})
    rows = list(csv.DictReader(open("Dataset/ATEPP/ATEPP-metadata-1.2.csv", encoding="utf-8")))
    # zip の名前は macOS の NFD で、Windows で使えない文字 (: や ") は展開で _ になっているので、揃えてから引く
    root = Path("Dataset/ATEPP/ATEPP-1.2")
    files = {_file_key(str(p.relative_to(root))): p for p in root.rglob("*") if p.suffix in (".mxl", ".musicxml")}
    seen = set()
    for row in rows:
        score = row["score_path"]
        if score and score not in seen:
            seen.add(score)
            path = files.get(_file_key(score), root / score)
            items.append({"source": "atepp", "path": str(path), "name": str(Path(score).parent)})
    return items


def _file_key(path: str) -> str:
    path = unicodedata.normalize("NFC", path.replace("\\", "/"))
    return "".join("_" if c in ':"*?<>|' else c for c in path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default="data/piano_score/pretraining")
    parser.add_argument("--out", default="data/piano_score/eval_overlap.json")
    parser.add_argument("--min-containment", type=float, default=0.2, help="これ以上を重なりの候補として残す")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    items = eval_scores()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_score_masks, [item["path"] for item in items], chunksize=4))
    query_hashes: list[np.ndarray] = []
    for item, masks in zip(items, results):
        if isinstance(masks, str):
            item["error"] = masks
            query_hashes.append(np.zeros(0, dtype=np.uint64))
            continue
        hashes = np.unique(ngram_hashes(masks))
        hashes = hashes[hashes != np.uint64(2**64 - 1)]
        item["ngrams"] = int(len(hashes))
        query_hashes.append(hashes)
    print(f"評価曲の楽譜 {len(items)} (読めない {sum('error' in i for i in items)})")

    # n-gram -> それを持つ評価曲
    all_hashes = np.concatenate(query_hashes)
    owners = np.concatenate([np.full(len(h), q) for q, h in enumerate(query_hashes)])
    order = np.argsort(all_hashes)
    all_hashes, owners = all_hashes[order], owners[order]
    vocabulary = np.unique(all_hashes)

    cache = ScoreCache(args.cache_dir)
    tokenizer = ScoreTokenizer(cache.tokenizer_config)
    masks, songs = pretraining_masks(cache, tokenizer)
    hashes = ngram_hashes(masks, songs)
    songs = songs[: len(hashes)]
    hit = np.isin(hashes, vocabulary)
    # (事前学習の曲, n-gram) の重複を除いてから、評価曲ごとに数える
    pairs = np.unique(np.stack([songs[hit].astype(np.uint64), hashes[hit]], axis=1), axis=0)
    left = np.searchsorted(all_hashes, pairs[:, 1], side="left")
    right = np.searchsorted(all_hashes, pairs[:, 1], side="right")
    lengths = right - left
    pair_song = np.repeat(pairs[:, 0].astype(np.int64), lengths)
    # 組 i の評価曲は owners[left[i]:right[i]]
    first = np.repeat(left - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths)
    pair_query = owners[first + np.arange(lengths.sum())]
    keys, key_counts = np.unique(pair_query * len(cache.ids) + pair_song, return_counts=True)
    query_of, song_of = keys // len(cache.ids), keys % len(cache.ids)
    sizes = np.array([max(item.get("ngrams", 0), 1) for item in items])
    containments = key_counts / sizes[query_of]
    keep = containments >= args.min_containment

    titles = json.loads((cache.cache_dir / "titles.json").read_text(encoding="utf-8"))
    for item in items:
        item["matches"] = []
    for q, song, containment in zip(query_of[keep].tolist(), song_of[keep].tolist(), containments[keep].tolist()):
        song_id = str(cache.ids[song])
        items[q]["matches"].append(
            {
                "id": song_id,
                "containment": round(containment, 3),
                "is_val": bool(cache.is_val[song]),
                "title": titles.get(song_id, ""),
            }
        )
    for item in items:
        item["matches"].sort(key=lambda m: -m["containment"])
    Path(args.out).write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")

    for source in ("asap", "atepp"):
        group = [i for i in items if i["source"] == source and "error" not in i]
        for threshold in (0.2, 0.5, 0.8):
            found = sum(any(m["containment"] >= threshold for m in i["matches"]) for i in group)
            print(f"{source}: containment {threshold} 以上の事前学習の曲がある楽譜 {found} / {len(group)}")
    print(f"出力: {args.out}")


if __name__ == "__main__":
    main()
