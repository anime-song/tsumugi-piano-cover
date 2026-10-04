"""実演奏と楽譜の組のキャッシュ (data.PairCache で読む形) を作るときの共通の部品。asap.py と periscope.py で使う。

    1 つの演奏の、弾いた順の小節 (Measure) と各小節の開始時刻 (秒) と演奏の音・ペダル
      -> 楽譜と演奏が合っているかを確かめる (match_counts)
      -> 語彙にない小節や、楽譜と演奏が合わない小節の前後で切る (make_pairs)
      -> 全部の組を 1 つのキャッシュに書く (write_cache)

キャッシュ:
    tokens.npy    小節ごとのトークン列 (MTIME を除く、終端トークン込み) を連結した int16
    measures.npz  token_offsets [M+1] / starts [M] (各小節の開始時刻 (秒)。組の演奏の最初の音 = 0) / lengths [M] (4 分音符単位)
    songs.npz     measure_offsets / note_offsets / pedal_offsets [S+1]、ids (組)、pieces (曲)、sources、is_val、match
    notes.npy     [N, 4] 打鍵・離鍵の時刻 (秒、組の最初の音 = 0)・音高・ベロシティ
    pedal.npy     [K, 2] 時刻・踏んでいるか
    meta.json     トークナイザー設定・曲数・使えなかった理由
"""

from __future__ import annotations

import json
import zlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from piano_score.musicxml import ScoreError, _tie_stops
from piano_score.score import Measure
from piano_score.tokenizer import ScoreTokenizer

from .data import SEGMENT_MARGIN, pedal_between

MIN_SEGMENT = 8  # 切ったとき、組として残す最小の小節数


class PairError(Exception):
    pass


def match_counts(measures: list[Measure], starts: np.ndarray, notes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """小節ごとの (近くに同じ音高の打鍵がある楽譜の音の数, 楽譜の音の数)。
    楽譜の各音 (タイで続く音を除く) は、小節の中の位置から時刻に直す (小節の中は一定の速さとみなす)"""
    found = np.zeros(len(measures))
    total = np.zeros(len(measures))
    if len(measures) < 2:
        return found, total
    tie_stops = _tie_stops(measures)
    durations = np.diff(starts)
    durations = np.append(durations, durations[-1] * measures[-1].length / max(measures[-2].length, 1e-3))
    by_pitch = {int(p): np.sort(notes[notes[:, 2] == p, 0]) for p in np.unique(notes[:, 2])}
    for mi, (measure, start, duration) in enumerate(zip(measures, starts, durations)):
        tolerance = max(0.1, 0.12 * duration)
        for g in measure.groups:
            t = start + float(g.onset) / max(float(measure.length), 1e-3) * duration
            for note in g.notes:
                if (mi, id(g), note.pitch) in tie_stops:
                    continue
                total[mi] += 1
                onsets = by_pitch.get(note.pitch)
                if onsets is None:
                    continue
                i = np.searchsorted(onsets, t)
                near = [abs(onsets[j] - t) for j in (i - 1, i) if 0 <= j < len(onsets)]
                found[mi] += bool(near) and min(near) <= tolerance
    return found, total


def match_rate(measures: list[Measure], starts: np.ndarray, notes: np.ndarray) -> float:
    """楽譜の音のうち、近くに同じ音高の打鍵がある割合"""
    found, total = match_counts(measures, starts, notes)
    return float(found.sum() / max(total.sum(), 1))


def score_error(e: ScoreError) -> str:
    message = str(e)
    return "楽譜: " + (" ".join(message.split(" ")[:2]) if message.startswith("語彙にない") else message.split(" ")[0])


def runs(flags: list[bool]) -> list[tuple[int, int]]:
    """True が続く区間 [a, b) の一覧"""
    out, a = [], None
    for i, flag in enumerate([*flags, False]):
        if flag and a is None:
            a = i
        elif not flag and a is not None:
            out.append((a, i))
            a = None
    return out


def make_pairs(
    tokenizer: ScoreTokenizer,
    measures: list[Measure],
    starts: np.ndarray,
    notes: np.ndarray,
    pedal: np.ndarray,
    end: float,
    *,
    name: str,
    info: dict,
    usable: list[bool] | None = None,
    errors: list[str],
) -> list[dict]:
    """1 つの演奏を組にする。語彙にない小節と usable が False の小節の前後で切り、MIN_SEGMENT 小節以上の区間を
    別々の組にする (時刻はそれぞれの最初の音 = 0)。end より後に始まる音は使わない。info は各組にそのまま入れる"""
    bins = tokenizer.mtime_bins(starts)
    encoded: list[list[int] | None] = []
    for i, (measure, b) in enumerate(zip(measures, bins)):
        try:
            encoded.append(tokenizer.encode_measure(measure, b, i == len(measures) - 1))
        except ScoreError as e:
            encoded.append(None)
            errors.append(score_error(e) + " (小節)")
    keep = [e is not None and (usable is None or usable[i]) for i, e in enumerate(encoded)]
    if usable is not None:
        errors += ["楽譜と演奏が合わない (小節)"] * sum(not u for u in usable)
    pairs = []
    for a, b in runs(keep):
        if b - a < MIN_SEGMENT:
            errors.append("切った残りが短い")
            continue
        t0 = starts[a] - SEGMENT_MARGIN if a > 0 else -np.inf
        t1 = starts[b] - SEGMENT_MARGIN if b < len(starts) else end
        inside = (notes[:, 0] >= t0) & (notes[:, 0] < t1)
        if not inside.any():
            continue
        origin = float(notes[inside, 0].min())
        segment_notes = notes[inside].copy()
        segment_notes[:, :2] -= origin
        segment_pedal = pedal_between(pedal, t0, t1) - np.array([origin, 0.0])
        pairs.append(
            {
                "id": name if (a, b) == (0, len(measures)) else f"{name}#{a}",
                "performance": name,
                "tokens": [np.asarray(e[1:], dtype=np.int16) for e in encoded[a:b]],  # MTIME は学習時に入れ直す
                "starts": (starts[a:b] - origin).astype(np.float32),
                "lengths": np.array([float(m.length) for m in measures[a:b]], dtype=np.float32),
                "notes": segment_notes.astype(np.float32),
                "pedal": segment_pedal.astype(np.float32).reshape(-1, 2),
                **info,
            }
        )
    return pairs


def write_cache(pairs: list[dict], errors: Counter, out_dir: Path, val_percent: float, extra_meta: dict) -> dict:
    """組をキャッシュに書いて meta を返す。検証は曲 (piece) ごとに、曲名のハッシュで val_percent % を選ぶ"""
    tokenizer = ScoreTokenizer()
    pairs = sorted(pairs, key=lambda p: p["id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    measure_tokens = [t for p in pairs for t in p["tokens"]]
    token_offsets = np.concatenate([[0], np.cumsum([len(t) for t in measure_tokens])]).astype(np.int64)
    np.save(out_dir / "tokens.npy", np.concatenate(measure_tokens))
    np.savez(
        out_dir / "measures.npz",
        token_offsets=token_offsets,
        starts=np.concatenate([p["starts"] for p in pairs]),
        lengths=np.concatenate([p["lengths"] for p in pairs]),
    )
    np.save(out_dir / "notes.npy", np.concatenate([p["notes"] for p in pairs]))
    np.save(out_dir / "pedal.npy", np.concatenate([p["pedal"] for p in pairs]))

    def offsets(key: str) -> np.ndarray:
        return np.concatenate([[0], np.cumsum([len(p[key]) for p in pairs])]).astype(np.int64)

    piece_names = [p["piece"] for p in pairs]
    is_val = np.array([zlib.crc32(n.encode()) % 10000 < val_percent * 100 for n in piece_names])
    np.savez(
        out_dir / "songs.npz",
        measure_offsets=offsets("tokens"),
        note_offsets=offsets("notes"),
        pedal_offsets=offsets("pedal"),
        ids=np.asarray([p["id"] for p in pairs]),
        pieces=np.asarray(piece_names),
        sources=np.asarray([p.get("source", "") for p in pairs]),
        is_val=is_val,
        match=np.asarray([p["match"] for p in pairs], dtype=np.float32),
    )
    hours = sum(float(p["notes"][:, 0].max()) for p in pairs) / 3600
    match = np.asarray([p["match"] for p in pairs])
    meta = {
        "tokenizer": asdict(tokenizer.config),
        "visible_only": True,
        "vocab_size": tokenizer.vocab_size,
        "songs": len(pairs),
        "performances": len({p["performance"] for p in pairs}),
        "val_songs": int(is_val.sum()),
        "pieces": len(set(piece_names)),
        "val_pieces": len({n for n, v in zip(piece_names, is_val) if v}),
        "hours": round(hours, 2),
        "measures": len(measure_tokens),
        "tokens": int(token_offsets[-1]),
        "match_percentiles_5_50": [round(float(x), 3) for x in np.percentile(match, [5, 50])],
        "skipped": dict(errors.most_common()),
        **extra_meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"組 {len(pairs)} (検証 {int(is_val.sum())}) / 演奏 {meta['performances']} / 曲 {meta['pieces']}"
        f" (検証 {meta['val_pieces']}) / {hours:.1f} 時間 / 小節 {len(measure_tokens)}"
    )
    print(f"楽譜と演奏の一致 5/50% = {meta['match_percentiles_5_50']}")
    print("使えなかった演奏・小節 (末尾が (小節) のものは小節の数):")
    for reason, count in errors.most_common(25):
        print(f"  {count:6d} {reason}")
    print(f"出力: {out_dir}")
    return meta
