"""ASAP (Dataset/asap) の演奏と楽譜を、3 段目の学習の組のキャッシュにする。

    python -m piano_perf2score.asap

ASAP には演奏ごとに、小節線の時刻 (performance_downbeats) と、それが楽譜 (xml_score.musicxml) の何小節目か
(downbeats_score_map、MusicXML の measure の並びの番号 (0 から)。number 属性ではない) が付いている。ここから次の 2 つを作り、楽譜をその順に読んで
トークン列にする (piano_score.musicxml.read_musicxml の order)。
    - 弾いた順の小節の並び (繰り返しを省いた演奏も、弾いたとおりに並ぶ)
    - 各小節の開始時刻

補う所:
    "97-98" のように 1 つの小節線に複数の小節が対応する所 (繰り返しの境目で 1 小節が 2 つに分かれて書かれている所) は、
    次の小節線までの拍の時刻 (performance_beats) から、内側の小節の開始を補う。
    最初の小節線より前の小節 (弱起) と、最後の小節線より後の小節は、隣の小節のテンポで延ばして開始を見積もる。

楽譜と演奏が合っているかは、楽譜の各音を小節の中の位置から時刻に直し、同じ音高の打鍵が近くにあるかで確かめる
(match)。小節の対応がずれている演奏はここが大きく下がるので、MIN_MATCH 未満は使わない。
語彙にない小節 (カデンツァの細かい連符など) がある演奏は、その小節の前後で切って別々の組にする (切った組の時刻は
それぞれの最初の音 = 0 秒。曲の途中から始まる組も、曲の頭から始まるものとして扱う)。
曲 (同じ曲の *_no_repeat などの別版も同じ曲とみなす) ごとに検証に回すかを決める。

出力 (data/piano_score/asap、PairCache で読む):
    tokens.npy    小節ごとのトークン列 (MTIME を除く、終端トークン込み) を連結した int16
    measures.npz  token_offsets [M+1] / starts [M] (各小節の開始時刻 (秒)。演奏の最初の音 = 0) / lengths [M] (4 分音符単位)
    songs.npz     measure_offsets / note_offsets / pedal_offsets [S+1]、ids (演奏のパス)、pieces (曲)、is_val、match
    notes.npy     [N, 4] 打鍵・離鍵の時刻 (秒、最初の音 = 0)・音高・ベロシティ
    pedal.npy     [K, 2] 時刻・踏んでいるか
    meta.json     トークナイザー設定・曲数・使えなかった理由
"""

from __future__ import annotations

import argparse
import json
import re
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pretty_midi

from piano_score.musicxml import ScoreError, _tie_stops, parse_xml, read_musicxml
from piano_score.score import Measure
from piano_score.tokenizer import ScoreTokenizer

from .data import pedal_between
from .generate import read_performance

MIN_MATCH = 0.8  # 楽譜の音のうち、近くに同じ音高の打鍵がある割合の下限
MAX_TAIL = 2  # 最後の小節線より後に足してよい小節の数
MIN_SEGMENT = 8  # 語彙にない小節の前後で切ったとき、組として残す最小の小節数
SEGMENT_MARGIN = 0.1  # 切った所の小節線より前に始まる音 (装飾音・ずれ) も、この秒数までは後ろの組に入れる


class PairError(Exception):
    pass


def piece_name(performance: str) -> str:
    """演奏のパス -> 曲 (同じ曲の別版のフォルダ (17-1_no_repeat など) も同じ曲にする)"""
    folder = str(Path(performance).parent).replace("\\", "/")
    return re.sub(r"_no_repeat.*$|_repeat.*$", "", folder)


def played_order(annotation: dict, lengths: list[float]) -> tuple[list[int], np.ndarray]:
    """弾いた順の小節の並びの番号 (0 から) と、各小節の開始時刻 (MIDI の秒)"""
    entries = annotation["downbeats_score_map"]
    downbeats = np.asarray(annotation["performance_downbeats"], dtype=np.float64)
    beats = np.asarray(annotation["performance_beats"], dtype=np.float64)
    if not isinstance(entries, list) or len(entries) != len(downbeats) or len(entries) < 2:
        raise PairError("小節の対応がない")
    order: list[int] = []
    starts: list[float] = []
    for k, entry in enumerate(entries):
        # まとめて書かれた所は先頭の番号から続く小節とみなす ("808-809-910" のような書き誤りもあるため)
        parts = str(entry).split("-")
        group = list(range(int(parts[0]), int(parts[0]) + len(parts)))
        if group[-1] >= len(lengths):
            raise PairError("楽譜にない小節番号")
        if len(group) > 1:
            if k + 1 >= len(downbeats):
                raise PairError("最後の小節線に複数の小節")
            # 次の小節線までの拍の時刻を、小節の長さの割合で読む
            t0, t1 = downbeats[k], downbeats[k + 1]
            knots = np.concatenate([[t0], beats[(beats > t0 + 1e-6) & (beats < t1 - 1e-6)], [t1]])
            cumulative = np.cumsum([0.0] + [lengths[i] for i in group])
            position = cumulative[:-1] / cumulative[-1] * (len(knots) - 1)
            starts += np.interp(position, np.arange(len(knots)), knots).tolist()
        else:
            starts.append(float(downbeats[k]))
        order += group
    if any(b <= a for a, b in zip(starts, starts[1:])):
        raise PairError("小節線の時刻が増えていない")

    # 最初の小節線より前の小節 (弱起): 次の小節のテンポで逆算する
    if order[0] > 1:
        raise PairError("最初の小節線より前に 2 小節以上")
    if order[0] == 1:
        seconds_per_quarter = (starts[1] - starts[0]) / max(lengths[order[0]], 1e-3)
        order.insert(0, 0)
        starts.insert(0, starts[0] - lengths[0] * seconds_per_quarter)
    # 最後の小節線より後の小節: 前の小節のテンポで延ばす。多ければ途中で終わる演奏 (Fine など) とみなして足さない
    tail = list(range(order[-1] + 1, len(lengths)))
    if len(tail) <= MAX_TAIL:
        for index in tail:
            seconds_per_quarter = (starts[-1] - starts[-2]) / max(lengths[order[-2]], 1e-3)
            starts.append(starts[-1] + lengths[order[-1]] * seconds_per_quarter)
            order.append(index)
    return order, np.asarray(starts)


def match_rate(measures: list[Measure], starts: np.ndarray, notes: np.ndarray) -> float:
    """楽譜の各音 (タイで続く音を除く) を小節の中の位置から時刻に直し、近くに同じ音高の打鍵がある割合"""
    if len(measures) < 2:
        return 0.0
    tie_stops = _tie_stops(measures)
    durations = np.diff(starts)
    durations = np.append(durations, durations[-1] * measures[-1].length / max(measures[-2].length, 1e-3))
    by_pitch: dict[int, np.ndarray] = {}
    for p in np.unique(notes[:, 2]).astype(int):
        by_pitch[p] = np.sort(notes[notes[:, 2] == p, 0])
    found = total = 0
    for mi, (measure, start, duration) in enumerate(zip(measures, starts, durations)):
        tolerance = max(0.1, 0.12 * duration)
        for g in measure.groups:
            t = start + float(g.onset) / max(float(measure.length), 1e-3) * duration
            for note in g.notes:
                if (mi, id(g), note.pitch) in tie_stops:
                    continue
                total += 1
                onsets = by_pitch.get(note.pitch)
                if onsets is None:
                    continue
                i = np.searchsorted(onsets, t)
                near = [abs(onsets[j] - t) for j in (i - 1, i) if 0 <= j < len(onsets)]
                found += bool(near) and min(near) <= tolerance
    return found / max(total, 1)


def _load_piece(xml: str, performances: list[tuple[str, dict]]) -> list[dict | str]:
    """1 つの楽譜とその演奏すべて -> 演奏ごとの組 (dict) か、使えなかった理由 (str)"""
    tokenizer = ScoreTokenizer()
    try:
        root = parse_xml(xml)
        lengths = [float(m.length) for m in read_musicxml(root, unfold=False, visible_only=True)]
    except ScoreError as e:
        return [f"楽譜: {str(e).split(' ')[0]}"] * len(performances)
    except Exception as e:  # 想定していない壊れ方
        return [f"楽譜: 例外 {type(e).__name__}"] * len(performances)

    results: list[dict | str] = []
    for name, annotation in performances:
        try:
            if annotation.get("score_and_performance_aligned") not in (True, "True"):
                raise PairError("楽譜と揃っていない印")
            order, starts = played_order(annotation, lengths)
            measures = read_musicxml(root, unfold=False, visible_only=True, order=order)
            path = Path("Dataset/asap") / name
            performance = read_performance(path)
            # read_performance は最初の音を 0 秒にするので、小節線の時刻も同じだけずらす
            starts = starts - min(n.start for inst in pretty_midi.PrettyMIDI(str(path)).instruments for n in inst.notes)
            notes, pedal = performance.notes, performance.pedal
            # 最後の小節の終わりより後に始まる音は捨てる (注釈が途中で終わっている演奏)
            end = starts[-1] + (starts[-1] - starts[-2]) * lengths[order[-1]] / max(lengths[order[-2]], 1e-3) + 0.5
            dropped = int((notes[:, 0] >= end).sum())
            match = match_rate(measures, starts, notes[notes[:, 0] < end])
            if match < MIN_MATCH:
                raise PairError("楽譜と演奏が合わない")
        except PairError as e:
            results.append(str(e))
            continue
        except ScoreError as e:
            results.append(_score_error(e))
            continue
        # 語彙にない小節 (カデンツァの細かい連符など) があれば、その前後で切って別々の組にする
        bins = tokenizer.mtime_bins(starts)
        encoded: list[list[int] | None] = []
        for i, (measure, b) in enumerate(zip(measures, bins)):
            try:
                encoded.append(tokenizer.encode_measure(measure, b, i == len(measures) - 1))
            except ScoreError as e:
                encoded.append(None)
                results.append(_score_error(e) + " (小節)")
        for a, b in _runs([e is not None for e in encoded]):
            if b - a < MIN_SEGMENT:
                results.append("切った残りが短い")
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
            results.append(
                {
                    "id": name if (a, b) == (0, len(measures)) else f"{name}#{a}",
                    "performance": name,
                    "piece": piece_name(name),
                    "tokens": [np.asarray(e[1:], dtype=np.int16) for e in encoded[a:b]],  # MTIME は学習時に入れ直す
                    "starts": (starts[a:b] - origin).astype(np.float32),
                    "lengths": np.array([float(m.length) for m in measures[a:b]], dtype=np.float32),
                    "notes": segment_notes.astype(np.float32),
                    "pedal": segment_pedal.astype(np.float32).reshape(-1, 2),
                    "match": match,
                    "dropped_notes": dropped if b == len(measures) else 0,
                }
            )
    return results


def _score_error(e: ScoreError) -> str:
    message = str(e)
    return "楽譜: " + (" ".join(message.split(" ")[:2]) if message.startswith("語彙にない") else message.split(" ")[0])


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    """True が続く区間 [a, b) の一覧"""
    runs, a = [], None
    for i, flag in enumerate([*flags, False]):
        if flag and a is None:
            a = i
        elif not flag and a is not None:
            runs.append((a, i))
            a = None
    return runs


def _load_piece_args(args: tuple[str, list[tuple[str, dict]]]) -> list[dict | str]:
    return _load_piece(*args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="Dataset/asap")
    parser.add_argument("--out-dir", default="data/piano_score/asap")
    parser.add_argument("--val-percent", type=float, default=8.0, help="検証に回す曲の割合 (曲名のハッシュで決める)")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    annotations = json.loads((Path(args.dir) / "asap_annotations.json").read_text(encoding="utf-8"))
    pieces: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for name, annotation in annotations.items():
        pieces[str(Path(args.dir) / Path(name).parent / "xml_score.musicxml")].append((name, annotation))
    print(f"楽譜 {len(pieces)} / 演奏 {len(annotations)}")

    tokenizer = ScoreTokenizer()
    pairs: list[dict] = []
    errors: Counter = Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for results in pool.map(_load_piece_args, sorted(pieces.items())):
            for result in results:
                if isinstance(result, str):
                    errors[result] += 1
                else:
                    pairs.append(result)
    pairs.sort(key=lambda p: p["id"])

    out_dir = Path(args.out_dir)
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
    is_val = np.array([zlib.crc32(n.encode()) % 10000 < args.val_percent * 100 for n in piece_names])
    np.savez(
        out_dir / "songs.npz",
        measure_offsets=offsets("tokens"),
        note_offsets=offsets("notes"),
        pedal_offsets=offsets("pedal"),
        ids=np.asarray([p["id"] for p in pairs]),
        pieces=np.asarray(piece_names),
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
        "dropped_notes": int(sum(p["dropped_notes"] for p in pairs)),
        "skipped": dict(errors.most_common()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"組 {len(pairs)} (検証 {int(is_val.sum())}) / 演奏 {meta['performances']} / 曲 {meta['pieces']} (検証 {meta['val_pieces']})"
        f" / {hours:.1f} 時間 / 小節 {len(measure_tokens)}"
    )
    print(f"楽譜と演奏の一致 5/50% = {meta['match_percentiles_5_50']} / 最後の小節より後で捨てた音 {meta['dropped_notes']}")
    print("使えなかった演奏・小節 (末尾が (小節) のものは小節の数):")
    for reason, count in errors.most_common():
        print(f"  {count:5d} {reason}")
    print(f"出力: {out_dir}")


if __name__ == "__main__":
    main()
