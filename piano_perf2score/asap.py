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

出力は data/piano_score/asap (形は pairs.py)。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pretty_midi

from piano_score.musicxml import ScoreError, parse_xml, read_musicxml
from piano_score.tokenizer import ScoreTokenizer

from .generate import read_performance
from .pairs import PairError, make_pairs, match_rate, score_error, write_cache

MIN_MATCH = 0.8  # 楽譜の音のうち、近くに同じ音高の打鍵がある割合の下限
MAX_TAIL = 2  # 最後の小節線より後に足してよい小節の数


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


def _load_piece(xml: str, performances: list[tuple[str, dict]]) -> list[dict | str]:
    """1 つの楽譜とその演奏すべて -> 演奏ごとの組 (dict) か、使えなかった理由 (str)"""
    tokenizer = ScoreTokenizer()
    try:
        root = parse_xml(xml)
        lengths = [float(m.length) for m in read_musicxml(root, unfold=False, visible_only=True)]
    except ScoreError as e:
        return [score_error(e)] * len(performances)
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
            match = match_rate(measures, starts, notes[notes[:, 0] < end])
            if match < MIN_MATCH:
                raise PairError("楽譜と演奏が合わない")
        except PairError as e:
            results.append(str(e))
            continue
        except ScoreError as e:
            results.append(score_error(e))
            continue
        # 語彙にない小節 (カデンツァの細かい連符など) があれば、その前後で切って別々の組にする
        errors: list[str] = []
        info = {"piece": piece_name(name), "source": "ASAP", "match": match}
        results += make_pairs(tokenizer, measures, starts, notes, pedal, end, name=name, info=info, errors=errors)
        results += errors
    return results


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

    pairs: list[dict] = []
    errors: Counter = Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for results in pool.map(_load_piece_args, sorted(pieces.items())):
            for result in results:
                if isinstance(result, str):
                    errors[result] += 1
                else:
                    pairs.append(result)
    write_cache(pairs, errors, Path(args.out_dir), args.val_percent, {})


if __name__ == "__main__":
    main()
