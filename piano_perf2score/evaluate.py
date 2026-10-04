"""generate --val-songs が書き出した楽譜 (*_reference.musicxml と *_generated.musicxml の組) を比べる。

    python -m piano_perf2score.evaluate outputs/perf2score_eval/stage2 outputs/perf2score_eval/stage3 ...

書き方が 1 つに決まらない所 (タイでつなぐか付点で書くか・声部の分け方・強弱の記号など) で点が変わらないよう、
楽譜を「どの位置でどの音を打鍵し、何拍鳴らすか」に直してから比べる。
    打鍵        タイで続く音を除いた (位置, 音高)。位置は曲の頭からの 4 分音符単位。装飾音は除く
    鳴る長さ    打鍵からタイでつながる音をたどった合計の長さ

指標 (曲ごとに出して平均する):
    区間 F1        4 分音符 8 個ずつの区間ごとに、生成をずらして最もよく合う所で測った打鍵 F1 (小節線のずれに強い)
    打鍵 F1        (位置, 音高) が完全に一致する打鍵の F1
    倍率込み F1    生成の位置を 1/2・2/3・1・3/2・2 倍したうちの最良 (テンポの倍・半分の読み違えだけなら高い)
    音の並び       位置を無視した、同時に打鍵する音の組の並びの一致 (LCS の F1)。リズムが崩れても音が合っていれば高い
    長さ一致       一致した打鍵のうち、鳴る長さも一致する割合
    小節線 F1      小節の開始位置の F1 (2/4 と 4/4 の読み違えでは下がる)
    段一致         一致した打鍵のうち、上下の段も一致する割合
    拍子一致       最初の小節の拍子が同じか
長さ・段・小節線は、倍率込み F1 でいちばん合った倍率に生成をそろえてから比べる。
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

from piano_score.musicxml import read_musicxml
from piano_score.score import Measure

SCALES = (Fraction(1, 2), Fraction(2, 3), Fraction(1), Fraction(3, 2), Fraction(2))


def attacks(measures: list[Measure]) -> dict[tuple[Fraction, int], tuple[Fraction, int]]:
    """打鍵 (位置, 音高) -> (鳴る長さ, 段)。タイで続く音はつながる前の打鍵の長さに足す"""
    out: dict[tuple[Fraction, int], tuple[Fraction, int]] = {}
    open_ties: dict[tuple[int, int], tuple[tuple[Fraction, int], Fraction]] = {}  # (段, 音高) -> (打鍵, 続く位置)
    offset = Fraction(0)
    for measure in measures:
        groups = sorted((g for g in measure.groups if not g.duration.grace), key=lambda g: g.onset)
        for g in groups:
            onset = offset + g.onset
            length = g.duration.quarters
            for note in g.notes:
                key = (g.staff, note.pitch)
                tie = open_ties.pop(key, None)
                if tie is not None and tie[1] == onset:
                    attack = tie[0]
                    total, staff = out[attack]
                    out[attack] = (total + length, staff)
                else:
                    attack = (onset, note.pitch)
                    out[attack] = (length, g.staff)
                if note.tie:
                    open_ties[key] = (attack, onset + length)
        offset += measure.length
    return out


def barlines(measures: list[Measure]) -> set[Fraction]:
    out, offset = set(), Fraction(0)
    for measure in measures:
        out.add(offset)
        offset += measure.length
    return out


def f1(a: set, b: set) -> float:
    return 2 * len(a & b) / max(len(a) + len(b), 1)


def chord_sequence_f1(a: dict, b: dict) -> float:
    def chords(x: dict) -> list[frozenset[int]]:
        by: dict[Fraction, set[int]] = {}
        for onset, pitch in x:
            by.setdefault(onset, set()).add(pitch)
        return [frozenset(by[t]) for t in sorted(by)]

    ca, cb = chords(a), chords(b)
    previous = [0] * (len(cb) + 1)
    for x in ca:
        current = [0]
        for j, y in enumerate(cb):
            current.append(previous[j] + 1 if x == y else max(previous[j + 1], current[j]))
        previous = current
    return 2 * previous[-1] / max(len(ca) + len(cb), 1)


def local_f1(ref: set, gen: set, end: Fraction, window: int = 8, reach: int = 16) -> float:
    """元の楽譜を 4 分音符 window 個ずつの区間に分け、区間ごとに生成を 16 分単位で ±reach 拍までずらした最良の F1 を、
    区間の打鍵の数で重みづけて平均する。途中で小節線がずれて位置が全部ずれても、区間の中が合っていれば高い"""
    shifts = [Fraction(k, 4) for k in range(-4 * reach, 4 * reach + 1)]
    total = score = 0.0
    start = Fraction(0)
    while start < end:
        stop = start + window
        a = {x for x in ref if start <= x[0] < stop}
        if a:
            near = [x for x in gen if start - reach <= x[0] < stop + reach]
            best = max(
                (f1(a, {(t + s, p) for t, p in near if start <= t + s < stop}), -abs(s)) for s in shifts
            )[0]
            score += best * len(a)
            total += len(a)
        start = stop
    return score / max(total, 1)


def compare(reference: list[Measure], generated: list[Measure]) -> dict[str, float]:
    ref, gen = attacks(reference), attacks(generated)
    end = sum((m.length for m in reference), Fraction(0))
    # 元の楽譜より後ろの生成は比べない (生成は演奏の終わりまで続くことがある)
    exact = {k for k in gen if k[0] < end}
    best_scale, best = Fraction(1), -1.0
    for scale in SCALES:
        scaled = {(t * scale, p) for t, p in gen if t * scale < end}
        value = f1(set(ref), scaled)
        if value > best:
            best_scale, best = scale, value
    # 長さ・段・小節線は、いちばん合う倍率にそろえてから比べる
    scaled_gen = {(t * best_scale, p): (d * best_scale, staff) for (t, p), (d, staff) in gen.items() if t * best_scale < end}
    matched = set(ref) & set(scaled_gen)
    ref_bars = barlines(reference)
    gen_bars = {t * best_scale for t in barlines(generated) if t * best_scale < end}
    return {
        "local_f1": local_f1(set(ref), set(scaled_gen), end),
        "attack_f1": f1(set(ref), exact),
        "scaled_f1": best,
        "sequence": chord_sequence_f1(ref, scaled_gen),
        "duration": sum(ref[k][0] == scaled_gen[k][0] for k in matched) / max(len(matched), 1),
        "barline_f1": f1(ref_bars, gen_bars),
        "staff": sum(ref[k][1] == scaled_gen[k][1] for k in matched) / max(len(matched), 1),
        "meter": float(reference[0].time_signature == generated[0].time_signature) if generated else 0.0,
        "scale": float(best_scale),
    }


METRICS = ("local_f1", "attack_f1", "scaled_f1", "sequence", "duration", "barline_f1", "staff", "meter")
LABELS = {
    "local_f1": "区間F1",
    "attack_f1": "打鍵F1",
    "scaled_f1": "倍率込",
    "sequence": "音の並び",
    "duration": "長さ一致",
    "barline_f1": "小節線",
    "staff": "段一致",
    "meter": "拍子一致",
}


def evaluate_dir(directory: Path, verbose: bool = False) -> dict[str, float]:
    rows = []
    for reference_path in sorted(directory.glob("*_reference.musicxml")):
        generated_path = Path(str(reference_path).replace("_reference", "_generated"))
        if not generated_path.exists():
            continue
        row = compare(read_musicxml(reference_path), read_musicxml(generated_path))
        rows.append(row)
        if verbose:
            name = reference_path.name.removesuffix("_reference.musicxml")
            print(f"  {name[:48]:48s} " + " ".join(f"{row[m]:.2f}" for m in METRICS) + f" x{row['scale']:.2g}")
    return {m: sum(r[m] for r in rows) / max(len(rows), 1) for m in METRICS} | {"songs": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--verbose", action="store_true", help="曲ごとの値も出す")
    args = parser.parse_args()
    print(f"{'':24s} " + " ".join(f"{LABELS[m]:>6s}" for m in METRICS) + "  曲数")
    for directory in args.dirs:
        if args.verbose:
            print(directory)
        result = evaluate_dir(Path(directory), args.verbose)
        print(f"{Path(directory).name[:24]:24s} " + " ".join(f"{result[m]:8.3f}" for m in METRICS) + f"  {result['songs']}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
