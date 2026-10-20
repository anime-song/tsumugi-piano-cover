"""楽譜の書き方の誤りを MusicXML から数え、自動変換 (MIDI の取り込み) や初心者が書いたような低品質の楽譜を見つける。

    python -m piano_score.quality --out data/piano_score/quality.json --workers 8
    python -m piano_score.quality --ids 4303651 5775550 3834116      # 1 曲ずつ見る

楽譜コーパス (BEAT の MuseScore 由来の MusicXML) には評価などのメタデータがないので、書き方そのものから判断する。
数える規則 (どれも「当てはまる箇所の数 / 音の数」。odd_measure だけは小節の数で割る):
    tuplet_misaligned  連符のまとまりの頭が、そのまとまりの長さの倍数の位置にない (拍の途中から始まる 16 分の 3 連など。
                       演奏のタイミングをそのまま音符にした MIDI の取り込みに多い)
    redundant_tie      同じ声部でタイでつないだ 2 音を、同じ位置から 1 つの音符 (付点も可) で書ける
    rest_split         同じ声部の続く休符を、拍の中で 1 つの休符にまとめられる
    beat_cross         音が拍の頭をまたぐのに、拍の頭からも拍の半分の位置からも始まらない
                       (8 分裏から始まる 4 分音符のようなシンコペーションの定型は許す)
    staff_split        1 本の旋律の線を 2 つの段に休符で分けた書き方。拍の半分以下の休符の位置でもう片方の段が単音を
                       出し、その音が休符の直前のこちらの段の単音から 1 オクターブ以内 (線がつながっている) で、同じ拍の
                       中でもう片方の段にも休符がある (分散和音の 4 つ目だけを右手の段に書くなど。音高で段を機械的に
                       分けた取り込みや初心者の書き方)。低音と後打ちの和音を分け合う伴奏 (和音か、離れた音) は数えない
    ledger             加線が 6 本以上の音 (音部記号・オクターブ記号 8va などを考えた見た目の位置で数える)。ある小節から先が
                       オクターブ違う・音部記号を間違えた楽譜で多い。高い音域をヘ音記号のまま書く程度 (加線 4〜5 本) は数えない
    odd_measure        拍子より長い小節 (和音の印が落ちて音が順に並んだ壊れた MusicXML など)。短い小節は、弱起や繰り返しの
                       区切りで 1 小節を 2 つに分けた正しい書き方が多いので数えない
曲ごとの印 (0 / 1):
    tab                TAB 譜 (TAB の音部記号かフレットの番号がある)
    no_time            拍子記号がどこにもない

low_quality が外す基準。ユーザーが良い・微妙・悪いを付けた見本 (2026-10-21、outputs/score_label の 1〜50 番 + 確認済みの
悪い 3 曲) に合わせた: 悪い楽譜は、加線の多い音 (ある小節から先がオクターブ違う・音部記号の誤り) と TAB 譜がいちばん目立ち、
拍をまたぐ連符・タイの鎖・旋律の線の段の分け方がそれに続く。細切れの休符と拍をまたぐ音価は、良い楽譜にも多いので使わない
(休符は楽譜のトークンに入れず書き出しで付け直すので、元の楽譜の休符の書き方は学習に関係しない)。
拍子より長い小節のある楽譜と拍子記号のない楽譜は、見た目が良くても小節の拍子を学べないので外す。
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
DIATONIC = "CDEFGAB"
# 段のいちばん上の線といちばん下の線の位置 (全音階の番号 = オクターブ x 7 + 音名の番号)。G: F5 / E4、F: A3 / G2、C (アルト): G4 / F3
STAFF_LINES = {"G": (5 * 7 + 3, 4 * 7 + 2), "F": (3 * 7 + 5, 2 * 7 + 4), "C": (4 * 7 + 4, 3 * 7 + 3)}
LEDGER_LINES = 6
# 1 つの音符で書ける長さ (4 分音符単位): 全音符〜64 分音符と、その付点・複付点
NOTE_VALUES = {Fraction(4, 2**k) * m for k in range(7) for m in (Fraction(1), Fraction(3, 2), Fraction(7, 4))}
RULES = ("tuplet_misaligned", "redundant_tie", "rest_split", "beat_cross", "staff_split", "ledger")
# low_quality で外すしきい値 (この値を超えたら外す)
LIMITS = {
    "ledger": 0.01,
    "tuplet_misaligned": 0.01,
    "redundant_tie": 0.05,
    "staff_split": 0.02,
    "odd_measure": 0.02,  # 1 小節だけ長いような楽譜は残す (学習の窓で、その小節を含む窓を外す)
}
EPS = Fraction(1, 256)


def _same_beat(start: Fraction, end: Fraction, beat: Fraction) -> bool:
    return start // beat == (end - EPS) // beat


def lint(path: str | Path) -> dict:
    root = ET.parse(path).getroot()
    tab = any((sign.text or "") == "TAB" for sign in root.iter("sign")) or root.find(".//fret") is not None
    no_time = root.find(".//time") is None
    counts: dict[str, int] = defaultdict(int)
    notes_total = 0
    measures_total = 0
    for part in root.findall("part"):
        divisions = 1
        beats, beat_type = 4, 4
        clefs: dict[str, str] = {}
        clef_octaves: dict[str, int] = {}  # 段 -> 音部記号のオクターブの変更 (ト音記号の下の 8 など)
        octave_shift: dict[str, int] = {}  # 段 -> 8va などで見た目を実音から動かすオクターブ数
        measures = part.findall("measure")
        measures_total += len(measures)
        for mi, measure in enumerate(measures):
            pos = Fraction(0)
            end_pos = Fraction(0)
            # 声部ごと: [(位置, 長さ, 休符か, タイの始まりか, 音高, 連符か)]、段ごと: 打鍵の位置と休符の (位置, 長さ)
            voice_items: dict[tuple[str, str], list] = defaultdict(list)
            staff_onsets: dict[str, dict] = defaultdict(dict)  # 段 -> 打鍵の位置 -> 音高の一覧
            staff_rests: dict[str, list] = defaultdict(list)
            beat = Fraction(1)
            nominal = Fraction(4)
            for el in measure:
                if el.tag == "attributes":
                    if el.find("divisions") is not None:
                        divisions = int(el.findtext("divisions"))
                    time = el.find("time")
                    if time is not None and (time.findtext("beats") or "").isdigit():
                        beats, beat_type = int(time.findtext("beats")), int(time.findtext("beat-type"))
                    for clef in el.findall("clef"):
                        clefs[clef.get("number", "1")] = clef.findtext("sign") or "G"
                        clef_octaves[clef.get("number", "1")] = int(clef.findtext("clef-octave-change") or 0)
                if el.tag == "direction":
                    for shift in el.iter("octave-shift"):
                        octaves = {8: 1, 15: 2, 22: 3}.get(int(shift.get("size", "8")), 1)
                        kind = shift.get("type")
                        # type="down" は 8va (実音より下に書いて見せる)。音高は実音で入っている
                        offset = -octaves if kind == "down" else octaves if kind == "up" else 0
                        octave_shift[el.findtext("staff") or "1"] = offset
                compound = beat_type in (8, 16) and beats % 3 == 0 and beats > 3
                beat = Fraction(4, beat_type) * (3 if compound else 1)
                nominal = Fraction(4 * beats, beat_type)
                if el.tag == "backup":
                    pos -= Fraction(int(el.findtext("duration") or 0), divisions)
                elif el.tag == "forward":
                    pos += Fraction(int(el.findtext("duration") or 0), divisions)
                elif el.tag == "note":
                    if el.find("grace") is not None or el.find("cue") is not None:
                        continue
                    duration = Fraction(int(el.findtext("duration") or 0), divisions)
                    chord = el.find("chord") is not None
                    at = pos - duration if chord else pos
                    if not chord:
                        pos += duration
                    staff = el.findtext("staff") or "1"
                    voice = (staff, el.findtext("voice") or "1")
                    rest = el.find("rest") is not None
                    # 小節全体の休符を全休符の長さで書いたもの (6/8 の小節に 4 拍分) は、表示は正しいので長さに数えない
                    if not (rest and at == 0 and duration >= nominal):
                        end_pos = max(end_pos, pos)
                    modification = el.find("time-modification")
                    in_tuplet = modification is not None
                    if in_tuplet and not chord:
                        for tuplet in el.iter("tuplet"):
                            if tuplet.get("type") != "start":
                                continue
                            actual = int(modification.findtext("actual-notes") or 1)
                            normal = int(modification.findtext("normal-notes") or 1)
                            # まとまりの長さ: 書いた音価 x normal (16 分の 3:2 なら 8 分)
                            span = duration * Fraction(actual, normal) * normal
                            if span > 0 and at % span != 0:
                                counts["tuplet_misaligned"] += 1
                    pitch = None
                    if not rest:
                        step = el.findtext("pitch/step")
                        if step is None:  # 打楽器の音符など
                            continue
                        alter = round(float(el.findtext("pitch/alter") or 0))
                        pitch = 12 * (int(el.findtext("pitch/octave") or 4) + 1) + STEP[step] + alter
                        sign = clefs.get(staff, "G" if staff == "1" else "F")
                        if sign in STAFF_LINES:
                            octave = (
                                int(el.findtext("pitch/octave") or 4)
                                + octave_shift.get(staff, 0)
                                - clef_octaves.get(staff, 0)
                            )
                            position = octave * 7 + DIATONIC.index(step)
                            top, bottom = STAFF_LINES[sign]
                            if max(position - top, bottom - position) // 2 >= LEDGER_LINES:
                                counts["ledger"] += 1
                        staff_onsets[staff].setdefault(at, []).append(pitch)
                        if not chord:
                            notes_total += 1
                    if chord:
                        continue
                    if rest and duration > 0:
                        staff_rests[staff].append((at, duration))
                    tie_start = any(t.get("type") == "start" for t in el.findall("tie"))
                    voice_items[voice].append((at, duration, rest, tie_start, pitch, in_tuplet))
            if end_pos > nominal:
                counts["odd_measure"] += 1
            for items in voice_items.values():
                items.sort(key=lambda x: x[0])
                for i, (at, duration, rest, tie_start, pitch, in_tuplet) in enumerate(items):
                    following = items[i + 1] if i + 1 < len(items) else None
                    if not rest and not in_tuplet and duration > 0:
                        crosses = (at // beat + 1) * beat < at + duration and at % beat != 0
                        half = beat / 2
                        if crosses and not (at % half == 0 and duration % half == 0 and duration <= 2 * beat):
                            counts["beat_cross"] += 1
                    if following is None or following[0] != at + duration or following[5] or in_tuplet:
                        continue
                    total = duration + following[1]
                    # 拍の頭から始まり拍の倍数の長さ、または全体が 1 拍の中に収まるなら 1 つの音符で書ける
                    one_note = total in NOTE_VALUES and (
                        (at % beat == 0 and total % beat == 0) or _same_beat(at, at + total, beat)
                    )
                    if tie_start and not rest and following[4] == pitch and one_note:
                        counts["redundant_tie"] += 1
                    if (
                        rest
                        and following[2]
                        and one_note
                        and at % min(beat, total) == 0
                        and _same_beat(at, at + total, beat)
                    ):
                        counts["rest_split"] += 1
            staves = sorted(staff_onsets.keys() | staff_rests.keys())
            if len(staves) >= 2:
                for a in staves:
                    for b in staves:
                        if a == b:
                            continue
                        for at, duration in staff_rests[a]:
                            other = staff_onsets[b].get(at)
                            if duration > beat / 2 + EPS or other is None or len(other) != 1:
                                continue
                            # 休符の直前の a の段の打鍵が単音で、b の段の音と 1 オクターブ以内 (1 本の線)
                            before = [t for t in staff_onsets[a] if t < at]
                            if not before or len(staff_onsets[a][max(before)]) != 1:
                                continue
                            if abs(staff_onsets[a][max(before)][0] - other[0]) > 12:
                                continue
                            if any(
                                _same_beat(min(at, r_at), max(at, r_at) + EPS, beat)
                                for r_at, r_duration in staff_rests[b]
                                if r_duration <= beat
                            ):
                                counts["staff_split"] += 1
    n = max(1, notes_total)
    out = {rule: counts[rule] / n for rule in RULES}
    out["odd_measure"] = counts["odd_measure"] / max(1, measures_total)
    out["tab"] = int(tab)
    out["no_time"] = int(no_time)
    out["notes"] = notes_total
    out["measures"] = measures_total
    return out


def low_quality(result: dict) -> bool:
    """lint の結果から、学習に使わない低品質の楽譜か"""
    if "error" in result or result.get("tab") or result.get("no_time"):
        return True
    return any(result[name] > limit for name, limit in LIMITS.items())


def _lint_file(path: str) -> tuple[str, dict]:
    try:
        return Path(path).stem, lint(path)
    except Exception as error:  # noqa: BLE001  壊れた MusicXML は記録だけして続ける
        return Path(path).stem, {"error": repr(error)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="Dataset/piano_musicxml")
    parser.add_argument("--ids", nargs="*", help="この番号の楽譜だけを数えて表示する")
    parser.add_argument("--out", default="data/piano_score/quality.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.ids:
        for score_id in args.ids:
            _, result = _lint_file(str(Path(args.dir) / f"{score_id}.musicxml"))
            values = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in result.items())
            print(score_id, "低品質" if low_quality(result) else "良い", values)
        return
    paths = sorted(str(p) for p in Path(args.dir).glob("*.musicxml"))
    results = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (score_id, result) in enumerate(pool.map(_lint_file, paths, chunksize=64), 1):
            results[score_id] = result
            if i % 10000 == 0:
                print(f"{i}/{len(paths)}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results), encoding="utf-8")
    errors = sum("error" in r for r in results.values())
    low = sum(low_quality(r) for r in results.values())
    print(
        f"{len(results)} 曲 (読めなかった {errors}、低品質 {low} = {low / max(1, len(results)):.1%}) を {out} に書いた"
    )


if __name__ == "__main__":
    main()
