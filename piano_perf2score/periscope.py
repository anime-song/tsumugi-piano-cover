"""PERiScoPe v1.1 (Dataset/PERiScoPe/PERiScoPe-1.1) の演奏と楽譜を、3 段目の学習の組のキャッシュにする。

    python -m piano_perf2score.periscope

PERiScoPe は (n)ASAP・ATEPP・Web から集めて Transkun V2 で採譜した演奏を、楽譜と音ごとに対応づけたもの。
    data/          楽譜 (score.musicxml / score.mxl) と元の演奏の MIDI (raw)
    data_aligned/  楽譜の MIDI (score_*.mid、partitura で反復を展開したもの)、楽譜と対応づけた演奏の MIDI と
                   対応 (*.npy: 楽譜の音 i -> 演奏の音の番号。どちらも (打鍵の時刻, 音高) で並べたときの番号)
対応づけた演奏 (aligned) は、弾かれなかった音を補い、間 (ま) の長い所を詰めてあるなど、元の演奏から手が入っている。
モデルの入力には元の演奏 (raw) を使い、aligned は小節線の時刻を求めるのにだけ使う:

    楽譜の MIDI の各音の位置 (4 分音符単位) -- こちらで MusicXML を読んだ小節の位置とのずれ (弱起など) を合わせる
      -> aligned の演奏の時刻 (npy)
      -> 元の演奏の時刻 (同じ音高の近い打鍵。aligned は区間ごとにずれているので、区間ごとにずれを求める)
      -> 位置 -> 時刻の対応 (単調になるように) から、各小節の開始時刻を読む

確かめること:
    楽譜の MIDI とこちらの小節の音が MIN_SCORE_MATCH 以上合う (反復の展開のしかたの違いなどを弾く)
    楽譜の音のうち、元の演奏の近くに同じ音高の打鍵がある割合 (pairs.match_counts) が MIN_MATCH 以上
    小節ごとにもその割合を見て、MIN_MEASURE_MATCH 未満の小節 (繰り返しを省いた所など) の前後で切る
出力は data/piano_score/periscope (形は pairs.py)。曲は楽譜ごと (同じ楽譜の演奏は同じ曲)。
"""

from __future__ import annotations

import argparse
import csv
import os
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
import pretty_midi

from piano_score.musicxml import ScoreError, _tie_stops, read_musicxml
from piano_score.score import Measure
from piano_score.tokenizer import ScoreTokenizer

from .generate import read_performance
from .pairs import PairError, make_pairs, match_counts, score_error, write_cache

ROOT = Path("Dataset/PERiScoPe/PERiScoPe-1.1")
MIN_SCORE_MATCH = 0.9  # 楽譜の MIDI の音のうち、こちらで読んだ小節にも同じ位置・音高である割合の下限
MIN_MATCH = 0.75  # 楽譜の音のうち、元の演奏の近くに同じ音高の打鍵がある割合の下限 (採譜の誤りがあるので ASAP より低め)
MIN_MEASURE_MATCH = 0.4  # 小節ごとの同じ割合の下限
WINDOW = 48  # aligned と元の演奏のずれを求める区間 (aligned の音の数)
TOLERANCE = 0.03  # aligned の音と元の演奏の音を同じとみなす時刻の差 (秒)
SPECIAL = ':"*?<>|'


def file_key(path: str) -> str:
    """ファイルのパス -> メタデータのパスと比べる形。Windows で展開すると : や " は私用領域の文字 (U+F000 + 文字コード)
    になり、é などは分解された形 (NFD) のことがあるので、元に戻して NFC にそろえる"""
    for c in SPECIAL:
        path = path.replace(chr(0xF000 + ord(c)), c)
    return unicodedata.normalize("NFC", path.replace("\\", "/"))


def index_files(root: Path) -> dict[str, str]:
    files = {}
    for base in ("data", "data_aligned"):
        for dirpath, _, names in os.walk(root / base):
            for name in names:
                full = os.path.join(dirpath, name)
                files[file_key(os.path.relpath(full, root))] = full
    return files


def midi_notes(path: str) -> tuple[pretty_midi.PrettyMIDI, list[pretty_midi.Note]]:
    midi = pretty_midi.PrettyMIDI(path)
    notes = sorted((n for i in midi.instruments for n in i.notes), key=lambda n: (n.start, n.pitch))
    return midi, notes


def score_positions(measures: list[Measure], score_midi: str) -> tuple[np.ndarray, np.ndarray, float]:
    """楽譜の MIDI の各音 (npy の番号の順) の、こちらの小節での位置 (4 分音符単位、曲の頭から) と音高と、
    こちらの小節にも同じ位置・音高の音がある割合。弱起などで全体がずれている分は合わせる"""
    midi, notes = midi_notes(score_midi)
    positions = np.array([midi.time_to_tick(n.start) / midi.resolution for n in notes])
    pitches = np.array([n.pitch for n in notes])
    ours: Counter = Counter()
    tie_stops = _tie_stops(measures)
    offset = Fraction(0)
    for mi, measure in enumerate(measures):
        for g in measure.groups:
            if g.duration.grace:
                continue
            for note in g.notes:
                if (mi, id(g), note.pitch) not in tie_stops:
                    ours[(round(float(offset + g.onset), 3), note.pitch)] += 1
        offset += measure.length
    theirs = Counter(zip(np.round(positions, 3).tolist(), pitches.tolist()))
    best_shift, best = 0.0, -1.0
    for shift in [k / 4 for k in range(-32, 33)]:
        shifted = Counter({(round(t + shift, 3), p): c for (t, p), c in theirs.items()})
        rate = sum((ours & shifted).values()) / max(sum(theirs.values()), 1)
        if rate > best:
            best_shift, best = shift, rate
    return positions + best_shift, pitches, best


def raw_times(aligned: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """aligned の演奏の各音 [n, 2] (時刻, 音高) に対応する元の演奏の打鍵の時刻 (見つからなければ nan)。
    aligned は間を詰めるなどで区間ごとに元の演奏からずれているので、WINDOW 音ずつ、ずれの最頻値を求めて合わせる"""
    by_pitch = {int(p): np.sort(raw[raw[:, 1] == p, 0]) for p in np.unique(raw[:, 1])}
    order = np.argsort(aligned[:, 0], kind="stable")
    out = np.full(len(aligned), np.nan)
    previous = None
    for start in range(0, len(order), WINDOW):
        window = order[start : start + WINDOW]
        diffs = []
        for t, p in aligned[window]:
            onsets = by_pitch.get(int(p))
            if onsets is None:
                continue
            center = t + (previous if previous is not None else 0.0)
            reach = 4.0 if previous is not None else 30.0
            near = onsets[(onsets > center - reach) & (onsets < center + reach)]
            diffs += (near - t).tolist()
        if not diffs:
            continue
        diffs = np.asarray(diffs)
        bins = np.round(diffs / 0.01).astype(np.int64)
        values, counts = np.unique(bins, return_counts=True)
        # 隣の 10ms も合わせて数えた最頻値
        smoothed = counts + np.interp(values - 1, values, counts, left=0, right=0) * np.isin(values - 1, values)
        smoothed += np.interp(values + 1, values, counts, left=0, right=0) * np.isin(values + 1, values)
        mode = values[np.argmax(smoothed)] * 0.01
        close = diffs[np.abs(diffs - mode) < TOLERANCE]
        if len(close) < max(3, 0.3 * len(window)):
            continue
        previous = float(np.median(close))
        for i in window:
            t, p = aligned[i]
            onsets = by_pitch.get(int(p))
            if onsets is None:
                continue
            j = np.searchsorted(onsets, t + previous)
            candidates = [onsets[k] for k in (j - 1, j) if 0 <= k < len(onsets)]
            nearest = min(candidates, key=lambda x: abs(x - t - previous))
            if abs(nearest - t - previous) < TOLERANCE:
                out[i] = nearest
    return out


def measure_starts(measures: list[Measure], positions: np.ndarray, times: np.ndarray) -> np.ndarray:
    """位置 -> 時刻の点 (音ごと) から、各小節の開始時刻。同じ位置は中央値にし、時刻が戻る点は除いて単調にする。
    点の外側の小節は、端の 8 点のテンポで延ばす"""
    keep = np.isfinite(times)
    positions, times = positions[keep], times[keep]
    unique = np.unique(positions)
    medians = np.array([np.median(times[positions == q]) for q in unique])
    qs, ts = [], []
    for q, t in zip(unique, medians):
        if not ts or t > ts[-1]:
            qs.append(q)
            ts.append(t)
    if len(qs) < 8:
        raise PairError("対応する音が少ない")
    qs, ts = np.asarray(qs), np.asarray(ts)
    bars = np.cumsum([0.0] + [float(m.length) for m in measures[:-1]])
    starts = np.interp(bars, qs, ts)
    head = np.polyfit(qs[:8], ts[:8], 1)[0]
    tail = np.polyfit(qs[-8:], ts[-8:], 1)[0]
    starts = np.where(bars < qs[0], ts[0] - (qs[0] - bars) * max(head, 1e-3), starts)
    starts = np.where(bars > qs[-1], ts[-1] + (bars - qs[-1]) * max(tail, 1e-3), starts)
    return starts


def _load_piece(item: tuple[str, str, str, list[tuple[str, str, str]]]) -> list[dict | str]:
    """1 つの楽譜とその演奏すべて -> 演奏ごとの組 (dict) か、使えなかった理由 (str)"""
    piece, xml, score_midi, performances = item
    tokenizer = ScoreTokenizer()
    try:
        measures = read_musicxml(xml, unfold=True, visible_only=True)
        positions, pitches, score_match = score_positions(measures, score_midi)
    except ScoreError as e:
        return [score_error(e)] * len(performances)
    except Exception as e:  # 想定していない壊れ方
        return [f"楽譜: 例外 {type(e).__name__}"] * len(performances)
    if score_match < MIN_SCORE_MATCH:
        return ["楽譜の MIDI と小節が合わない"] * len(performances)

    results: list[dict | str] = []
    for name, aligned_path, raw_path in performances:
        try:
            _, aligned_notes = midi_notes(aligned_path)
            mapping = np.load(aligned_path.removesuffix(".mid") + ".npy")
            if len(mapping) != len(positions) or mapping.max() >= len(aligned_notes):
                raise PairError("対応の数が楽譜の MIDI と違う")
            aligned = np.array([[n.start, n.pitch] for n in aligned_notes])[mapping]
            if (aligned[:, 1] != pitches).mean() > 0.01:
                raise PairError("対応の音高が楽譜と違う")
            performance = read_performance(raw_path)
            notes, pedal = performance.notes, performance.pedal
            if len(notes) < 16:
                raise PairError("演奏の音が少ない")
            # 時刻は read_performance がそろえた後 (最初の音 = 0) のもの。aligned とのずれは raw_times が求める
            times = raw_times(aligned, np.c_[notes[:, 0], notes[:, 2]])
            starts = measure_starts(measures, positions, times)
            span = starts[-1] - starts[-2] if len(starts) > 1 else 1.0
            end = starts[-1] + span * float(measures[-1].length) / max(float(measures[-2].length), 1e-3) + 0.5
            found, total = match_counts(measures, starts, notes[notes[:, 0] < end])
            match = float(found.sum() / max(total.sum(), 1))
            if match < MIN_MATCH:
                raise PairError("楽譜と演奏が合わない")
            # 音のない小節 (全休符) は、前後と同じ扱いにする
            rate = np.where(total > 0, found / np.maximum(total, 1), 1.0)
            usable = (rate >= MIN_MEASURE_MATCH).tolist()
        except PairError as e:
            results.append(str(e))
            continue
        except ScoreError as e:
            results.append(score_error(e))
            continue
        except Exception as e:  # 壊れた MIDI など
            results.append(f"演奏: 例外 {type(e).__name__}")
            continue
        errors: list[str] = []
        source = Path(name).name.split("_")[0]
        info = {"piece": piece, "source": source, "match": match}
        results += make_pairs(
            tokenizer, measures, starts, notes, pedal, end, name=name, info=info, usable=usable, errors=errors
        )
        results += errors
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out-dir", default="data/piano_score/periscope")
    parser.add_argument("--val-percent", type=float, default=3.0, help="検証に回す曲の割合 (曲名のハッシュで決める)")
    parser.add_argument("--min-cleaner", type=float, default=0.0, help="メタデータの match_cleaner の下限")
    parser.add_argument("--limit", type=int, default=0, help="動作確認用に、先頭から何曲だけ使うか (0 なら全部)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    files = index_files(root)
    pieces: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    errors: Counter = Counter()
    with open(root / "metadata.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["match_cleaner"]:
                continue  # 対応づけられなかった演奏
            if float(row["match_cleaner"]) < args.min_cleaner:
                errors["match_cleaner が低い"] += 1
                continue
            performance = file_key(row["performance"])
            paths = [files.get(k) for k in ("data/" + file_key(row["xml"]), "data_aligned/" + file_key(row["score"]),
                                            "data_aligned/" + performance, "data/" + performance)]  # fmt: skip
            if None in paths:
                errors["ファイルがない"] += 1
                continue
            xml, score_midi, aligned, raw = paths
            pieces[(xml, score_midi)].append((performance, aligned, raw))
    items = [
        (file_key(str(Path(os.path.relpath(xml, root / "data")).parent)), xml, score_midi, performances)
        for (xml, score_midi), performances in sorted(pieces.items(), key=lambda kv: -len(kv[1]))
    ]
    if args.limit:
        items = items[: args.limit]
    print(f"楽譜 {len(items)} / 演奏 {sum(len(i[3]) for i in items)}", flush=True)

    pairs: list[dict] = []
    done = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_load_piece, item) for item in items]
        for future in as_completed(futures):
            for result in future.result():
                if isinstance(result, str):
                    errors[result] += 1
                else:
                    pairs.append(result)
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(items)} 曲 / 組 {len(pairs)}", flush=True)
    sources = Counter(p["source"] for p in pairs)
    write_cache(pairs, errors, Path(args.out_dir), args.val_percent, {"sources": dict(sources)})
    print("出どころ:", dict(sources))


if __name__ == "__main__":
    main()
