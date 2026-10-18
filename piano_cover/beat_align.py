"""カバーの拍と原曲の拍を対応させる (拍単位のアラインメント)。

    python -m piano_cover.beat_align --cache-dir data/piano_cover_v5 [--limit N]

カバーの拍 (カバーの音声、音声がなければ MIDI の合成音の SheetSage2-Refined の拍) を、音声の DTW で求めた時刻の対応
(piano_cover.prepare の align) で原曲に写し、原曲の拍の番号に読む。拍の追跡はカバーと原曲で階層が違うことがある
(local/probe_beats.py の (c): カバー 1 拍が原曲 2 拍 2.2%、1/2 拍 1.8%、曲の途中で変わる拍が 10% 以上のカバー 16.8%)
ので、区間ごとにカバーの拍を原曲の階層にそろえる (2 拍分なら間に拍を足し、1/2 拍分なら 1 つおきに間引く)。
そろえたあとの拍が原曲の拍の上に乗れば (±0.3 拍)、その原曲の拍に対応させる。小節の頭は、対応した拍では原曲のもの、
対応しない拍 (カバーだけの前奏や間奏など) ではカバー自身のものを使う。

生成では、カバーの拍 = 原曲の拍 (対応は恒等) として弾き、原曲の拍の時刻に置いて鳴らす。学習でカバーの拍の階層と
小節の区切りを原曲にそろえておくのは、生成と同じ形で学習するため。

CLI は、カバーの音符を原曲の時刻に写したときに原曲の同じ音名の onset と合う割合を、拍の対応と音声の DTW で比べる
(同じ音符の集合で)。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ALIGN_STEP = 10  # piano_cover.prepare.ALIGN_STEP
LEVELS = np.array([0.5, 1.0, 2.0])  # カバーの 1 拍が原曲の何拍か
LEVEL_TOLERANCE = np.log(1.15)
ON_BEAT = 0.3  # そろえた拍が原曲の拍の上に乗るとみなす範囲 (拍)


def beat_index(beats: np.ndarray, times: np.ndarray) -> np.ndarray:
    """時刻を拍の番号 (小数) にする。最初と最後の拍の外は、端の拍の間隔で延ばす"""
    if len(beats) < 2:
        return np.full(len(times), np.nan)
    index = np.arange(len(beats), dtype=np.float64)
    first, last = beats[1] - beats[0], beats[-1] - beats[-2]
    knots_t = np.concatenate([[beats[0] - first * 1e4], beats, [beats[-1] + last * 1e4]])
    knots_i = np.concatenate([[-1e4], index, [index[-1] + 1e4]])
    return np.interp(times, knots_t, knots_i)


def _majority(values: np.ndarray, width: int) -> np.ndarray:
    """values (小さい整数) の前後 width//2 の多数決"""
    out = values.copy()
    half = width // 2
    for i in range(len(values)):
        window = values[max(0, i - half) : i + half + 1]
        counts = np.bincount(window, minlength=len(LEVELS) + 1)
        out[i] = int(np.argmax(counts))
    return out


@dataclass
class BeatAlignment:
    """カバーの拍 (原曲の階層にそろえたもの) と、それぞれが対応する原曲の拍"""

    times: np.ndarray  # [B] カバーの拍の時刻 (秒、カバーの MIDI と同じ基準)
    downbeat: np.ndarray  # [B] 小節の頭か
    source_beat: np.ndarray  # [B] 原曲の拍の番号 (小数。写した位置そのもの)
    ok: np.ndarray  # [B] 原曲の拍に対応したか (このときは round(source_beat) が対応する原曲の拍)
    level: np.ndarray  # [B] その拍の区間の階層 (LEVELS の番号、len(LEVELS) は判定できない)


def align_beats(
    cover_beats: np.ndarray,
    cover_downbeats: np.ndarray,
    source_beats: np.ndarray,
    source_downbeats: np.ndarray,
    to_source,
    usable,
    smooth_beats: int = 8,
) -> BeatAlignment:
    """to_source(秒の配列) はカバーの時刻を原曲の時刻 (秒) に写す関数、usable(秒の配列) はその時刻の対応が使えるか"""
    cover_beats = np.asarray(cover_beats, dtype=np.float64)
    n = len(cover_beats)
    unknown = len(LEVELS)
    if n < 4 or len(source_beats) < 4:
        empty = np.zeros(n, dtype=bool)
        return BeatAlignment(cover_beats, empty, np.full(n, np.nan), empty, np.full(n, unknown))

    # 1. カバーの拍を原曲の拍の番号に写し、拍の間ごとの比 (原曲の何拍か) を前後の中央値でならして階層を決める
    mapped = beat_index(source_beats, to_source(cover_beats))
    step = np.diff(mapped)
    half = smooth_beats // 2
    local = np.array([np.median(step[max(0, k - half) : k + half + 1]) for k in range(n - 1)])
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.abs(np.log(np.maximum(local, 1e-6))[:, None] - np.log(LEVELS)[None])
    level = np.where(distance.min(1) <= LEVEL_TOLERANCE, distance.argmin(1), unknown)
    level[~(usable(cover_beats[:-1]) & usable(cover_beats[1:]))] = unknown
    level = _majority(level, smooth_beats + 1)

    # 2. 区間ごとに原曲の階層にそろえる
    times, levels, keep_downbeat = [], [], []
    k = 0
    while k < n - 1:
        if level[k] == 0:  # カバーの 1 拍 = 原曲の 1/2 拍: 1 つおきに間引く (原曲の拍に近い方の偶奇を残す)
            end = k
            while end < n - 1 and level[end] == 0:
                end += 1
            run = np.arange(k, end)
            offsets = [
                np.mean(np.abs(mapped[run[p::2]] - np.round(mapped[run[p::2]]))) if len(run[p::2]) else np.inf
                for p in (0, 1)
            ]
            for i in run[int(np.argmin(offsets)) :: 2]:
                times.append(cover_beats[i])
                levels.append(0)
                keep_downbeat.append(True)
            k = end
            continue
        times.append(cover_beats[k])
        levels.append(level[k])
        keep_downbeat.append(True)
        if level[k] == 2:  # カバーの 1 拍 = 原曲の 2 拍: 間に拍を足す
            times.append((cover_beats[k] + cover_beats[k + 1]) / 2)
            levels.append(2)
            keep_downbeat.append(False)
        k += 1
    times.append(cover_beats[-1])
    levels.append(level[-1])
    keep_downbeat.append(True)
    times = np.asarray(times)
    levels = np.asarray(levels)

    # 3. そろえた拍を原曲の拍に対応させる (原曲の拍の上に乗り、対応する拍の番号が増えていく所だけ)
    source_beat = beat_index(source_beats, to_source(times))
    nearest = np.round(source_beat)
    ok = (levels != unknown) & usable(times) & (np.abs(source_beat - nearest) < ON_BEAT)
    last = -np.inf
    for i in np.flatnonzero(ok):
        if nearest[i] <= last:
            ok[i] = False
        else:
            last = nearest[i]

    # 4. 小節の頭: 対応した拍は原曲のもの、対応しない拍はカバー自身のもの (足した拍は頭にしない)
    source_down = set(np.round(beat_index(source_beats, np.asarray(source_downbeats))).astype(int).tolist())
    cover_down = np.asarray(cover_downbeats, dtype=np.float64)
    if len(cover_down):
        gap = np.abs(times[:, None] - cover_down[None]).min(1)
        own = (gap < 0.07) & np.asarray(keep_downbeat)
    else:
        own = np.zeros(len(times), dtype=bool)
    downbeat = np.where(ok, np.isin(nearest, list(source_down)), own)
    return BeatAlignment(times, downbeat, source_beat, ok, levels)


def beat_mapping(alignment: BeatAlignment, source_beats: np.ndarray, max_gap: int = 2):
    """拍の対応から、カバーの時刻 -> 原曲の時刻の関数を作る (対応した拍の間を線形に結ぶ)。
    対応した拍の間に max_gap より多く対応しない拍がある所と、両端の外は NaN"""
    index = np.flatnonzero(alignment.ok)
    cover_t = alignment.times[index]
    source_t = source_beats[np.round(alignment.source_beat[index]).astype(int).clip(0, len(source_beats) - 1)]

    def mapping(times: np.ndarray) -> np.ndarray:
        if len(index) < 2:
            return np.full(len(times), np.nan)
        out = np.interp(times, cover_t, source_t)
        position = np.searchsorted(cover_t, times)
        inside = (position > 0) & (position < len(cover_t))
        wide = np.zeros(len(times), dtype=bool)
        if len(index) >= 2:
            span = np.diff(index)
            wide[inside] = span[position[inside] - 1] > max_gap + 1
        out[~inside | wide] = np.nan
        return out

    return mapping


def _onset_hits(mapped: np.ndarray, pitch: np.ndarray, source_onsets: np.ndarray, source_pc: np.ndarray, tol: float):
    order = np.argsort(source_onsets)
    onsets, pcs = source_onsets[order], source_pc[order]
    lo = np.searchsorted(onsets, mapped - tol)
    hi = np.searchsorted(onsets, mapped + tol, side="right")
    return np.array([(pcs[a:b] == p % 12).any() for a, b, p in zip(lo, hi, pitch)], dtype=bool)


def main() -> None:
    from piano_ar.config import TokenizerConfig
    from piano_ar.tokenizer import KIND, KIND_NOTE, ONSET, PITCH, PianoTokenizer

    from .data import CoverCache
    from .source import DRUM_ID, ROW_A, ROW_D, ROW_ONSET, tsumugi_notes

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default="data/piano_cover_v5")
    parser.add_argument(
        "--cover-sheetsage", default="data/sheetsage/cover", help="カバーの音声の注釈 (<原曲>/<カバー>.json)"
    )
    parser.add_argument("--midi-sheetsage", default="data/sheetsage/pretraining", help="音声のないカバーの合成音の注釈")
    parser.add_argument("--source-sheetsage", default="data/sheetsage/original")
    parser.add_argument("--cover-midi", default="Dataset/pianos_midi")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.05, help="onset が合うとみなす範囲 (秒)")
    parser.add_argument("--out", default="outputs/beat_align")
    args = parser.parse_args()

    cache = CoverCache(args.cache_dir)
    fr = cache.tokenizer_config.frame_rate
    tok = PianoTokenizer(TokenizerConfig())
    order = np.random.default_rng(0).permutation(len(cache.video_ids))
    rows_out, level_counts = [], np.zeros(len(LEVELS) + 1)
    for cover in order:
        original_id = str(cache.source_ids[cache.source_index[cover]])
        piano_id = str(cache.video_ids[cover])
        annotation = Path(args.cover_sheetsage) / original_id / f"{piano_id}.json"
        kind = "audio"
        if not annotation.exists():
            annotation, kind = Path(args.midi_sheetsage) / f"{piano_id}.json", "render"
        source_annotation = Path(args.source_sheetsage) / f"{original_id}.json"
        midi = Path(args.cover_midi) / original_id / f"{piano_id}.mid"
        if not annotation.exists() or not source_annotation.exists() or not midi.exists():
            continue
        cover_ann = json.loads(annotation.read_text(encoding="utf-8"))
        source_ann = json.loads(source_annotation.read_text(encoding="utf-8"))
        source_beats = np.asarray(source_ann["beats"])
        # キャッシュのカバーは先頭の無音を詰めてある (shift フレーム)。注釈と MIDI は詰める前の時刻
        events, _ = tok.midi_to_events(midi, trim=False)
        shift = int(events[:, ONSET].min())
        align = cache.cover_align(cover).astype(np.float64)
        ok_steps = cache.cover_align_ok(cover)
        ok_steps = np.ones(len(align), bool) if ok_steps is None else ok_steps.astype(bool)

        def to_source(seconds: np.ndarray) -> np.ndarray:
            return np.interp((seconds * fr - shift) / ALIGN_STEP, np.arange(len(align)), align) / fr

        def usable(seconds: np.ndarray) -> np.ndarray:
            step = (seconds * fr - shift) / ALIGN_STEP
            inside = (step >= 0) & (step <= len(align) - 1)
            return inside & ok_steps[np.clip(np.round(step).astype(int), 0, len(align) - 1)]

        result = align_beats(
            np.asarray(cover_ann["beats"]), np.asarray(cover_ann["downbeats"]), source_beats,
            np.asarray(source_ann["downbeats"]), to_source, usable,
        )  # fmt: skip
        level_counts += np.bincount(result.level, minlength=len(LEVELS) + 1)
        by_beat = beat_mapping(result, source_beats)

        notes = events[events[:, KIND] == KIND_NOTE]
        t = notes[:, ONSET] / fr
        beat_t = by_beat(t)
        both = ~np.isnan(beat_t) & usable(t)
        rows = cache.source_rows(int(cache.source_index[cover]))
        reference = tsumugi_notes(rows)
        reference = reference[reference[:, ROW_D] != DRUM_ID]
        ref_t, ref_pc = reference[:, ROW_ONSET] / fr, reference[:, ROW_A] % 12
        pitch = notes[both, PITCH]
        hit_dtw = _onset_hits(to_source(t[both]), pitch, ref_t, ref_pc, args.tolerance)
        hit_beat = _onset_hits(beat_t[both], pitch, ref_t, ref_pc, args.tolerance)
        # 拍の対応から読んだ局所のテンポ比 (原曲の拍の長さ / カバーの拍の長さ) の揺れ
        idx = np.flatnonzero(result.ok)
        rows_out.append(
            {
                "cover": piano_id,
                "kind": kind,
                "beats_ok": float(result.ok.mean()),
                "covered": float(both.mean()),
                "hit_dtw": float(hit_dtw.mean()) if len(pitch) else np.nan,
                "hit_beat": float(hit_beat.mean()) if len(pitch) else np.nan,
                "downbeats": int(result.downbeat.sum()),
                "beats": len(result.times),
                "consecutive": int(np.sum(np.diff(idx) == 1)) if len(idx) > 1 else 0,
            }
        )
        if args.limit and len(rows_out) >= args.limit:
            break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# 拍単位のアラインメント ({len(rows_out)} カバー)", ""]
    for kind in ("audio", "render"):
        sub = [r for r in rows_out if r["kind"] == kind]
        if not sub:
            continue
        lines.append(f"## カバーの拍: {kind} ({len(sub)} 本)")
        for key, label in (
            ("beats_ok", "原曲の拍に対応した拍の割合"),
            ("covered", "拍の対応で写せたカバーの音の割合"),
            (
                "hit_dtw",
                f"音声の DTW で写した onset が原曲の同じ音名の onset と ±{args.tolerance * 1000:.0f}ms で合う割合",
            ),
            ("hit_beat", "同じ音を拍の対応で写したとき"),
        ):
            values = np.array([r[key] for r in sub], dtype=np.float64)
            lines.append(
                f"- {label}: 平均 {np.nanmean(values):.3f}、中央値 {np.nanmedian(values):.3f}、"
                f"10% {np.nanpercentile(values, 10):.3f}"
            )
        diff = np.array([r["hit_beat"] - r["hit_dtw"] for r in sub])
        lines.append(
            f"- 拍の対応 − DTW の差: 平均 {np.nanmean(diff):+.3f}、拍の対応の方が良いカバー {np.nanmean(diff > 0):.1%}"
        )
        lines.append("")
    total = level_counts.sum()
    lines.append(
        "拍の間の階層 (カバーの 1 拍が原曲の何拍か): "
        + ", ".join(
            f"{name} {count / total:.1%}" for name, count in zip(("1/2", "1", "2", "判定できない"), level_counts)
        )
    )
    report = "\n".join(lines)
    print(report)
    (out / "report.md").write_text(report + "\n", encoding="utf-8")
    (out / "covers.json").write_text(json.dumps(rows_out, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
