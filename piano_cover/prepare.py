"""カバー学習用のキャッシュを作る (原曲の行・カバーのイベント・カバー -> 原曲の時刻の対応)。

    python -m piano_cover.prepare

入力:
    data/metadata/dataset.json                  曲ごとの原曲とカバーの動画 ID
    Dataset/original_midis_v2/merged/<原曲>.mid  tsumugi で採譜した原曲 (まだない曲は飛ばす)
    Dataset/pianos_midi/<原曲>/<カバー>.mid      Transkun で採譜したカバー
    data/alignments/<原曲>/<カバー>.pt           音声同士の DTW で求めたカバー -> 原曲の時刻の対応
    data/alignments/source_offsets.json         アラインメントの原曲側の時刻の基準 (下記)
    data/metadata/piano_to_performer.json        カバー -> チャンネル ID (事前学習と同じ番号)

出力 (--out-dir):
    source_rows.npy / sources.npz   原曲の行 (piano_cover.source) を連結したものと曲ごとの範囲
    cover_events.npy / covers.npz   カバーのイベント (先頭の無音を詰めたもの) と曲ごとの範囲・原曲の番号・分割
    align.npy                       カバーの ALIGN_STEP フレームごとの、対応する原曲のフレーム (平滑化済み, float32)
    meta.json

アラインメントのキャッシュは旧モデルが作ったもので、時刻はどちらも「そのときの MIDI の最初の音」が 0 になっている。
    カバー側: カバー MIDI (今と同じ Transkun のもの) の最初の音の onset
    原曲側  : 旧 tsumugi の原曲 MIDI (Dataset/original_midis/merged) の最初の音の onset -> source_offsets.json
新しい原曲 MIDI は音声の絶対時刻なので、原曲側に source_offsets の秒数を足して戻す。

音声の DTW は音符の単位では粗い (平均 40ms 早く、±40ms 揺れる) ので、音符の onset 同士が合うよう
窓ごとに細かく補正し (refine_alignment)、最後に原曲の拍ごとに対応を決め直す (beat_alignment)。

DTW の経路をそのまま使うとガタつく (4 秒の平滑化との差が上位 5% で 0.14 秒) ので、
--smooth-seconds の移動平均をかける。カバーのリズムは伸縮せず、この対応は cross-attention の位置にだけ使う。
"""

from __future__ import annotations

import argparse
import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from piano_ar.config import TokenizerConfig
from piano_ar.tokenizer import KIND, KIND_NOTE, ONSET, PianoTokenizer

from .source import load_source

PAIR_DATASET = Path("data/metadata/dataset.json")
PIANO_TO_PERFORMER = Path("data/metadata/piano_to_performer.json")
# アラインメントを持つ間隔 (フレーム)。10 フレーム = 0.1 秒
ALIGN_STEP = 10


def split_of(original_id: str, val_percent: float, test_percent: float) -> int:
    """原曲の動画 ID のハッシュで 0 学習 / 1 検証 / 2 テストに分ける (同じ原曲のカバーは同じ側に入る)"""
    bucket = zlib.crc32(original_id.encode()) % 10000 / 100
    if bucket < val_percent:
        return 1
    if bucket < val_percent + test_percent:
        return 2
    return 0


def cover_alignment(
    alignment_path: Path,
    raw_note_start: float,
    source_offset: float,
    shift_frames: int,
    num_frames: int,
    frame_rate: int,
    smooth: float,
) -> np.ndarray:
    """カバー (先頭を shift_frames 詰めたもの) の ALIGN_STEP フレームごとに、原曲 (音声の絶対時刻) のフレームを返す。

    raw_note_start: カバー MIDI の最初の音の onset (秒)、source_offset: 旧原曲 MIDI の最初の音の onset (秒)
    """
    import torch

    payload = torch.load(alignment_path, map_location="cpu", weights_only=False)
    target = payload["target_time_knots_seconds"].numpy().astype(np.float64)
    source = payload["source_time_knots_seconds"].numpy().astype(np.float64)
    grid = np.arange(0, num_frames + ALIGN_STEP, ALIGN_STEP)
    seconds = (grid + shift_frames) / frame_rate - raw_note_start
    mapped = (np.interp(seconds, target, source) + source_offset) * frame_rate
    width = max(1, int(round(smooth * frame_rate / ALIGN_STEP)))
    if width > 1 and len(mapped) > 1:
        padded = np.pad(mapped, (width // 2, width - 1 - width // 2), mode="edge")
        mapped = np.convolve(padded, np.ones(width) / width, mode="valid")
    return mapped.astype(np.float32)


def refine_alignment(
    align: np.ndarray,
    cover_notes: np.ndarray,
    source_rows: np.ndarray,
    window: int = 1000,
    hop: int = 500,
    max_lag: int = 25,
) -> tuple[np.ndarray, float]:
    """音声の DTW で求めた対応を、音符の onset 同士が合うように細かく補正する。

    音声の DTW を平滑化したものは、onset の単位で見ると平均 40ms 早く、カバーの中でも ±40ms ほど揺れていた。
    Local の cross-attention は原曲の音の正確な時刻を見るので、このずれがあると onset の予測がぼやけて、
    生成のリズムがばらつく (和音が 10〜40ms に割れるなど)。
    カバーの window フレームの窓ごとに、写した onset を ±max_lag フレームずらして、原曲 (ドラム以外) の
    同じ音名の onset と ±1 フレームで一致する数が最も多いずれを探し、窓の間を補間して足す。
    はっきりした山がない窓 (音が少ない・原曲と違う弾き方) は使わない。
    返り値は補正後の対応と、使えた窓の割合。
    """
    from .source import DRUM_ID, ROW_A, ROW_D, ROW_ONSET, ROW_TYPE, TYPE_NOTE

    notes = source_rows[(source_rows[:, ROW_TYPE] == TYPE_NOTE) & (source_rows[:, ROW_D] != DRUM_ID)]
    source_onsets = [np.unique(notes[notes[:, ROW_A] % 12 == pc, ROW_ONSET]).astype(np.float64) for pc in range(12)]
    cover_onsets = cover_notes[:, ONSET].astype(np.float64)
    mapped = np.interp(cover_onsets / ALIGN_STEP, np.arange(len(align)), align)
    pitch_class = cover_notes[:, 2] % 12
    lags = np.arange(-max_lag, max_lag + 1)

    centers, found, total = [], [], 0
    for start in range(0, int(cover_onsets.max(initial=0)) + 1, hop):
        selected = (cover_onsets >= start) & (cover_onsets < start + window)
        if selected.sum() < 30:
            continue
        total += 1
        hits = np.zeros(len(lags))
        for pc in range(12):
            targets = source_onsets[pc]
            x = mapped[selected & (pitch_class == pc)]
            if len(targets) < 2 or len(x) == 0:
                continue
            shifted = x[None, :] + lags[:, None]
            k = np.clip(np.searchsorted(targets, shifted), 1, len(targets) - 1)
            distance = np.minimum(np.abs(targets[k] - shifted), np.abs(targets[k - 1] - shifted))
            hits += (distance <= 1).sum(axis=1)
        rate = hits / selected.sum()
        if rate.max() > 0.15 and rate.max() > 2 * np.median(rate):
            centers.append(start + window / 2)
            found.append(lags[rate.argmax()])
    if not found:
        return align, 0.0
    found = np.asarray(found, dtype=np.float64)
    # 1 つだけ外れた窓に引っ張られないよう、隣と合わせた 3 つの中央値にする
    if len(found) >= 3:
        padded = np.pad(found, 1, mode="edge")
        found = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)
    grid = np.arange(len(align)) * ALIGN_STEP
    correction = np.interp(grid, np.asarray(centers), found)
    return (align + correction).astype(np.float32), len(centers) / max(total, 1)


def beat_alignment(
    align: np.ndarray,
    cover_notes: np.ndarray,
    source_rows: np.ndarray,
    max_shift: int = 20,
    shift_cost: float = 0.03,
    prior_cost: float = 0.002,
    max_anchor_gap: int = 60,
) -> tuple[np.ndarray, float]:
    """原曲の拍ごとに、カバーのどの時刻に当たるかを決め直す (拍の単位のアラインメント)。

    窓ごとに一定量ずらす補正 (refine_alignment) では、演奏者の揺れや 1 拍の食い違いを拍の途中の位置でつなぐので、
    区間によってカバーのリズムが原曲の拍の格子から 1/8〜1/4 拍ずれる (学習データの約 1/3 の区間で 20ms 超)。
    モデルはそのずれも学ぶので、生成でも原曲の拍に対する位相が数秒ごとに飛ぶ (拍が抜ける・増えるように聞こえる)。

    原曲の拍 (間隔が max_anchor_gap フレームより長ければ間に目印を足す) を目印にして、今の対応で写した
    カバーの時刻の ±max_shift フレームを候補にする。各候補は、その目印のまわり (前後の目印との中点まで) の
    カバーの音を候補に合わせてずらしたとき、原曲 (ドラム以外) の同じ音名の onset と ±1 フレームで一致する割合で採点する。
    隣の目印とのずらし量の差 (テンポの変化) に shift_cost、ずらし量そのものに prior_cost の罰を付けて、
    目印の列全体で最もよい組み合わせを Viterbi で選び、目印の間は直線でつなぐ。
    拍が倍テン・半テンで付いていても、目印の間隔が変わるだけなので問題ない。
    返り値は新しい対応と、音で位置がはっきり決まった目印の割合。
    """
    from .source import DRUM_ID, ROW_A, ROW_D, ROW_ONSET, ROW_TYPE, TYPE_BEAT, TYPE_NOTE

    beats = np.unique(source_rows[source_rows[:, ROW_TYPE] == TYPE_BEAT, ROW_ONSET]).astype(np.float64)
    grid = np.arange(len(align)) * ALIGN_STEP
    # 対応を単調にしてから逆向き (原曲 -> カバー) に引く
    forward = np.maximum.accumulate(align.astype(np.float64))
    inside = beats[(beats > forward[0]) & (beats < forward[-1])]
    if len(inside) < 4 or len(cover_notes) == 0:
        return align, 0.0
    anchors = [inside[0]]
    for a, b in pairwise(inside):
        pieces = max(1, int(np.ceil((b - a) / max_anchor_gap)))
        anchors.extend(a + (b - a) * np.arange(1, pieces + 1) / pieces)
    anchors = np.asarray(anchors)
    start = np.interp(anchors, forward, grid)  # 今の対応での、目印のカバーの時刻
    shifts = np.arange(-max_shift, max_shift + 1)

    # 採点: 各音を最も近い目印に割り当て、候補ごとに原曲の同じ音名の onset と一致するかを数える
    notes = source_rows[(source_rows[:, ROW_TYPE] == TYPE_NOTE) & (source_rows[:, ROW_D] != DRUM_ID)]
    onsets = cover_notes[:, ONSET].astype(np.float64)
    pitch_class = cover_notes[:, 2] % 12
    boundaries = (start[:-1] + start[1:]) / 2
    owner = np.searchsorted(boundaries, onsets)
    hits = np.zeros((len(anchors), len(shifts)))
    for pc in range(12):
        targets = np.unique(notes[notes[:, ROW_A] % 12 == pc, ROW_ONSET]).astype(np.float64)
        selected = pitch_class == pc
        if len(targets) < 2 or not selected.any():
            continue
        k = owner[selected]
        mapped = onsets[selected] + anchors[k] - start[k]
        shifted = mapped[:, None] - shifts[None, :]
        index = np.clip(np.searchsorted(targets, shifted), 1, len(targets) - 1)
        distance = np.minimum(np.abs(targets[index] - shifted), np.abs(targets[index - 1] - shifted))
        np.add.at(hits, k, (distance <= 1).astype(np.float64))
    counts = np.bincount(owner, minlength=len(anchors)).astype(np.float64)
    # 音の少ない目印は当てにならないので、点の重みを下げる
    score = hits / np.maximum(counts, 1)[:, None] * np.minimum(counts, 8)[:, None] / 8
    score -= prior_cost * np.abs(shifts)[None, :]

    # Viterbi: 目印の間隔 (カバー側) は今の対応の 0.5〜1.5 倍までにして、順番の入れ替わりや急なテンポ変化を防ぐ
    step_cost = shift_cost * np.abs(shifts[None, :] - shifts[:, None])  # [前, 今]
    total = score[0].copy()
    back = np.zeros((len(anchors), len(shifts)), dtype=np.int64)
    for i in range(1, len(anchors)):
        base = start[i] - start[i - 1]
        gap = base + shifts[None, :] - shifts[:, None]
        allowed = (gap >= 0.5 * base) & (gap <= 1.5 * base) & (gap > 0)
        candidates = np.where(allowed, total[:, None] - step_cost, -np.inf)
        back[i] = candidates.argmax(axis=0)
        total = candidates.max(axis=0) + score[i]
    chosen = np.zeros(len(anchors), dtype=np.int64)
    chosen[-1] = total.argmax()
    for i in range(len(anchors) - 1, 0, -1):
        chosen[i - 1] = back[i, chosen[i]]
    cover_times = start + shifts[chosen]

    # 目印の間は直線、最初と最後の目印の外側は端のずらし量をそのまま使う
    new_align = np.interp(grid, cover_times, anchors)
    before, after = grid < cover_times[0], grid > cover_times[-1]
    new_align[before] = align[before] + (anchors[0] - np.interp(cover_times[0], grid, align))
    new_align[after] = align[after] + (anchors[-1] - np.interp(cover_times[-1], grid, align))
    evidence = counts >= 4
    sharp = (score.max(axis=1) - np.median(score, axis=1)) > 0.15
    return new_align.astype(np.float32), float(np.mean(sharp & evidence))


def _load_song(args: tuple) -> dict | str:
    original_id, covers, source_dir, cover_dir, alignment_dir, source_offset, config, smooth, beat = args
    try:
        tokenizer = PianoTokenizer(config)
        rows, source_end = load_source(source_dir / f"{original_id}.mid", config.frame_rate)
        results = []
        for piano_id in covers:
            midi_path = cover_dir / original_id / f"{piano_id}.mid"
            alignment_path = alignment_dir / original_id / f"{piano_id}.pt"
            if not midi_path.exists() or not alignment_path.exists():
                continue
            events, end_frame = tokenizer.midi_to_events(midi_path, trim=False)
            notes = events[events[:, KIND] == KIND_NOTE]
            if len(notes) == 0:
                continue
            shift = int(events[:, ONSET].min())
            events[:, ONSET] -= shift
            end_frame -= shift
            align = cover_alignment(
                alignment_path,
                notes[:, ONSET].min() / config.frame_rate,
                source_offset,
                shift,
                end_frame,
                config.frame_rate,
                smooth,
            )
            notes = events[events[:, KIND] == KIND_NOTE]  # 先頭を詰めた時刻で取り直す
            align, refined = refine_alignment(align, notes, rows)
            if beat:
                align, refined = beat_alignment(align, notes, rows)
            results.append((piano_id, events, end_frame, align, refined))
        return {"original_id": original_id, "rows": rows, "end_frame": source_end, "covers": results}
    except Exception as e:  # 壊れた MIDI などは飛ばす
        return f"{original_id}: {e!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", default="Dataset/original_midis_v2/merged")
    parser.add_argument("--cover-dir", default="Dataset/pianos_midi")
    parser.add_argument("--alignment-dir", default="data/alignments")
    parser.add_argument("--out-dir", default="data/piano_cover")
    parser.add_argument("--smooth-seconds", type=float, default=4.0, help="アラインメントの移動平均の幅")
    parser.add_argument("--val-percent", type=float, default=5.0)
    parser.add_argument("--test-percent", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-beat-alignment", action="store_true", help="拍の単位の対応の決め直しをしない")
    args = parser.parse_args()

    config = TokenizerConfig()
    source_dir, cover_dir, alignment_dir = Path(args.source_dir), Path(args.cover_dir), Path(args.alignment_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(PAIR_DATASET.read_text(encoding="utf-8"))
    performer = json.loads(PIANO_TO_PERFORMER.read_text(encoding="utf-8"))
    source_offsets = json.loads((alignment_dir / "source_offsets.json").read_text(encoding="utf-8"))

    jobs = []
    missing_source = 0
    for entry in pairs.values():
        original_id = entry["original"]
        if not (source_dir / f"{original_id}.mid").exists() or original_id not in source_offsets:
            missing_source += 1
            continue
        jobs.append(
            (
                original_id,
                entry["pianos"],
                source_dir,
                cover_dir,
                alignment_dir,
                source_offsets[original_id],
                config,
                args.smooth_seconds,
                not args.no_beat_alignment,
            )
        )
    print(f"原曲 {len(jobs)} 曲 (原曲の MIDI がまだない曲 {missing_source})")

    source_rows, source_ids, source_ends, source_lengths = [], [], [], []
    cover_events, cover_ids, cover_ends, cover_lengths, cover_source, channels, splits = [], [], [], [], [], [], []
    aligns, align_lengths = [], []
    refined_rates: list[float] = []
    failed = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(_load_song, jobs, chunksize=4):
            if isinstance(result, str):
                failed += 1
                print(f"失敗 {result}")
                continue
            if not result["covers"]:
                continue
            source_index = len(source_ids)
            source_ids.append(result["original_id"])
            source_rows.append(result["rows"])
            source_lengths.append(len(result["rows"]))
            source_ends.append(result["end_frame"])
            split = split_of(result["original_id"], args.val_percent, args.test_percent)
            for piano_id, events, end_frame, align, refined in result["covers"]:
                refined_rates.append(refined)
                cover_ids.append(piano_id)
                cover_events.append(events)
                cover_lengths.append(len(events))
                cover_ends.append(end_frame)
                cover_source.append(source_index)
                channels.append(int(performer.get(piano_id, 0)))
                splits.append(split)
                aligns.append(align)
                align_lengths.append(len(align))

    def offsets(lengths: list[int]) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    np.save(out_dir / "source_rows.npy", np.concatenate(source_rows).astype(np.int32))
    np.savez(
        out_dir / "sources.npz",
        offsets=offsets(source_lengths),
        end_frames=np.asarray(source_ends, dtype=np.int64),
        video_ids=np.asarray(source_ids),
    )
    np.save(out_dir / "cover_events.npy", np.concatenate(cover_events).astype(np.int32))
    np.save(out_dir / "align.npy", np.concatenate(aligns).astype(np.float32))
    np.savez(
        out_dir / "covers.npz",
        offsets=offsets(cover_lengths),
        align_offsets=offsets(align_lengths),
        end_frames=np.asarray(cover_ends, dtype=np.int64),
        video_ids=np.asarray(cover_ids),
        source_index=np.asarray(cover_source, dtype=np.int64),
        channels=np.asarray(channels, dtype=np.int64),
        split=np.asarray(splits, dtype=np.int64),
    )
    meta = {
        "tokenizer": asdict(config),
        "align_step": ALIGN_STEP,
        "smooth_seconds": args.smooth_seconds,
        "beat_alignment": not args.no_beat_alignment,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    splits_array = np.asarray(splits)
    hours = sum(cover_ends) / config.frame_rate / 3600
    print(
        f"原曲 {len(source_ids)} 曲 / カバー {len(cover_ids)} 本 ({hours:.0f} 時間) / "
        f"学習 {(splits_array == 0).sum()} 検証 {(splits_array == 1).sum()} テスト {(splits_array == 2).sum()} / 失敗 {failed}"
    )
    what = "拍の目印のうち音で位置が決まったもの" if not args.no_beat_alignment else "音符の onset で補正できた窓"
    print(f"{what}の割合: 平均 {np.mean(refined_rates):.0%}")
    print(f"出力: {out_dir}")


if __name__ == "__main__":
    main()
