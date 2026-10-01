"""カバー学習用のキャッシュを作る (原曲の行・カバーのイベント・カバー -> 原曲の時刻の対応)。

    python -m piano_cover.prepare

入力:
    data/metadata/dataset.json                  曲ごとの原曲とカバーの動画 ID
    Dataset/original_midis_v2/merged/<原曲>.mid  tsumugi で採譜した原曲 (まだない曲は飛ばす)
    Dataset/pianos_midi/<原曲>/<カバー>.mid      Transkun で採譜したカバー
    data/alignments/<原曲>/<カバー>.pt           音声同士の DTW で求めたカバー -> 原曲の時刻の対応
    data/alignments/source_offsets.json         アラインメントの原曲側の時刻の基準 (下記)
    data/metadata/piano_to_performer.json        カバー -> チャンネル ID (事前学習と同じ番号)
    data/metadata/dataset_mined.json            事前学習のカバーから足した組 (local/pair_review.py、あれば)

出力 (--out-dir):
    source_rows.npy / sources.npz   原曲の行 (piano_cover.source) を連結したものと曲ごとの範囲
    cover_events.npy / covers.npz   カバーのイベント (先頭の無音を詰めたもの) と曲ごとの範囲・原曲の番号・分割
    align.npy                       カバーの ALIGN_STEP フレームごとの、対応する原曲のフレーム (平滑化済み, float32)
    meta.json

アラインメントのキャッシュは旧モデルが作ったもので、時刻はどちらも「そのときの MIDI の最初の音」が 0 になっている。
    カバー側: カバー MIDI (今と同じ Transkun のもの) の最初の音の onset
    原曲側  : 旧 tsumugi の原曲 MIDI (Dataset/original_midis/merged) の最初の音の onset -> source_offsets.json
新しい原曲 MIDI は音声の絶対時刻なので、原曲側に source_offsets の秒数を足して戻す。
あとから作った対応 (version 3、local/align_mined.py) は原曲側が最初から音声の絶対時刻なので戻さない。

dataset_mined.json の組のカバーは事前学習に入っているので、原曲の分け方に関係なく学習側に入れる
(検証・テスト側の原曲の曲なら入れない)。チャンネルは事前学習の一覧の演奏者から引く。
原曲と長さが大きく違うカバー (カバー / 原曲 が --min-length-ratio〜--max-length-ratio の外) は入れない。
--exclude にカバーの動画 ID の一覧 (JSON の配列) を渡すと、そのカバーは入れない (組の品質を確かめて弾いたものなど)。
今の組では、範囲外のカバーはメロディの一致率 (下記) の中央値が 0.3 前後しかなく (範囲内は 0.69)、
TV サイズ・ショート版・メドレーなどで原曲の一部しか対応しない。
カバーごとに、原曲のメロディの音のうち、対応する時刻 (±50ms) に同じ音名のカバーの音がある割合
(melody_match) を測って保存する (アラインメントの失敗や別アレンジの組を見つけるため)。

カバーの ALIGN_STEP ごとに、原曲に沿っているか (align_ok) も保存する (sync_ok)。学習ではそうでない所の損失を取らない。
    - 前後 1 秒の対応の傾き (原曲が進む速さ) が曲全体の中央値の 0.75〜1.33 倍の外: カバーだけのイントロや間奏を
      原曲の一点に押し込んだ所など、対応が崩れている所。曲の冒頭 4 秒では 18% (曲の途中は 3〜6%)
    - 前後 2 秒の原曲のメロディの音 (3 音以上) のうち、カバーが同じ音名を ±50ms で弾いている割合が 0.2 未満:
      原曲を離れて弾いている所。
学習データは全体で 13%、冒頭 4 秒では 23% が外れる。原曲を無視した冒頭を学ぶと、生成の冒頭で原曲を無視して
弾き出す (しばらくして急に原曲に沿い出す)。

音声の DTW は音符の単位では粗い (平均 40ms 早く、±40ms 揺れる) ので、最後に音符の onset 同士が合うよう
窓ごとに細かく補正する (refine_alignment で 10 秒ごと、refine_fine で原曲のメロディとベースに 3 秒ごと)。

DTW の経路をそのまま使うとガタつく (4 秒の平滑化との差が上位 5% で 0.14 秒) ので、
--smooth-seconds の移動平均をかける。カバーのリズムは伸縮せず、この対応は cross-attention の位置にだけ使う。
移動平均の端は、端を中心に点対称に折り返して埋める (直線の傾きを保つ)。端の値で埋めると曲の冒頭と終わりの
2 秒ほどで傾きが寝て、冒頭 5 秒の 2 秒ごとのテンポ比が 0.9-1.1 から外れる割合が 37% になっていた (折り返すと 26%)。
"""

from __future__ import annotations

import argparse
import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from piano_ar.config import TokenizerConfig
from piano_ar.tokenizer import KIND, KIND_NOTE, ONSET, PianoTokenizer

from .source import load_source

PAIR_DATASET = Path("data/metadata/dataset.json")
MINED_DATASET = Path("data/metadata/dataset_mined.json")
PIANO_TO_PERFORMER = Path("data/metadata/piano_to_performer.json")
PRETRAINING_MANIFEST = Path("data/metadata/pretraining_manifest.csv")
CHANNEL_INDEX = Path("data/metadata/channel_index.json")
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

    raw_note_start: カバー MIDI の最初の音の onset (秒)、source_offset: 旧原曲 MIDI の最初の音の onset (秒)。
    version 3 の対応は原曲側が音声の絶対時刻なので source_offset を使わない (None でもよい)
    """
    import torch

    payload = torch.load(alignment_path, map_location="cpu", weights_only=False)
    if payload.get("version", 0) >= 3:
        source_offset = 0.0
    elif source_offset is None:
        raise ValueError(f"{alignment_path} は旧原曲 MIDI の時刻だが source_offsets.json に原曲がない")
    target = payload["target_time_knots_seconds"].numpy().astype(np.float64)
    source = payload["source_time_knots_seconds"].numpy().astype(np.float64)
    grid = np.arange(0, num_frames + ALIGN_STEP, ALIGN_STEP)
    seconds = (grid + shift_frames) / frame_rate - raw_note_start
    mapped = (np.interp(seconds, target, source) + source_offset) * frame_rate
    width = max(1, int(round(smooth * frame_rate / ALIGN_STEP)))
    if width > 1 and len(mapped) > 1:
        padded = np.pad(mapped, (width // 2, width - 1 - width // 2), mode="reflect", reflect_type="odd")
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


# 細かい補正 (refine_fine) で合わせる原曲の楽器。全部の音だと密すぎて (1 秒に 16 個ほど) 偶然の一致が多い
ANCHOR_INSTRUMENTS = ("melody", "vocal_harmony", "acoustic_bass", "electric_bass", "slap_bass", "synth_bass")


def anchor_rows(source_rows: np.ndarray) -> np.ndarray:
    """refine_fine で合わせる原曲の音 (メロディとベース)"""
    from .source import INSTRUMENT_ID, ROW_D, ROW_TYPE, TYPE_NOTE

    ids = [INSTRUMENT_ID[name] for name in ANCHOR_INSTRUMENTS]
    return source_rows[(source_rows[:, ROW_TYPE] == TYPE_NOTE) & np.isin(source_rows[:, ROW_D], ids)]


def refine_fine(
    align: np.ndarray,
    cover_notes: np.ndarray,
    anchors: np.ndarray,
    window: int = 300,
    hop: int = 50,
    max_lag: int = 10,
    tolerance: int = 2,
    min_hits: int = 3,
) -> tuple[np.ndarray, float]:
    """refine_alignment の後に、短い窓 (window フレーム、hop ずつ) で対応を細かく補正する。

    refine_alignment (10 秒の窓) と 4 秒の移動平均では、2 秒単位で見ると対応が 20〜30ms ほど揺れていた
    (演奏のためや走りが均されている)。メロディの音を半分に分け、片方とベースで補正を決めて残りの半分で測ると、
    写したカバーの音と同じ音名の原曲のメロディが ±30ms に入る割合は 0.543 -> 0.619、±10ms は 0.278 -> 0.321
    (窓 4 秒・6 秒、一致 ±10ms などより、この設定が一番よかった)。

    窓ごとに、写したカバーの onset を ±max_lag フレームずらして、原曲のメロディとベース (anchors) の同じ音名の
    onset と ±tolerance フレームで一致する数が最も多いずれを探す。はっきりした山 (min_hits 以上で、ずらしの中央値の
    2 倍以上) のない窓は使わず、1 つだけ外れた窓に引っ張られないよう前後 2 つずつと合わせた 5 つの中央値にしてから
    窓の間を補間して足す。返り値は補正後の対応と、使えた窓の割合。
    """
    from .source import ROW_A, ROW_ONSET

    if len(anchors) == 0 or len(cover_notes) == 0:
        return align, 0.0
    targets = [np.unique(anchors[anchors[:, ROW_A] % 12 == pc, ROW_ONSET]).astype(np.float64) for pc in range(12)]
    onsets = cover_notes[:, ONSET].astype(np.float64)
    mapped = np.interp(onsets / ALIGN_STEP, np.arange(len(align)), align)
    pitch_class = cover_notes[:, 2] % 12
    lags = np.arange(-max_lag, max_lag + 1)
    centers, found, total = [], [], 0
    for start in range(0, int(onsets.max()) + 1, hop):
        selected = (onsets >= start) & (onsets < start + window)
        if selected.sum() < 8:
            continue
        total += 1
        hits = np.zeros(len(lags))
        for pc in range(12):
            t = targets[pc]
            x = mapped[selected & (pitch_class == pc)]
            if len(t) < 2 or len(x) == 0:
                continue
            shifted = x[None, :] + lags[:, None]
            k = np.clip(np.searchsorted(t, shifted), 1, len(t) - 1)
            distance = np.minimum(np.abs(t[k] - shifted), np.abs(t[k - 1] - shifted))
            hits += (distance <= tolerance).sum(axis=1)
        if hits.max() >= min_hits and hits.max() >= 2 * np.median(hits):
            centers.append(start + window / 2)
            found.append(lags[hits.argmax()])
    if not found:
        return align, 0.0
    found = np.asarray(found, dtype=np.float64)
    if len(found) >= 5:
        padded = np.pad(found, 2, mode="edge")
        found = np.median(np.stack([padded[i : i + len(found)] for i in range(5)]), axis=0)
    grid = np.arange(len(align)) * ALIGN_STEP
    correction = np.interp(grid, np.asarray(centers), found)
    return (align + correction).astype(np.float32), len(centers) / max(total, 1)


def melody_match(align: np.ndarray, cover_notes: np.ndarray, source_rows: np.ndarray, tolerance: int = 5) -> float:
    """原曲のメロディの音のうち、対応する時刻の ±tolerance フレームに同じ音名のカバーの音がある割合
    (カバーが弾いている範囲のメロディだけで数える)"""
    from .source import INSTRUMENT_ID, ROW_A, ROW_D, ROW_ONSET, ROW_TYPE, TYPE_NOTE

    melody = source_rows[(source_rows[:, ROW_TYPE] == TYPE_NOTE) & (source_rows[:, ROW_D] == INSTRUMENT_ID["melody"])]
    mapped = np.interp(cover_notes[:, ONSET].astype(np.float64) / ALIGN_STEP, np.arange(len(align)), align)
    melody = melody[(melody[:, ROW_ONSET] >= mapped.min()) & (melody[:, ROW_ONSET] <= mapped.max())]
    if len(melody) == 0:
        return float("nan")
    order = np.argsort(mapped)
    mapped, pitch_class = mapped[order], cover_notes[order, 2] % 12
    lo = np.searchsorted(mapped, melody[:, ROW_ONSET] - tolerance)
    hi = np.searchsorted(mapped, melody[:, ROW_ONSET] + tolerance, side="right")
    hits = [(pitch_class[a:b] == p % 12).any() for a, b, p in zip(lo, hi, melody[:, ROW_A])]
    return float(np.mean(hits))


def sync_ok(
    align: np.ndarray,
    cover_notes: np.ndarray,
    source_rows: np.ndarray,
    slope_range: tuple[float, float] = (0.75, 1.33),
    min_melody: float = 0.2,
    tolerance: int = 5,
) -> np.ndarray:
    """カバーの ALIGN_STEP ごとに、原曲に沿っているか (uint8、1 = 沿っている) を返す (上の説明)"""
    from .source import INSTRUMENT_ID, ROW_A, ROW_D, ROW_ONSET, ROW_TYPE, TYPE_NOTE

    n = len(align)
    align = align.astype(np.float64)
    index = np.arange(n)
    # 前後 1 秒の傾き (曲全体の中央値との比)
    half = max(1, 100 // ALIGN_STEP)
    lo, hi = np.clip(index - half, 0, n - 1), np.clip(index + half, 0, n - 1)
    slope = (align[hi] - align[lo]) / np.maximum((hi - lo) * ALIGN_STEP, 1)
    median = np.median(slope)
    bad = (slope < slope_range[0] * median) | (slope > slope_range[1] * median) if median > 0 else np.zeros(n, bool)

    # 前後 2 秒のメロディの一致
    melody = source_rows[(source_rows[:, ROW_TYPE] == TYPE_NOTE) & (source_rows[:, ROW_D] == INSTRUMENT_ID["melody"])]
    melody = melody[(melody[:, ROW_ONSET] >= align[0]) & (melody[:, ROW_ONSET] <= align[-1])]
    if len(melody) and len(cover_notes):
        mapped = np.interp(cover_notes[:, ONSET].astype(np.float64) / ALIGN_STEP, index, align)
        order = np.argsort(mapped)
        mapped, pitch_class = mapped[order], cover_notes[order, 2] % 12
        start = np.searchsorted(mapped, melody[:, ROW_ONSET] - tolerance)
        end = np.searchsorted(mapped, melody[:, ROW_ONSET] + tolerance, side="right")
        hit = np.array([(pitch_class[a:b] == p % 12).any() for a, b, p in zip(start, end, melody[:, ROW_A])], float)
        # メロディの音を、カバーの時刻 (対応の逆) に置いて数える。対応が単調でない所があっても近くに落ちればよい
        position = np.clip(
            np.interp(melody[:, ROW_ONSET], np.maximum.accumulate(align), index).astype(np.int64), 0, n - 1
        )
        kernel = np.ones(2 * max(1, 200 // ALIGN_STEP) + 1)
        count = np.convolve(np.bincount(position, minlength=n).astype(np.float64), kernel, mode="same")
        hits = np.convolve(np.bincount(position, weights=hit, minlength=n), kernel, mode="same")
        bad |= (count >= 3) & (hits < min_melody * count)
    return (~bad).astype(np.uint8)


def _load_song(args: tuple) -> dict | str:
    original_id, covers, source_dir, cover_dir, alignment_dir, source_offset, config, smooth = args
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
            note_events = events[events[:, KIND] == KIND_NOTE]
            align, refined = refine_alignment(align, note_events, rows)
            align, _ = refine_fine(align, note_events, anchor_rows(rows))
            match = melody_match(align, note_events, rows)
            results.append((piano_id, events, end_frame, align, refined, match, sync_ok(align, note_events, rows)))
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
    parser.add_argument("--min-length-ratio", type=float, default=0.75, help="カバー / 原曲 の長さの比の下限")
    parser.add_argument("--max-length-ratio", type=float, default=1.33, help="カバー / 原曲 の長さの比の上限")
    parser.add_argument("--no-mined", action="store_true", help="dataset_mined.json の組を使わない")
    parser.add_argument("--exclude", default=None, help="入れないカバーの動画 ID の一覧 (JSON の配列)")
    args = parser.parse_args()

    config = TokenizerConfig()
    source_dir, cover_dir, alignment_dir = Path(args.source_dir), Path(args.cover_dir), Path(args.alignment_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(PAIR_DATASET.read_text(encoding="utf-8"))
    performer = json.loads(PIANO_TO_PERFORMER.read_text(encoding="utf-8"))
    source_offsets = json.loads((alignment_dir / "source_offsets.json").read_text(encoding="utf-8"))

    # 原曲ごとのカバー。足した組 (mined) は事前学習の一覧の演奏者 (チャンネル ID) から番号を引く
    covers_of: dict[str, list[str]] = {}
    for entry in pairs.values():
        covers_of.setdefault(entry["original"], []).extend(entry["pianos"])
    mined: set[str] = set()
    if not args.no_mined and MINED_DATASET.exists():
        import csv

        channel_index = json.loads(CHANNEL_INDEX.read_text(encoding="utf-8"))
        channel_of = {
            row["id"]: row["performer"] for row in csv.DictReader(PRETRAINING_MANIFEST.open(encoding="utf-8-sig"))
        }
        for entry in json.loads(MINED_DATASET.read_text(encoding="utf-8")).values():
            new = [p for p in entry["pianos"] if p not in covers_of.get(entry["original"], [])]
            covers_of.setdefault(entry["original"], []).extend(new)
            mined.update(new)
            for piano_id in new:
                if channel_of.get(piano_id) in channel_index:
                    performer[piano_id] = channel_index[channel_of[piano_id]]
        print(f"足した組のカバー {len(mined)} 本")

    if args.exclude:
        excluded = set(json.loads(Path(args.exclude).read_text(encoding="utf-8")))
        before = sum(len(covers) for covers in covers_of.values())
        covers_of = {original: [p for p in covers if p not in excluded] for original, covers in covers_of.items()}
        print(f"--exclude で外したカバー {before - sum(len(covers) for covers in covers_of.values())} 本")

    jobs = []
    missing_source = 0
    for original_id, covers in covers_of.items():
        if not (source_dir / f"{original_id}.mid").exists():
            missing_source += 1
            continue
        jobs.append(
            (
                original_id,
                covers,
                source_dir,
                cover_dir,
                alignment_dir,
                source_offsets.get(original_id),
                config,
                args.smooth_seconds,
            )
        )
    print(f"原曲 {len(jobs)} 曲 (原曲の MIDI がまだない曲 {missing_source})")

    source_rows, source_ids, source_ends, source_lengths = [], [], [], []
    cover_events, cover_ids, cover_ends, cover_lengths, cover_source, channels, splits = [], [], [], [], [], [], []
    aligns, align_lengths = [], []
    refined_rates: list[float] = []
    matches, is_mined, align_ok = [], [], []
    dropped = {"length": [0, 0], "mined_eval": 0}  # 長さで外した本数 [前の組, 足した組]、検証・テスト側の曲の足した組
    failed = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(_load_song, jobs, chunksize=4):
            if isinstance(result, str):
                failed += 1
                print(f"失敗 {result}")
                continue
            split = split_of(result["original_id"], args.val_percent, args.test_percent)
            kept = []
            for cover in result["covers"]:
                piano_id, end_frame = cover[0], cover[2]
                if piano_id in mined and split != 0:
                    dropped["mined_eval"] += 1
                    continue
                ratio = end_frame / max(result["end_frame"], 1)
                if not args.min_length_ratio <= ratio <= args.max_length_ratio:
                    dropped["length"][piano_id in mined] += 1
                    continue
                kept.append(cover)
            result["covers"] = kept
            if not result["covers"]:
                continue
            source_index = len(source_ids)
            source_ids.append(result["original_id"])
            source_rows.append(result["rows"])
            source_lengths.append(len(result["rows"]))
            source_ends.append(result["end_frame"])
            for piano_id, events, end_frame, align, refined, match, ok in result["covers"]:
                align_ok.append(ok)
                refined_rates.append(refined)
                matches.append(match)
                is_mined.append(piano_id in mined)
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
    # align と同じ並び・同じ範囲 (align_offsets) で、原曲に沿っているか
    np.save(out_dir / "align_ok.npy", np.concatenate(align_ok).astype(np.uint8))
    np.savez(
        out_dir / "covers.npz",
        offsets=offsets(cover_lengths),
        align_offsets=offsets(align_lengths),
        end_frames=np.asarray(cover_ends, dtype=np.int64),
        video_ids=np.asarray(cover_ids),
        source_index=np.asarray(cover_source, dtype=np.int64),
        channels=np.asarray(channels, dtype=np.int64),
        split=np.asarray(splits, dtype=np.int64),
        melody_match=np.asarray(matches, dtype=np.float32),
        mined=np.asarray(is_mined, dtype=bool),
    )
    meta = {"tokenizer": asdict(config), "align_step": ALIGN_STEP, "smooth_seconds": args.smooth_seconds}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    splits_array = np.asarray(splits)
    hours = sum(cover_ends) / config.frame_rate / 3600
    print(
        f"原曲 {len(source_ids)} 曲 / カバー {len(cover_ids)} 本 ({hours:.0f} 時間) / "
        f"学習 {(splits_array == 0).sum()} 検証 {(splits_array == 1).sum()} テスト {(splits_array == 2).sum()} / 失敗 {failed}"
    )
    print(f"音符の onset で補正できた窓の割合: 平均 {np.mean(refined_rates):.0%}")
    print(
        f"長さの比で外したカバー: 前の組 {dropped['length'][0]} / 足した組 {dropped['length'][1]}、"
        f"検証・テスト側の曲なので外した足した組 {dropped['mined_eval']}"
    )
    ok_all = np.concatenate(align_ok)
    ok_start = np.concatenate([ok[: 400 // ALIGN_STEP] for ok in align_ok])
    print(f"原曲に沿っていない (損失を取らない) 所: 全体 {1 - ok_all.mean():.0%} / 冒頭 4 秒 {1 - ok_start.mean():.0%}")
    matches_array, mined_array = np.asarray(matches), np.asarray(is_mined, dtype=bool)
    for name, selected in (("前の組", ~mined_array), ("足した組", mined_array)):
        values = matches_array[selected & ~np.isnan(matches_array)]
        if len(values):
            print(
                f"メロディの一致率 {name} {selected.sum()} 本: 中央値 {np.median(values):.2f} / "
                f"0.3 未満 {(values < 0.3).sum()} 本"
            )
    print(f"出力: {out_dir}")


if __name__ == "__main__":
    main()
