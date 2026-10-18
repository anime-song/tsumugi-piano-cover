"""作ってあるカバーのキャッシュの原曲の行に、SheetSage2-Refined の拍とメロディを足した別のキャッシュを作る。

    python -m piano_cover.add_sheetsage data/piano_cover_v5 data/piano_cover_v5s [--no-beats] [--no-melody]

カバーのイベントとアラインメントは元のキャッシュのまま写すので、原曲の入力以外は同じ学習になる
(アラインメントの細かい補正と align_ok は tsumugi の行で求めたもの)。原曲の行は piano_cover.source.with_sheetsage で、
拍は SheetSage のものに置き換え、メロディは別の楽器 ID の音として足す。注釈 (piano_ar.sheetsage の JSON) がない原曲は
元の行のまま。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .source import ROW_ONSET, with_sheetsage

COPIED = ("cover_events.npy", "align.npy", "align_ok.npy", "covers.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--sheetsage-dir", type=Path, default=Path("data/sheetsage/original"))
    parser.add_argument("--no-beats", action="store_true", help="拍は tsumugi のまま")
    parser.add_argument("--no-melody", action="store_true", help="SheetSage のメロディを足さない")
    args = parser.parse_args()

    meta = json.loads((args.src / "meta.json").read_text(encoding="utf-8"))
    frame_rate = int(meta["tokenizer"]["frame_rate"])
    sources = np.load(args.src / "sources.npz")
    rows_all = np.load(args.src / "source_rows.npy", mmap_mode="r")
    offsets, end_frames, video_ids = sources["offsets"], sources["end_frames"], sources["video_ids"]

    out_rows, lengths, missing = [], [], []
    for i, original_id in enumerate(video_ids):
        rows = np.asarray(rows_all[offsets[i] : offsets[i + 1]])
        path = args.sheetsage_dir / f"{original_id}.json"
        if path.exists():
            annotation = json.loads(path.read_text(encoding="utf-8"))
            rows = with_sheetsage(rows, annotation, frame_rate, beats=not args.no_beats, melody=not args.no_melody)
            # load_source と同じく、曲の終端 (tsumugi の最後の音の終わり) より後の行は入れない
            rows = rows[rows[:, ROW_ONSET] < end_frames[i]]
        else:
            missing.append(str(original_id))
        out_rows.append(rows.astype(np.int32))
        lengths.append(len(rows))

    args.dst.mkdir(parents=True, exist_ok=True)
    np.save(args.dst / "source_rows.npy", np.concatenate(out_rows))
    np.savez(
        args.dst / "sources.npz",
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        end_frames=end_frames,
        video_ids=video_ids,
    )
    for name in COPIED:
        if (args.src / name).exists():
            shutil.copyfile(args.src / name, args.dst / name)
    meta["sheetsage"] = {
        "from": str(args.src),
        "beats": not args.no_beats,
        "melody": not args.no_melody,
        "missing": missing,
    }
    (args.dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"原曲 {len(video_ids)} 曲のうち、注釈がなく元のままの曲 {len(missing)}")


if __name__ == "__main__":
    main()
