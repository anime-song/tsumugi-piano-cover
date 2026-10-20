"""カバーのキャッシュ (prepare_beats の出力) に、原曲の音声の特徴 (piano_cover.audio_frames extract の出力) を足す。

    python -m piano_cover.prepare_frames data/piano_cover_v6b [--frames-dir data/audio_frames/original]

出力 (キャッシュの中):
    frames.npy         [N, D] float16  全原曲の特徴をキャッシュの原曲の順につなげたもの
    frames_index.npz   offsets [S + 1] (原曲ごとの範囲。特徴のない原曲は空)、frame_rate (Hz)
    frames_pca.npz     特徴を作った PCA (audio_frames の pca.npz の写し。生成で同じ特徴を作るのに要る)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from .data import CoverCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--frames-dir", type=Path, default=Path("data/audio_frames/original"))
    args = parser.parse_args()

    cache = CoverCache(args.cache_dir)
    pca = np.load(args.frames_dir / "pca.npz")
    dims = int(pca["components"].shape[1])
    paths = [args.frames_dir / f"{video_id}.npy" for video_id in cache.source_ids]
    lengths = [len(np.load(p, mmap_mode="r")) if p.exists() else 0 for p in paths]
    out = np.lib.format.open_memmap(
        args.cache_dir / "frames.npy", mode="w+", dtype=np.float16, shape=(int(sum(lengths)), dims)
    )
    position = 0
    for path, length in zip(paths, lengths):
        if length:
            out[position : position + length] = np.load(path)
            position += length
    out.flush()
    np.savez(
        args.cache_dir / "frames_index.npz",
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        frame_rate=np.float64(pca["frame_rate"]),
    )
    shutil.copy2(args.frames_dir / "pca.npz", args.cache_dir / "frames_pca.npz")
    have = sum(length > 0 for length in lengths)
    print(f"原曲 {len(paths)} のうち特徴のある原曲 {have}、{position} フレーム ({out.nbytes / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
