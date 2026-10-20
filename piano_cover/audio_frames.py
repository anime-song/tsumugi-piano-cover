"""原曲の音声の特徴 (カバーモデルに足すフレームの流れ) を、SheetSage2-Refined のエンコーダ (MERT) から作る。

    python -m piano_cover.audio_frames fit AUDIO_ROOT OUT_DIR [--songs 200]
    python -m piano_cover.audio_frames extract AUDIO_ROOT OUT_DIR

エンコーダの 25 層の出力 (25 Hz、1024 次元) を、拍のヘッドとメロディのヘッドがそれぞれ学習した層の混ぜ方 (FrameHead.mix) で
2 通りに混ぜ、それぞれ layer_norm してつなげる (2048 次元)。拍のヘッドは中ほどの層 (15〜19)、メロディのヘッドは上の層
(19〜22) を重く使う。そのままだと原曲 2,346 本で約 29GB (float16) になるので、原曲から取った PCA で 256 次元にする
(分散の約 90%。30 曲で 80% に 127 次元、90% に 248 次元、95% に 428 次元)。

fit:     AUDIO_ROOT から --songs 本を選び、5 フレームおきの特徴の平均と共分散から PCA を求めて OUT_DIR/pca.npz に書く
extract: AUDIO_ROOT の音声ごとに OUT_DIR/<名前>.npy ([フレーム, 256]、float16、25 Hz、0 フレーム目は音声の頭) を書く。
         書き済みのものは飛ばす

エンコーダの窓の切り方は piano_ar.sheetsage.SheetSageAnnotator.outputs と同じ (300 秒ずつ、0.5 秒未満の余りは捨てる)。
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AUDIO_SUFFIXES = (".mp3", ".wav", ".flac")
FRAME_RATE = 25
DIMS = 256


class FrameExtractor:
    def __init__(self, model: str | Path | None = None, device: str | None = None) -> None:
        from sheetsage2_refined.heads import FrameHead

        from piano_ar.sheetsage import SheetSageAnnotator

        annotator = SheetSageAnnotator(model, device=device)
        self.encoder = annotator.beats.encoder
        self.device = annotator.beats.device
        self.sample_rate = annotator.sample_rate
        weights = []
        for module in (annotator.beats.head, annotator.melody.head):
            mixes = [torch.softmax(m.mix.detach().float(), -1).reshape(-1, m.mix.shape[-1])
                     for m in module.modules() if isinstance(m, FrameHead)]  # fmt: skip
            weights.append(torch.cat(mixes).mean(0))
        self.weights = torch.stack(weights).to(self.device)  # [2, 25]
        # ヘッドは使わない (エンコーダと混ぜ方だけ)
        del annotator

    @torch.no_grad()
    def features(self, wave: torch.Tensor) -> torch.Tensor:
        """sample_rate のモノラルの波形 [samples] -> [フレーム, 2048] (float32、GPU の上)"""
        from sheetsage2_refined.model import WINDOW_SECONDS
        from sheetsage2_refined.transcribe import autocast

        window = round(WINDOW_SECONDS * self.sample_rate)
        parts = []
        for start in range(0, len(wave), window):
            if start > 0 and len(wave) - start < self.sample_rate // 2:
                break
            with autocast(self.device):
                states = self.encoder(wave[start : start + window][None].to(self.device))
            mixed = [sum(w * s[0].float() for w, s in zip(row, states)) for row in self.weights]
            parts.append(torch.cat([F.layer_norm(m, m.shape[-1:]) for m in mixed], -1))
        return torch.cat(parts)

    def load(self, path: Path) -> torch.Tensor:
        from sheetsage2_refined.transcribe import load_waveform

        return load_waveform(path, self.sample_rate)


def audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)


def fit(extractor: FrameExtractor, files: list[Path], out: Path, stride: int = 5) -> None:
    total = torch.zeros(2048, dtype=torch.float64, device=extractor.device)
    outer = torch.zeros(2048, 2048, dtype=torch.float64, device=extractor.device)
    count = 0
    for i, path in enumerate(files):
        x = extractor.features(extractor.load(path))[::stride].double()
        total += x.sum(0)
        outer += x.T @ x
        count += len(x)
        print(f"[fit] {i + 1}/{len(files)} {path.name} {len(x)} 行", flush=True)
    mean = total / count
    covariance = outer / count - mean[:, None] * mean[None]
    values, vectors = torch.linalg.eigh(covariance)
    order = values.argsort(descending=True)
    values, vectors = values[order], vectors[:, order]
    explained = float(values[:DIMS].sum() / values.sum())
    np.savez(
        out / "pca.npz",
        mean=mean.float().cpu().numpy(),
        components=vectors[:, :DIMS].float().cpu().numpy(),  # [2048, DIMS]
        variance=values[:DIMS].float().cpu().numpy(),
        explained=np.float32(explained),
        frame_rate=np.int64(FRAME_RATE),
        weights=extractor.weights.cpu().numpy(),
    )
    print(f"[fit] {count} 行、{DIMS} 次元で分散の {explained:.1%}")


def extract(extractor: FrameExtractor, files: list[Path], root: Path, out: Path) -> None:
    pca = np.load(out / "pca.npz")
    mean = torch.from_numpy(pca["mean"]).to(extractor.device)
    components = torch.from_numpy(pca["components"]).to(extractor.device)
    started, done = time.time(), 0
    for i, path in enumerate(files):
        target = out / path.relative_to(root).with_suffix(".npy")
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        x = (extractor.features(extractor.load(path)) - mean) @ components
        tmp = target.with_suffix(".tmp.npy")
        np.save(tmp, x.half().cpu().numpy())
        tmp.replace(target)
        done += 1
        if done % 20 == 0:
            rate = (time.time() - started) / done
            print(
                f"[extract] {i + 1}/{len(files)} ({rate:.1f} 秒/曲、残り約 {rate * (len(files) - i - 1) / 60:.0f} 分)",
                flush=True,
            )
    print(f"[extract] 終わり ({done} 本書いた)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fit", "extract"))
    parser.add_argument("audio_root", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--model", default=None, help="Hugging Face の repository か手元のモデルのフォルダ")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--songs", type=int, default=200, help="fit に使う曲数")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    files = audio_files(args.audio_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    extractor = FrameExtractor(args.model, args.device)
    if args.command == "fit":
        random.Random(args.seed).shuffle(files)
        fit(extractor, files[: args.songs], args.out_dir)
    else:
        extract(extractor, files, args.audio_root, args.out_dir)


if __name__ == "__main__":
    main()
