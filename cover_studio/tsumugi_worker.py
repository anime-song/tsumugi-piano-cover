"""音源 1 曲を tsumugi (https://github.com/anime-song/tsumugi) で採譜して MIDI にする。cover_studio が別プロセスで呼ぶ。

tsumugi は依存 (ステム分離など) が多いので、このリポジトリの .venv ではなく tsumugi の環境の Python で動かす。
このファイルはこのリポジトリのパッケージを import しない (tsumugi の環境には入っていない)。

    <tsumugi>/.venv/Scripts/python.exe cover_studio/tsumugi_worker.py --tsumugi-dir <tsumugi> --audio song.mp3 --out source.mid

設定はカバーモデルの学習に使った原曲の MIDI と同じ (ステム分離 → 採譜 → 楽器の再判定 → マージ → ベロシティ →
ビート・コード・キー)。違う設定の MIDI では原曲エンコーダが学習で見たものと変わってしまう。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tsumugi-dir", required=True, help="tsumugi のリポジトリ (チェックポイントもここに置かれる)")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True, help="書き出す MIDI")
    parser.add_argument("--work", default=None, help="作業フォルダ (終わったら消す)。省略で --out の隣")
    parser.add_argument("--ffmpeg-dir", default=None, help="FFmpeg の共有ライブラリ版の bin (Windows の torchcodec 用)")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument("--compile", action="store_true", help="torch.compile を使う (初回の準備に時間がかかる)")
    parser.add_argument("--window-batch-size", type=int, default=4)
    parser.add_argument(
        "--gpu-memory-limit", type=float, default=0.9, help="起動時の空き VRAM のうち使ってよい割合 (0 で無制限)"
    )
    args = parser.parse_args()

    audio = Path(args.audio).resolve()
    out = Path(args.out).resolve()
    work = Path(args.work).resolve() if args.work else out.parent / "_tsumugi_work"
    shutil.rmtree(work, ignore_errors=True)
    tsumugi_dir = Path(args.tsumugi_dir).resolve()

    if args.ffmpeg_dir:
        # torchaudio (torchcodec) は FFmpeg の共有ライブラリ版が要る。システムの FFmpeg には触らず、このプロセスだけに読ませる
        os.add_dll_directory(str(Path(args.ffmpeg_dir).resolve()))
        os.environ["PATH"] = f"{Path(args.ffmpeg_dir).resolve()}{os.pathsep}{os.environ['PATH']}"
    # tsumugi はチェックポイントを作業フォルダ相対 (checkpoints/ など) に置くので、リポジトリの中で動かす
    os.chdir(tsumugi_dir)
    sys.path.insert(0, str(tsumugi_dir))
    import torch
    from instrument_agnostic_amt.cli import infer_stem

    if args.gpu_memory_limit > 0 and torch.cuda.is_available():
        # Windows のドライバは VRAM が足りないとメインメモリにあふれさせ、エラーにならずに極端に遅くなるので上限を設ける
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(free * args.gpu_memory_limit / total)
        print(f"VRAM の上限 {free * args.gpu_memory_limit / 1e9:.1f}GB", flush=True)

    try:
        result = infer_stem.run_stem_separated_transcription(
            audio,
            output_root=work,
            refine_instruments=True,
            predict_velocity=True,
            predict_beat_chord=True,
            amp=True,
            window_batch_size=args.window_batch_size,
            merge_onset_ms=50.0,
            semi_crf_backend="auto",
            compile_model=args.compile,
            compile_velocity=args.compile,
            device=args.device,
        )
        merged = Path(result["merged_midi_path"])
        # ベロシティ予測とコード推定は内部で失敗しても警告だけで続行するので、最後まで進んだかをファイル名で確かめる
        if not merged.stem.endswith("_beat_chord"):
            print(f"警告: ベロシティかビート・コードの推定が途中で失敗しました ({merged.name})", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".mid.tmp")
        shutil.copy2(merged, tmp)
        os.replace(tmp, out)
        print(f"採譜しました: {out}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
