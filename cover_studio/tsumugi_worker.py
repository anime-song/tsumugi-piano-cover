"""音源 1 曲を tsumugi (https://github.com/anime-song/tsumugi) で採譜して MIDI にする。cover_studio が別プロセスで呼ぶ。

tsumugi は依存 (ステム分離など) が多いので、このリポジトリの .venv ではなく tsumugi の環境の Python で動かす。
このファイルはこのリポジトリのパッケージを import しない (tsumugi の環境には入っていない)。

    <tsumugi>/.venv/Scripts/python.exe cover_studio/tsumugi_worker.py --tsumugi-dir <tsumugi> --audio song.mp3 --out source.mid

設定はカバーモデルの学習に使った原曲の MIDI と同じ (ステム分離 → 採譜 → 楽器の再判定 → マージ → ベロシティ →
ビート・コード・キー)。違う設定の MIDI では原曲エンコーダが学習で見たものと変わってしまう。

採譜中は、確定したノートを "@@notes {json}" の行で stdout に流す (tsumugi の streaming。UI がピアノロールに描く)。
ステムの始まりと終わりは "@@stem {json}" / "@@stem_done {json}"。楽器の再判定・ベロシティの前の値なので、最終の MIDI とは
楽器とベロシティが違う。
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
import shutil
import sys
from pathlib import Path


def emit(kind: str, data: dict) -> None:
    print(f"@@{kind} {json.dumps(data, separators=(',', ':'))}", flush=True)


def stream_notes(infer_stem) -> None:
    """ステムごとの採譜 (infer.run_inference) に tsumugi の StreamingNoteEmitter をつなぎ、確定したノートを流す。
    run_stem_separated_transcription はこのフックを通さないので、呼び出す側の関数を包む。採譜の結果そのものは変えない"""
    try:
        from instrument_agnostic_amt.streaming import StreamingNoteEmitter
    except ImportError:  # streaming のない古い tsumugi では流さない (採譜はできる)
        return
    infer = infer_stem.infer
    original_run, original_resolve = infer.run_inference, infer_stem.resolve_stem_model_type
    current = {"stem": None}

    def resolve_stem_model_type(stem_name):
        current["stem"] = stem_name  # この直後に、そのステムの run_inference が呼ばれる
        return original_resolve(stem_name)

    def run_inference(*args, **kwargs):
        stem = current["stem"] or "stem"
        sample_rate = int(kwargs["model_config"].sample_rate)
        emit("stem", {"stem": stem, "duration": round(kwargs["waveform"].shape[-1] / sample_rate, 3)})
        if "drum" in stem.lower():  # ドラムは音高がないので描かない
            result = original_run(*args, **kwargs)
            emit("stem_done", {"stem": stem})
            return result
        state = {"emitter": None}

        def send(events, position):
            notes = [[e.id, e.start, e.end, e.pitch, int(e.final)] for e in events]
            emit("notes", {"stem": stem, "pos": round(position, 3), "notes": notes})

        def on_window(stitcher, window_start_sample):
            if state["emitter"] is None:
                state["emitter"] = StreamingNoteEmitter(
                    stitcher, stem=stem, sample_rate=sample_rate, instrument_label=str
                )
            send(state["emitter"].on_window(window_start_sample), window_start_sample / sample_rate)

        result = original_run(*args, on_window_consumed=on_window, **kwargs)
        if state["emitter"] is not None:
            send(state["emitter"].finish(), kwargs["waveform"].shape[-1] / sample_rate)
        emit("stem_done", {"stem": stem})
        return result

    infer.run_inference = run_inference
    infer_stem.resolve_stem_model_type = resolve_stem_model_type


def stream_separation(infer_stem) -> None:
    """ステム分離の進み具合を "@@progress {json}" の行で流す。

    stem_splitter 0.0.9 からは _separate_one_file がバッチごとに on_batch(done, total) を呼ぶので、
    tsumugi が呼ぶ _separate_one_file に on_batch を足して渡す。done は GPU の計算が終わってから数えられる。
    on_batch のない古い stem_splitter では何もしない (採譜はできる)。
    """
    original = getattr(infer_stem, "_separate_one_file", None)
    if original is None:
        return
    try:
        if "on_batch" not in inspect.signature(original).parameters:
            return
    except (TypeError, ValueError):
        return

    def on_batch(done: int, total: int) -> None:
        emit("progress", {"stage": "separate", "done": int(done), "total": int(total)})

    infer_stem._separate_one_file = functools.partial(original, on_batch=on_batch)


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
    stream_separation(infer_stem)
    stream_notes(infer_stem)

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
