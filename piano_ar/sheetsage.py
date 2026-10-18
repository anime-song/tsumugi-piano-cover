"""SheetSage2-Refined で音声から拍 (拍・小節の頭・拍子) とメロディを取る。

    python -m piano_ar.sheetsage AUDIO_ROOT OUT_ROOT [--model REPO_OR_FOLDER] [--shard i/n]
    python -m piano_ar.sheetsage MIDI_ROOT OUT_ROOT --soundfont SF [--render-python PY] [--render-workers 4]

AUDIO_ROOT の下の音声 (*.mp3 / *.wav / *.flac) を、同じ相対パスで OUT_ROOT/<相対パス>.json に書く。書き済みのものは飛ばす。
MIDI (*.mid / *.midi) は、音声のない演奏 (事前学習の採譜など) のためにピアノの音で合成してから同じように通す。
合成は子プロセス (piano_ar.render_midi) が先回りして行い、GPU を待たせない。合成音と実音声の拍の F 値は
0.955 (カバー 40 本、±70ms、local/probe_beats.py の (d))。Windows の symusic 0.6.0 は合成で落ちるので、
symusic 0.5.7 を入れた別の Python を --render-python に渡す。

    {"duration": s, "beats": [s, ...], "downbeats": [s, ...],
     "bars": [{"start": s, "beats": n, "unit": d, "meter": "n/d"}, ...], "meters": [...],
     "melody": [{"start": s, "end": s, "pitch": midi, "track": 0 歌 / 1 楽器, "steps": n, "pos": n}, ...]}

拍のヘッド (HeadBeats) もメロディのヘッド (HeadMelody) も、同じ凍結したエンコーダ (MERT) の出力を読むだけなので、
エンコーダは 1 回だけ通して、2 つのヘッドに渡す。中身の後処理 (拍の追跡・メロディの経路) は sheetsage2_refined のまま。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

AUDIO_SUFFIXES = (".mp3", ".wav", ".flac")
MIDI_SUFFIXES = (".mid", ".midi")


class SheetSageAnnotator:
    def __init__(self, model: str | Path | None = None, device: str | None = None) -> None:
        from sheetsage2_refined.beats import HeadBeats
        from sheetsage2_refined.heads import load_head
        from sheetsage2_refined.melody import HeadMelody, melody_files
        from sheetsage2_refined.pipeline import DEFAULT_MODEL, model_folder

        model = model or DEFAULT_MODEL
        self.beats = HeadBeats.load(model, device=device)
        folder = model_folder(model, None, melody_files)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        tracker = config["melody_tracker"]
        if tracker.get("beat_head"):
            raise ValueError("拍のヘッドの出力を読むメロディのヘッドには対応していない")
        head = load_head(folder / tracker["head"], self.beats.device)
        # エンコーダは拍の側のものを共有する (2.7GB を 2 回読まない)
        self.melody = HeadMelody(
            self.beats.encoder, head, self.beats.device, head.line is not None, **(tracker.get("decoding") or {})
        )

    @property
    def sample_rate(self) -> int:
        return self.beats.encoder.config.sample_rate

    @torch.no_grad()
    def outputs(self, wave: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """拍とメロディのヘッドの 25 Hz の出力 (logits)。sheetsage2_refined.beats.frame_outputs と同じ窓で通す"""
        from sheetsage2_refined.heads import CHUNK
        from sheetsage2_refined.model import WINDOW_SECONDS
        from sheetsage2_refined.transcribe import autocast

        encoder, device = self.beats.encoder, self.beats.device
        window = round(WINDOW_SECONDS * self.sample_rate)
        beat_outs, melody_outs = [], []
        for start in range(0, len(wave), window):
            # 窓で割った余りがごく短い (メルの反射パディングの 1024 サンプル以下) とエンコーダが通らないので、
            # 0.5 秒未満の余りは捨てる (sheetsage2_refined.beats.frame_outputs も同じ所で落ちる)
            if start > 0 and len(wave) - start < self.sample_rate // 2:
                break
            with autocast(device):
                states = encoder(wave[start : start + window][None].to(device))
            beat_outs.append(self.beats.head(states, CHUNK))
            melody_outs.append(self.melody.head(states, CHUNK))

        def join(outs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            return {k: torch.cat([o[k] for o in outs], 1)[0].float().cpu() for k in outs[0]}

        return join(beat_outs), join(melody_outs)

    def annotate(self, audio: str | Path) -> dict:
        from sheetsage2_refined.transcribe import load_waveform

        return self.annotate_wave(load_waveform(Path(audio), self.sample_rate))

    def annotate_wave(self, wave: torch.Tensor) -> dict:
        """sample_rate のモノラルの波形 [samples] (float32) から"""
        from sheetsage2_refined.heads import LINE, OFFSETS

        beat_out, melody_out = self.outputs(wave)

        # HeadBeats.probabilities と HeadMelody.probabilities の変換
        beat_probs = {k: torch.sigmoid(beat_out[k]).numpy() for k in ("beat", "downbeat")}
        beat_probs |= {
            k: torch.softmax(beat_out[k], -1).numpy() for k in ("bar_beats", "beats_left", "bar_unit") if k in beat_out
        }
        beat_probs |= {k: beat_out[k].numpy() for k in OFFSETS if k in beat_out}
        melody_probs = {
            k: torch.sigmoid(melody_out[k]).numpy() for k in ("melody", "melody_on", "melody_vocal") if k in melody_out
        }
        melody_probs |= {k: torch.softmax(melody_out[k], -1).numpy() for k in LINE if k in melody_out}
        melody_probs["melody_offset"] = melody_out["melody_offset"].numpy()

        result = {"duration": len(wave) / self.sample_rate}
        result |= self.beats.track_probabilities(beat_probs)
        result["melody"] = self.melody.notes(melody_probs)
        return result


def load_annotation(path: str | Path) -> dict:
    """書き出した JSON を読み、拍と小節の頭を np.ndarray (秒) にして返す"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["beats"] = np.asarray(data["beats"], dtype=np.float64)
    data["downbeats"] = np.asarray(data["downbeats"], dtype=np.float64)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio_root", type=Path)
    parser.add_argument("out_root", type=Path)
    parser.add_argument("--model", default=None, help="Hugging Face の repository か手元のモデルのフォルダ")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--list", type=Path, help="対象を AUDIO_ROOT からの相対パスの一覧 (1 行 1 つ) に絞る")
    parser.add_argument("--shard", default="0/1", help="i/n: 並べた一覧の i 番目から n 個おきだけを処理する")
    parser.add_argument("--memory-fraction", type=float, default=0.6, help="この処理が使う GPU メモリの上限の割合")
    parser.add_argument("--soundfont", type=Path, help="MIDI の合成に使う SoundFont (*.sf2 / *.sf3)")
    parser.add_argument("--render-python", help="合成の子プロセスの Python (既定はこの Python)")
    parser.add_argument("--render-workers", type=int, default=4)
    args = parser.parse_args()

    if args.list:
        names = [line.strip() for line in args.list.read_text(encoding="utf-8").splitlines() if line.strip()]
        files = [args.audio_root / name for name in names]
    else:
        suffixes = AUDIO_SUFFIXES + MIDI_SUFFIXES
        files = sorted(p for p in args.audio_root.rglob("*") if p.suffix.lower() in suffixes)
    index, count = (int(v) for v in args.shard.split("/"))
    files = files[index::count]

    def out_path(path: Path) -> Path:
        return (args.out_root / path.relative_to(args.audio_root)).with_suffix(".json")

    todo = [p for p in files if not out_path(p).exists()]
    print(f"{len(files)} 本のうち {len(todo)} 本が未処理", flush=True)
    if not todo:
        return
    audio_todo = [p for p in todo if p.suffix.lower() in AUDIO_SUFFIXES]
    midi_todo = [p for p in todo if p.suffix.lower() in MIDI_SUFFIXES]
    if midi_todo and args.soundfont is None:
        parser.error("MIDI を合成するには --soundfont が要る")

    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, torch.device(args.device).index or 0)
    annotator = SheetSageAnnotator(args.model, args.device)
    started = time.perf_counter()
    done = 0

    def write(path: Path, result: dict) -> None:
        nonlocal done
        out = out_path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out)
        done += 1
        elapsed = time.perf_counter() - started
        print(
            f"[{done}/{len(todo)}] {path.name}: {len(result['beats'])} 拍, {len(result['bars'])} 小節, "
            f"メロディ {len(result['melody'])} 音 ({elapsed / done:.1f} 秒/本)",
            flush=True,
        )

    for audio in audio_todo:
        try:
            result = annotator.annotate(audio)
        except Exception as error:  # 壊れた音声などは飛ばして続ける
            print(f"失敗: {audio}: {error!r}", flush=True)
            continue
        write(audio, result)

    if midi_todo:
        for path, result in rendered(midi_todo, args, annotator):
            write(path, result)


def rendered(midi: list[Path], args, annotator: SheetSageAnnotator):
    """MIDI を子プロセスで合成しながら、出来た順 (一覧の順) に注釈を付ける"""
    queue = (args.out_root / "_render_queue").resolve()
    queue.mkdir(parents=True, exist_ok=True)
    for stale in queue.iterdir():
        stale.unlink()
    listing = queue / "list.txt"
    listing.write_text("".join(f"{p.resolve()}\n" for p in midi), encoding="utf-8")
    # 子プロセスはこのリポジトリの直下で動かすので、パスはすべて絶対にする
    python = str(Path(args.render_python).resolve()) if args.render_python else sys.executable
    workers = [
        subprocess.Popen(
            [
                python,
                "-m",
                "piano_ar.render_midi",
                str(listing),
                str(queue),
                "--soundfont",
                str(args.soundfont.resolve()),
                "--sample-rate",
                str(annotator.sample_rate),
                "--worker",
                f"{w}/{args.render_workers}",
            ],
            cwd=Path(__file__).resolve().parents[1],
        )  # fmt: skip
        for w in range(args.render_workers)
    ]
    try:
        for i, path in enumerate(midi):
            audio, error = queue / f"{i}.npy", queue / f"{i}.err"
            while not audio.exists() and not error.exists():
                if workers[i % len(workers)].poll() is not None:
                    raise RuntimeError(f"合成の子プロセス {i % len(workers)} が止まった")
                time.sleep(0.05)
            if error.exists():
                print(f"失敗 (合成): {path}: {error.read_text(encoding='utf-8')}", flush=True)
                error.unlink()
                continue
            wave = torch.from_numpy(np.load(audio).astype(np.float32))
            audio.unlink()
            try:
                result = annotator.annotate_wave(wave)
            except Exception as failure:
                print(f"失敗: {path}: {failure!r}", flush=True)
                continue
            yield path, result
    finally:
        for worker in workers:
            worker.kill()


if __name__ == "__main__":
    main()
