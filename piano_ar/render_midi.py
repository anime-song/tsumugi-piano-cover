"""MIDI をピアノの音で合成して、piano_ar.sheetsage に渡す波形を書き出す (piano_ar.sheetsage が子プロセスとして動かす)。

    python -m piano_ar.render_midi LIST QUEUE_DIR --soundfont SF --sample-rate 24000 --worker i/n

LIST の i 番目から n 個おきの MIDI を合成し、QUEUE_DIR/<一覧の番号>.npy (float16 のモノラル) に置く。
合成できなかったものは <番号>.err に理由を書く。置いたまま読まれていないものが --max-pending 個あれば待つ。

torch を読まないので、symusic の合成が動く別の Python からも動かせる
(Windows の symusic 0.6.0 は合成の返り値の変換で落ちるが、0.5.7 は動く)。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def render(path: Path, synth) -> np.ndarray:
    from symusic import Score

    score = Score(str(path))
    score.tracks = [track for track in score.tracks if not track.is_drum]
    for track in score.tracks:
        track.program = 0
    audio = np.asarray(synth.render(score, stereo=True), dtype=np.float32).mean(axis=0)
    return audio / max(1e-6, float(np.abs(audio).max())) * 0.9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("list", type=Path, help="MIDI のパスの一覧 (1 行 1 つ)")
    parser.add_argument("queue_dir", type=Path)
    parser.add_argument("--soundfont", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--worker", default="0/1")
    parser.add_argument("--max-pending", type=int, default=4)
    args = parser.parse_args()

    from symusic import Synthesizer

    synth = Synthesizer(sf_path=str(args.soundfont), sample_rate=args.sample_rate)
    paths = [line.strip() for line in args.list.read_text(encoding="utf-8").splitlines()]
    index, count = (int(v) for v in args.worker.split("/"))
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    pending: list[Path] = []
    for i in range(index, len(paths), count):
        # 自分が置いて、まだ読まれていないものが多ければ待つ
        while True:
            pending = [p for p in pending if p.exists()]
            if len(pending) < args.max_pending:
                break
            time.sleep(0.2)
        try:
            audio = render(Path(paths[i]), synth)
        except Exception as error:  # 壊れた MIDI などは飛ばす
            (args.queue_dir / f"{i}.err").write_text(repr(error), encoding="utf-8")
            continue
        tmp = args.queue_dir / f"{i}.tmp.npy"
        np.save(tmp, audio.astype(np.float16))
        tmp.replace(args.queue_dir / f"{i}.npy")
        pending.append(args.queue_dir / f"{i}.npy")


if __name__ == "__main__":
    main()
