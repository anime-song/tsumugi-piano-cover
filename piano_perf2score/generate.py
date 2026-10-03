"""演奏 (MIDI) から楽譜 (MusicXML) を生成する。

    python -m piano_perf2score.generate --checkpoint checkpoints/piano_perf2score/best.pt --midi 演奏.mid --out 楽譜.musicxml
    python -m piano_perf2score.generate --checkpoint ... --val-songs 4 --out-dir outputs/perf2score

--val-songs を付けると、2 段目のキャッシュの検証曲を合成演奏にして楽譜に戻し、元の楽譜と並べて保存する
(合成演奏 .mid・元の楽譜 _reference.musicxml・生成した楽譜 _generated.musicxml)。

小節を 1 つずつ生成する。小節 p を生成し始めるときに分かっているのは、それまでに生成した MTIME から戻した各小節の開始時刻
だけなので、学習と同じく「前の小節の開始」を基準にして演奏を見る (piano_perf2score.model)。
演奏の最後の音より後に始まる小節を生成したら、そこで止める。--temperature 0 (既定) なら毎回いちばん確率の高いトークンを選ぶ。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from piano_ar.config import ModelConfig
from piano_ar.model import _sample
from piano_score.config import ScoreTokenizerConfig
from piano_score.generate import unseen_tokens
from piano_score.grammar import ScoreGrammar
from piano_score.musicxml import write_musicxml
from piano_score.tokenizer import EOS, PAD, ScoreTokenizer

from .data import FRAMES_PER_SECOND, PerformanceVocab, collate, patch_rows
from .model import Perf2ScoreConfig, Perf2ScoreModel
from .render import Performance, measure_qpm, render

PEDAL_THRESHOLD = 64


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[Perf2ScoreModel, ScoreTokenizer, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint["tokenizer_config"])
    config["fraction_denominators"] = tuple(config["fraction_denominators"])
    tokenizer = ScoreTokenizer(ScoreTokenizerConfig(**config))
    model = Perf2ScoreModel(
        ModelConfig.from_dict(checkpoint["model_config"]), Perf2ScoreConfig(**checkpoint["perf2score_config"]), tokenizer
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), tokenizer, checkpoint


def read_performance(path: str | Path) -> Performance:
    """MIDI を演奏にする (全トラックの音と、サステインペダル CC64 の踏み・離し)。最初の音を 0 秒にする"""
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(path))
    notes, pedal = [], []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes += [[n.start, n.end, n.pitch, n.velocity] for n in instrument.notes]
        down = False
        for cc in sorted(instrument.control_changes, key=lambda c: c.time):
            if cc.number == 64 and (cc.value >= PEDAL_THRESHOLD) != down:
                down = not down
                pedal.append([cc.time, float(down)])
    notes = np.array(sorted(notes), dtype=np.float64).reshape(-1, 4)
    pedal = np.array(sorted(pedal), dtype=np.float64).reshape(-1, 2)
    origin = notes[:, 0].min() if len(notes) else 0.0
    notes[:, :2] -= origin
    if len(pedal):
        pedal[:, 0] -= origin
    return Performance(notes, pedal, np.zeros(0))


def performance_batch(performance: Performance, max_rows: int = 160) -> dict[str, torch.Tensor]:
    """演奏を、モデルの演奏エンコーダに入れる形 (data.collate の perf_* と同じ、バッチ 1) にする"""
    vocab = PerformanceVocab()
    features, onset = vocab.rows(performance)
    end_frame = int(onset.max()) + 1 if len(onset) else 1
    perf_features, perf_onset, perf_valid = patch_rows(features, onset, max_rows, end_frame)
    item = {
        "tokens": torch.zeros(1, 1, dtype=torch.long),
        "perf_features": torch.from_numpy(perf_features),
        "perf_onset": torch.from_numpy(perf_onset),
        "perf_valid": torch.from_numpy(perf_valid),
    }
    batch = collate([item])
    del batch["tokens"]
    return batch


# 曲の最初の小節で試すテンポ (4 分音符/分)。前の小節がないので、4 分音符の長さはこの中からモデルに選ばせる
SEARCH_TEMPOS = (40, 48, 56, 66, 76, 88, 100, 116, 132, 152, 176, 208)


@torch.no_grad()
def transcribe(
    model: Perf2ScoreModel,
    tokenizer: ScoreTokenizer,
    performance: Performance,
    *,
    max_measures: int = 1000,
    context_measures: int = 32,
    temperature: float = 0.0,
    top_p: float = 0.95,
    banned: torch.Tensor | None = None,
    first_qpm: float | None = None,
    search_seconds: float = 10.0,
) -> list[list[int]]:
    """演奏から楽譜を生成して、小節ごとのトークン列を返す。

    曲の最初の小節の 4 分音符の長さ (first_qpm) を渡さなければ、SEARCH_TEMPOS のそれぞれで冒頭 search_seconds 秒を
    いちばん確率の高いトークンで生成し、1 秒あたりの対数尤度がいちばん高いテンポを選ぶ。演奏だけではテンポの倍・半分
    (8 分で書くか 16 分で書くか) が決まらないので、楽譜としてどちらが自然かをモデルの確率で決める。
    """
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in performance_batch(performance).items()}
    memory = model.encode_performance(batch)
    end_frame = float(performance.notes[:, 0].max() * FRAMES_PER_SECOND) if len(performance.notes) else 0.0
    common = {"memory": memory, "end_frame": end_frame, "context_measures": context_measures, "banned": banned}
    if first_qpm is None:
        scores = {}
        for qpm in SEARCH_TEMPOS:
            _, logprob, covered = _decode(
                model, tokenizer, first_spq=60.0 * FRAMES_PER_SECOND / qpm, max_measures=max_measures,
                stop_frame=search_seconds * FRAMES_PER_SECOND, temperature=0.0, top_p=1.0, **common,
            )  # fmt: skip
            scores[qpm] = logprob / max(covered, 1.0)
        first_qpm = max(scores, key=scores.get)
    patches, _, _ = _decode(
        model, tokenizer, first_spq=60.0 * FRAMES_PER_SECOND / first_qpm, max_measures=max_measures,
        stop_frame=None, temperature=temperature, top_p=top_p, **common,
    )  # fmt: skip
    return patches


def _decode(
    model: Perf2ScoreModel,
    tokenizer: ScoreTokenizer,
    *,
    memory,
    end_frame: float,
    first_spq: float,
    max_measures: int,
    context_measures: int,
    stop_frame: float | None,
    temperature: float,
    top_p: float,
    banned: torch.Tensor | None,
) -> tuple[list[list[int]], float, float]:
    """小節を 1 つずつ生成する。(小節ごとのトークン列, 選んだトークンの対数尤度の和, 生成した小節が覆う時間 (フレーム))。
    stop_frame を渡すと、そこより後に始まる小節の手前で止める"""
    device = memory.song.device
    decoder = model.decoder
    c = model.config
    song = torch.zeros(1, dtype=torch.long, device=device)
    mtime_values = torch.from_numpy(tokenizer.mtime_values).float() * FRAMES_PER_SECOND
    mtime_first = tokenizer.ids[("mtime", 0)]

    patches: list[list[int]] = []
    summaries: list[torch.Tensor] = []
    starts: list[float] = []  # 生成した各小節の開始 (フレーム)
    refs: list[float] = []
    lengths: list[float] = []
    open_slurs = (0, 0)
    total_logprob = 0.0
    covered = 0.0
    for p in range(max_measures):
        ref = starts[-1] if p > 0 else 0.0
        refs.append(ref)
        first = max(0, p - context_measures + 1)
        stacked = torch.stack(summaries[first:p] + [torch.zeros(1, decoder.config.dim, device=device)], dim=1)
        pedal = torch.zeros(1, stacked.shape[1], dtype=torch.long, device=device)
        song_start = torch.tensor([int(first == 0)], device=device)
        channel = torch.zeros(1, dtype=torch.long, device=device)
        global_pos = torch.tensor([refs[first : p + 1]], device=device) / 200.0
        delta = memory.song_pos[:, None, :] - global_pos[:, :, None]
        global_mask = (delta >= -c.global_back) & (delta <= c.global_ahead) & memory.song_valid[:, None, :]

        def global_hook(i: int, x: torch.Tensor) -> torch.Tensor:
            if i not in model.global_cross_blocks:
                return x
            block = model.global_cross[model.global_cross_blocks.index(i)]
            return block(x, global_pos, memory.song, memory.song_pos, global_mask)

        context = decoder.global_forward(stacked, song_start, pedal, channel, global_hook)[:, -1]

        ref_t = torch.tensor([ref], device=device)
        rows, k_pos, valid = model.local_memory(memory, song, ref_t)
        mask = valid[:, None, :]
        grammar = ScoreGrammar(tokenizer, open_slurs, banned)
        sequence = torch.zeros(1, 0, dtype=torch.long, device=device)
        start = torch.tensor([ref], device=device)
        spq = torch.tensor([first_spq], device=device)
        measure_logprob = 0.0
        while not grammar.finished:
            q_pos = model.query_times(sequence, ref_t, start, spq)

            def local_hook(i: int, x: torch.Tensor, q_pos: torch.Tensor = q_pos) -> torch.Tensor:
                if i not in model.local_cross_blocks:
                    return x
                block = model.local_cross[model.local_cross_blocks.index(i)]
                return block(x, q_pos, rows, k_pos, mask)

            logits = decoder.local_forward(sequence, context, local_hook)[:, -1].float()
            allowed = grammar.allowed().to(device)[None]
            logits = logits.masked_fill(~allowed, float("-inf"))
            token = _sample(logits, temperature, top_p)
            measure_logprob += float(logits.log_softmax(-1)[0, token])
            grammar.update(int(token))
            sequence = torch.cat((sequence, token[:, None]), dim=1)
            if sequence.shape[1] == 1:
                # MTIME から小節の開始と 4 分音符の長さの見積もりを決める (学習の data.py と同じ)
                value = float(mtime_values[int(token) - mtime_first])
                start_value = -value if p == 0 else ref + value
                start = torch.tensor([start_value], device=device)
                if p > 0:
                    spq = torch.tensor([(start_value - ref) / max(lengths[-1], 1e-3)], device=device)
        tokens = [t for t in sequence[0].tolist() if t != PAD]
        measure_start = float(start)
        if p > 0:
            covered = measure_start - starts[0]  # 前の小節までが覆う時間
        if p > 0 and measure_start > end_frame + 50:
            # 演奏の最後の音より後に始まる小節は作らず、前の小節で曲を終える
            patches[-1] = patches[-1][:-1] + [EOS]
            break
        if stop_frame is not None and p > 0 and measure_start > stop_frame:
            break
        patches.append(tokens)
        starts.append(measure_start)
        total_logprob += measure_logprob
        open_slurs = tuple(grammar.open_slurs)
        decoded, _ = tokenizer.decode(patches[-1:])
        lengths.append(float(decoded[0].length) if decoded else 4.0)
        summaries.append(decoder.summarize_patches(sequence))
        if grammar.song_end:
            covered = max(covered, end_frame - starts[0])
            break
    return patches, total_logprob, covered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--midi", default=None, help="楽譜にする演奏の MIDI")
    parser.add_argument("--out", default=None, help="--midi のときの出力 (省略で MIDI と同じ名前の .musicxml)")
    parser.add_argument("--val-songs", type=int, default=0, help="検証曲を合成演奏にして楽譜に戻す曲数")
    parser.add_argument("--val-measures", type=int, default=16, help="--val-songs で使う冒頭の小節数")
    parser.add_argument("--cache-dir", default="data/piano_score/synth")
    parser.add_argument("--out-dir", default="outputs/perf2score")
    parser.add_argument("--max-measures", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--first-qpm", type=float, default=None, help="最初の小節のテンポ (省略でモデルに選ばせる)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)
    banned = unseen_tokens(checkpoint["token_counts"], tokenizer) if checkpoint.get("token_counts") is not None else None
    options = {"temperature": args.temperature, "top_p": args.top_p, "banned": banned, "first_qpm": args.first_qpm}

    def run(performance: Performance, max_measures: int) -> list[list[int]]:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return transcribe(model, tokenizer, performance, max_measures=max_measures, **options)

    if args.midi:
        patches = run(read_performance(args.midi), args.max_measures)
        out = Path(args.out or Path(args.midi).with_suffix(".musicxml"))
        write_musicxml(tokenizer.decode(patches)[0], out)
        print(f"{len(patches)} 小節 -> {out}")
    if args.val_songs:
        import pretty_midi

        from piano_score.data import ScoreCache

        from .data import SynthWindowDataset

        cache = ScoreCache(args.cache_dir)
        dataset = SynthWindowDataset(cache, split="val", window_measures=args.val_measures, context_measures=0)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for k in range(args.val_songs):
            song = int(dataset.songs[k])
            n = min(cache.num_measures(song), args.val_measures)
            bodies = cache.measure_tokens(song, 0, n)
            reference, _ = tokenizer.decode([[tokenizer.ids[("mtime", 0)], *b.tolist()] for b in bodies])
            performance = render(
                reference, np.random.default_rng(song), qpm=measure_qpm(reference, cache.song_seconds(song)[: n + 1])
            )
            patches = run(performance, n + 4)
            name = f"val{k}_{cache.ids[song]}"
            write_musicxml(reference, out_dir / f"{name}_reference.musicxml")
            write_musicxml(tokenizer.decode(patches)[0], out_dir / f"{name}_generated.musicxml")
            midi = pretty_midi.PrettyMIDI()
            piano = pretty_midi.Instrument(0)
            piano.notes = [pretty_midi.Note(int(v), int(p), float(a), float(b)) for a, b, p, v in performance.notes]
            piano.control_changes = [
                pretty_midi.ControlChange(64, 127 if d else 0, float(max(t, 0))) for t, d in performance.pedal
            ]
            midi.instruments.append(piano)
            midi.write(str(out_dir / f"{name}.mid"))
            print(f"{name}: 元 {n} 小節 / 生成 {len(patches)} 小節")


if __name__ == "__main__":
    main()
