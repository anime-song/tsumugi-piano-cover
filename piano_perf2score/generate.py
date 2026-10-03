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
    いちばん確率の高いトークンで生成し (テンポごとにバッチの 1 行として同時に)、1 秒あたりの対数尤度がいちばん高い
    テンポを選ぶ。演奏だけではテンポの倍・半分 (8 分で書くか 16 分で書くか) が決まらないので、楽譜としてどちらが
    自然かをモデルの確率で決める。
    """
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in performance_batch(performance).items()}
    memory = model.encode_performance(batch)
    end_frame = float(performance.notes[:, 0].max() * FRAMES_PER_SECOND) if len(performance.notes) else 0.0
    common = {"memory": memory, "end_frame": end_frame, "context_measures": context_measures, "banned": banned}
    if first_qpm is None:
        spq = [60.0 * FRAMES_PER_SECOND / qpm for qpm in SEARCH_TEMPOS]
        _, logprob, covered = _decode(
            model, tokenizer, first_spq=spq, max_measures=max_measures,
            stop_frame=search_seconds * FRAMES_PER_SECOND, temperature=0.0, top_p=1.0, **common,
        )  # fmt: skip
        scores = [lp / max(cv, 1.0) for lp, cv in zip(logprob, covered)]
        first_qpm = SEARCH_TEMPOS[int(np.argmax(scores))]
    patches, _, _ = _decode(
        model, tokenizer, first_spq=[60.0 * FRAMES_PER_SECOND / first_qpm], max_measures=max_measures,
        stop_frame=None, temperature=temperature, top_p=top_p, **common,
    )  # fmt: skip
    return patches[0]


class _StepCondition:
    """Local を 1 トークンずつ生成するときの cross-attention (piano_ar.model.StepCondition)。演奏の行の k / v は
    小節ごとに作り置き (tensors)、onset には各行のクエリの時刻 (基準の時刻からの相対フレーム) を渡す"""

    def __init__(self, model: Perf2ScoreModel) -> None:
        self.model = model

    def tensors(self, rows: torch.Tensor, k_pos: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        tensors = {"mask": valid[:, None, :]}
        for n, block in enumerate(self.model.local_cross):
            k, v = getattr(block, "_orig_mod", block).memory_kv(rows, k_pos)
            tensors[f"k{n}"], tensors[f"v{n}"] = k, v
        return tensors

    def step_cross(self, tensors: dict, i: int, x: torch.Tensor, onset: torch.Tensor) -> torch.Tensor:
        model = self.model
        if i not in model.local_cross_blocks:
            return x
        n = model.local_cross_blocks.index(i)
        block = getattr(model.local_cross[n], "_orig_mod", model.local_cross[n])
        return block.attend(x, onset[:, None], tensors[f"k{n}"], tensors[f"v{n}"], tensors["mask"])

    def step_output(self, tensors: dict, h: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return logits


def _decode(
    model: Perf2ScoreModel,
    tokenizer: ScoreTokenizer,
    *,
    memory,
    end_frame: float,
    first_spq: list[float],
    max_measures: int,
    context_measures: int,
    stop_frame: float | None,
    temperature: float,
    top_p: float,
    banned: torch.Tensor | None,
) -> tuple[list[list[list[int]]], list[float], list[float]]:
    """小節を 1 つずつ生成する。first_spq の数だけの行 (曲の最初の小節の 4 分音符の長さだけが違う) を、バッチとして同時に
    進める。Local は 1 トークンずつ KV cache で計算し、演奏の行の k / v は小節ごとに作り置く。文法は行ごとに CPU で確かめる。
    行ごとに (小節ごとのトークン列, 選んだトークンの対数尤度の和, 生成した小節が覆う時間 (フレーム)) を返す。
    stop_frame を渡すと、そこより後に始まる小節の手前でその行を止める"""
    device = memory.song.device
    decoder = model.decoder
    c = model.config
    R = len(first_spq)
    L = tokenizer.config.max_patch_tokens
    rows_index = torch.zeros(R, dtype=torch.long, device=device)
    mtime_values = tokenizer.mtime_values * FRAMES_PER_SECOND
    mtime_first = tokenizer.ids[("mtime", 0)]
    condition = _StepCondition(model)
    transformer = decoder.local_transformer
    block0 = getattr(transformer.blocks[0], "_orig_mod", transformer.blocks[0])
    autocast = device.type == "cuda" and torch.is_autocast_enabled("cuda")
    dtype = torch.get_autocast_dtype("cuda") if autocast else decoder.token_embedding.weight.dtype
    shape = (R, block0.heads, L, transformer.head_dim)
    cache = [
        (torch.zeros(shape, dtype=dtype, device=device), torch.zeros(shape, dtype=dtype, device=device))
        for _ in transformer.blocks
    ]
    song = memory.song.expand(R, -1, -1)
    song_pos, song_valid = memory.song_pos.expand(R, -1), memory.song_valid.expand(R, -1)

    patches: list[list[list[int]]] = [[] for _ in range(R)]
    summaries: list[torch.Tensor] = []
    starts: list[list[float]] = [[] for _ in range(R)]
    refs: list[list[float]] = [[] for _ in range(R)]
    lengths: list[list[float]] = [[] for _ in range(R)]
    open_slurs = [(0, 0)] * R
    active = [True] * R
    total_logprob = [0.0] * R
    covered = [0.0] * R
    for p in range(max_measures):
        if not any(active):
            break
        for r in range(R):
            refs[r].append(starts[r][-1] if p > 0 else 0.0)
        ref = [refs[r][-1] for r in range(R)]
        first = max(0, p - context_measures + 1)
        stacked = torch.stack(summaries[first:p] + [torch.zeros(R, decoder.config.dim, device=device)], dim=1)
        pedal = torch.zeros(R, stacked.shape[1], dtype=torch.long, device=device)
        song_start = torch.full((R,), int(first == 0), device=device)
        channel = torch.zeros(R, dtype=torch.long, device=device)
        global_pos = torch.tensor([refs[r][first : p + 1] for r in range(R)], device=device) / 200.0
        delta = song_pos[:, None, :] - global_pos[:, :, None]
        global_mask = (delta >= -c.global_back) & (delta <= c.global_ahead) & song_valid[:, None, :]

        def global_hook(i: int, x: torch.Tensor) -> torch.Tensor:
            if i not in model.global_cross_blocks:
                return x
            block = model.global_cross[model.global_cross_blocks.index(i)]
            return block(x, global_pos, song, song_pos, global_mask)

        context = decoder.global_forward(stacked, song_start, pedal, channel, global_hook)[:, -1]
        ref_t = torch.tensor(ref, device=device)
        tensors = condition.tensors(*model.local_memory(memory, rows_index, ref_t))

        grammars = [ScoreGrammar(tokenizer, open_slurs[r], banned) for r in range(R)]
        for r in range(R):
            grammars[r].finished = not active[r]
        inactive = torch.tensor([not a for a in active], device=device)
        sequence = torch.full((R, L), PAD, dtype=torch.long, device=device)
        token = torch.zeros(R, dtype=torch.long, device=device)
        step = torch.zeros(1, dtype=torch.long, device=device)
        start = list(ref)
        spq = [first_spq[r] if p == 0 else 0.0 for r in range(R)]
        measure_logprob = [0.0] * R
        for s in range(L):
            # 各行のクエリの時刻 (学習の Perf2ScoreModel.query_times と同じ。MTIME を予測する位置は 0)
            q = [0.0 if s == 0 else start[r] - ref[r] + float(grammars[r].position or 0) * spq[r] for r in range(R)]
            logits = decoder.local_step(
                token, step, context, cache, torch.tensor(q, device=device), condition, tensors
            ).float()
            allowed = torch.stack([g.allowed() for g in grammars]).to(device)
            logits = logits.masked_fill(~allowed, float("-inf"))
            next_token = _sample(logits, temperature, top_p)
            chosen = logits.log_softmax(-1).gather(1, next_token[:, None])[:, 0].tolist()
            values = next_token.tolist()
            for r, g in enumerate(grammars):
                if g.finished:
                    continue
                measure_logprob[r] += chosen[r]
                g.update(values[r])
                if s == 0:
                    # MTIME から小節の開始と 4 分音符の長さの見積もりを決める (学習の data.py と同じ)
                    value = float(mtime_values[values[r] - mtime_first])
                    start[r] = -value if p == 0 else ref[r] + value
                    if p > 0:
                        spq[r] = (start[r] - ref[r]) / max(lengths[r][-1], 1e-3)
            sequence[:, s] = torch.where(inactive, PAD, next_token)
            token = next_token
            step += 1
            if all(g.finished for g in grammars):
                break
        summaries.append(decoder.summarize_patches(sequence[:, : s + 1]))
        rows = sequence[:, : s + 1].tolist()
        for r in range(R):
            if not active[r]:
                continue
            tokens = [t for t in rows[r] if t != PAD]
            if p > 0:
                covered[r] = start[r] - starts[r][0]  # 前の小節までが覆う時間
            if p > 0 and start[r] > end_frame + 50:
                # 演奏の最後の音より後に始まる小節は作らず、前の小節で曲を終える
                patches[r][-1] = patches[r][-1][:-1] + [EOS]
                active[r] = False
                continue
            if stop_frame is not None and p > 0 and start[r] > stop_frame:
                active[r] = False
                continue
            patches[r].append(tokens)
            starts[r].append(start[r])
            total_logprob[r] += measure_logprob[r]
            open_slurs[r] = tuple(grammars[r].open_slurs)
            decoded, _ = tokenizer.decode([tokens])
            lengths[r].append(float(decoded[0].length) if decoded else 4.0)
            if grammars[r].song_end:
                covered[r] = max(covered[r], end_frame - starts[r][0])
                active[r] = False
        for r in range(R):
            # 止めた行も、他の行と小節の数をそろえて starts と lengths を埋めておく (次の小節の基準に使う)
            if len(starts[r]) <= p:
                starts[r].append(starts[r][-1] if starts[r] else 0.0)
                lengths[r].append(lengths[r][-1] if lengths[r] else 4.0)
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
