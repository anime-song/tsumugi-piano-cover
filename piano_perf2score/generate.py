"""演奏 (MIDI) から楽譜 (MusicXML) を生成する。

    python -m piano_perf2score.generate --checkpoint checkpoints/piano_perf2score/best.pt --midi 演奏.mid --out 楽譜.musicxml
    python -m piano_perf2score.generate --checkpoint ... --val-songs 4 --out-dir outputs/perf2score

--val-songs を付けると、2 段目のキャッシュの検証曲を合成演奏にして楽譜に戻し、元の楽譜と並べて保存する
(合成演奏 .mid・元の楽譜 _reference.musicxml・生成した楽譜 _generated.musicxml)。

小節を 1 つずつ生成する。小節 p を生成し始めるときに分かっているのは、それまでに生成した MTIME から戻した各小節の開始時刻
だけなので、学習と同じく「前の小節の開始」を基準にして演奏を見る (piano_perf2score.model)。
演奏の最後の音より後に始まる小節を生成したら、そこで止める。--temperature 0 (既定) なら毎回いちばん確率の高いトークンを選ぶ。

拍の上の演奏 (V2 のカバー、beat_input) は --beat-midi と --sheetsage で渡す。拍の格子から小節線が決まるので、
MTIME は与えた小節線から決めてモデルには予測させず (transcribe_on_beats)、最初の小節は拍が楽譜の何分音符かだけを選ぶ。
書き出す前に、小節をまたぐ記譜の不整合 (タイ・ペダル・強弱の繰り返し・調号と綴り) を直し (tidy.tidy_score)、
片手で弾けない和音を直す (playability.make_playable、--no-playable で外す)。
テンポ (メトロノーム記号と再生のテンポ) は、モデルの見積もりではなく原曲の小節線の秒から決める (beat_tempo)。
加線の多い所は、小節ごとに 8va などのオクターブ記号と音部記号を選び直して五線に近づける (staves.readable_staves)。
--chords に原曲の MIDI (Cover Studio の source.mid) を渡すと、tsumugi のコードマーカーをコード記号として置く
(chords.place_chords)。省略したときは --beat-midi にマーカーがあればそれを使う。
スラーは確率が半分を超えないと選ばれず少なくなるので、--slur-bias でスラーの始まりと終わりのトークンの logit に足せる。

    python -m piano_perf2score.generate --checkpoint checkpoints/piano_perf2score_beat/best.pt \\
        --beat-midi projects/<曲>/takes/t0001/cover.mid --sheetsage projects/<曲>/sheetsage.json --out 楽譜.musicxml
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
from piano_score.tokenizer import EOS, MLEN, PAD, SLUR_START, SLUR_STOP, ScoreTokenizer

from .beat_input import BEAT_SECONDS, melody_from_annotation, on_beats
from .beat_tempo import apply_bar_tempos
from .chords import place_chords, read_chord_markers
from .data import FRAMES_PER_SECOND, PerformanceVocab, collate, patch_rows
from .model import Perf2ScoreModel, perf2score_config
from .playability import make_playable, remove_unisons
from .render import Performance, measure_qpm, render
from .staves import readable_staves
from .tidy import tidy_score

PEDAL_THRESHOLD = 64


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[Perf2ScoreModel, ScoreTokenizer, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint["tokenizer_config"])
    config["fraction_denominators"] = tuple(config["fraction_denominators"])
    tokenizer = ScoreTokenizer(ScoreTokenizerConfig(**config))
    model = Perf2ScoreModel(
        ModelConfig.from_dict(checkpoint["model_config"]), perf2score_config(checkpoint["perf2score_config"]), tokenizer
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


def performance_batch(
    performance: Performance,
    max_rows: int = 160,
    vocab: PerformanceVocab | None = None,
    drop_durations: bool = False,
) -> dict[str, torch.Tensor]:
    """演奏を、モデルの演奏エンコーダに入れる形 (data.collate の perf_* と同じ、バッチ 1) にする。
    vocab はモデルのもの (メロディの印の列の有無)。drop_durations なら音の長さの列を消す"""
    vocab = vocab or PerformanceVocab()
    features, onset = vocab.rows(performance)
    if drop_durations:
        features[:, 3] = 0
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


# 拍の上の演奏の最初の小節で試す、1 拍の長さ (4 分音符いくつ分): 4 分・付点 4 分 (複合拍子)。
# 8 分は小節の拍が 6 つ以上 (6/8 を 8 分で数えた拍)、2 分は 2 つ以下 (2/2) のときだけ試す。
# 小節の長さを拍の数で決めるので、ここで選んだ値が楽譜の拍子になる。モデルの尤度は密な曲 (16 分の多いカバー) で
# 8 分を選びやすい (2/4 で 32 分音符を書く) ので、4 分を基本にし、他の値は 1 拍あたり SEARCH_MARGIN 以上よいときだけ選ぶ
SEARCH_BEAT_QUARTERS = (1.0, 1.5, 2.0, 0.5)
SEARCH_MARGIN = 0.5
BAR_MATCH_SECONDS = 0.1  # 格子の小節の頭と注釈の小節 (bars) の頭を同じとみなすずれ


@torch.no_grad()
def transcribe_on_beats(
    model: Perf2ScoreModel,
    tokenizer: ScoreTokenizer,
    performance: Performance,
    *,
    context_measures: int = 32,
    temperature: float = 0.0,
    top_p: float = 0.95,
    banned: torch.Tensor | None = None,
    search_measures: int = 8,
    drop_durations: bool = True,
    slur_bias: float = 0.0,
) -> list[list[int]]:
    """拍の時刻の演奏 (beat_input.on_beats、measure_starts に小節線) から楽譜を生成する。小節の開始は与えたものを使い、
    MTIME はそこから決める。拍子は performance.meters (SheetSage2-Refined の小節の拍子) があればそれに決め、1 拍は
    その拍子の分母の音符にする。なければ、1 拍が SEARCH_BEAT_QUARTERS のどれかとして冒頭 search_measures 小節を
    生成し、1 拍あたりの対数尤度がいちばん高いものを選ぶ (スラーの足し値は掛けずに比べる)。
    drop_durations なら音の長さを消して入れる (カバーの短い打鍵をそのまま音価にしない)"""
    device = next(model.parameters()).device
    batch = performance_batch(performance, vocab=model.vocab, drop_durations=drop_durations)
    batch = {k: v.to(device) for k, v in batch.items()}
    memory = model.encode_performance(batch)
    starts = np.asarray(performance.measure_starts, dtype=np.float64) * FRAMES_PER_SECOND
    end_frame = float(performance.notes[:, 0].max() * FRAMES_PER_SECOND) if len(performance.notes) else 0.0
    common = {"memory": memory, "end_frame": end_frame, "context_measures": context_measures, "banned": banned}
    common["slur_bias"] = slur_bias
    beat_frames = BEAT_SECONDS * FRAMES_PER_SECOND
    bar_beats = np.median(np.diff(starts)) / beat_frames if len(starts) > 1 else 4.0
    quarters = [q for q in SEARCH_BEAT_QUARTERS if (q != 0.5 or bar_beats >= 6) and (q != 2.0 or bar_beats <= 2)]
    meters = list(performance.meters) if len(performance.meters) == len(starts) else []
    if meters and meters[0] is not None:
        quarters = [4 / meters[0][1]]
        common["time_signatures"] = meters
    spq = [beat_frames / q for q in quarters]
    best = 0
    if len(starts) > 1 and len(spq) > 1:
        stop = float(starts[min(search_measures, len(starts) - 1)])
        _, logprob, covered = _decode(
            model, tokenizer, first_spq=spq, max_measures=len(starts), stop_frame=stop, temperature=0.0, top_p=1.0,
            given_starts=starts, **(common | {"slur_bias": 0.0}),
        )  # fmt: skip
        per_beat = np.array([lp / max(cv, 1.0) for lp, cv in zip(logprob, covered)]) * beat_frames
        per_beat[1:] -= SEARCH_MARGIN
        best = int(np.argmax(per_beat))
    patches, _, _ = _decode(
        model, tokenizer, first_spq=[spq[best]], max_measures=len(starts), stop_frame=None,
        temperature=temperature, top_p=top_p, given_starts=starts, **common,
    )  # fmt: skip
    return patches[0]


def beat_performance(midi: str | Path, annotation: dict, chords: list[tuple[float, str]] | None = None) -> Performance:
    """拍の上で作った演奏の MIDI (V2 のカバー) と、その拍 (SheetSage2-Refined の注釈) から、拍の時刻の演奏を作る。
    小節は注釈の小節の頭で区切り、音のある小節だけを残す。注釈のメロディから、メロディの印も付ける。
    chords (原曲の秒のコード、chords.read_chord_markers) を渡すと、拍の時刻にして performance.chords に入れる。
    残した小節の本当の小節線の秒 (と最後の小節の終わり) を performance.bar_seconds に、注釈の小節の拍子
    ((拍の数, 拍の音符)) を performance.meters に入れる"""
    from piano_ar.beat_tokenizer import BeatGrid, BeatTokenizer
    from piano_ar.config import TokenizerConfig
    from piano_ar.tokenizer import PianoTokenizer

    fr = FRAMES_PER_SECOND
    events, _ = PianoTokenizer(TokenizerConfig(frame_rate=fr)).midi_to_events(midi, trim=False)
    tokenizer = BeatTokenizer()
    first, last = float(events[:, 0].min()), float((events[:, 0] + events[:, 3]).max())
    grid = BeatGrid.build(annotation["beats"], annotation["downbeats"], fr, first, last, tokenizer.config.max_bar_beats)
    if grid is None:
        raise ValueError("拍が足りない")
    bar_frames = grid.beats[grid.bar_starts]  # 各小節の頭と、最後の小節の終わり
    keep = (bar_frames[1:] > first) & (bar_frames[:-1] <= last)
    performance, origin = on_beats(events, grid, bar_frames[:-1][keep], tokenizer)
    performance.melody = melody_from_annotation(performance, annotation, grid, origin)
    kept = np.flatnonzero(keep)
    performance.bar_seconds = np.append(bar_frames[kept], bar_frames[kept[-1] + 1]) / fr if len(kept) else np.zeros(0)
    bars = annotation.get("bars") or []
    if bars:
        bar_starts = np.array([float(b["start"]) for b in bars])
        nearest = [int(np.argmin(np.abs(bar_starts - t))) for t in bar_frames[kept] / fr]
        performance.meters = [
            (int(bars[i]["beats"]), int(bars[i]["unit"])) if abs(bar_starts[i] - t) < BAR_MATCH_SECONDS else None
            for i, t in zip(nearest, bar_frames[kept] / fr)
        ]
    if chords:
        times = grid.beat_position(np.array([t for t, _ in chords]) * fr) * BEAT_SECONDS - origin
        performance.chords = [(float(t), text) for t, (_, text) in zip(times, chords)]
    return performance


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
        """ScoreOnsetHead: MTIME を予測する位置では MTIME の候補、それ以降は BEAT の候補、拍の直後の行では FRAC の候補"""
        head = self.model.onset_head
        if head is None:
            return logits
        frac_rows = tensors.get("head_frac_rows")
        where = None
        if frac_rows is not None:
            n = frac_rows.nonzero().squeeze(1)
            where = (n, torch.zeros_like(n))
        return head(
            h, logits, tensors.get("head_mtime"), tensors.get("head_beat"),
            tensors["head_frac"][where[0]] if where is not None else None, where,
        ).to(logits.dtype)  # fmt: skip


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
    given_starts: np.ndarray | None = None,
    slur_bias: float = 0.0,
    time_signatures: list[tuple[int, int] | None] | None = None,
) -> tuple[list[list[list[int]]], list[float], list[float]]:
    """小節を 1 つずつ生成する。first_spq の数だけの行 (曲の最初の小節の 4 分音符の長さだけが違う) を、バッチとして同時に
    進める。Local は 1 トークンずつ KV cache で計算し、演奏の行の k / v は小節ごとに作り置く。文法は行ごとに CPU で確かめる。
    行ごとに (小節ごとのトークン列, 選んだトークンの対数尤度の和, 生成した小節が覆う時間 (フレーム)) を返す。
    stop_frame を渡すと、そこより後に始まる小節の手前でその行を止める。
    given_starts (各小節の開始のフレーム) を渡すと、MTIME はそこから決めて (予測させず)、小節の開始と基準の時刻もその値を
    そのまま使い (学習の data.exact_measure_times と同じ)、その小節の数だけ生成する。小節の長さも、小節線の間の拍の数 x
    行ごとの 1 拍の 4 分音符の長さ (最初の小節の first_spq から) に決め、その長さの拍子だけを選ばせて MLEN は使わせない
    (拍の時刻の演奏 beat_input 用。拍の数の決まった小節を、途中から別の拍子や短い小節で書かないように)。
    time_signatures (小節ごとの拍子、SheetSage2-Refined の小節の meter) を渡すと、その小節はその拍子に決める"""
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
    measure_quarters = None  # [小節][行] 決めた小節の長さ (4 分音符単位)。決めない小節は None
    if given_starts is not None:
        beat_frames = BEAT_SECONDS * FRAMES_PER_SECOND
        beats = np.round(np.diff(given_starts) / beat_frames).astype(int).tolist()
        beats.append(beats[-1] if beats else 0)
        measure_quarters = [[b * beat_frames / spq for spq in first_spq] if b > 0 else None for b in beats]
    ts_ids = np.array([i for (kind, _), i in tokenizer.ids.items() if kind == "ts"])
    ts_quarters = np.array([4 * tokenizer.values[i][0] / tokenizer.values[i][1] for i in ts_ids])
    given_ts = [None] * max_measures  # 小節ごとに決めた拍子のトークン
    for p, ts in enumerate((time_signatures or [])[:max_measures]):
        if ts is not None and ("ts", tuple(ts)) in tokenizer.ids:
            given_ts[p] = tokenizer.ids[("ts", tuple(ts))]
            measure_quarters[p] = [4 * ts[0] / ts[1]] * R
    total_logprob = [0.0] * R
    covered = [0.0] * R
    for p in range(max_measures):
        if given_starts is not None and p >= len(given_starts):
            break
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
        rows_memory, k_pos, valid, channels = model.local_window(memory, rows_index, ref_t)
        tensors = condition.tensors(rows_memory, k_pos, valid)
        head = model.onset_head
        if head is not None:
            smoothed = head.smoothed(k_pos, valid, channels)
            first_rows = torch.full((R,), p == 0, device=device)
            tensors["head_mtime"], _ = head.candidates(
                smoothed, torch.zeros(R, device=device), torch.ones(R, device=device), first_rows
            )

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
            if head is not None and s > 0:
                # 拍のトークンの直後の行 (小節の長さの拍は除く) は、その拍のあとの FRAC の候補を足す
                after_beat = [g.stage == "after_beat" and not g.finished for g in grammars]
                tensors.pop("head_frac_rows", None)
                if any(after_beat):
                    relative = torch.tensor([start[r] - ref[r] for r in range(R)], device=device)
                    beats = torch.tensor([float(g.beat) for g in grammars], device=device)
                    tensors["head_frac"] = head.frac_candidates(
                        smoothed, relative, torch.tensor(spq, device=device), beats
                    )
                    tensors["head_frac_rows"] = torch.tensor(after_beat, device=device)
            logits = decoder.local_step(
                token, step, context, cache, torch.tensor(q, device=device), condition, tensors
            ).float()
            if slur_bias:
                logits[:, [SLUR_START, SLUR_STOP]] += slur_bias
            masks = [g.allowed() for g in grammars]
            if measure_quarters is not None and measure_quarters[p] is not None:
                for r, g in enumerate(grammars):
                    fitting = ts_ids[np.abs(ts_quarters - measure_quarters[p][r]) < 1e-6]
                    if given_ts[p] is not None and masks[r][given_ts[p]]:
                        fitting = np.array([given_ts[p]])
                    if g.stage == "ts" and masks[r][fitting].any():
                        keep = torch.zeros_like(masks[r])
                        keep[fitting] = masks[r][fitting]
                        masks[r] = keep
                    elif g.stage == "header" and len(fitting):
                        masks[r][MLEN] = False
            allowed = torch.stack(masks).to(device)
            logits = logits.masked_fill(~allowed, float("-inf"))
            if s == 0 and given_starts is not None:
                # 与えた小節線から MTIME を決める (曲の最初の小節は最初の音より前の秒数、それ以降は前の小節の開始から)
                seconds = -given_starts[0] if p == 0 else given_starts[p] - given_starts[p - 1]
                forced = mtime_first + tokenizer.mtime_bin(max(float(seconds) / FRAMES_PER_SECOND, 0.0))
                next_token = torch.full((R,), forced, dtype=torch.long, device=device)
            else:
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
                    if given_starts is not None:
                        start[r] = float(given_starts[p])
                    if p > 0:
                        spq[r] = (start[r] - ref[r]) / max(lengths[r][-1], 1e-3)
            if head is not None and s == 0:
                # MTIME が決まったので、BEAT の候補をこの小節の開始と 4 分音符の長さから作る
                tensors.pop("head_mtime", None)
                relative = torch.tensor([start[r] - ref[r] for r in range(R)], device=device)
                _, tensors["head_beat"] = head.candidates(
                    smoothed, relative, torch.tensor(spq, device=device), first_rows
                )
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
    if given_starts is not None:
        # 与えた小節を全部作ったら、最後の小節で曲を終える
        for r in range(R):
            if active[r] and patches[r] and patches[r][-1][-1] != EOS:
                patches[r][-1] = patches[r][-1][:-1] + [EOS]
    return patches, total_logprob, covered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--midi", default=None, help="楽譜にする演奏の MIDI")
    parser.add_argument(
        "--beat-midi", default=None, help="拍の上で作った演奏の MIDI (V2 のカバー)。--sheetsage と一緒に"
    )
    parser.add_argument("--sheetsage", default=None, help="--beat-midi の拍 (SheetSage2-Refined の注釈の JSON)")
    parser.add_argument(
        "--slur-bias", type=float, default=0.0, help="--beat-midi で、スラーのトークンの logit に足す値"
    )
    parser.add_argument("--no-playable", action="store_true", help="--beat-midi で、片手で弾けない和音を直さない")
    parser.add_argument(
        "--chords", default=None, help="--beat-midi で、コードマーカーを読む MIDI (原曲。省略で --beat-midi)"
    )
    parser.add_argument("--out", default=None, help="--midi のときの出力 (省略で MIDI と同じ名前の .musicxml)")
    parser.add_argument("--val-songs", type=int, default=0, help="検証曲を合成演奏にして楽譜に戻す曲数")
    parser.add_argument("--val-measures", type=int, default=16, help="--val-songs で使う冒頭の小節数")
    parser.add_argument("--cache-dir", default="data/piano_score/synth")
    parser.add_argument("--true-tempo", action="store_true", help="--val-songs で、最初の小節の本当のテンポを渡す")
    parser.add_argument(
        "--val-real-cache", default=None, help="実演奏の組のキャッシュ (asap.py)。渡すと --val-songs はその検証の組を使う"
    )
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

    def run(performance: Performance, max_measures: int, first_qpm: float | None = None) -> list[list[int]]:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            given = {"first_qpm": first_qpm} if first_qpm is not None else {}
            return transcribe(model, tokenizer, performance, max_measures=max_measures, **(options | given))

    if args.beat_midi:
        import json

        annotation = json.loads(Path(args.sheetsage).read_text(encoding="utf-8"))
        chords = read_chord_markers(args.chords or args.beat_midi)
        performance = beat_performance(args.beat_midi, annotation, chords)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            patches = transcribe_on_beats(
                model, tokenizer, performance, temperature=args.temperature, top_p=args.top_p, banned=banned,
                slur_bias=args.slur_bias,
            )  # fmt: skip
        out = Path(args.out or Path(args.beat_midi).with_suffix(".musicxml"))
        measures = tokenizer.decode(patches)[0]
        tidied = tidy_score(measures)
        print(tidied)
        if not args.no_playable:
            print(make_playable(measures))
            print(f"両手で重なる同じ音 {remove_unisons(measures)} 個を消した")
        print(readable_staves(measures))
        apply_bar_tempos(measures, performance, tidied.rests)
        if performance.chords:
            print(f"コード記号 {place_chords(measures, performance, performance.chords, tidied.rests)} 個")
        write_musicxml(measures, out)
        print(f"{len(patches)} 小節 -> {out}")
    if args.midi:
        patches = run(read_performance(args.midi), args.max_measures)
        out = Path(args.out or Path(args.midi).with_suffix(".musicxml"))
        write_musicxml(tokenizer.decode(patches)[0], out)
        print(f"{len(patches)} 小節 -> {out}")
    if args.val_songs:
        import pretty_midi

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        items = real_val_items(args) if args.val_real_cache else synth_val_items(args)
        for k, (song_id, reference, performance) in enumerate(items):
            n = len(reference)
            first_qpm = None
            if args.true_tempo and len(performance.measure_starts) > 1:
                # 最初の小節の本当のテンポを渡す (テンポの選び方と、その後の採譜の良し悪しを分けて見る)
                span = float(performance.measure_starts[1] - performance.measure_starts[0])
                first_qpm = 60.0 * float(reference[0].length) / max(span, 1e-3)
            patches = run(performance, 4 * n, first_qpm)  # 拍子を短く読むと小節が増えるので多めに
            name = f"val{k}_{song_id}"
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


def synth_val_items(args: argparse.Namespace):
    """合成演奏の検証曲: (名前, 冒頭 val_measures 小節の楽譜, それを描き出した演奏)"""
    from piano_score.data import ScoreCache

    from .data import SynthWindowDataset

    cache = ScoreCache(args.cache_dir)
    tokenizer = ScoreTokenizer(cache.tokenizer_config)
    dataset = SynthWindowDataset(cache, split="val", window_measures=args.val_measures, context_measures=0)
    for k in range(min(args.val_songs, len(dataset))):
        song = int(dataset.songs[k])
        n = min(cache.num_measures(song), args.val_measures)
        bodies = cache.measure_tokens(song, 0, n)
        reference, _ = tokenizer.decode([[tokenizer.ids[("mtime", 0)], *b.tolist()] for b in bodies])
        performance = render(
            reference, np.random.default_rng(song), qpm=measure_qpm(reference, cache.song_seconds(song)[: n + 1])
        )
        yield str(cache.ids[song]), reference, performance


def real_val_items(args: argparse.Namespace):
    """実演奏の検証の組: (名前, 冒頭 val_measures 小節の楽譜, その小節までの演奏)。
    曲が偏らないよう、曲ごとに 1 つずつ順に選ぶ"""
    from .data import SEGMENT_MARGIN, PairCache, pedal_between

    cache = PairCache(args.val_real_cache)
    tokenizer = ScoreTokenizer(cache.tokenizer_config)
    by_piece: dict[str, list[int]] = {}
    for song in np.flatnonzero(cache.is_val):
        by_piece.setdefault(str(cache.pieces[song]), []).append(int(song))
    order = [songs[i] for i in range(max(map(len, by_piece.values()))) for songs in by_piece.values() if i < len(songs)]
    for song in order[: args.val_songs]:
        n = min(cache.num_measures(song), args.val_measures)
        bodies = cache.measure_tokens(song, 0, n)
        reference, _ = tokenizer.decode([[tokenizer.ids[("mtime", 0)], *b.tolist()] for b in bodies])
        starts = cache.song_starts(song)
        end = starts[n] - SEGMENT_MARGIN if n < len(starts) else np.inf
        notes, pedal = cache.performance(song)
        performance = Performance(notes[notes[:, 0] < end], pedal_between(pedal, -np.inf, end), starts[:n])
        song_id = str(cache.ids[song]).replace(".mid", "").replace("/", "_").replace("#", "_")
        yield song_id, reference, performance


if __name__ == "__main__":
    main()
