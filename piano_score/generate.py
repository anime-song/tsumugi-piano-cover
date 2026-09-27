"""学習したモデルで楽譜を生成して MusicXML に保存する。

    python -m piano_score.generate --checkpoint checkpoints/piano_score/best.pt --measures 32 --num-samples 2
    python -m piano_score.generate --checkpoint ... --prompt 曲.musicxml --prompt-measures 4

--prompt を渡すと、その楽譜の冒頭 --prompt-measures 小節をそのまま使って続きを生成する。
--render を付けると MuseScore で PNG にも書き出す (MuseScore 4 が入っている場合)。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from piano_ar.config import ModelConfig
from piano_ar.model import GenerationConditionSource, PianoARModel, _sample

from .config import ScoreTokenizerConfig
from .grammar import ScoreGrammar
from .musicxml import read_musicxml, write_musicxml
from .tokenizer import PAD, ScoreTokenizer

MUSESCORE_CANDIDATES = (
    "C:/Program Files/MuseScore 4/bin/MuseScore4.exe",
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    "mscore",
    "musescore",
)


@torch.no_grad()
def generate(
    model: PianoARModel,
    tokenizer: ScoreTokenizer,
    *,
    num_measures: int,
    num_samples: int = 1,
    temperature: float = 1.0,
    top_p: float = 0.95,
    context_measures: int = 32,
    prompts: list[list[list[int]]] | None = None,
    condition: GenerationConditionSource | None = None,
    banned: torch.Tensor | None = None,
) -> list[list[list[int]]]:
    """曲の冒頭から小節ごとに生成し、サンプルごとに小節のトークン列のリストを返す。

    context_measures を超えたら、学習時の「途中から始まる窓」と同じ形 (BOS(途中) + 直近の小節) で続ける。
    prompts[i] (小節ごとのトークン列) を渡すと、サンプル i の先頭の小節はそのトークンで埋めて続きを生成する。
    condition (演奏など) を渡すと cross-attention で条件を入れる。各トークンは ScoreGrammar で文法に合うものに絞る。
    banned [vocab] (学習データに出てこないトークンなど) は出さない (プロンプトの小節には効かない)。
    """
    device = model.token_embedding.weight.device
    bound = condition.bind([True] * num_samples) if condition is not None else None
    channels = torch.zeros(num_samples, dtype=torch.long, device=device)
    patches: list[list[list[int]]] = [[] for _ in range(num_samples)]
    summaries: list[torch.Tensor] = []
    open_slurs = [0] * num_samples
    done = [False] * num_samples

    for p in range(num_measures):
        if all(done):
            break
        first = max(0, p - context_measures + 1)
        # global_forward は最後の要約を捨てるので、ダミーを 1 つ足して位置 p まで計算する
        stacked = torch.stack(summaries[first:p] + [torch.zeros(num_samples, model.config.dim, device=device)], dim=1)
        pedal = torch.zeros(num_samples, stacked.shape[1], dtype=torch.long, device=device)
        song_start = torch.full((num_samples,), int(first == 0), dtype=torch.long, device=device)
        global_cross = bound.global_cross(first, p) if bound is not None else None
        context = model.global_forward(stacked, song_start, pedal, channels, global_cross)[:, -1]

        grammars = [ScoreGrammar(tokenizer, open_slurs[i], banned) for i in range(num_samples)]
        for i in range(num_samples):
            grammars[i].finished = done[i]
        forced = [prompts[i][p] if prompts is not None and p < len(prompts[i]) else None for i in range(num_samples)]
        sequence = torch.zeros(num_samples, 0, dtype=torch.long, device=device)
        while not all(g.finished for g in grammars):
            local_cross = bound.local_cross(p, sequence) if bound is not None else None
            logits = model.local_forward(sequence, context, local_cross)[:, -1].float()
            allowed = torch.stack([g.allowed() for g in grammars]).to(device)
            next_token = _sample(logits.masked_fill(~allowed, float("-inf")), temperature, top_p)
            step = sequence.shape[1]
            for i, prompt in enumerate(forced):
                if prompt is not None and not grammars[i].finished:
                    next_token[i] = prompt[step] if step < len(prompt) else PAD
            for i, g in enumerate(grammars):
                g.update(int(next_token[i]))
            sequence = torch.cat((sequence, next_token[:, None]), dim=1)

        for i, g in enumerate(grammars):
            if done[i]:
                continue
            patches[i].append([t for t in sequence[i].tolist() if t != PAD])
            open_slurs[i] = g.open_slurs
            done[i] = g.song_end
        summaries.append(model.summarize_patches(sequence))
    return patches


def unseen_tokens(counts: np.ndarray, tokenizer: ScoreTokenizer) -> torch.Tensor:
    """学習データに一度も出てこないトークン [vocab] (bool)。MTIME は学習時に入れるのでキャッシュの回数によらず残す"""
    banned = torch.from_numpy(np.asarray(counts) == 0)
    banned[[i for i, kind in enumerate(tokenizer.kinds) if kind in ("mtime", "special")]] = False
    return banned


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[PianoARModel, ScoreTokenizer, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint["tokenizer_config"])
    config["fraction_denominators"] = tuple(config["fraction_denominators"])
    tokenizer = ScoreTokenizer(ScoreTokenizerConfig(**config))
    model = PianoARModel(ModelConfig(**checkpoint["model_config"]), tokenizer).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model, tokenizer, checkpoint


def score_stats(patches: list[list[int]], tokenizer: ScoreTokenizer) -> dict[str, float]:
    """生成した楽譜の傾向 (学習データと比べる用)"""
    measures, _ = tokenizer.decode(patches)
    if not measures:
        return {}
    groups = [g for m in measures for g in m.groups]
    notes = sum(len(g.notes) for g in groups)
    voices = [len({(g.staff, g.voice) for g in m.groups}) for m in measures]
    return {
        "measures": len(measures),
        "notes_per_measure": notes / len(measures),
        "chord_size": notes / max(len(groups), 1),
        "voices_per_measure": float(np.mean(voices)),
        "tuplet_rate": sum(g.duration.tuplet is not None for g in groups) / max(len(groups), 1),
        "tokens_per_measure": float(np.mean([len(p) for p in patches])),
        "empty_measure_rate": float(np.mean([not m.groups for m in measures])),
    }


def find_musescore() -> str | None:
    for candidate in MUSESCORE_CANDIDATES:
        if Path(candidate).exists() or shutil.which(candidate):
            return candidate
    return None


def render_png(musicxml: Path, musescore: str, timeout: float = 180.0) -> list[Path]:
    """MuseScore で MusicXML をページごとの PNG (xxx-1.png, xxx-2.png, ...) にする。失敗したら空"""
    target = musicxml.with_suffix(".png")
    try:
        subprocess.run([musescore, "-o", str(target), str(musicxml)], check=True, timeout=timeout, capture_output=True)
    except (subprocess.SubprocessError, OSError):
        return []
    return sorted(musicxml.parent.glob(f"{musicxml.stem}-*.png"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="outputs/piano_score")
    parser.add_argument("--measures", type=int, default=32, help="生成する小節数の上限 (EOS が出たらそこで終わる)")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--prompt", default=None, help="冒頭をそのまま使う楽譜 (MusicXML)")
    parser.add_argument("--prompt-measures", type=int, default=4)
    parser.add_argument("--context-measures", type=int, default=None, help="Global が見る小節数。省略で学習時の窓")
    parser.add_argument("--render", action="store_true", help="MuseScore で PNG にも書き出す")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)
    prompts = None
    if args.prompt:
        prompt = tokenizer.encode(read_musicxml(args.prompt))[: args.prompt_measures]
        prompts = [prompt] * args.num_samples
    banned = unseen_tokens(checkpoint["token_counts"], tokenizer) if "token_counts" in checkpoint else None
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        samples = generate(
            model,
            tokenizer,
            num_measures=args.measures,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            context_measures=args.context_measures or checkpoint["args"]["window_measures"],
            prompts=prompts,
            banned=banned,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    musescore = find_musescore() if args.render else None
    for i, patches in enumerate(samples):
        measures, _ = tokenizer.decode(patches)
        path = out_dir / f"{Path(args.checkpoint).stem}_{i}.musicxml"
        write_musicxml(measures, path)
        stats = score_stats(patches, tokenizer)
        print(f"{path}: " + " ".join(f"{k} {v:.2f}" for k, v in stats.items()))
        if musescore:
            for png in render_png(path, musescore):
                print(f"  {png}")


if __name__ == "__main__":
    main()
