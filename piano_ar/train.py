"""ピアノ生成の事前学習。

    python -m piano_ar.train --cache-dir data/piano_ar/pretraining --out-dir checkpoints/piano_ar --wandb-project piano-ar

途中から再開する場合は --resume checkpoints/piano_ar/latest.pt (wandb も同じ run に続けて記録する)。
--init-from は学習済みの重みから新しく学習を始める (optimizer と学習率の予定は新しくする)。チャンネル数が増えたキャッシュでも、
既存のチャンネルの埋め込みはそのまま使い、増えた分だけ新しく作る (channel_index は追記だけなので番号は変わらない)。

--memory-seconds の長さの記憶を窓の前に付ける。記憶のパッチは要約を勾配なしで作って Global に入れるだけなので、
曲全体を文脈にしても学習の重さはほとんど変わらない (PianoARModel.forward)。検証では曲の途中 (記憶の長さの位置) から
始まる窓で、記憶ありとなしの損失を比べる (val_long / val_long_nomem)。

--dynamics-bins の段階で、パッチごとの強さと音の多さ (曲の中で標準化したもの) を条件に入れる (patch_dynamics)。
--dynamics-dropout の確率で外して学習するので、条件なしでも生成できる。
検証の損失がそれまでで最も低くなったら out-dir/best.pt にも保存する。
--sample-every ごとに曲を生成して out-dir/samples に MIDI で保存し、wandb には音声とピアノロールを送る。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import AugmentConfig, PianoWindowDataset, PretrainingCache, collate, make_sampler
from .evaluation import piano_roll_image, sample_stats, synthesize
from .model import PianoARModel
from .tokenizer import PAD, TOKEN_GROUPS, PianoTokenizer


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default="data/piano_ar/pretraining")
    parser.add_argument("--out-dir", default="checkpoints/piano_ar")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init-from", default=None, help="学習済みのチェックポイントの重みから始める (モデルの大きさもそれに合わせる)"
    )
    parser.add_argument("--window-seconds", type=float, default=64.0, help="損失を取る窓の長さ")
    parser.add_argument(
        "--memory-seconds", type=float, default=192.0, help="窓の前に付ける記憶 (文脈) の長さ。窓と合わせて曲全体を見る"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50_000, help="optimizer の更新回数")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--song-start-prob", type=float, default=0.1, help="窓を曲の冒頭から切り出す確率")
    parser.add_argument("--channel-dropout", type=float, default=0.15, help="チャンネル条件を落とす確率")
    parser.add_argument("--dynamics-dropout", type=float, default=0.2, help="強弱と音の多さの条件を落とす確率")
    parser.add_argument(
        "--channel-alpha",
        type=float,
        default=0.5,
        help="曲の多い演奏者をどれだけ抑えるか (1 で抑えない、0 でしきい値の時間まで)",
    )
    parser.add_argument(
        "--performer-balance-hours", type=float, default=10.0, help="これより総時間の長い演奏者だけを抑える"
    )
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--source-weights",
        default="",
        help="データセットごとの選ばれやすさの倍率。例: maestro=0.5,pijama=1.5 (名前は prepare の sources)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument(
        "--no-grad-checkpoint", action="store_true", help="gradient checkpointing を切る (速いがメモリを大きく使う)"
    )
    parser.add_argument("--length-buckets", type=int, default=8, help="パッチを長さ順に何個の塊に分けて処理するか")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Transformer を torch.compile する (triton-windows と、日本語版 Windows では PYTHONUTF8=1 が必要)",
    )
    parser.add_argument(
        "--gpu-memory-limit",
        type=float,
        default=0.9,
        help="起動時の空き VRAM のうち使ってよい割合。超えたらメモリ不足エラーにする (0 で無制限)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    # 学習中の生成評価
    parser.add_argument("--sample-every", type=int, default=5000, help="0 で生成評価をしない")
    parser.add_argument("--sample-count", type=int, default=2, help="無条件生成と続き生成、それぞれの本数")
    parser.add_argument("--sample-seconds", type=float, default=30.0)
    parser.add_argument("--prompt-seconds", type=float, default=8.0, help="続き生成で検証曲の冒頭から与える長さ")
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    # モデルの大きさ (ModelConfig の各項目を --dim などで上書きできる)
    for field in fields(ModelConfig):
        if field.name != "num_channels":
            parser.add_argument(f"--{field.name.replace('_', '-')}", type=type(field.default), default=field.default)
    # ModelConfig の既定値 0 は以前のチェックポイントを読むためのもの。新しく学習するときは強弱の条件を入れる
    parser.set_defaults(dynamics_bins=24)
    return parser.parse_args()


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = min(1.0, (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps))
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def make_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    # 行列の重み (線形層・埋め込み) だけ weight decay をかけ、正規化層のスケールやバイアスにはかけない
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (decay if param.dim() >= 2 and "norm" not in name else no_decay).append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )


def context_patches(args: dict | argparse.Namespace, patch_seconds: float) -> int:
    """生成で見る文脈のパッチ数 (学習の窓 + 記憶)。記憶のない以前のチェックポイントは窓だけ"""
    values = args if isinstance(args, dict) else vars(args)
    return round((values["window_seconds"] + values.get("memory_seconds", 0.0)) / patch_seconds)


def adapt_state(state: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """以前のチェックポイントの重みを、今の model に読めるようにする。

    チャンネルの埋め込みが model より少なければ、既存の分をそのまま使い、増えた分は model の初期値にする
    (channel_index は追記だけなので番号は変わらない)。あとから足した部分 (強弱の条件など) がなければ model の初期値
    (0 で始まるので、足す前と同じ出力) を使う。
    """
    state = dict(state)
    current = model.state_dict()
    key = "channel_embedding.weight"
    old, new = state[key], current[key]
    if old.shape[0] < new.shape[0]:
        state[key] = torch.cat([old.to(new.dtype), new[old.shape[0] :].to(old.device)])
        print(f"チャンネルの埋め込みを {old.shape[0]} -> {new.shape[0]} に増やした")
    for key in current.keys() - state.keys():
        if key.startswith("dynamics_embedding"):
            state[key] = current[key]
            print(f"{key} はチェックポイントにないので新しく作る")
    key = "dynamics_embedding.weight"
    if key in state and state[key].shape[0] < current[key].shape[0]:
        # 条件の列を後ろに足した: 既存の列はそのまま、足した列は model の初期値 (0) にする
        old = state[key]
        state[key] = torch.cat([old.to(current[key].dtype), current[key][old.shape[0] :].to(old.device)])
        print(f"強弱の条件の表を {old.shape[0]} -> {current[key].shape[0]} 行に増やした")
    return state


def limit_gpu_memory(fraction_of_free: float) -> None:
    """VRAM の使用量に上限を設ける。

    Windows のドライバは VRAM が足りなくなるとメインメモリ (共有 GPU メモリ) にあふれさせるため、
    PyTorch はメモリ不足のエラーを出さずに、RAM を大量に使いながら極端に遅く動き続けてしまう。
    上限を超えた確保を PyTorch 側で OutOfMemoryError にして、すぐに気付けるようにする。
    """
    if fraction_of_free <= 0 or not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(free * fraction_of_free / total)
    print(f"VRAM の上限 {free * fraction_of_free / 1e9:.1f}GB (空き {free / 1e9:.1f}GB / 全体 {total / 1e9:.1f}GB)")


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: PianoARModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(to_device(batch, device))
        tokens = int(output["tokens"])
        for key in ("loss", *[f"loss_{g}" for g in TOKEN_GROUPS]):
            sums[key] = sums.get(key, 0.0) + float(output[key]) * tokens
        count += tokens
    model.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


def sample_evaluation(
    model: PianoARModel,
    tokenizer: PianoTokenizer,
    val_set: PianoWindowDataset,
    args: argparse.Namespace,
    step: int,
    out_dir: Path,
    wandb_run,
) -> None:
    """無条件で冒頭から生成したものと、検証曲の冒頭をプロンプトにして続きを生成したものを残す。

    続き生成のプロンプトは毎回同じ検証曲を使うので、学習が進むにつれて同じ入力への応答がどう変わるかを比べられる。
    """
    c = tokenizer.config
    num_patches = math.ceil(args.sample_seconds / c.patch_seconds)
    prompt_patches = round(args.prompt_seconds / c.patch_seconds)
    cache = val_set.cache
    # 生成する長さより長い検証曲から、固定で選ぶ
    long_songs = [s for s in val_set.songs.tolist() if cache.end_frames[s] > num_patches * tokenizer.patch_frames]
    songs = long_songs[: args.sample_count]
    prompts = []
    for song in songs:
        window = tokenizer.tokenize_window(cache.song_events(song), int(cache.end_frames[song]), 0, prompt_patches)
        prompts.append([[t for t in row.tolist() if t != PAD] for row in window["tokens"]])

    model.eval()
    started = time.time()
    common = {
        "num_patches": num_patches,
        "temperature": args.sample_temperature,
        "top_p": args.sample_top_p,
        "context_patches": context_patches(args, c.patch_seconds),
    }
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        free = model.generate(tokenizer, num_samples=args.sample_count, **common)
        continued = model.generate(tokenizer, num_samples=len(prompts), prompts=prompts, **common) if prompts else []
    model.train()

    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    logs: dict[str, object] = {}
    stats: list[dict[str, float]] = []
    if wandb_run:
        import wandb

    def add(name: str, events: np.ndarray, prompt_frames: int = 0) -> None:
        path = sample_dir / f"step{step:07d}_{name}.mid"
        tokenizer.events_to_midi(events, path)
        if wandb_run:
            wandb_run.save(str(path), base_path=str(out_dir), policy="now")
            logs[f"samples/{name}_audio"] = wandb.Audio(synthesize(events, c.frame_rate), sample_rate=16000)
            logs[f"samples/{name}_roll"] = wandb.Image(piano_roll_image(events, tokenizer, prompt_frames=prompt_frames))

    for i, patches in enumerate(free):
        events = tokenizer.patches_to_events(patches)
        add(f"free{i}", events)
        stats.append(sample_stats(events, c.frame_rate))
    for i, (song, patches) in enumerate(zip(songs, continued)):
        add(f"continuation{i}", tokenizer.patches_to_events(patches), prompt_patches * tokenizer.patch_frames)
        # 比較用に同じ区間の元の曲も残す (最初の評価のときだけ)
        if not (sample_dir / f"reference{i}.mid").exists():
            window = tokenizer.tokenize_window(cache.song_events(song), int(cache.end_frames[song]), 0, num_patches)
            reference = tokenizer.patches_to_events([row.tolist() for row in window["tokens"]])
            tokenizer.events_to_midi(reference, sample_dir / f"reference{i}.mid")
            if wandb_run:
                logs[f"samples/reference{i}_audio"] = wandb.Audio(
                    synthesize(reference, c.frame_rate), sample_rate=16000
                )
                logs[f"samples/reference{i}_roll"] = wandb.Image(
                    piano_roll_image(reference, tokenizer, prompt_frames=prompt_patches * tokenizer.patch_frames)
                )

    # 無条件生成が学習データと似た傾向か (音数・音域・ペダルの量など) を数値でも追う
    mean_stats = {key: float(np.mean([s.get(key, 0.0) for s in stats])) for key in stats[0]} if stats else {}
    mean_stats["ended_rate"] = float(np.mean([len(p) < num_patches for p in free])) if free else 0.0
    logs.update({f"sample_stats/{k}": v for k, v in mean_stats.items()})
    print(
        f"[sample] step {step} {time.time() - started:.0f}s "
        + " ".join(f"{k} {v:.2f}" for k, v in mean_stats.items())
        + f" -> {sample_dir}"
    )
    if wandb_run:
        wandb_run.log(logs, step=step)


def reference_stats(val_set: PianoWindowDataset, seconds: float) -> dict[str, float]:
    """生成の統計量と比べる基準として、検証曲の冒頭 seconds 秒の統計量の平均を出す"""
    fr = val_set.tokenizer.config.frame_rate
    stats = []
    for song in val_set.songs.tolist():
        events = val_set.cache.song_events(song)
        stats.append(sample_stats(events[events[:, 0] < seconds * fr], fr))
    return {key: float(np.mean([s.get(key, 0.0) for s in stats])) for key in stats[0]} if stats else {}


def main() -> None:
    args = build_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    limit_gpu_memory(args.gpu_memory_limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = PretrainingCache(args.cache_dir)
    tokenizer = PianoTokenizer(cache.tokenizer_config)
    window_patches = round(args.window_seconds / cache.tokenizer_config.patch_seconds)
    memory_patches = round(args.memory_seconds / cache.tokenizer_config.patch_seconds)
    model_config = ModelConfig(
        **{f.name: getattr(args, f.name) for f in fields(ModelConfig) if f.name != "num_channels"},
        num_channels=cache.meta["num_channels"],
    )
    init = None
    if args.init_from and not args.resume:
        init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model_config = replace(
            ModelConfig.from_dict(init["model_config"]),
            num_channels=cache.meta["num_channels"],
            dynamics_bins=args.dynamics_bins,
        )
    source_weights = {
        name: float(value) for name, value in (item.split("=") for item in args.source_weights.split(",") if item)
    }

    train_set = PianoWindowDataset(
        cache,
        split="train",
        window_patches=window_patches,
        augment=None if args.no_augment else AugmentConfig(),
        song_start_prob=args.song_start_prob,
        channel_dropout=args.channel_dropout,
        memory_patches=memory_patches,
        dynamics_bins=model_config.dynamics_bins,
        dynamics_dropout=args.dynamics_dropout,
    )
    dynamics = {"dynamics_bins": model_config.dynamics_bins}
    val_set = PianoWindowDataset(
        cache, split="val", window_patches=window_patches, memory_patches=memory_patches, **dynamics
    )
    # 曲の途中から始まる窓で、記憶ありとなしを比べる (記憶がどれだけ効いているか)
    long_sets = (
        {
            name: PianoWindowDataset(
                cache,
                split="val",
                window_patches=window_patches,
                memory_patches=memory,
                val_start_patch=memory_patches,
                **dynamics,
            )
            for name, memory in (("long", memory_patches), ("long_nomem", 0))
        }
        if memory_patches
        else {}
    )
    if model_config.dynamics_bins:
        # 強弱の条件を外した検証 (条件がどれだけ効いているか)
        long_sets["nodyn"] = PianoWindowDataset(
            cache, split="val", window_patches=window_patches, memory_patches=memory_patches
        )
    micro_steps = args.steps * args.grad_accum
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=make_sampler(
            train_set, micro_steps * args.batch_size, args.channel_alpha, source_weights, args.performer_balance_hours
        ),
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
    long_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
        for name, ds in long_sets.items()
    }
    print(
        f"学習 {len(train_set)} 曲 / 検証 {len(val_set)} 曲 / 窓 {window_patches} パッチ + 記憶 {memory_patches} パッチ"
        f" / 語彙 {tokenizer.vocab_size}"
    )

    model = PianoARModel(model_config, tokenizer).to(device)
    model.set_gradient_checkpointing(not args.no_grad_checkpoint)
    print(f"パラメータ数 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    optimizer = make_optimizer(model, args)
    step = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        print(f"{args.resume} の step {step} から再開 (これまでの最良の検証損失 {best_val_loss:.4f})")
    elif init is not None:
        model.load_state_dict(adapt_state(init["model"], model))
        print(f"{args.init_from} (step {init['step']}) の重みから始める")
        del init
    if args.compile:
        # inductor は内部のテンプレートを既定の文字コードで読むため、日本語版 Windows (cp932) では失敗する
        if not sys.flags.utf8_mode:
            raise SystemExit(
                "--compile には Python の UTF-8 モードが必要です。PYTHONUTF8=1 を設定するか python -X utf8 で起動してください"
            )
        model.compile_blocks()

    wandb_run = None
    if args.wandb_project:
        import wandb

        # 再開時は同じ run に続けて記録する
        wandb_id = checkpoint.get("wandb_id") if args.resume else None
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=wandb_id,
            resume="allow" if wandb_id else None,
            config={**vars(args), "model": asdict(model_config), "tokenizer": asdict(cache.tokenizer_config)},
        )
        wandb_run.summary["parameters_M"] = sum(p.numel() for p in model.parameters()) / 1e6
        if args.sample_every and len(val_set):
            reference = reference_stats(val_set, args.sample_seconds)
            wandb_run.summary.update({f"reference_stats/{k}": v for k, v in reference.items()})
            print("[reference] " + " ".join(f"{k} {v:.2f}" for k, v in reference.items()))

    def save(path: Path) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "model_config": asdict(model_config),
                "tokenizer_config": asdict(cache.tokenizer_config),
                "channel_index": cache.meta.get("channel_index"),
                "args": vars(args),
                "wandb_id": wandb_run.id if wandb_run else None,
                "best_val_loss": best_val_loss,
            },
            path,
        )

    # 再開時は、サンプラーが作る曲の並びのうち済んだ分を読み飛ばさずに新しい乱数で続ける
    loader_iter = iter(train_loader)
    model.train()
    running: dict[str, float] = {}
    running_tokens = 0
    last_log = time.time()
    while step < args.steps:
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)
        for _ in range(args.grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(to_device(batch, device), args.length_buckets)
            (output["loss"] / args.grad_accum).backward()
            tokens = int(output["tokens"])
            running_tokens += tokens
            for key in ("loss", *[f"loss_{g}" for g in TOKEN_GROUPS]):
                running[key] = running.get(key, 0.0) + float(output[key].detach()) * tokens
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            elapsed = time.time() - last_log
            logs = {f"train/{k}": v / running_tokens for k, v in running.items()}
            logs.update({"train/lr": lr_at(step, args), "train/grad_norm": float(grad_norm)})
            logs["train/tokens_per_sec"] = running_tokens / elapsed
            print(
                f"step {step} loss {logs['train/loss']:.4f} "
                + " ".join(f"{g} {logs[f'train/loss_{g}']:.3f}" for g in TOKEN_GROUPS)
                + f" lr {logs['train/lr']:.2e} {logs['train/tokens_per_sec']:.0f} tok/s"
            )
            if wandb_run:
                wandb_run.log(logs, step=step)
            running, running_tokens, last_log = {}, 0, time.time()

        if args.val_every and step % args.val_every == 0 and len(val_set):
            val = evaluate(model, val_loader, device)
            extra = {name: evaluate(model, loader, device) for name, loader in long_loaders.items()}
            long = {name: result["loss"] for name, result in extra.items()}
            print(
                f"[val] step {step} loss {val['loss']:.4f} "
                + " ".join(f"{g} {val[f'loss_{g}']:.3f}" for g in TOKEN_GROUPS)
                + (f" | 曲の途中 記憶あり {long['long']:.4f} なし {long['long_nomem']:.4f}" if "long" in long else "")
                + (f" | 強弱の条件なし {long['nodyn']:.4f}" if "nodyn" in long else "")
            )
            if wandb_run:
                logs = {f"val/{k}": v for k, v in val.items()}
                logs.update({f"val/loss_{name}": v for name, v in long.items()})
                if "long" in long:
                    logs["val/memory_gain"] = long["long_nomem"] - long["long"]
                if "nodyn" in extra:
                    # 条件で主に変わるのはベロシティと音の数 (パッチの終わり・TIME) なので、それぞれの差も見る
                    logs["val/dynamics_gain"] = long["nodyn"] - val["loss"]
                    for g in ("velocity", "time", "end"):
                        logs[f"val/dynamics_gain_{g}"] = extra["nodyn"][f"loss_{g}"] - val[f"loss_{g}"]
                wandb_run.log(logs, step=step)
            if val["loss"] < best_val_loss:
                best_val_loss = val["loss"]
                save(out_dir / "best.pt")
                print(f"[val] 最良を更新したので best.pt に保存 (step {step})")
                if wandb_run:
                    wandb_run.summary.update({"best/val_loss": best_val_loss, "best/step": step})

        if args.sample_every and step % args.sample_every == 0:
            sample_evaluation(model, tokenizer, val_set, args, step, out_dir, wandb_run)

        if step % args.save_every == 0 or step == args.steps:
            save(out_dir / "latest.pt")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
