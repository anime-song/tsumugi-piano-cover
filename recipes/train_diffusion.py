from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from recipes.data.dataset import SegmentDiffusionDataset, collate_diffusion_samples
from recipes.generation_eval import evaluate_generation
from recipes.losses import diffusion_mse_loss
from recipes.train_common import (
    ExponentialMovingAverage,
    ResumeState,
    build_lr_scheduler,
    build_splits,
    clear_cuda_cache,
    get_autocast_context,
    load_training_checkpoint,
    move_batch_to_device,
    resolve_device,
    save_checkpoint,
    set_global_seed,
    skip_to_batch,
)
from recipes.wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb_images,
    log_wandb_metrics,
    update_wandb_summary,
)
from tsumugi_piano_cover.config import ExperimentConfig, load_experiment_config
from tsumugi_piano_cover.latent_scaling import LatentNormalizer
from tsumugi_piano_cover.models.diffusion import ConditionalSegmentDiffusionModel
from tsumugi_piano_cover.models.segment_autoencoder import SegmentLatentAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train source-conditioned segment-latent diffusion.")
    parser.add_argument("--config", default="configs/piano_cover_v2.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-train-pairs", type=int)
    parser.add_argument("--limit-val-pairs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--autoencoder-checkpoint")
    parser.add_argument("--latent-scale", type=float)
    parser.add_argument(
        "--latent-no-whiten",
        action="store_true",
        help="latent_std による次元ごとの白色化を使わず、スカラー latent_scale だけで割る",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name")
    return parser.parse_args()


def resolve_autoencoder_checkpoint(config: ExperimentConfig, override: str | None) -> Path:
    if override is not None:
        return Path(override).expanduser()
    if config.runtime.autoencoder_checkpoint is not None:
        return Path(config.runtime.autoencoder_checkpoint).expanduser()
    return Path(config.runtime.autoencoder_work_dir) / "vae" / "best.pt"


def load_frozen_autoencoder(
    config: ExperimentConfig, checkpoint_path: Path, device: torch.device
) -> tuple[SegmentLatentAutoencoder, dict[str, object]]:
    # オートエンコーダーの重みをフリーズ
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"autoencoder checkpoint not found: {checkpoint_path}")
    model = SegmentLatentAutoencoder(config.autoencoder_model, config.target_roll).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


class OnPolicyReplayBuffer:
    """Keep recent self-generated latent states per song without retaining computation graphs."""

    def __init__(self, max_songs: int) -> None:
        self.max_songs = int(max_songs)
        self._states: OrderedDict[str, list[tuple[torch.Tensor, int]]] = OrderedDict()

    def get(self, key: str) -> list[tuple[torch.Tensor, int]] | None:
        states = self._states.get(key)
        if states is not None:
            self._states.move_to_end(key)
        return states

    def put(self, key: str, states: list[tuple[torch.Tensor, int]]) -> None:
        # Replay tensors are detached and kept in CPU float16 so full-song training does not
        # retain GPU memory for every song in the dataset.
        stored = [(state.detach().to(device="cpu", dtype=torch.float16), timestep) for state, timestep in states]
        self._states[key] = stored
        self._states.move_to_end(key)
        while len(self._states) > self.max_songs:
            self._states.popitem(last=False)


def on_policy_batch_key(batch: dict[str, object]) -> str:
    metadata = batch.get("metadata")
    if not isinstance(metadata, list) or len(metadata) != 1 or not isinstance(metadata[0], tuple):
        raise ValueError("on-policy training requires batch_size=1 with song metadata")
    song_name, piano_id = metadata[0]
    return f"{song_name}::{piano_id}"


def on_policy_loss_counts(config: ExperimentConfig) -> tuple[int, int, int]:
    """Return standard, replay, and fresh loss counts for one optimizer accumulation unit."""
    total = int(config.diffusion_training.timesteps_per_sample)
    on_policy = config.diffusion_training.on_policy
    if not on_policy.enabled:
        return total, 0, 0
    replay_count = int(round(total * on_policy.replay_fraction))
    fresh_count = int(round(total * on_policy.fresh_fraction))
    standard_count = total - replay_count - fresh_count
    if standard_count < 1 or replay_count + fresh_count < 1:
        raise ValueError(
            "on-policy fractions must leave at least one standard sample and one on-policy sample: "
            f"timesteps_per_sample={total}, replay={replay_count}, fresh={fresh_count}"
        )
    return standard_count, replay_count, fresh_count


@torch.no_grad()
def collect_on_policy_states(
    model: ConditionalSegmentDiffusionModel,
    batch: dict[str, object],
    device: torch.device,
    config: ExperimentConfig,
    seed: int,
) -> list[tuple[torch.Tensor, int]]:
    """Capture self-generated x_t states at the late reverse-diffusion steps used for replay."""
    on_policy = config.diffusion_training.on_policy
    was_training = model.training
    model.eval()
    try:
        latent_shape = (
            1,
            int(batch["segment_mask"].shape[1]),  # type: ignore[index]
            config.diffusion_model.latent_dim,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        current = torch.randn(latent_shape, device=device, generator=generator)
        step_indices = torch.linspace(
            config.diffusion_model.num_train_timesteps - 1,
            0,
            steps=on_policy.sampling_steps,
            device=device,
        ).long()
        capture_positions: dict[int, int] = {}
        for requested_timestep in on_policy.capture_timesteps:
            position = int((step_indices - requested_timestep).abs().argmin().item())
            capture_positions[position] = int(step_indices[position].item())

        states: list[tuple[torch.Tensor, int]] = []
        with get_autocast_context(device, config.diffusion_training.mixed_precision):
            conditioning = model.prepare_conditioning(batch)  # type: ignore[arg-type]
            for step_position, timestep in enumerate(step_indices):
                if step_position in capture_positions:
                    states.append((current.float().detach().clone(), capture_positions[step_position]))
                timestep_batch = torch.full((1,), int(timestep.item()), device=device, dtype=torch.long)
                model_output = model.denoise(batch, current, timestep_batch, conditioning)  # type: ignore[arg-type]
                pred_x0, pred_noise = model.predict_x0_and_noise(current, model_output, timestep_batch)
                if step_position == len(step_indices) - 1:
                    break
                next_timestep = step_indices[step_position + 1]
                alpha_bar_next = model.alpha_bars[next_timestep]
                current = torch.sqrt(alpha_bar_next) * pred_x0 + torch.sqrt(1.0 - alpha_bar_next) * pred_noise
        if not states:
            raise RuntimeError("on-policy rollout did not capture any latent states")
        return states
    finally:
        model.train(was_training)


@torch.no_grad()
def on_policy_training_target(
    model: ConditionalSegmentDiffusionModel,
    clean_latents: torch.Tensor,
    noisy_latents: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Build the v-target that brings an off-policy x_t back to the clean training latent."""
    sqrt_alpha = model.sqrt_alpha_bars[timestep].view(-1, 1, 1).float()
    sqrt_sigma = model.sqrt_one_minus_alpha_bars[timestep].view(-1, 1, 1).float().clamp_min(1.0e-5)
    effective_noise = (noisy_latents.float() - sqrt_alpha * clean_latents.float()) / sqrt_sigma
    return model.compute_training_target(clean_latents, effective_noise, timestep)


def diffusion_latents_from_conditioning(
    latents: torch.Tensor,
    conditioning: dict[str, torch.Tensor | None],
) -> torch.Tensor:
    """Convert absolute target latents to residual latents when coarse latents are enabled."""
    coarse_latent = conditioning.get("coarse_latent")
    if coarse_latent is None:
        return latents
    return latents - coarse_latent


def build_validation_timesteps(config: ExperimentConfig) -> list[int]:
    """検証で毎回使う固定タイムステップを返す。

    タイムステップを乱択すると val loss が引いた t に強く依存し、エポック間で比較できず
    best チェックポイントの選択がほぼ運になる。各ノイズ帯域の中央を等間隔に固定する。
    """
    count = int(config.diffusion_training.val_timesteps)
    num_train_timesteps = int(config.diffusion_model.num_train_timesteps)
    return [min(num_train_timesteps - 1, int((index + 0.5) * num_train_timesteps / count)) for index in range(count)]


@torch.no_grad()
def run_validation(
    model: ConditionalSegmentDiffusionModel,
    autoencoder: SegmentLatentAutoencoder,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    epoch: int,
    normalizer: LatentNormalizer,
    validation_timesteps: list[int],
) -> tuple[float, dict[int, float]]:
    model.eval()
    totals = {timestep: 0.0 for timestep in validation_timesteps}
    batch_count = 0
    # 毎エポック同じノイズを引くための専用ジェネレータ
    generator = torch.Generator(device=device)

    val_progress = tqdm(loader, desc=f"Epoch {epoch} val", dynamic_ncols=True, leave=False)
    for batch_index, batch in enumerate(val_progress):
        batch = move_batch_to_device(batch, device)
        # ピアノカバーを潜在表現にエンコード
        latents, _, _ = autoencoder.encode(batch["target_segments"], sample_posterior=False)
        latents = normalizer.normalize(latents)

        with get_autocast_context(device, config.diffusion_training.mixed_precision):
            # 原曲エンコードは全タイムステップで共有する
            conditioning = model.prepare_conditioning(batch)
            diffusion_latents = diffusion_latents_from_conditioning(latents, conditioning)
            for timestep_index, timestep_value in enumerate(validation_timesteps):
                generator.manual_seed(config.seed * 1_000_003 + batch_index * 1009 + timestep_index)
                noise = torch.randn(
                    diffusion_latents.shape,
                    generator=generator,
                    device=device,
                    dtype=diffusion_latents.dtype,
                )
                timesteps = torch.full((latents.shape[0],), timestep_value, device=device, dtype=torch.long)

                noisy_latents = model.q_sample(diffusion_latents, timesteps, noise)
                model_output = model.denoise(batch, noisy_latents, timesteps, conditioning)
                target = model.compute_training_target(diffusion_latents, noise, timesteps)
                loss = diffusion_mse_loss(model_output, target, batch["segment_mask"])
                totals[timestep_value] += float(loss.detach().item())

        batch_count += 1
        val_progress.set_postfix(val_total=f"{sum(totals.values()) / (batch_count * len(validation_timesteps)):.4f}")

    if batch_count == 0:
        return float("nan"), {}
    per_timestep = {timestep: value / batch_count for timestep, value in totals.items()}
    return sum(per_timestep.values()) / len(per_timestep), per_timestep


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    set_global_seed(config.seed)
    if args.wandb:
        config.wandb.enabled = True
    if args.wandb_name is not None:
        config.wandb.name = args.wandb_name
    device = resolve_device(args.device)
    wandb_run = init_wandb_run(config, job_type="diffusion")

    try:
        train_pairs, val_pairs, test_pairs = build_splits(
            config,
            limit_train_pairs=args.limit_train_pairs,
            limit_val_pairs=args.limit_val_pairs,
        )
        # 演奏者IDの語彙サイズ。末尾の1つは無条件（null）埋め込み用にモデル側で確保される
        performer_vocab_size = max(pair.performer_id for pair in (train_pairs + val_pairs + test_pairs)) + 1
        null_performer_id = performer_vocab_size
        train_performer_ids = {pair.performer_id for pair in train_pairs}
        unseen_val_pairs = sum(1 for pair in val_pairs if pair.performer_id not in train_performer_ids)
        work_dir = Path(config.runtime.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # フリーズされたオートエンコーダーのロード
        autoencoder_checkpoint = resolve_autoencoder_checkpoint(config, args.autoencoder_checkpoint)
        autoencoder, autoencoder_payload = load_frozen_autoencoder(config, autoencoder_checkpoint, device)
        normalizer = LatentNormalizer.from_autoencoder_checkpoint(
            autoencoder_payload,
            scale_override=args.latent_scale,
            whiten=not args.latent_no_whiten,
        ).to(device)

        train_dataset = SegmentDiffusionDataset(
            train_pairs,
            config.target_roll,
            alignment_cache_dir=config.dataset.alignment_cache_dir,
            audio_sample_rate=config.tsumugi_model.sample_rate,
            audio_channels=config.tsumugi_model.audio_channels,
        )
        val_dataset = SegmentDiffusionDataset(
            val_pairs,
            config.target_roll,
            alignment_cache_dir=config.dataset.alignment_cache_dir,
            audio_sample_rate=config.tsumugi_model.sample_rate,
            audio_channels=config.tsumugi_model.audio_channels,
            # 学習に出てこない演奏者の埋め込みは未学習のままなので null へ寄せる
            known_performer_ids=train_performer_ids,
            unknown_performer_id=null_performer_id,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.diffusion_training.batch_size,
            shuffle=True,
            num_workers=config.runtime.num_workers,
            collate_fn=collate_diffusion_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.diffusion_training.batch_size,
            shuffle=False,
            num_workers=config.runtime.num_workers,
            collate_fn=collate_diffusion_samples,
        )

        # 生成チェック用に val の先頭数曲を固定で取り出しておく（毎エポック同じ曲を見る）
        generation_every = config.diffusion_training.generation_eval_every_epochs
        generation_batches: list[dict] = []
        if generation_every > 0:
            probe_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_diffusion_samples,
            )
            for probe_index, probe_batch in enumerate(probe_loader):
                if probe_index >= config.diffusion_training.generation_eval_songs:
                    break
                generation_batches.append(probe_batch)

        model = ConditionalSegmentDiffusionModel(
            config.source_model,
            config.diffusion_model,
            performer_vocab_size,
            config.tsumugi_model,
        ).to(device)
        model.set_gradient_checkpointing(config.diffusion_model.gradient_checkpointing)
        optimizer = AdamW(
            model.parameters(),
            lr=config.diffusion_training.learning_rate,
            weight_decay=config.diffusion_training.weight_decay,
        )
        ema = (
            ExponentialMovingAverage(model, config.diffusion_training.ema_decay)
            if config.diffusion_training.ema_enabled
            else None
        )

        # GradScalerの初期化
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and config.diffusion_training.mixed_precision == "fp16"
        )

        steps_per_epoch = math.ceil(len(train_loader) / config.diffusion_training.grad_accum_steps)
        total_steps = config.diffusion_training.max_steps
        if total_steps is None:
            total_steps = steps_per_epoch * config.diffusion_training.max_epochs

        scheduler = build_lr_scheduler(optimizer, config.diffusion_training, total_steps)
        timesteps_per_sample = int(config.diffusion_training.timesteps_per_sample)
        standard_loss_count, replay_loss_count, fresh_loss_count = on_policy_loss_counts(config)
        on_policy_config = config.diffusion_training.on_policy
        on_policy_replay = OnPolicyReplayBuffer(on_policy_config.replay_max_songs)
        # ある曲のロールアウトを何エポック使い回したか。global_step で数えると、
        # 1曲は1エポックに1回しか学習に出てこないので条件が常に成立してしまう
        on_policy_epochs_since_refresh: dict[str, int] = {}
        validation_timesteps = build_validation_timesteps(config)

        resume_state = ResumeState()
        if args.resume_from is not None:
            resume_state, extra_state = load_training_checkpoint(
                Path(args.resume_from).expanduser(),
                model,
                optimizer,
                scaler,
                device,
                scheduler=scheduler,
                ema=ema,
            )
            checkpoint_config = extra_state.get("experiment_config")
            if isinstance(checkpoint_config, dict):
                checkpoint_diffusion_config = checkpoint_config.get("diffusion_model")
                if isinstance(checkpoint_diffusion_config, dict):
                    checkpoint_prediction_type = str(checkpoint_diffusion_config.get("prediction_type", "epsilon"))
                    if checkpoint_prediction_type != config.diffusion_model.prediction_type:
                        raise ValueError(
                            "resume checkpoint prediction_type does not match current config: "
                            f"{checkpoint_prediction_type!r} != {config.diffusion_model.prediction_type!r}"
                        )
            checkpoint_vocab_size = extra_state.get("performer_vocab_size")
            if checkpoint_vocab_size is not None and int(checkpoint_vocab_size) != performer_vocab_size:
                raise ValueError(
                    "resume checkpoint performer_vocab_size does not match current dataset: "
                    f"{int(checkpoint_vocab_size)} != {performer_vocab_size}"
                )
            checkpoint_normalizer = LatentNormalizer.from_state_dict(extra_state)
            if not normalizer.matches(checkpoint_normalizer.to(device)):
                raise ValueError(
                    "resume checkpoint latent normalization does not match current setting: "
                    f"{checkpoint_normalizer.describe()} != {normalizer.describe()}"
                )

        print(f"experiment={config.experiment_name}")
        print(f"device={device}")
        print(f"seed={config.seed}")
        print(f"train_pairs={len(train_pairs)}")
        print(f"val_pairs={len(val_pairs)}")
        print(f"test_pairs={len(test_pairs)}")
        print(f"performer_vocab_size={performer_vocab_size} (null_performer_id={null_performer_id})")
        print(f"val_pairs_with_unseen_performer={unseen_val_pairs} (remapped to null_performer_id)")
        print(f"autoencoder_checkpoint={autoencoder_checkpoint}")
        print(f"latent_normalization={normalizer.describe()}")
        print(f"diffusion_parameters={sum(parameter.numel() for parameter in model.parameters())}")
        print(f"prediction_type={config.diffusion_model.prediction_type}")
        print(f"zero_terminal_snr={config.diffusion_model.zero_terminal_snr}")
        print(f"mixed_precision={config.diffusion_training.mixed_precision}")
        print(f"timesteps_per_sample={timesteps_per_sample}")
        print(
            "on_policy="
            f"{on_policy_config.enabled} standard={standard_loss_count} replay={replay_loss_count} "
            f"fresh={fresh_loss_count} refresh_every_epochs={on_policy_config.refresh_every_epochs} "
            f"sampling_steps={on_policy_config.sampling_steps}"
        )
        print(f"validation_timesteps={validation_timesteps}")
        if generation_every > 0:
            print(
                f"generation_eval: {len(generation_batches)}曲 / {generation_every}エポックごと / "
                f"{config.diffusion_training.generation_eval_sampling_steps}ステップ"
            )
        else:
            print("generation_eval=disabled")
        if ema is None:
            print("ema=disabled")
        else:
            print(f"ema_decay={config.diffusion_training.ema_decay}")
        print(
            f"gradient_checkpointing: denoiser={config.diffusion_model.gradient_checkpointing} tsumugi={config.tsumugi_model.gradient_checkpointing}"
        )
        print(f"optimizer_steps_per_epoch={steps_per_epoch}")
        if scheduler is None:
            print("lr_scheduler=none")
        else:
            warmup_epochs = config.diffusion_training.lr_warmup_steps / max(steps_per_epoch, 1)
            print(f"lr_scheduler={config.diffusion_training.lr_scheduler_type}")
            print(f"lr_warmup_steps={config.diffusion_training.lr_warmup_steps}")
            print(f"lr_warmup_epochs~={warmup_epochs:.2f}")
            print(f"total_optimizer_steps={total_steps}")

        update_wandb_summary(
            wandb_run,
            train_pairs=len(train_pairs),
            val_pairs=len(val_pairs),
            test_pairs=len(test_pairs),
            performer_vocab_size=performer_vocab_size,
            val_pairs_with_unseen_performer=unseen_val_pairs,
            autoencoder_checkpoint=str(autoencoder_checkpoint),
            latent_normalization=normalizer.describe(),
            diffusion_parameters=sum(parameter.numel() for parameter in model.parameters()),
            prediction_type=config.diffusion_model.prediction_type,
            zero_terminal_snr=config.diffusion_model.zero_terminal_snr,
            mixed_precision=config.diffusion_training.mixed_precision,
            timesteps_per_sample=timesteps_per_sample,
            on_policy_enabled=on_policy_config.enabled,
            on_policy_replay_fraction=on_policy_config.replay_fraction,
            on_policy_fresh_fraction=on_policy_config.fresh_fraction,
            on_policy_refresh_every_epochs=on_policy_config.refresh_every_epochs,
            on_policy_sampling_steps=on_policy_config.sampling_steps,
            ema_decay=config.diffusion_training.ema_decay if ema is not None else None,
            denoiser_gradient_checkpointing=config.diffusion_model.gradient_checkpointing,
            tsumugi_gradient_checkpointing=config.tsumugi_model.gradient_checkpointing,
            device=str(device),
            optimizer_steps_per_epoch=steps_per_epoch,
            total_optimizer_steps=total_steps,
        )

        if args.dry_run:
            return

        # 再開時のバッチ位置調整
        if resume_state.start_batch_index == len(train_loader):
            resume_state.start_epoch += 1
            resume_state.start_batch_index = 0
        if resume_state.start_batch_index > len(train_loader):
            raise ValueError(
                f"resume batch index {resume_state.start_batch_index} exceeds train loader length {len(train_loader)}"
            )

        global_step = resume_state.global_step
        best_val_loss = resume_state.best_val_loss
        optimizer.zero_grad(set_to_none=True)

        def checkpoint_extra_state() -> dict[str, object]:
            return {
                "autoencoder_checkpoint": str(autoencoder_checkpoint),
                "performer_vocab_size": performer_vocab_size,
                **normalizer.state_dict(),
            }

        def apply_optimizer_step() -> None:
            # 勾配クリップ → optimizer.step → scheduler / EMA 更新までを1か所にまとめる
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.diffusion_training.grad_clip_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        for epoch in range(resume_state.start_epoch, config.diffusion_training.max_epochs):
            model.train()
            running_total = 0.0
            running_standard = 0.0
            running_on_policy = 0.0
            batch_count = 0
            epoch_start_batch = resume_state.start_batch_index if epoch == resume_state.start_epoch else 0
            epoch_step_limit = config.diffusion_training.max_steps_per_epoch
            epoch_step_count = epoch_start_batch // config.diffusion_training.grad_accum_steps
            # 勾配累積の残量。batch_index に依存しないので途中再開でもズレない
            pending_accumulation = 0
            stopped_by_epoch_step_limit = False
            train_progress = tqdm(
                skip_to_batch(train_loader, epoch_start_batch),
                total=len(train_loader),
                initial=epoch_start_batch,
                desc=f"Epoch {epoch} train",
                dynamic_ncols=True,
            )

            for batch_index, batch in train_progress:
                if epoch_step_limit is not None and epoch_step_count >= epoch_step_limit:
                    stopped_by_epoch_step_limit = True
                    break
                batch = move_batch_to_device(batch, device)

                # ピアノロールを潜在表現にエンコード
                with torch.no_grad():
                    latents, _, _ = autoencoder.encode(batch["target_segments"], sample_posterior=False)
                    latents = normalizer.normalize(latents)

                replay_states: list[tuple[torch.Tensor, int]] | None = None
                fresh_states: list[tuple[torch.Tensor, int]] | None = None
                if on_policy_config.enabled:
                    replay_key = on_policy_batch_key(batch)
                    previous_states = on_policy_replay.get(replay_key)
                    reused_epochs = on_policy_epochs_since_refresh.get(replay_key)
                    should_refresh = (
                        previous_states is None
                        or reused_epochs is None
                        or reused_epochs >= on_policy_config.refresh_every_epochs
                    )
                    if should_refresh:
                        fresh_states = collect_on_policy_states(
                            model,
                            batch,
                            device,
                            config,
                            seed=config.seed * 1_000_003 + epoch * 1_009 + batch_index,
                        )
                        on_policy_replay.put(replay_key, fresh_states)
                        on_policy_epochs_since_refresh[replay_key] = 1
                    else:
                        fresh_states = on_policy_replay.get(replay_key)
                        on_policy_epochs_since_refresh[replay_key] = reused_epochs + 1
                    replay_states = previous_states if previous_states is not None else fresh_states
                    if not replay_states or not fresh_states:
                        raise RuntimeError(f"on-policy state buffer is empty for {replay_key}")

                with get_autocast_context(device, config.diffusion_training.mixed_precision):
                    # 1. 原曲エンコードと alignment bias は1曲につき1回だけ計算し、
                    #    複数のタイムステップで共有する（計算時間の大半がここなので安い）
                    conditioning = model.prepare_conditioning(batch)
                    diffusion_latents = diffusion_latents_from_conditioning(latents, conditioning)

                    loss_terms: list[torch.Tensor] = []
                    standard_terms: list[torch.Tensor] = []
                    on_policy_terms: list[torch.Tensor] = []
                    for _ in range(standard_loss_count):
                        timesteps = torch.randint(
                            0,
                            config.diffusion_model.num_train_timesteps,
                            (diffusion_latents.shape[0],),
                            device=device,
                            dtype=torch.long,
                        )
                        noise = torch.randn_like(diffusion_latents)

                        # 2. 潜在変数にノイズを付与し、prediction_type に応じたターゲットを予測
                        noisy_latents = model.q_sample(diffusion_latents, timesteps, noise)
                        model_output = model.denoise(batch, noisy_latents, timesteps, conditioning)
                        target = model.compute_training_target(diffusion_latents, noise, timesteps)
                        timestep_loss = diffusion_mse_loss(model_output, target, batch["segment_mask"])
                        loss_terms.append(timestep_loss)
                        standard_terms.append(timestep_loss)

                    if replay_states is not None and fresh_states is not None:
                        for sample_index in range(replay_loss_count):
                            stored_state, timestep_value = replay_states[sample_index % len(replay_states)]
                            noisy_latents = stored_state.to(device=device, dtype=diffusion_latents.dtype)
                            timesteps = torch.full(
                                (diffusion_latents.shape[0],), timestep_value, device=device, dtype=torch.long
                            )
                            target = on_policy_training_target(model, diffusion_latents, noisy_latents, timesteps)
                            model_output = model.denoise(batch, noisy_latents, timesteps, conditioning)
                            timestep_loss = diffusion_mse_loss(model_output, target, batch["segment_mask"])
                            loss_terms.append(timestep_loss)
                            on_policy_terms.append(timestep_loss)

                        for sample_index in range(fresh_loss_count):
                            stored_state, timestep_value = fresh_states[sample_index % len(fresh_states)]
                            noisy_latents = stored_state.to(device=device, dtype=diffusion_latents.dtype)
                            timesteps = torch.full(
                                (diffusion_latents.shape[0],), timestep_value, device=device, dtype=torch.long
                            )
                            target = on_policy_training_target(model, diffusion_latents, noisy_latents, timesteps)
                            model_output = model.denoise(batch, noisy_latents, timesteps, conditioning)
                            timestep_loss = diffusion_mse_loss(model_output, target, batch["segment_mask"])
                            loss_terms.append(timestep_loss)
                            on_policy_terms.append(timestep_loss)

                    loss = sum(loss_terms) / len(loss_terms)
                    standard_loss = sum(standard_terms) / len(standard_terms) if standard_terms else loss.new_zeros(())
                    on_policy_loss = (
                        sum(on_policy_terms) / len(on_policy_terms) if on_policy_terms else loss.new_zeros(())
                    )
                    step_loss = loss / config.diffusion_training.grad_accum_steps

                if scaler.is_enabled():
                    scaler.scale(step_loss).backward()
                else:
                    step_loss.backward()

                running_total += loss.detach().item()
                running_standard += standard_loss.detach().item()
                running_on_policy += on_policy_loss.detach().item()
                batch_count += 1
                pending_accumulation += 1
                train_progress.set_postfix(
                    total=f"{running_total / batch_count:.4f}",
                    standard=f"{running_standard / batch_count:.4f}",
                    on_policy=f"{running_on_policy / batch_count:.4f}",
                    step=global_step,
                )

                if pending_accumulation >= config.diffusion_training.grad_accum_steps:
                    apply_optimizer_step()
                    pending_accumulation = 0
                    global_step += 1
                    epoch_step_count += 1

                    if global_step % config.runtime.log_every_steps == 0:
                        tqdm.write(
                            f"epoch={epoch} step={global_step} train_total={running_total / batch_count:.4f} "
                            f"standard={running_standard / batch_count:.4f} "
                            f"on_policy={running_on_policy / batch_count:.4f}"
                        )
                        log_wandb_metrics(
                            wandb_run,
                            {
                                "train/total": running_total / batch_count,
                                "train/standard": running_standard / batch_count,
                                "train/on_policy": running_on_policy / batch_count,
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "epoch": epoch,
                            },
                            step=global_step,
                        )

                    # チェックポイント保存
                    if global_step % config.runtime.save_every_steps == 0:
                        next_epoch = epoch
                        next_batch_index = batch_index + 1
                        if next_batch_index >= len(train_loader):
                            next_epoch += 1
                            next_batch_index = 0
                        save_checkpoint(
                            work_dir / "last.pt",
                            model,
                            optimizer,
                            scaler,
                            config,
                            epoch,
                            global_step,
                            best_val_loss,
                            next_epoch,
                            next_batch_index,
                            "last",
                            extra_state=checkpoint_extra_state(),
                            scheduler=scheduler,
                            ema=ema,
                        )

                    if args.max_steps is not None and global_step >= args.max_steps:
                        break
                    if epoch_step_limit is not None and epoch_step_count >= epoch_step_limit:
                        stopped_by_epoch_step_limit = True
                        break

            if not stopped_by_epoch_step_limit and pending_accumulation > 0:
                apply_optimizer_step()
                pending_accumulation = 0
                global_step += 1

            # 検証とチェックポイント更新。EMA重みで評価する（推論で使うのはEMA側のため）
            evaluation_context = ema.as_active(model) if ema is not None else nullcontext()
            with evaluation_context:
                val_loss, val_per_timestep = run_validation(
                    model,
                    autoencoder,
                    val_loader,
                    device,
                    config,
                    epoch,
                    normalizer,
                    validation_timesteps,
                )
                # 実際に逆拡散を最後まで回して生成品質を見る。MSE では破綻を検知できないため
                generation_metrics: dict[str, float] = {}
                generation_images: list = []
                if generation_batches and epoch % generation_every == 0:
                    generation_metrics, generation_images = evaluate_generation(
                        model,
                        autoencoder,
                        generation_batches,
                        device,
                        config,
                        normalizer,
                        epoch,
                        sample_dir=work_dir / "samples",
                    )
            is_best = val_loss <= best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            val_metrics: dict[str, float | int] = {"val/total": val_loss, "epoch": epoch}
            for timestep_value, timestep_loss in val_per_timestep.items():
                val_metrics[f"val/t{timestep_value:04d}"] = timestep_loss
            val_metrics.update(generation_metrics)
            log_wandb_metrics(wandb_run, val_metrics, step=global_step)
            log_wandb_images(wandb_run, generation_images, key="generation/piano_roll", step=global_step)
            update_wandb_summary(wandb_run, best_val_loss=best_val_loss)
            if generation_metrics:
                tqdm.write(
                    "epoch={} gen onset_f1={:.3f} (recon上限 {:.3f}) 密度比={:.2f} "
                    "ピッチクラス一致={:.3f} latent_std={:.2f}".format(
                        epoch,
                        generation_metrics.get("gen/onset_f1", float("nan")),
                        generation_metrics.get("recon/onset_f1", float("nan")),
                        generation_metrics.get("gen/note_density_ratio", float("nan")),
                        generation_metrics.get("gen/pitch_class_overlap", float("nan")),
                        generation_metrics.get("gen/latent_std", float("nan")),
                    )
                )
            per_timestep_text = " ".join(f"t{k}={v:.4f}" for k, v in sorted(val_per_timestep.items()))
            tqdm.write(f"epoch={epoch} val_total={val_loss:.4f} {per_timestep_text}")

            save_checkpoint(
                work_dir / "last.pt",
                model,
                optimizer,
                scaler,
                config,
                epoch,
                global_step,
                best_val_loss,
                epoch + 1,
                0,
                "last",
                extra_state=checkpoint_extra_state(),
                scheduler=scheduler,
                ema=ema,
            )
            if is_best:
                save_checkpoint(
                    work_dir / "best.pt",
                    model,
                    optimizer,
                    scaler,
                    config,
                    epoch,
                    global_step,
                    best_val_loss,
                    epoch + 1,
                    0,
                    "best",
                    extra_state=checkpoint_extra_state(),
                    scheduler=scheduler,
                    ema=ema,
                )

            # CUDAキャッシュのクリア
            clear_cuda_cache(device, epoch)

            if args.max_steps is not None and global_step >= args.max_steps:
                break
            if config.diffusion_training.max_steps is not None and global_step >= config.diffusion_training.max_steps:
                break

    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
