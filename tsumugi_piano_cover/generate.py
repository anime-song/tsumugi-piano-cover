from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from tsumugi_piano_cover.config import load_experiment_config
from tsumugi_piano_cover.data.audio import load_audio
from tsumugi_piano_cover.data.segment import roll_to_score, segment_grid_from_duration
from tsumugi_piano_cover.latent_scaling import LatentNormalizer
from tsumugi_piano_cover.models.diffusion import ConditionalSegmentDiffusionModel
from tsumugi_piano_cover.models.segment_autoencoder import SegmentLatentAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate piano-cover MIDI with segment-latent diffusion.")
    parser.add_argument("--config", default="configs/piano_cover_v2.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--autoencoder-checkpoint", default=None)
    parser.add_argument("--source-audio", required=True)
    parser.add_argument("--output-midi", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--performer-id", type=int)
    parser.add_argument("--piano-id")
    parser.add_argument(
        "--null-performer",
        action="store_true",
        help="演奏者を指定せず、学習時に使った無条件（null）埋め込みで生成する",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="演奏者条件の classifier-free guidance 強度。1.0 で guidance なし",
    )
    parser.add_argument("--latent-scale", type=float)
    parser.add_argument("--no-ema", action="store_true", help="EMA重みではなく生の学習重みで生成する")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def restore_config_section(section: object, payload: object) -> None:
    """チェックポイント保存時の設定でモデル構造に関わる項目を上書きする。

    ノイズスケジュールや alignment bias の幅は学習時と一致していないと推論が壊れるため、
    YAML よりチェックポイント側を優先する。
    """
    if not isinstance(payload, dict):
        return
    for item in fields(section):  # type: ignore[arg-type]
        if item.name in payload:
            setattr(section, item.name, payload[item.name])


def resolve_performer_id(args: argparse.Namespace, config, null_performer_id: int) -> int:
    if args.null_performer:
        return null_performer_id
    if args.performer_id is not None:
        return int(args.performer_id)
    if args.piano_id is None:
        raise ValueError("either --performer-id, --piano-id or --null-performer is required")
    performer_map = json.loads(Path(config.dataset.piano_to_performer_json).read_text(encoding="utf-8"))
    if args.piano_id not in performer_map:
        raise KeyError(f"piano_id not found in performer map: {args.piano_id}")
    return int(performer_map[args.piano_id])


def build_source_batch(source_audio_path: str, config, performer_id: int, device: torch.device):
    # モデル入力用バッチの構築
    source_audio = load_audio(
        source_audio_path,
        sample_rate=config.tsumugi_model.sample_rate,
        num_channels=config.tsumugi_model.audio_channels,
    )
    source_audio_length = int(source_audio.shape[-1])

    # セグメント分割用グリッドの作成
    source_duration = source_audio_length / config.tsumugi_model.sample_rate
    segment_grid = segment_grid_from_duration(source_duration, config.target_roll)
    num_segments = int(segment_grid.segments.shape[0])
    segment_times = segment_grid.segment_times.unsqueeze(0).to(device)
    # 推論時は target 側 alignment が無いので同時刻を guide にする。
    # 学習側でも一定確率でこの条件を与えているため、ここでのズレは吸収されるはず
    alignment_source_times = segment_times.clone()
    alignment_mask = torch.ones((1, num_segments), dtype=torch.bool, device=device)
    batch = {
        "source_audio": source_audio.unsqueeze(0).to(device),
        "source_audio_lengths": torch.tensor([source_audio_length], dtype=torch.long, device=device),
        "segment_times": segment_times,
        "segment_mask": torch.ones((1, num_segments), dtype=torch.bool, device=device),
        "alignment_source_times": alignment_source_times,
        "alignment_mask": alignment_mask,
        "performer_ids": torch.tensor([performer_id], dtype=torch.long, device=device),
    }
    return batch, segment_grid


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    device = resolve_device(args.device)

    # 1. Diffusionモデルのロード
    diffusion_checkpoint = Path(args.checkpoint or (Path(config.runtime.work_dir) / "best.pt"))
    if not diffusion_checkpoint.is_file():
        raise FileNotFoundError(f"diffusion checkpoint not found: {diffusion_checkpoint}")
    diffusion_payload = torch.load(diffusion_checkpoint, map_location=device)

    # 1-1. 生成時はcheckpoint保存時のモデル設定を優先する
    checkpoint_config = diffusion_payload.get("experiment_config")
    if isinstance(checkpoint_config, dict):
        restore_config_section(config.diffusion_model, checkpoint_config.get("diffusion_model"))
        restore_config_section(config.source_model, checkpoint_config.get("source_model"))

    # 2. オートエンコーダー（VAE）のチェックポイントの特定
    autoencoder_checkpoint = args.autoencoder_checkpoint
    if autoencoder_checkpoint is None:
        autoencoder_checkpoint = diffusion_payload.get("autoencoder_checkpoint")
    if autoencoder_checkpoint is None:
        fallback = Path(config.runtime.autoencoder_work_dir) / "vae" / "best.pt"
        autoencoder_checkpoint = config.runtime.autoencoder_checkpoint or str(fallback)
    autoencoder_checkpoint = Path(autoencoder_checkpoint)
    if not autoencoder_checkpoint.is_file():
        raise FileNotFoundError(f"autoencoder checkpoint not found: {autoencoder_checkpoint}")

    # 3. 演奏者情報の解決。埋め込みの最終行は無条件（null）用に確保されている
    performer_vocab_size = diffusion_payload.get("performer_vocab_size")
    if performer_vocab_size is None:
        # 重み形状から演奏者数を推測（null ぶんの1行を差し引く）
        embedding_rows = int(diffusion_payload["model_state"]["denoiser.performer_embedding.weight"].shape[0])
        performer_vocab_size = embedding_rows - 1
    performer_vocab_size = int(performer_vocab_size)
    null_performer_id = performer_vocab_size
    performer_id = resolve_performer_id(args, config, null_performer_id)
    if not 0 <= performer_id <= null_performer_id:
        raise ValueError(f"performer_id out of range: {performer_id} (vocab size {performer_vocab_size})")

    # 4. オートエンコーダーモデルのロード
    autoencoder = SegmentLatentAutoencoder(config.autoencoder_model, config.target_roll).to(device)
    autoencoder_payload = torch.load(autoencoder_checkpoint, map_location=device)
    autoencoder.load_state_dict(autoencoder_payload["model_state"])
    autoencoder.eval()
    # 潜在の正規化は学習時と厳密に一致させる必要があるので diffusion checkpoint を優先する
    if args.latent_scale is not None:
        normalizer = LatentNormalizer(scale=float(args.latent_scale))
    elif diffusion_payload.get("latent_scale") is not None:
        normalizer = LatentNormalizer.from_state_dict(diffusion_payload)
    else:
        normalizer = LatentNormalizer.from_autoencoder_checkpoint(autoencoder_payload)
    normalizer = normalizer.to(device)

    # 5. Diffusionモデルのロード。既定では推論品質が安定するEMA重みを使う
    model = ConditionalSegmentDiffusionModel(
        config.source_model,
        config.diffusion_model,
        performer_vocab_size,
        config.tsumugi_model,
    ).to(device)
    model_state = dict(diffusion_payload["model_state"])
    ema_state = diffusion_payload.get("ema_state")
    used_ema = False
    if ema_state is not None and not args.no_ema:
        shadow = ema_state.get("shadow") if isinstance(ema_state, dict) else None
        if isinstance(shadow, dict) and shadow:
            for name, tensor in shadow.items():
                if name in model_state:
                    model_state[name] = torch.as_tensor(tensor).to(model_state[name].dtype)
            used_ema = True
    model.load_state_dict(model_state)
    model.eval()

    # 6. 入力バッチの構築と推論（サンプリング＆デコード）の実行
    batch, segment_grid = build_source_batch(args.source_audio, config, performer_id, device)
    latent_shape = (
        1,
        int(batch["segment_mask"].shape[1]),
        config.diffusion_model.latent_dim,
    )

    with torch.no_grad():
        # Diffusionモデルから潜在変数をサンプリング
        latents = model.sample(
            batch,
            latent_shape=latent_shape,
            sampling_steps=args.sampling_steps,
            guidance_scale=args.guidance_scale,
        )
        latents = normalizer.denormalize(latents)
        # 潜在変数をオートエンコーダーでデコード
        decoded = autoencoder.decode(latents)
        recon_roll = autoencoder.reconstruct_roll(
            decoded,
            segment_grid.segment_times,
            segment_grid.segment_valid_lengths,
            segment_grid.num_frames,
        )

    # 7. 再構成されたピアノロールをMIDIスコアに変換して書き出し
    output_midi = Path(args.output_midi)
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    roll_to_score(recon_roll, config.target_roll).dump_midi(str(output_midi))

    # 8. 実行結果のレポート作成と保存
    report = {
        "source_audio": args.source_audio,
        "diffusion_checkpoint": str(diffusion_checkpoint),
        "autoencoder_checkpoint": str(autoencoder_checkpoint),
        "performer_id": performer_id,
        "null_performer_id": null_performer_id,
        "used_ema_weights": used_ema,
        "prediction_type": config.diffusion_model.prediction_type,
        "zero_terminal_snr": config.diffusion_model.zero_terminal_snr,
        "latent_normalization": normalizer.describe(),
        "guidance_scale": args.guidance_scale,
        "source_audio_samples": int(batch["source_audio_lengths"].item()),
        "num_segments": int(batch["segment_mask"].sum().item()),
        "num_frames": segment_grid.num_frames,
        "sampling_steps": args.sampling_steps or config.diffusion_model.sampling_steps,
        "output_midi": str(output_midi),
    }
    if args.output_json is not None:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
