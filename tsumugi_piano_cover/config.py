from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml


@dataclass
class DatasetConfig:
    # データセット情報のJSONパス
    dataset_json: str = "data/metadata/dataset.json"
    # 原曲（ソース）の音源ディレクトリ
    original_audio_dir: str = "Dataset/original"
    # ピアノカバーのMIDIディレクトリ
    piano_midi_dir: str = "Dataset/pianos_midi"
    # ピアノカバーの音源ディレクトリ
    piano_audio_dir: str = "Dataset/pianos"
    # ピアノと演奏者の対応マッピングJSONのパス
    piano_to_performer_json: str = "data/metadata/piano_to_performer.json"
    # source/target の時刻対応を保存したキャッシュディレクトリ
    alignment_cache_dir: str | None = None
    # 訓練/検証/テストの分割割合
    train_fraction: float = 0.9
    val_fraction: float = 0.05
    test_fraction: float = 0.05


@dataclass
class TargetRollConfig:
    # 1フレームあたりの時間幅（秒）
    frame_seconds: float = 0.05
    # セグメント（フレーズ）の時間長（秒）
    phrase_seconds: float = 2.56
    # セグメントの切り出し間隔（秒）
    phrase_hop_seconds: float = 1.28
    # 最小ピッチ番号（MIDIノート番号）
    pitch_min: int = 21
    # 最大ピッチ番号
    pitch_max: int = 108
    # 最小音符長（秒）
    min_duration_seconds: float = 0.03
    # 曲あたりの最大セグメント数
    max_phrases_per_song: int | None = None
    # 評価時に使う onset / sustain / pedal の二値化閾値
    onset_threshold: float = 0.5
    sustain_threshold: float = 0.5
    pedal_threshold: float = 0.5

    @property
    def pitch_count(self) -> int:
        return self.pitch_max - self.pitch_min + 1

    @property
    def feature_dim(self) -> int:
        # 特徴量次元数: (onset + sustain + velocity) * pitch_count + pedal
        return self.pitch_count * 3 + 1

    @property
    def frames_per_phrase(self) -> int:
        return max(1, int(round(self.phrase_seconds / self.frame_seconds)))

    @property
    def frames_per_hop(self) -> int:
        return max(1, int(round(self.phrase_hop_seconds / self.frame_seconds)))


@dataclass
class SourceEncoderConfig:
    # Tsumugi adapterとdiffusion memoryの次元数
    d_model: int = 256
    # alignment 由来の cross-attention bias の強さ
    alignment_bias_strength: float = 1.0
    # alignment bias の時間幅（秒）。大きいほど弱く広く source を見る
    alignment_bias_sigma_seconds: float = 6.0
    # 0なら元の同時刻グリッド、1ならalignment時刻だけを中心にする
    alignment_bias_aligned_time_weight: float = 0.5
    # 学習時にalignment biasをsegment単位で落とす確率
    alignment_bias_dropout: float = 0.0
    # 学習時にalignment biasを曲単位で丸ごと落とす確率（bias無しでも動く状態を保つ）
    alignment_bias_song_dropout: float = 0.1
    # 学習時にguide時刻を推論と同じ「同時刻」に置き換える確率
    # （推論ではalignmentが無く同時刻を使うため、そのズレた条件も学習させる）
    alignment_bias_identity_probability: float = 0.3


@dataclass
class TsumugiConfig:
    model_id: str = "anime-song/tsumugi-mrl"
    revision: str | None = None
    hidden_dim: int = 512
    sample_rate: int = 22050
    audio_channels: int = 2
    lora_enabled: bool = True
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0", "net.1", "net.4"]
    )
    lora_layers: list[int] | None = None
    gradient_checkpointing: bool = True


@dataclass
class SegmentAutoencoderConfig:
    d_model: int = 256
    # 潜在表現（latent）の次元数
    latent_dim: int = 64
    # 潜在表現を集約するクエリ数
    num_latent_queries: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 4
    num_heads: int = 4
    ff_multiplier: int = 4
    dropout: float = 0.1
    # KL損失の重み
    kl_beta: float = 1.0e-4
    # 補助損失の重み
    onset_loss_weight: float = 1.0
    sustain_loss_weight: float = 1.0
    pedal_loss_weight: float = 1.0
    velocity_loss_weight: float = 1.0
    # onset のクラス重み
    onset_positive_class_weight: float = 0.5
    onset_negative_class_weight: float = 0.5
    # sustain のクラス重み
    sustain_positive_class_weight: float = 0.5
    sustain_negative_class_weight: float = 0.5

    @property
    def head_dim(self) -> int:
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}")
        return self.d_model // self.num_heads


@dataclass
class DiffusionConfig:
    d_model: int = 256
    latent_dim: int = 64
    # Diffusionの予測対象 ("epsilon" または "v")
    prediction_type: str = "epsilon"
    num_layers: int = 6
    num_heads: int = 4
    ff_multiplier: int = 4
    dropout: float = 0.1
    # Performer条件を null ID へ置き換える確率（classifier-free guidance 用の無条件埋め込み）
    performer_dropout: float = 0.1
    # セグメント絶対時刻の正弦波埋め込みがカバーする周期（秒）
    segment_time_min_period_seconds: float = 0.5
    segment_time_max_period_seconds: float = 1024.0
    # 訓練のタイムステップ数
    num_train_timesteps: int = 1000
    beta_start: float = 1.0e-4
    beta_end: float = 0.02
    # 終端のSNRを0にリスケールするか（学習時のx_Tと推論時の純ノイズを一致させる）
    zero_terminal_snr: bool = True
    # 推論のサンプリングステップ数
    sampling_steps: int = 50
    # denoiser の gradient checkpointing。timesteps_per_sample の回数だけ再計算が走るので
    # VRAM に余裕があるなら切った方が速い（実測 1.4x）
    gradient_checkpointing: bool = False
    # Learn a source-conditioned coarse latent center and diffuse only the residual.
    coarse_latent_enabled: bool = False

    @property
    def head_dim(self) -> int:
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}")
        return self.d_model // self.num_heads


@dataclass
class RuntimeConfig:
    # Diffusionの出力先
    work_dir: str = "artifacts/piano_cover_v2_diffusion"
    # Autoencoderの出力先
    autoencoder_work_dir: str = "artifacts/piano_cover_v2_autoencoder"
    # Autoencoderの学習済みチェックポイントパス
    autoencoder_checkpoint: str | None = None
    # DataLoaderのワーカー数
    num_workers: int = 0
    # Autoencoder 側だけ num_workers を上書きしたい場合に使う
    autoencoder_num_workers: int | None = None
    # Autoencoder dataset 内で保持する song 単位の LRU cache 数
    autoencoder_max_cached_songs: int | None = 16
    log_every_steps: int = 20
    # 保存ステップ間隔
    save_every_steps: int = 500


@dataclass
class AlignmentConfig:
    # alignment cache の保存先。未指定なら dataset.alignment_cache_dir または既定値を使う
    cache_dir: str | None = None
    # 前計算方法。音声同士の同期を使う
    method: str = "audio_sync"
    # alignment 用の時間フレーム幅（秒）
    frame_seconds: float = 0.1
    # audio-sync 前処理で読み込むサンプルレート
    audio_sample_rate: int = 22050
    # synctoolbox の特徴量フレームレート
    sync_feature_rate: int = 50
    # synctoolbox の step weight
    sync_step_weights: tuple[float, float, float] = (1.5, 1.5, 2.0)
    # synctoolbox の recursion threshold
    sync_threshold_rec: int = 10**6


@dataclass
class WandbConfig:
    # W&Bロギングの有効化
    enabled: bool = False
    project: str = "piano-cover-v2"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"


@dataclass
class OnPolicyConfig:
    # Mix standard diffusion states with states visited by the model during sampling.
    enabled: bool = False
    replay_fraction: float = 0.25
    fresh_fraction: float = 0.25
    refresh_steps: int = 10
    sampling_steps: int = 50
    # These are target timesteps; the nearest steps in the sampling schedule are captured.
    capture_timesteps: tuple[int, int, int, int] = (591, 387, 183, 81)
    # Replay states are keyed by song and kept on CPU in float16.
    replay_max_songs: int = 128


@dataclass
class TrainingConfig:
    batch_size: int = 1
    # 勾配累積ステップ数
    grad_accum_steps: int = 1
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    max_epochs: int = 200
    max_steps: int | None = None
    max_steps_per_epoch: int | None = None
    # 勾配クリッピング閾値
    grad_clip_norm: float = 1.0
    # 混合精度訓練の種別
    mixed_precision: str = "bf16"
    # ピッチシフトの範囲（半音）
    pitch_shift_min_semitones: int = 0
    pitch_shift_max_semitones: int = 0
    # 学習率スケジューラー
    lr_scheduler_type: str = "cosine_with_warmup"
    lr_warmup_steps: int = 1000
    lr_min: float = 1.0e-6
    # 重みのEMA（Diffusionでは事実上必須。検証・保存はEMA重みで行う）
    ema_enabled: bool = True
    ema_decay: float = 0.999
    # 1サンプルあたりに引くタイムステップ数。原曲エンコードを共有して勾配分散を下げる
    timesteps_per_sample: int = 1
    # 検証で使う固定タイムステップの本数（毎エポック同じ値を使い、val lossを比較可能にする）
    val_timesteps: int = 5
    # 実際に逆拡散を回して生成品質を測る間隔（エポック）。0 で無効
    # MSE は生成品質と相関しないため、破綻を早期に検知するのに使う
    generation_eval_every_epochs: int = 1
    generation_eval_songs: int = 4
    generation_eval_sampling_steps: int = 50
    on_policy: OnPolicyConfig = field(default_factory=OnPolicyConfig)


@dataclass
class ExperimentConfig:
    experiment_name: str = "piano_cover_v2_segment_diffusion"
    seed: int = 7
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    target_roll: TargetRollConfig = field(default_factory=TargetRollConfig)
    source_model: SourceEncoderConfig = field(default_factory=SourceEncoderConfig)
    tsumugi_model: TsumugiConfig = field(default_factory=TsumugiConfig)
    autoencoder_model: SegmentAutoencoderConfig = field(default_factory=SegmentAutoencoderConfig)
    diffusion_model: DiffusionConfig = field(default_factory=DiffusionConfig)
    autoencoder_training: TrainingConfig = field(default_factory=TrainingConfig)
    diffusion_training: TrainingConfig = field(default_factory=TrainingConfig)


T = TypeVar("T")


def _is_dataclass_type(tp: Any) -> bool:
    return isinstance(tp, type) and is_dataclass(tp)


def _coerce_value(tp: Any, value: Any) -> Any:
    origin = get_origin(tp)
    if _is_dataclass_type(tp):
        return _build_dataclass(tp, value)
    if origin is list:
        (item_type,) = get_args(tp)
        return [_coerce_value(item_type, item) for item in value]
    if origin is tuple:
        item_types = get_args(tp)
        return tuple(_coerce_value(item_type, item) for item_type, item in zip(item_types, value, strict=False))
    if origin is not None and type(None) in get_args(tp):
        inner = next(arg for arg in get_args(tp) if arg is not type(None))
        return None if value is None else _coerce_value(inner, value)
    return value


def _build_dataclass(cls: type[T], data: dict[str, Any]) -> T:
    # 辞書からdataclassインスタンスを再帰的に生成
    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in data:
            continue
        kwargs[item.name] = _coerce_value(type_hints[item.name], data[item.name])
    return cls(**kwargs)


def _validate_experiment_config(config: ExperimentConfig) -> None:
    # 0. Diffusionの予測対象が対応済みの方式か確認
    if config.diffusion_model.prediction_type not in {"epsilon", "v"}:
        raise ValueError(
            "diffusion_model.prediction_type must be one of {'epsilon', 'v'}: "
            f"{config.diffusion_model.prediction_type!r}"
        )

    # 1. オートエンコーダーとDiffusionで潜在次元が一致しているか確認
    if config.autoencoder_model.latent_dim != config.diffusion_model.latent_dim:
        raise ValueError(
            "autoencoder_model.latent_dim and diffusion_model.latent_dim must match: "
            f"{config.autoencoder_model.latent_dim} != {config.diffusion_model.latent_dim}"
        )

    # 2. 原曲エンコーダーとDiffusionで隠れ次元が一致しているか確認
    if config.source_model.d_model != config.diffusion_model.d_model:
        raise ValueError(
            "source_model.d_model and diffusion_model.d_model must match: "
            f"{config.source_model.d_model} != {config.diffusion_model.d_model}"
        )

    if config.tsumugi_model.hidden_dim <= 0:
        raise ValueError(f"tsumugi_model.hidden_dim must be positive: {config.tsumugi_model.hidden_dim}")
    if config.tsumugi_model.sample_rate <= 0:
        raise ValueError(f"tsumugi_model.sample_rate must be positive: {config.tsumugi_model.sample_rate}")
    if config.tsumugi_model.audio_channels != 2:
        raise ValueError("tsumugi_model.audio_channels must be 2 for Tsumugi-MRL")
    if config.tsumugi_model.sample_rate != config.alignment.audio_sample_rate:
        raise ValueError(
            "tsumugi_model.sample_rate and alignment.audio_sample_rate must match: "
            f"{config.tsumugi_model.sample_rate} != {config.alignment.audio_sample_rate}"
        )
    if config.tsumugi_model.lora_rank <= 0:
        raise ValueError(f"tsumugi_model.lora_rank must be positive: {config.tsumugi_model.lora_rank}")
    if config.tsumugi_model.lora_alpha <= 0.0:
        raise ValueError(f"tsumugi_model.lora_alpha must be positive: {config.tsumugi_model.lora_alpha}")
    if not 0.0 <= config.tsumugi_model.lora_dropout < 1.0:
        raise ValueError(f"tsumugi_model.lora_dropout must be in [0, 1): {config.tsumugi_model.lora_dropout}")
    if not config.tsumugi_model.lora_target_modules:
        raise ValueError("tsumugi_model.lora_target_modules must not be empty")

    # 4. alignment attention bias の設定範囲を確認
    if config.source_model.alignment_bias_strength < 0.0:
        raise ValueError("source_model.alignment_bias_strength must be non-negative")
    if config.source_model.alignment_bias_sigma_seconds <= 0.0:
        raise ValueError("source_model.alignment_bias_sigma_seconds must be positive")
    if not 0.0 <= config.source_model.alignment_bias_aligned_time_weight <= 1.0:
        raise ValueError("source_model.alignment_bias_aligned_time_weight must be in [0, 1]")
    if not 0.0 <= config.source_model.alignment_bias_dropout <= 1.0:
        raise ValueError("source_model.alignment_bias_dropout must be in [0, 1]")
    if not 0.0 <= config.source_model.alignment_bias_song_dropout <= 1.0:
        raise ValueError("source_model.alignment_bias_song_dropout must be in [0, 1]")
    if not 0.0 <= config.source_model.alignment_bias_identity_probability <= 1.0:
        raise ValueError("source_model.alignment_bias_identity_probability must be in [0, 1]")
    if not 0.0 <= config.diffusion_model.performer_dropout <= 1.0:
        raise ValueError("diffusion_model.performer_dropout must be in [0, 1]")
    if (
        not 0.0
        < config.diffusion_model.segment_time_min_period_seconds
        < config.diffusion_model.segment_time_max_period_seconds
    ):
        raise ValueError(
            "diffusion_model.segment_time_min_period_seconds must be positive and smaller than "
            "segment_time_max_period_seconds"
        )

    # 6. 学習ループ側の設定範囲を確認
    for name, training in (
        ("autoencoder_training", config.autoencoder_training),
        ("diffusion_training", config.diffusion_training),
    ):
        if not 0.0 < training.ema_decay < 1.0:
            raise ValueError(f"{name}.ema_decay must be in (0, 1): {training.ema_decay}")
        if training.timesteps_per_sample < 1:
            raise ValueError(f"{name}.timesteps_per_sample must be >= 1: {training.timesteps_per_sample}")
        if training.val_timesteps < 1:
            raise ValueError(f"{name}.val_timesteps must be >= 1: {training.val_timesteps}")
        if training.generation_eval_every_epochs < 0:
            raise ValueError(f"{name}.generation_eval_every_epochs must be >= 0")
        if training.generation_eval_every_epochs > 0:
            if training.generation_eval_songs < 1:
                raise ValueError(f"{name}.generation_eval_songs must be >= 1")
            if training.generation_eval_sampling_steps < 1:
                raise ValueError(f"{name}.generation_eval_sampling_steps must be >= 1")
        on_policy = training.on_policy
        if not 0.0 <= on_policy.replay_fraction <= 1.0:
            raise ValueError(f"{name}.on_policy.replay_fraction must be in [0, 1]")
        if not 0.0 <= on_policy.fresh_fraction <= 1.0:
            raise ValueError(f"{name}.on_policy.fresh_fraction must be in [0, 1]")
        if on_policy.replay_fraction + on_policy.fresh_fraction >= 1.0:
            raise ValueError(f"{name}.on_policy replay_fraction + fresh_fraction must be < 1")
        if on_policy.refresh_steps < 1:
            raise ValueError(f"{name}.on_policy.refresh_steps must be >= 1")
        if on_policy.sampling_steps < 1:
            raise ValueError(f"{name}.on_policy.sampling_steps must be >= 1")
        if on_policy.replay_max_songs < 1:
            raise ValueError(f"{name}.on_policy.replay_max_songs must be >= 1")
        if len(on_policy.capture_timesteps) != 4:
            raise ValueError(f"{name}.on_policy.capture_timesteps must contain exactly 4 values")
        if any(
            timestep <= 0 or timestep >= config.diffusion_model.num_train_timesteps
            for timestep in on_policy.capture_timesteps
        ):
            raise ValueError(
                f"{name}.on_policy.capture_timesteps must be in (0, num_train_timesteps): "
                f"{on_policy.capture_timesteps}"
            )
        if on_policy.enabled and name != "diffusion_training":
            raise ValueError(f"{name}.on_policy.enabled is only supported for diffusion_training")
        if on_policy.enabled and training.batch_size != 1:
            raise ValueError("diffusion_training.on_policy currently requires batch_size=1")

    # 5. alignment 前計算の設定範囲を確認
    if config.alignment.method != "audio_sync":
        raise ValueError(f"alignment.method must be 'audio_sync': {config.alignment.method!r}")
    if config.alignment.frame_seconds <= 0.0:
        raise ValueError(f"alignment.frame_seconds must be positive: {config.alignment.frame_seconds}")
    if config.alignment.audio_sample_rate <= 0:
        raise ValueError(f"alignment.audio_sample_rate must be positive: {config.alignment.audio_sample_rate}")
    if config.alignment.sync_feature_rate <= 0:
        raise ValueError(f"alignment.sync_feature_rate must be positive: {config.alignment.sync_feature_rate}")
    if config.alignment.sync_threshold_rec <= 0:
        raise ValueError(f"alignment.sync_threshold_rec must be positive: {config.alignment.sync_threshold_rec}")
    if len(config.alignment.sync_step_weights) != 3:
        raise ValueError(
            f"alignment.sync_step_weights must contain exactly 3 values: {config.alignment.sync_step_weights}"
        )
    if any(weight <= 0.0 for weight in config.alignment.sync_step_weights):
        raise ValueError(f"alignment.sync_step_weights must be positive: {config.alignment.sync_step_weights}")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    # YAMLファイルから設定をロード
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = _build_dataclass(ExperimentConfig, raw)
    _validate_experiment_config(config)
    return config
