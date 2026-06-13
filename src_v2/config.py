from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml


@dataclass
class DatasetConfig:
    # データセット情報のJSONパス
    dataset_json: str = "dataset.json"
    # 原曲（ソース）のMIDIディレクトリ
    original_midi_dir: str = "Dataset/original_midis/merged"
    # ピアノカバーのMIDIディレクトリ
    piano_midi_dir: str = "Dataset/pianos_midi"
    # ピアノと演奏者の対応マッピングJSONのパス
    piano_to_performer_json: str = "piano_to_performer.json"
    # 訓練/検証/テストの分割割合
    train_fraction: float = 0.9
    val_fraction: float = 0.05
    test_fraction: float = 0.05


@dataclass
class SourceChunkConfig:
    # 原曲の切り出し窓サイズ（秒）
    window_seconds: float = 2.0
    # 窓のスライド幅（秒）
    hop_seconds: float = 0.5
    # チャンクあたりの最大音符数
    max_notes_per_chunk: int = 256
    # 最小音符長（秒）
    min_duration_seconds: float = 0.03
    # 曲あたりの最大チャンク数（制限しない場合はNone）
    max_chunks_per_song: int | None = None


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
    # 隠れ層の次元数（d_model）
    d_model: int = 256
    note_feature_dim: int = 4
    # プログラム（音色）の種類数
    num_programs: int = 129
    # トラックの役割の種類数
    num_source_track_roles: int = 4
    # チャンク表現を集約する query token 数
    num_chunk_queries: int = 1
    # チャンクエンコーダーのレイヤー数
    chunk_encoder_layers: int = 2
    # 曲エンコーダーのレイヤー数
    song_encoder_layers: int = 4
    num_heads: int = 4
    # FFNの拡大倍率
    ff_multiplier: int = 4
    dropout: float = 0.1

    @property
    def head_dim(self) -> int:
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}")
        return self.d_model // self.num_heads


@dataclass
class SegmentAutoencoderConfig:
    # デコーダーのモード ("semi-crf" または "frame")
    decoder_mode: str = "semi-crf"
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
    # semi-CRF 用の pitch/frame 特徴量次元
    semi_crf_pitch_feature_dim: int = 64
    # semi-CRF の query/key 次元
    semi_crf_head_dim: int = 64
    # 区間長に対するスコアリング方式
    semi_crf_length_scaling: str = "linear"
    semi_crf_length_penalty: float = 0.0
    # note の事前バイアス
    semi_crf_note_bias: float = 0.0
    # pitch ごとのデコードを分割するバッチサイズ
    semi_crf_track_batch_size: int = 128
    # 区間見逃し / 誤検出に対するコスト
    semi_crf_false_negative_cost: float = 0.0
    semi_crf_false_positive_cost: float = 0.0
    # 区間境界の onset / offset 存在を補助学習するか
    use_interval_boundary_head: bool = True
    # boundary loss の重み
    interval_presence_loss_weight: float = 1.0

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
    # Performer条件のドロップアウト率
    performer_dropout: float = 0.1
    # 訓練のタイムステップ数
    num_train_timesteps: int = 1000
    beta_start: float = 1.0e-4
    beta_end: float = 0.02
    # 推論のサンプリングステップ数
    sampling_steps: int = 50

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


@dataclass
class ExperimentConfig:
    experiment_name: str = "piano_cover_v2_segment_diffusion"
    seed: int = 7
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    source_chunks: SourceChunkConfig = field(default_factory=SourceChunkConfig)
    target_roll: TargetRollConfig = field(default_factory=TargetRollConfig)
    source_model: SourceEncoderConfig = field(default_factory=SourceEncoderConfig)
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


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    # YAMLファイルから設定をロード
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = _build_dataclass(ExperimentConfig, raw)
    _validate_experiment_config(config)
    return config
