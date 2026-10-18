from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class TokenizerConfig:
    # 時間グリッド。onset / duration はこの単位のフレームで表す (100 = 10ms)
    frame_rate: int = 100
    # Global が扱う固定長パッチ。ビートが分からないので秒数で切る
    patch_seconds: float = 2.0
    # 88 鍵
    pitch_min: int = 21
    pitch_max: int = 108
    # duration は短い音ほど細かく刻む。exact_duration_frames までは 1 フレーム刻み、その先は対数刻み
    num_duration_bins: int = 64
    exact_duration_frames: int = 16
    max_duration_frames: int = 1000
    num_velocity_bins: int = 32
    # 1 パッチの Local トークン列の上限 (終端トークン込み)。超えた分のイベントは切り捨てる
    max_patch_tokens: int = 384

    @property
    def patch_frames(self) -> int:
        return round(self.frame_rate * self.patch_seconds)


@dataclass(frozen=True)
class BeatTokenizerConfig:
    """拍のトークン (piano_ar.beat_tokenizer)。1 パッチ = 1 小節で、小節の中の位置と音の長さを拍の刻みで表す"""

    # イベント配列のフレーム (TokenizerConfig と同じ 10ms)。拍の時刻もこの単位で持つ
    frame_rate: int = 100
    # 1 拍の刻み。16 分 (3) と 3 連 (4) と 16 分の 3 連 (2) が乗る。カバーの音の 98.9% が刻みの ±30ms に入る
    # (local/probe_beats.py の (a))
    ticks_per_beat: int = 12
    # 1 小節の拍数の上限。超える小節は分ける
    max_bar_beats: int = 16
    pitch_min: int = 21
    pitch_max: int = 108
    # 長さは exact_duration_ticks までは 1 刻みずつ、その先は対数で max_duration_ticks まで
    num_duration_bins: int = 96
    exact_duration_ticks: int = 48
    max_duration_ticks: int = 768
    num_velocity_bins: int = 32
    # 1 パッチ (1 小節) のトークン列の上限 (終端込み)。4/4 の小節の 99.9% は 194 以下 (local/probe_beats.py の (b))
    max_patch_tokens: int = 512
    # 小節のテンポ (その小節の拍の長さの平均から読む BPM) の条件の段階。対数で tempo_min〜tempo_max
    num_tempo_bins: int = 32
    tempo_min: float = 30.0
    tempo_max: float = 300.0

    @property
    def max_bar_ticks(self) -> int:
        return self.max_bar_beats * self.ticks_per_beat


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 512
    heads: int = 8
    # パッチの中身を 1 本の要約ベクトルにまとめる双方向 Transformer (PatchSummarizer) の層数
    summary_layers: int = 3
    # パッチ列を因果的に見る Transformer
    global_layers: int = 12
    # パッチ内のトークンを生成する因果 Transformer
    local_layers: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    # 0 は「チャンネル指定なし」。事前学習で条件を落として学習するので無条件生成にも使える
    num_channels: int = 1
    # パッチごとの強さ (平均ベロシティ) と音の多さを、曲の中で標準化して何段階の条件にするか (0 で使わない)。
    # 以前のチェックポイントにはないので既定は 0 (新しく学習するときは piano_ar.train の既定値で入れる)
    dynamics_bins: int = 0
    # 条件の列の数。強さと音の多さの 2 列に、カバーでは編曲の性質 (piano_cover.arrangement) を後ろに足す
    dynamics_columns: int = 2
    # 拍のトークン (piano_ar.beat_tokenizer) のモデルなら、パッチ (小節) ごとの拍数 (1〜max_bar_beats) とテンポの段階
    # (1〜tempo_bins) を Global の入力に足す (0 は指定なし)。0 なら使わない (時刻のトークンのモデル)
    max_bar_beats: int = 0
    tempo_bins: int = 0

    @classmethod
    def from_dict(cls, values: dict) -> ModelConfig:
        """チェックポイントに保存した設定から作る。今はない項目 (以前のスタイル参照の設定など) は無視する"""
        names = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})
