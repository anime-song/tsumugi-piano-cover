from __future__ import annotations

from dataclasses import dataclass


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

