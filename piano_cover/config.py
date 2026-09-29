from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoverConfig:
    """カバーモデルで足す部分 (原曲エンコーダと cross-attention)。デコーダは ModelConfig のまま"""

    # 原曲エンコーダと cross-attention の中の幅。0 ならデコーダと同じ。デコーダを大きくしても足す部分を
    # 大きくしすぎない (カバーのデータは 150 時間ほどなので過学習しやすい) よう、細くできるようにする。
    # ヘッドの次元はデコーダと同じにして、ヘッド数を幅に合わせて減らす
    source_dim: int = 0
    # 原曲の 1 パッチ (デコーダと同じ 2 秒) の行を K 本の要約にまとめる双方向 Transformer
    source_patch_layers: int = 2
    source_latents: int = 4
    # 全曲の要約列を見る双方向 Transformer
    source_song_layers: int = 6
    # 1 パッチに入れる原曲の行 (音・拍・コード・キー) の上限。超えた分は切り捨てる
    max_source_rows: int = 256
    # Global / Local の何ブロックごとに cross-attention を挟むか
    global_cross_every: int = 2
    local_cross_every: int = 2
    # Global が見る原曲の範囲 (対応する時刻から ± 何パッチ)
    global_cross_window: int = 4
    # Local が見る原曲の音の範囲 (対応するパッチから ± 何パッチ)
    local_cross_radius: int = 1
