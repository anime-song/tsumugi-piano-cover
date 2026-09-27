from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreTokenizerConfig:
    # 88 鍵。範囲外の音は読み込み時に捨てる
    pitch_min: int = 21
    pitch_max: int = 108
    # 小節の長さの上限 (4 分音符単位)。位置は「何拍目 (4 分音符単位の整数部)」+「拍の中の分数」の 2 トークンで表す
    max_measure_quarters: int = 32
    # 拍の中の位置として許す分数の分母。これで表せない位置を含む曲は使わない
    fraction_denominators: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32)
    # 小節の開始時刻 (演奏上の秒)。前の小節の開始からの差を対数刻みのビンにする (0 秒のビンを別に持つ)
    num_mtime_bins: int = 128
    mtime_min_seconds: float = 0.02
    mtime_max_seconds: float = 30.0
    # 1 小節の Local トークン列の上限 (終端トークン込み)
    max_patch_tokens: int = 256
