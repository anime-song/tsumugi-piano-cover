from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AudioSyncPath:
    # target audio 時刻を source audio 時刻へ写すための knot 列
    target_time_knots_seconds: np.ndarray
    source_time_knots_seconds: np.ndarray


@dataclass(frozen=True)
class _AudioSyncFeatures:
    # synctoolbox に渡す特徴量一式
    chroma_quantized: np.ndarray
    dlnco_features: np.ndarray


@dataclass(frozen=True)
class _SyncToolboxFunctions:
    # synctoolbox の依存関数をひとまとめにして扱う
    sync_via_mrmsdtw: Any
    make_path_strictly_monotonic: Any
    pitch_to_chroma: Any
    quantize_chroma: Any
    pitch_onset_features_to_dlnco: Any
    audio_to_pitch_features: Any
    audio_to_pitch_onset_features: Any
    estimate_tuning: Any
    base_sample_rate: int


def compute_target_to_source_audio_sync_path(
    target_audio_path: str | Path,
    source_audio_path: str | Path,
    sample_rate: int,
    feature_rate: int,
    step_weights: tuple[float, float, float],
    threshold_rec: int,
) -> AudioSyncPath:
    # 1. 依存ライブラリを遅延 import して、通常の学習コードを壊さないようにする
    sync_fns = _load_synctoolbox_functions()
    _validate_synctoolbox_sample_rate(sample_rate, sync_fns.base_sample_rate)

    # 2. 同じ worker 内では decode / resample / 特徴抽出を path 単位で再利用する
    target_features = _load_cached_sync_features(target_audio_path, sample_rate, feature_rate)
    source_features = _load_cached_sync_features(source_audio_path, sample_rate, feature_rate)

    # 3. sync.py と同じ特徴量で MrMsDTW を実行する
    warping_path = sync_fns.sync_via_mrmsdtw(
        f_chroma1=target_features.chroma_quantized,
        f_onset1=target_features.dlnco_features,
        f_chroma2=source_features.chroma_quantized,
        f_onset2=source_features.dlnco_features,
        input_feature_rate=feature_rate,
        step_weights=np.asarray(step_weights, dtype=np.float64),
        threshold_rec=threshold_rec,
        verbose=False,
    )

    # 4. target 時刻 -> source 時刻の写像として使いやすいよう、strict monotonic に直す
    strict_path = sync_fns.make_path_strictly_monotonic(warping_path)
    target_knots = strict_path[0].astype(np.float32, copy=False) / float(feature_rate)
    source_knots = strict_path[1].astype(np.float32, copy=False) / float(feature_rate)
    return AudioSyncPath(
        target_time_knots_seconds=target_knots,
        source_time_knots_seconds=source_knots,
    )


@lru_cache(maxsize=1)
def _load_synctoolbox_functions() -> _SyncToolboxFunctions:
    # synctoolbox 依存を 1 度だけ解決して、以降は worker 内で再利用する
    try:
        from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
        from synctoolbox.dtw.utils import make_path_strictly_monotonic
        from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
        from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
        from synctoolbox.feature.pitch import FS_PITCH, audio_to_pitch_features
        from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
        from synctoolbox.feature.utils import estimate_tuning
    except ImportError as exc:
        raise ImportError(
            "audio_sync alignment requires synctoolbox. "
            "Install it before running recipes.tools.alignment.precompute_alignments "
            "with alignment.method=audio_sync."
        ) from exc

    return _SyncToolboxFunctions(
        sync_via_mrmsdtw=sync_via_mrmsdtw,
        make_path_strictly_monotonic=make_path_strictly_monotonic,
        pitch_to_chroma=pitch_to_chroma,
        quantize_chroma=quantize_chroma,
        pitch_onset_features_to_dlnco=pitch_onset_features_to_DLNCO,
        audio_to_pitch_features=audio_to_pitch_features,
        audio_to_pitch_onset_features=audio_to_pitch_onset_features,
        estimate_tuning=estimate_tuning,
        base_sample_rate=int(FS_PITCH[0]),
    )


def _validate_synctoolbox_sample_rate(sample_rate: int, base_sample_rate: int) -> None:
    # synctoolbox 1.4.x の pitch feature 実装は内部で固定の multi-rate 前提を持つ
    if int(sample_rate) != int(base_sample_rate):
        raise ValueError(
            "audio_sync alignment requires alignment.audio_sample_rate to match synctoolbox's base sample rate: "
            f"{sample_rate} != {base_sample_rate}. Set alignment.audio_sample_rate to {base_sample_rate}."
        )


def _normalize_audio_cache_key(audio_path: str | Path) -> str:
    # 相対/絶対 path の揺れで cache miss しないよう canonical path を使う
    return str(Path(audio_path).expanduser().resolve())


def _load_cached_sync_features(
    audio_path: str | Path,
    sample_rate: int,
    feature_rate: int,
) -> _AudioSyncFeatures:
    # Path 型の揺れを吸収してから cache 付き本体へ渡す
    return _load_cached_sync_features_impl(_normalize_audio_cache_key(audio_path), sample_rate, feature_rate)


@lru_cache(maxsize=64)
def _load_cached_sync_features_impl(
    audio_path: str,
    sample_rate: int,
    feature_rate: int,
) -> _AudioSyncFeatures:
    # 特徴抽出全体を cache して、同じ source 音源への再計算を避ける
    sync_fns = _load_synctoolbox_functions()
    audio = _load_cached_resampled_mono_audio(audio_path, sample_rate)
    tuning_offset = sync_fns.estimate_tuning(audio, sample_rate)
    chroma_quantized, dlnco_features = _build_sync_features(
        audio=audio,
        sample_rate=sample_rate,
        feature_rate=feature_rate,
        tuning_offset=tuning_offset,
        audio_to_pitch_features_fn=sync_fns.audio_to_pitch_features,
        pitch_to_chroma_fn=sync_fns.pitch_to_chroma,
        quantize_chroma_fn=sync_fns.quantize_chroma,
        audio_to_pitch_onset_features_fn=sync_fns.audio_to_pitch_onset_features,
        pitch_onset_features_to_dlnco_fn=sync_fns.pitch_onset_features_to_dlnco,
    )
    chroma_quantized.setflags(write=False)
    dlnco_features.setflags(write=False)
    return _AudioSyncFeatures(
        chroma_quantized=chroma_quantized,
        dlnco_features=dlnco_features,
    )


@lru_cache(maxsize=8)
def _load_cached_resampled_mono_audio(audio_path: str, sample_rate: int) -> np.ndarray:
    # 音声 decode と resample も cache して、同じ path の再読込を避ける
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "audio_sync alignment requires librosa. "
            "Install it before running recipes.tools.alignment.precompute_alignments "
            "with alignment.method=audio_sync."
        ) from exc

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    audio.setflags(write=False)
    return audio


def _build_sync_features(
    audio: np.ndarray,
    sample_rate: int,
    feature_rate: int,
    tuning_offset: int,
    audio_to_pitch_features_fn,
    pitch_to_chroma_fn,
    quantize_chroma_fn,
    audio_to_pitch_onset_features_fn,
    pitch_onset_features_to_dlnco_fn,
) -> tuple[np.ndarray, np.ndarray]:
    # sync.py と同じく chroma と DLNCO onset feature を組み合わせる
    pitch_features = audio_to_pitch_features_fn(
        f_audio=audio,
        Fs=sample_rate,
        tuning_offset=tuning_offset,
        feature_rate=feature_rate,
        verbose=False,
    )
    chroma_features = pitch_to_chroma_fn(f_pitch=pitch_features)
    chroma_quantized = quantize_chroma_fn(f_chroma=chroma_features)

    onset_features = audio_to_pitch_onset_features_fn(
        f_audio=audio,
        Fs=sample_rate,
        tuning_offset=tuning_offset,
        verbose=False,
    )
    dlnco_features = pitch_onset_features_to_dlnco_fn(
        f_peaks=onset_features,
        feature_rate=feature_rate,
        feature_sequence_length=chroma_quantized.shape[1],
        visualize=False,
    )
    return chroma_quantized, dlnco_features
