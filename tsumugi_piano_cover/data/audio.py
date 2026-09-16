from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def load_audio(path: str | Path, sample_rate: int, num_channels: int = 2) -> torch.Tensor:
    """Load one audio file as [channels, samples] at the requested sample rate."""
    waveform, input_sample_rate = torchaudio.load(str(path))
    if waveform.ndim != 2 or waveform.shape[0] == 0 or waveform.shape[1] == 0:
        raise ValueError(f"audio must have shape [channels, samples]: {path}")

    if input_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, input_sample_rate, sample_rate)

    if waveform.shape[0] == num_channels:
        return waveform.contiguous()
    if waveform.shape[0] == 1:
        return waveform.repeat(num_channels, 1).contiguous()

    # Tsumugi expects stereo. Downmix multichannel files before duplicating mono.
    mono = waveform.mean(dim=0, keepdim=True)
    return mono.repeat(num_channels, 1).contiguous()


def get_audio_duration_seconds(path: str | Path) -> float:
    """Return an audio file's duration without decoding the full waveform."""
    info = torchaudio.info(str(path))
    return float(info.num_frames) / float(info.sample_rate)
