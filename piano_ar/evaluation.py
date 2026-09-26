"""学習中の生成評価。サンプルを MIDI で保存し、wandb 用に音声・ピアノロール・統計量を作る。

音声は外部の音源に頼らず、減衰する倍音を足し合わせた簡易シンセで作る (音色の評価ではなく、
リズム・和声・ペダルの使い方を耳で確かめるためのもの)。
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .tokenizer import DURATION, KIND, KIND_NOTE, KIND_PEDAL_OFF, KIND_PEDAL_ON, ONSET, PITCH, VELOCITY, PianoTokenizer


def _pedal_intervals(events: np.ndarray, end_frame: int) -> np.ndarray:
    """ペダルの (on, off) フレームの組。閉じていないペダルは end_frame で離す"""
    intervals = []
    down = None
    for onset, kind in events[:, [ONSET, KIND]].tolist():
        if kind == KIND_PEDAL_ON and down is None:
            down = onset
        elif kind == KIND_PEDAL_OFF and down is not None:
            intervals.append((down, onset))
            down = None
    if down is not None:
        intervals.append((down, max(end_frame, down + 1)))
    return np.asarray(intervals, dtype=np.int64).reshape(-1, 2)


def events_end_frame(events: np.ndarray) -> int:
    if len(events) == 0:
        return 0
    notes = events[:, KIND] == KIND_NOTE
    return int(max((events[notes, ONSET] + events[notes, DURATION]).max(initial=0), events[:, ONSET].max() + 1))


def synthesize(events: np.ndarray, frame_rate: int, sample_rate: int = 16000) -> np.ndarray:
    """ペダルで離鍵が延びるところまで含めて、簡易的なピアノ風の波形 (float32, -1..1) を作る"""
    end_frame = events_end_frame(events)
    pedals = _pedal_intervals(events, end_frame)
    notes = events[events[:, KIND] == KIND_NOTE]
    total = int((end_frame / frame_rate + 1.0) * sample_rate)
    audio = np.zeros(total, dtype=np.float32)
    harmonics = np.array([1.0, 0.45, 0.2, 0.1])
    for onset, _, pitch, duration, velocity in notes.tolist():
        release = onset + duration
        # 離鍵の時点でペダルが踏まれていれば、そのペダルが離れるまで音が残る
        held = np.flatnonzero((pedals[:, 0] <= release) & (release < pedals[:, 1]))
        if len(held):
            release = int(pedals[held[0], 1])
        start = int(onset / frame_rate * sample_rate)
        release_sample = int((release - onset) / frame_rate * sample_rate)
        length = min(release_sample + int(0.3 * sample_rate), 6 * sample_rate, total - start)
        if length <= 0:
            continue
        t = np.arange(length) / sample_rate
        f0 = 440.0 * 2 ** ((pitch - 69) / 12)
        envelope = np.exp(-t * (1.5 + f0 / 400))
        after_release = np.maximum(0, np.arange(length) - release_sample) / sample_rate
        envelope *= np.exp(-after_release * 15)
        tone = np.zeros(length)
        for k, amplitude in enumerate(harmonics, 1):
            if f0 * k < sample_rate / 2:
                tone += amplitude * np.sin(2 * np.pi * f0 * k * t) * np.exp(-t * k * 0.5)
        audio[start : start + length] += (tone * envelope * (velocity / 127) ** 1.5).astype(np.float32)
    peak = float(np.abs(audio).max())
    return audio / peak * 0.9 if peak > 0 else audio


def write_wav(audio: np.ndarray, path: str | Path, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def piano_roll_image(
    events: np.ndarray,
    tokenizer: PianoTokenizer,
    *,
    frames_per_pixel: int = 5,
    pixels_per_key: int = 4,
    prompt_frames: int = 0,
) -> np.ndarray:
    """横が時間・縦が音高の RGB 画像 [H, W, 3] (uint8)。明るさがベロシティ、下の帯がペダル、赤線がプロンプトの終わり"""
    c = tokenizer.config
    end_frame = max(events_end_frame(events), prompt_frames, 1)
    width = end_frame // frames_per_pixel + 1
    keys = c.pitch_max - c.pitch_min + 1
    pedal_height = 6
    image = np.full((keys * pixels_per_key + pedal_height + 2, width, 3), 20, dtype=np.uint8)
    # オクターブごと (C) に薄い横線
    for pitch in range(c.pitch_min, c.pitch_max + 1):
        if pitch % 12 == 0:
            row = (c.pitch_max - pitch + 1) * pixels_per_key - 1
            image[row, :] = 45
    for onset, kind, pitch, duration, velocity in events.tolist():
        if kind != KIND_NOTE:
            continue
        top = (c.pitch_max - pitch) * pixels_per_key
        x0 = onset // frames_per_pixel
        x1 = max(x0 + 1, (onset + duration) // frames_per_pixel)
        brightness = 80 + int(175 * velocity / 127)
        image[top : top + pixels_per_key - 1, x0:x1] = (brightness, int(brightness * 0.8), 60)
        image[top : top + pixels_per_key - 1, x0] = 255
    for on, off in _pedal_intervals(events, end_frame).tolist():
        image[-pedal_height:, on // frames_per_pixel : max(on // frames_per_pixel + 1, off // frames_per_pixel)] = (
            90,
            160,
            230,
        )
    if prompt_frames:
        image[:, min(prompt_frames // frames_per_pixel, width - 1)] = (230, 60, 60)
    return image


def sample_stats(events: np.ndarray, frame_rate: int) -> dict[str, float]:
    """生成物の傾向を学習データと比べるための簡単な統計量"""
    end_frame = max(events_end_frame(events), 1)
    notes = events[events[:, KIND] == KIND_NOTE]
    seconds = end_frame / frame_rate
    if len(notes) == 0:
        return {"seconds": seconds, "notes_per_sec": 0.0}
    pedals = _pedal_intervals(events, end_frame)
    _, chord_sizes = np.unique(notes[:, ONSET], return_counts=True)
    return {
        "seconds": seconds,
        "notes_per_sec": len(notes) / seconds,
        "notes_per_onset": float(chord_sizes.mean()),
        "pitch_mean": float(notes[:, PITCH].mean()),
        "pitch_std": float(notes[:, PITCH].std()),
        "velocity_mean": float(notes[:, VELOCITY].mean()),
        "duration_median_sec": float(np.median(notes[:, DURATION]) / frame_rate),
        "pedal_ratio": float((pedals[:, 1] - pedals[:, 0]).sum() / end_frame) if len(pedals) else 0.0,
    }
