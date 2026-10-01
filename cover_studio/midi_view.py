"""画面のピアノロールと再生に渡す形 (秒) に MIDI を読む"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from piano_cover.source import (
    DRUM_ID,
    INSTRUMENTS,
    ROW_A,
    ROW_B,
    ROW_C,
    ROW_D,
    ROW_ONSET,
    ROW_TYPE,
    TYPE_BEAT,
    TYPE_NOTE,
    load_source,
)

FRAME_RATE = 100

# 原曲の音を色分けする大まかな種類
_GROUPS = {
    "melody": ("melody", "vocal_harmony"),
    "bass": ("acoustic_bass", "electric_bass", "slap_bass", "synth_bass"),
    "keys": ("piano", "electric_piano", "organ", "plucked_keyboard", "chromatic_percussion", "accordion_family"),
    "guitar": (
        "acoustic_guitar",
        "distorted_guitar",
        "electric_guitar_clean",
        "electric_guitar_muted",
        "guitar_harmonics",
    ),
}
GROUP_NAMES = ("melody", "bass", "keys", "guitar", "other")
_GROUP_OF = {INSTRUMENTS.index(name): GROUP_NAMES.index(g) for g, names in _GROUPS.items() for name in names}


def _chord_name(text: str) -> str | None:
    """tsumugi のコードマーカー ("A:m7/E"・"C:maj"・"N") を "Am7/E"・"C" にする"""
    text = text.strip()
    if not text or ":" not in text:
        return None if text in ("", "N") else text
    root, rest = text.split(":", 1)
    bass = ""
    if "/" in rest:
        rest, bass = rest.split("/", 1)
    quality = "" if rest == "maj" else rest
    return root + quality + (f"/{bass}" if bass else "")


def source_view(path: Path) -> dict:
    from symusic import Score

    rows, end_frame = load_source(path, FRAME_RATE)
    notes = rows[(rows[:, ROW_TYPE] == TYPE_NOTE) & (rows[:, ROW_D] != DRUM_ID)]
    groups = np.array([_GROUP_OF.get(int(i), len(GROUP_NAMES) - 1) for i in notes[:, ROW_D]], dtype=np.int64)
    beats = rows[rows[:, ROW_TYPE] == TYPE_BEAT]
    chords = []
    for marker in Score(str(path)).to("second").markers:
        name = _chord_name(marker.text)
        if name is not None:
            chords.append([round(float(marker.time), 3), name])
    return {
        "duration": end_frame / FRAME_RATE,
        "groups": list(GROUP_NAMES),
        # [onset 秒, 長さ 秒, pitch, velocity, 種類]
        "notes": [
            [o / FRAME_RATE, d / FRAME_RATE, p, v, g]
            for o, d, p, v, g in zip(
                notes[:, ROW_ONSET].tolist(),
                notes[:, ROW_B].tolist(),
                notes[:, ROW_A].tolist(),
                notes[:, ROW_C].tolist(),
                groups.tolist(),
            )
        ],
        "beats": (beats[:, ROW_ONSET] / FRAME_RATE).round(3).tolist(),
        "downbeats": (beats[beats[:, ROW_A] == 1, ROW_ONSET] / FRAME_RATE).round(3).tolist(),
        "chords": chords,
    }


def cover_view(path: Path) -> dict:
    """[onset, 長さ, pitch, velocity, 鳴り終わり] (鳴り終わりはペダルで延びた所まで。再生に使う) とペダル"""
    from symusic import Score

    score = Score(str(path)).to("second")
    notes, pedals = [], []
    for track in score.tracks:
        notes += [(float(n.time), float(n.duration), int(n.pitch), int(n.velocity)) for n in track.notes]
        pedals += [(float(p.time), float(p.time + p.duration)) for p in track.pedals]
    notes.sort()
    pedals.sort()
    starts = np.array([p[0] for p in pedals])
    out = []
    for onset, duration, pitch, velocity in notes:
        release = onset + duration
        i = int(np.searchsorted(starts, release, side="right")) - 1
        if i >= 0 and pedals[i][0] <= release < pedals[i][1]:
            release = pedals[i][1]
        out.append([round(onset, 3), round(duration, 3), pitch, velocity, round(release, 3)])
    end = max([n[4] for n in out], default=0.0)
    return {"duration": end, "notes": out, "pedals": [[round(a, 3), round(b, 3)] for a, b in pedals]}
