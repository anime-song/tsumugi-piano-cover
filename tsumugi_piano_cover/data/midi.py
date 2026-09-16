from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from symusic import Score


@dataclass(frozen=True)
class NoteEvent:
    start: float
    end: float
    pitch: int
    velocity: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


def _score_to_second(path: str | Path):
    return Score.from_file(str(path)).to("second")


def _sorted_target_events(
    path: str | Path,
    min_duration_seconds: float,
) -> tuple[list[NoteEvent], list[tuple[float, float]]]:
    score = _score_to_second(path)
    events: list[NoteEvent] = []
    pedals: list[tuple[float, float]] = []
    for track in score.tracks:
        for note in track.notes:
            start = float(note.time)
            end = float(note.time + note.duration)
            if end - start < min_duration_seconds:
                end = start + min_duration_seconds
            events.append(
                NoteEvent(
                    start=start,
                    end=end,
                    pitch=int(note.pitch),
                    velocity=int(note.velocity),
                )
            )
        for pedal in track.pedals:
            pedals.append((float(pedal.time), float(pedal.time + pedal.duration)))

    events.sort(key=lambda event: (event.start, event.pitch))
    pedals.sort()
    return events, pedals


def _trim_events(events: list[NoteEvent]) -> tuple[list[NoteEvent], float]:
    if not events:
        return [], 0.0
    trim = min(event.start for event in events)
    trimmed = [
        NoteEvent(
            start=event.start - trim,
            end=event.end - trim,
            pitch=event.pitch,
            velocity=event.velocity,
        )
        for event in events
    ]
    return trimmed, trim


def _trim_pedals(pedals: list[tuple[float, float]], trim: float) -> list[tuple[float, float]]:
    return [(start - trim, end - trim) for start, end in pedals]


def load_trimmed_target_events(
    path: str | Path,
    min_duration_seconds: float = 0.03,
) -> tuple[list[NoteEvent], list[tuple[float, float]]]:
    trimmed_events, trimmed_pedals, _ = load_trimmed_target_events_with_offset(
        path,
        min_duration_seconds=min_duration_seconds,
    )
    return trimmed_events, trimmed_pedals


def load_trimmed_target_events_with_offset(
    path: str | Path,
    min_duration_seconds: float = 0.03,
) -> tuple[list[NoteEvent], list[tuple[float, float]], float]:
    events, pedals = _sorted_target_events(path, min_duration_seconds=min_duration_seconds)
    trimmed_events, trim = _trim_events(events)
    trimmed_pedals = _trim_pedals(pedals, trim)
    return trimmed_events, trimmed_pedals, trim
