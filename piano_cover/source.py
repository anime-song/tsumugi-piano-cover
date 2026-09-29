"""原曲 (tsumugi で採譜した MIDI) を、カバーモデルの原曲エンコーダに入れる形にする。

原曲は「行」の配列 int32 [N, 6] = (onset, type, a, b, c, d) で持つ。時間はデコーダと同じフレーム単位。
    NOTE : a = pitch (0-127, ドラムは GM の打楽器番号), b = duration, c = velocity, d = 楽器 ID (INSTRUMENTS)
    BEAT : a = 1 なら小節の頭
    CHORD: a = root (0-11, 12 はコードなし N), b = 種類 ID (CHORD_QUALITIES), c = ベース音 (0-11, 12 は転回なし)
    KEY  : a = 主音 (0-11), b = 0 長調 / 1 短調
時刻は原曲の音声の先頭からの絶対時刻のまま (アラインメントの原曲側の時刻と同じ基準) で、先頭の無音も詰めない。

エンコーダに入れるときは source_features で 2 秒パッチごとに並べ、各行を埋め込み表の番号 5 個 (種類 + 属性 4 つ) にする。
パッチの先頭には「その時点で鳴っているコード」と「その時点のキー」の行を足す。キーは曲の途中で変わるので、
曲全体の条件にはせず、パッチごとに今のキーが分かるようにする。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from piano_ar.config import TokenizerConfig
from piano_ar.tokenizer import PianoTokenizer

ROW_ONSET, ROW_TYPE, ROW_A, ROW_B, ROW_C, ROW_D = range(6)
TYPE_NOTE, TYPE_BEAT, TYPE_CHORD, TYPE_KEY = range(4)
# パッチの先頭に足す「その時点のコード / キー」(source_features の中でだけ使う)
TYPE_CHORD_STATE, TYPE_KEY_STATE = 4, 5
NUM_TYPES = 6

# tsumugi の楽器クラス (instrument_agnostic_amt.taxonomy.instrument_classes) + それ以外
INSTRUMENTS = (
    "accordion_family", "acoustic_bass", "acoustic_guitar", "brass", "choir", "chromatic_percussion",
    "distorted_guitar", "drums", "electric_bass", "electric_guitar_clean", "electric_guitar_muted",
    "electric_piano", "ethnic", "flute_pipe", "guitar_harmonics", "harmonica", "orchestra_hit",
    "orchestral_harp", "orchestral_woodwind", "organ", "percussive_fx", "piano", "pizzicato_strings",
    "plucked_keyboard", "sax", "slap_bass", "sound_fx", "strings", "synth_bass", "synth_fx", "synth_lead",
    "synth_pad", "timpani", "melody", "vocal_harmony", "wind_chimes", "other",
)  # fmt: skip
INSTRUMENT_ID = {name: i for i, name in enumerate(INSTRUMENTS)}
DRUM_ID = INSTRUMENT_ID["drums"]

# tsumugi のコードマーカー ("Ab:add9/C" の ":" と "/" の間) の種類。長三和音は種類なしで書かれるので "maj" にする
CHORD_QUALITIES = (
    "N", "maj", "m", "7", "m7", "M7", "sus4", "m7-5", "m7(9)", "add9", "M7(9)", "6", "dim", "dim7", "7(b9)",
    "aug", "7(b13)", "m7(11)", "7(9)", "5", "7sus4", "m6", "69", "7(b9,b13)", "mM7", "7(#9)", "7(13)",
    "7(9,13)", "sus2", "madd9", "7-5", "augM7", "7(#9,b13)", "m7(9,11)", "M7(13)", "7(9,#11)", "aug7",
    "M7(#11)", "m69", "mM7(9)", "7(b9,13)", "other",
)  # fmt: skip
QUALITY_ID = {name: i for i, name in enumerate(CHORD_QUALITIES)}
NO_CHORD = 12
NO_BASS = 12

_NOTE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _pitch_class(name: str) -> int:
    value = _NOTE_NAMES[name[0]]
    for accidental in name[1:]:
        value += {"#": 1, "b": -1}[accidental]
    return value % 12


def parse_chord(text: str) -> tuple[int, int, int] | None:
    """ "Ab:add9/C" -> (root, 種類 ID, ベース音)。読めないものは None"""
    text = text.strip()
    if text == "N":
        return NO_CHORD, QUALITY_ID["N"], NO_BASS
    main, _, bass = text.partition("/")
    root_name, _, quality = main.partition(":")
    try:
        root = _pitch_class(root_name)
        bass_pc = _pitch_class(bass) if bass else NO_BASS
    except (KeyError, IndexError):
        return None
    return root, QUALITY_ID.get(quality or "maj", QUALITY_ID["other"]), bass_pc


class _TempoMap:
    """tick -> 秒 (MIDI のテンポ変化を積分する)"""

    def __init__(self, score) -> None:
        tpq = score.ticks_per_quarter
        tempos = sorted((int(t.time), float(t.mspq)) for t in score.tempos) or [(0, 500_000.0)]
        if tempos[0][0] > 0:
            tempos.insert(0, (0, 500_000.0))
        self.ticks = np.array([t for t, _ in tempos], dtype=np.float64)
        self.seconds_per_tick = np.array([m for _, m in tempos], dtype=np.float64) / 1e6 / tpq
        self.start_seconds = np.concatenate([[0.0], np.cumsum(np.diff(self.ticks) * self.seconds_per_tick[:-1])])

    def __call__(self, ticks: np.ndarray) -> np.ndarray:
        index = np.searchsorted(self.ticks, ticks, side="right") - 1
        return self.start_seconds[index] + (ticks - self.ticks[index]) * self.seconds_per_tick[index]


def _beat_rows(score, frame_rate: int) -> list[list[int]]:
    """拍子記号とテンポマップから拍と小節の頭を作る。tsumugi は推定した拍が MIDI の拍に乗るよう
    テンポマップと拍子を書き出すので、MIDI の拍をそのまま使う。拍の推定をしていない (拍子記号がない) 曲は拍なし"""
    signatures = sorted((int(t.time), t.numerator, t.denominator) for t in score.time_signatures)
    if not signatures:
        return []
    tempo_map = _TempoMap(score)
    end_tick = int(score.end())
    beat_ticks, downbeats = [], []
    for i, (start, numerator, denominator) in enumerate(signatures):
        stop = signatures[i + 1][0] if i + 1 < len(signatures) else end_tick
        step = score.ticks_per_quarter * 4 / denominator
        count = max(0, int(np.ceil((stop - start) / step - 1e-6)))
        ticks = start + np.arange(count) * step
        beat_ticks.append(ticks)
        downbeats.append(np.arange(count) % max(numerator, 1) == 0)
    if not beat_ticks:
        return []
    frames = np.round(tempo_map(np.concatenate(beat_ticks)) * frame_rate).astype(np.int64)
    flags = np.concatenate(downbeats)
    return [[int(f), TYPE_BEAT, int(d), 0, 0, 0] for f, d in zip(frames, flags)]


def load_source(path: str | Path, frame_rate: int) -> tuple[np.ndarray, int]:
    """tsumugi の MIDI を行の配列にして、曲の終端フレームと一緒に返す"""
    from symusic import Score

    score_ticks = Score(str(path))
    score = score_ticks.to("second")
    rows: list[list[int]] = []
    end = 0.0
    for track in score.tracks:
        name = track.name.strip()
        instrument = DRUM_ID if track.is_drum else INSTRUMENT_ID.get(name, INSTRUMENT_ID["other"])
        for n in track.notes:
            onset = int(round(n.time * frame_rate))
            duration = max(1, int(round(n.duration * frame_rate)))
            rows.append([onset, TYPE_NOTE, int(n.pitch), duration, int(np.clip(n.velocity, 1, 127)), instrument])
            end = max(end, n.time + n.duration)
    end_frame = int(np.ceil(end * frame_rate)) + 1

    for marker in score.markers:
        chord = parse_chord(marker.text)
        if chord is not None:
            rows.append([int(round(marker.time * frame_rate)), TYPE_CHORD, *chord, 0])
    for key in score.key_signatures:
        # key は調号の # の数 (負なら b)。長調の主音は 5 度圏で 7 半音ずつ進む
        tonic = (7 * int(key.key)) % 12
        minor = int(key.tonality) == 1
        rows.append(
            [int(round(key.time * frame_rate)), TYPE_KEY, (tonic + 9) % 12 if minor else tonic, int(minor), 0, 0]
        )
    rows.extend(_beat_rows(score_ticks, frame_rate))

    if not rows:
        return np.zeros((0, 6), dtype=np.int32), 0
    array = np.asarray(rows, dtype=np.int64)
    array = array[array[:, ROW_ONSET] < end_frame]
    return sort_rows(array).astype(np.int32), end_frame


# 学習できる onset-bias (CoverModel の OnsetHead) で原曲の onset を分ける種類
ONSET_GROUPS = ("melody", "keys", "guitar", "bass", "other", "kick", "snare", "drums", "beat", "downbeat", "chord")
_INSTRUMENT_ONSET_GROUP = {
    "melody": "melody", "vocal_harmony": "melody",
    "piano": "keys", "electric_piano": "keys", "organ": "keys", "plucked_keyboard": "keys",
    "chromatic_percussion": "keys", "accordion_family": "keys",
    "acoustic_guitar": "guitar", "distorted_guitar": "guitar", "electric_guitar_clean": "guitar",
    "electric_guitar_muted": "guitar", "guitar_harmonics": "guitar",
    "acoustic_bass": "bass", "electric_bass": "bass", "slap_bass": "bass", "synth_bass": "bass",
}  # fmt: skip
_KICK, _SNARE = (35, 36), (37, 38, 39, 40)


def onset_group_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """行から onset の種類 (ONSET_GROUPS の番号、-1 は使わない) を引く表。

    返り値は (楽器 -> 種類, ドラムの打楽器番号 -> 種類, 種類の番号 -> 種類)。
    行の種類が NOTE なら楽器 (ドラムは打楽器番号) で、BEAT は小節の頭かどうか、CHORD はコードの変わり目で分ける"""
    group = {name: i for i, name in enumerate(ONSET_GROUPS)}
    instrument = np.array([group[_INSTRUMENT_ONSET_GROUP.get(name, "other")] for name in INSTRUMENTS], dtype=np.int64)
    drum = np.full(128, group["drums"], dtype=np.int64)
    drum[list(_KICK)] = group["kick"]
    drum[list(_SNARE)] = group["snare"]
    row_type = np.full(NUM_TYPES, -1, dtype=np.int64)
    row_type[TYPE_CHORD] = group["chord"]
    return instrument, drum, row_type


def onset_time_bias(
    rows: np.ndarray, num_patches: int, patch_frames: int, strength: float, width_frames: float
) -> np.ndarray:
    """原曲の onset (全楽器の音と拍) に近いほど大きい値 [num_patches, patch_frames]。

    生成で TIME の logit に足して (PianoARModel.generate の time_bias)、出力の onset を原曲の onset に寄せる。
    strength * exp(-d^2 / (2 width^2))、d は一番近い原曲の onset までのフレーム数。
    カバーモデルは原曲のどこを見るかは合っているが、次の onset は自分のリズムで決めていて、
    テンポが少しずつずれては 8 分音符 1 つ分で戻る。毎回その場で原曲の onset に合わせ直すことで、ずれが溜まらなくなる。
    """
    frames = np.arange(num_patches * patch_frames)
    onsets = np.unique(rows[(rows[:, ROW_TYPE] == TYPE_NOTE) | (rows[:, ROW_TYPE] == TYPE_BEAT), ROW_ONSET])
    if len(onsets) == 0:
        return np.zeros((num_patches, patch_frames), dtype=np.float32)
    index = np.searchsorted(onsets, frames)
    after = onsets[np.minimum(index, len(onsets) - 1)]
    before = onsets[np.maximum(index - 1, 0)]
    distance = np.minimum(np.abs(after - frames), np.abs(frames - before))
    bias = strength * np.exp(-0.5 * (distance / width_frames) ** 2)
    return bias.reshape(num_patches, patch_frames).astype(np.float32)


def sort_rows(rows: np.ndarray) -> np.ndarray:
    return rows[np.lexsort((rows[:, ROW_A], rows[:, ROW_TYPE], rows[:, ROW_ONSET]))]


# ----------------------------------------------------------------------
# 拡張 (カバー側と同じ量をかける)
# ----------------------------------------------------------------------
def stretch_rows(rows: np.ndarray, factor: float) -> np.ndarray:
    rows = rows.astype(np.int64)
    rows[:, ROW_ONSET] = np.round(rows[:, ROW_ONSET] * factor)
    notes = rows[:, ROW_TYPE] == TYPE_NOTE
    rows[notes, ROW_B] = np.maximum(1, np.round(rows[notes, ROW_B] * factor))
    return rows


def transpose_rows(rows: np.ndarray, shift: int) -> np.ndarray:
    rows = rows.astype(np.int64)
    pitched = (rows[:, ROW_TYPE] == TYPE_NOTE) & (rows[:, ROW_D] != DRUM_ID)
    rows[pitched, ROW_A] += shift
    rows = rows[~pitched | ((rows[:, ROW_A] >= 0) & (rows[:, ROW_A] <= 127))]
    chords = rows[:, ROW_TYPE] == TYPE_CHORD
    for column, empty in ((ROW_A, NO_CHORD), (ROW_C, NO_BASS)):
        target = chords & (rows[:, column] != empty)
        rows[target, column] = (rows[target, column] + shift) % 12
    keys = rows[:, ROW_TYPE] == TYPE_KEY
    rows[keys, ROW_A] = (rows[keys, ROW_A] + shift) % 12
    return rows


# ----------------------------------------------------------------------
# エンコーダの入力
# ----------------------------------------------------------------------
class SourceVocab:
    """行の属性を 1 つの埋め込み表の番号にする。0 はパディング (ゼロベクトル)"""

    def __init__(self, tokenizer_config: TokenizerConfig) -> None:
        sizes = {
            "type": NUM_TYPES,
            "pitch": 128,
            "duration": tokenizer_config.num_duration_bins,
            "velocity": tokenizer_config.num_velocity_bins,
            "instrument": len(INSTRUMENTS),
            "downbeat": 2,
            "root": 13,
            "quality": len(CHORD_QUALITIES),
            "bass": 13,
            "tonic": 12,
            "mode": 2,
        }
        self.offset: dict[str, int] = {}
        next_index = 1
        for name, size in sizes.items():
            self.offset[name] = next_index
            next_index += size
        self.size = next_index


def source_features(
    rows: np.ndarray,
    end_frame: int,
    tokenizer: PianoTokenizer,
    vocab: SourceVocab,
    max_rows: int,
) -> dict[str, np.ndarray]:
    """行をパッチごとに並べる。

    返り値 (S = 曲のパッチ数, R = 1 パッチの行数の最大):
        features [S, R, 5] int16  埋め込み表の番号 (種類, 属性 4 つ)。0 はパディング
        onset    [S, R] int32     行の絶対フレーム (Local の cross-attention の位置に使う)
        valid    [S, R] bool
    """
    F = tokenizer.patch_frames
    num_patches = max(1, -(-end_frame // F))
    rows = rows.astype(np.int64)
    rows = rows[(rows[:, ROW_ONSET] >= 0) & (rows[:, ROW_ONSET] < num_patches * F)]

    # パッチの先頭に「その時点のコード / キー」を足す (パッチの先頭ちょうどで変わる場合はその行があるので足さない)
    starts = np.arange(num_patches) * F
    state_rows = []
    for row_type, state_type in ((TYPE_CHORD, TYPE_CHORD_STATE), (TYPE_KEY, TYPE_KEY_STATE)):
        changes = rows[rows[:, ROW_TYPE] == row_type]
        if len(changes) == 0:
            continue
        index = np.searchsorted(changes[:, ROW_ONSET], starts, side="left") - 1
        has = index >= 0
        state = changes[index[has]].copy()
        state[:, ROW_ONSET] = starts[has]
        state[:, ROW_TYPE] = state_type
        state_rows.append(state)
    if state_rows:
        rows = np.concatenate([rows, *state_rows])
    # 同じパッチの中はコード/キーの状態 -> 時刻順
    is_state = rows[:, ROW_TYPE] >= TYPE_CHORD_STATE
    rows = rows[np.lexsort((rows[:, ROW_TYPE], rows[:, ROW_ONSET], ~is_state, rows[:, ROW_ONSET] // F))]

    patch = rows[:, ROW_ONSET] // F
    first = np.searchsorted(patch, np.arange(num_patches))
    rank = np.arange(len(rows)) - first[patch]
    keep = rank < max_rows
    rows, patch, rank = rows[keep], patch[keep], rank[keep]
    width = max(1, int(rank.max(initial=-1)) + 1)

    o = vocab.offset
    t = rows[:, ROW_TYPE]
    a, b, c, d = rows[:, ROW_A], rows[:, ROW_B], rows[:, ROW_C], rows[:, ROW_D]
    feature = np.zeros((len(rows), 5), dtype=np.int64)
    feature[:, 0] = o["type"] + t
    note = t == TYPE_NOTE
    feature[note, 1] = o["pitch"] + a[note]
    feature[note, 2] = o["duration"] + tokenizer.duration_bin(b[note])
    feature[note, 3] = o["velocity"] + tokenizer.velocity_bin(c[note])
    feature[note, 4] = o["instrument"] + d[note]
    beat = t == TYPE_BEAT
    feature[beat, 1] = o["downbeat"] + a[beat]
    chord = (t == TYPE_CHORD) | (t == TYPE_CHORD_STATE)
    feature[chord, 1] = o["root"] + a[chord]
    feature[chord, 2] = o["quality"] + b[chord]
    feature[chord, 3] = o["bass"] + c[chord]
    key = (t == TYPE_KEY) | (t == TYPE_KEY_STATE)
    feature[key, 1] = o["tonic"] + a[key]
    feature[key, 2] = o["mode"] + b[key]

    features = np.zeros((num_patches, width, 5), dtype=np.int16)
    onset = np.zeros((num_patches, width), dtype=np.int32)
    valid = np.zeros((num_patches, width), dtype=bool)
    features[patch, rank] = feature
    onset[patch, rank] = rows[:, ROW_ONSET]
    valid[patch, rank] = True
    return {"features": features, "onset": onset, "valid": valid}
