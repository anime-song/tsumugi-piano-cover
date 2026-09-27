"""生成時に 1 小節分のトークン列が楽譜の文法 (tokenizer.py の冒頭) に従うよう、次に出せるトークンを絞る。

文法の並びに加えて、次のことも守らせる。これで生成したトークン列は、そのまま読める楽譜として書き出せる。
- 位置は小節の中で増え続け、小節の長さ (拍子か MLEN) を超えない。位置には指示か塊が 1 つ以上ある
- 同じ位置の指示 (音部記号・オクターブ記号・スイング・強弱など) は番号の昇順で 1 つずつ、塊より前。状態を変えない指示は出さない
- 塊は段・声部の順 (同じ声部は装飾音が先)。声部の中で前の音が鳴っている間は次の塊を始めない。音は小節からはみ出さない
- 和音の音高は昇順。スラーの終わりは開いているスラーがあるときだけ
- 1 小節のトークン数の上限を超えない (残りが少ないと新しい位置や塊を始めない)
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import torch

from .tokenizer import CROSS, EOM, EOS, GLISS, MLEN, PAD, SLUR_START, SLUR_STOP, TIE, ScoreTokenizer

# 新しい位置は BEAT [FRAC] SV DUR PITCH、新しい塊は SV DUR PITCH の分の残りが要る
NEW_ONSET_TOKENS = 5
NEW_GROUP_TOKENS = 3
MAX_SLUR_STARTS = 4
# ヘッダの指示の後に MLEN が来た、または塊が始まった後は、それより前に置く指示を出せないようにする
CLOSED = 10**9


class _Vocab:
    """文法の判定に使うトークン番号の表 (トークナイザーごとに 1 度だけ作る)"""

    def __init__(self, tok: ScoreTokenizer) -> None:
        def ids(kind: str) -> np.ndarray:
            return np.array([i for i, k in enumerate(tok.kinds) if k == kind], dtype=np.int64)

        self.mtime, self.key = ids("mtime"), ids("key")
        # 小節の長さが上限を超える拍子 (20/1 など) は位置を書けないので出さない
        limit = tok.config.max_measure_quarters
        self.ts = np.array([i for i in ids("ts") if 4 * tok.values[i][0] / tok.values[i][1] <= limit])
        self.clef = {s: np.array([i for i in ids("clef") if tok.values[i][0] == s]) for s in (1, 2)}
        self.beat = ids("beat")  # 並び = 拍 (0, 1, 2, ...)
        self.frac = ids("frac")
        self.frac_values = [tok.values[i] for i in self.frac]
        self.sv = ids("sv")
        self.dur = ids("dur")
        self.dur_quarters = [tok.values[i].quarters for i in self.dur]
        self.art = ids("art")
        # トレモロ (斜線の本数) は 1 つの塊に 1 つだけ。奏法記号の並びの最後にある
        self.first_tremolo = min(i for i in self.art if str(tok.values[i]).startswith("tremolo"))
        self.pitch = ids("pitch")
        self.pitch_midi = np.array([tok.values[i][0] for i in self.pitch])
        # 位置ごとの指示 (番号の昇順に並べる)。音部記号 < オクターブ記号 < スイング < 強弱など
        self.item = np.sort(np.concatenate([ids("clef"), ids("ottava"), ids("swing"), ids("direction")]))
        # ヘッダに置けるのは効いているオクターブ記号とスイングだけ (none は置かない)
        self.header_extra = np.array(
            [i for i in np.concatenate([ids("ottava"), ids("swing")]) if _state_of(tok, int(i))[1] != "none"]
        )


def _state_of(tok: ScoreTokenizer, token: int) -> tuple[str, str]:
    """状態を変える指示の (状態の名前, 値)。音部記号は clef1 / clef2、オクターブ記号は ottava1 / ottava2、スイングは swing"""
    kind, value = tok.kinds[token], tok.values[token]
    if kind in ("clef", "ottava"):
        return f"{kind}{value[0]}", value[1]
    return kind, value


_VOCABS: dict[int, _Vocab] = {}


class ScoreGrammar:
    """1 小節分の生成の状態。open_slurs は前の小節までに開いて閉じていないスラーの数"""

    def __init__(self, tokenizer: ScoreTokenizer, open_slurs: int = 0) -> None:
        self.tok = tokenizer
        if id(tokenizer) not in _VOCABS:
            _VOCABS[id(tokenizer)] = _Vocab(tokenizer)
        self.v = _VOCABS[id(tokenizer)]
        self.limit = tokenizer.config.max_patch_tokens
        self.open_slurs = open_slurs
        self.count = 0
        self.stage = "mtime"
        self.finished = False
        self.song_end = False
        # 小節の状態
        self.nominal = Fraction(4)
        self.length = Fraction(4)
        self.mlen_beat = 0
        self.mlen_used = False
        self.states: dict[str, str] = {"ottava1": "none", "ottava2": "none", "swing": "none"}
        self.position: Fraction | None = None
        self.previous_position: Fraction | None = None
        self.beat = 0
        self.items_here = 0  # 今の位置の指示と塊の数
        self.last_item = -1  # 今の位置 (かヘッダ) で最後に出した指示の番号
        self.set_here: set[str] = set()  # 今の位置 (かヘッダ) で変えた状態 (同じ状態は 1 つの位置で 1 回だけ)
        self.busy: dict[int, Fraction] = {}  # 声部 (SV の番号) -> 次の塊を始められる位置
        self.last_sv = -1
        # 今の塊
        self.sv = -1
        self.duration = Fraction(0)
        self.grace = False
        self.last_art = -1
        self.stops = 0
        self.starts = 0
        self.last_pitch = -1

    # ------------------------------------------------------------------
    # 次に出せるトークン
    # ------------------------------------------------------------------
    def allowed(self) -> torch.Tensor:
        mask = torch.zeros(self.tok.vocab_size, dtype=torch.bool)
        if self.finished:
            mask[PAD] = True
            return mask
        v = self.v
        stage = self.stage
        remaining = self.limit - 1 - self.count  # 終端トークンの分を除いた残り
        if remaining <= 0:
            mask[[EOM, EOS]] = True
            return mask
        if stage == "mtime":
            mask[v.mtime] = True
        elif stage == "ts":
            mask[v.ts] = True
        elif stage == "key":
            mask[v.key] = True
        elif stage in ("clef1", "clef2"):
            mask[v.clef[1 if stage == "clef1" else 2]] = True
        elif stage == "header":
            for token in v.header_extra[v.header_extra > self.last_item]:
                if _state_of(self.tok, int(token))[0] not in self.set_here:
                    mask[token] = True
            if not self.mlen_used:
                mask[MLEN] = True
            self._allow_next_onset(mask, remaining)
            mask[[EOM, EOS]] = True
        elif stage == "mlen_beat":
            mask[v.beat] = True
        elif stage == "mlen_frac":
            # 拍子どおりの長さちょうどは MLEN を使わない
            mask[[i for i, f in zip(v.frac, v.frac_values) if self.mlen_beat + f != self.nominal]] = True
            if 0 < self.mlen_beat != self.nominal:  # 分数なし (長さ = 拍) で先へ進む
                self._allow_next_onset(mask, remaining)
                mask[[EOM, EOS]] = True
        elif stage in ("after_beat", "items"):
            if stage == "after_beat":
                previous = self.previous_position
                mask[
                    [
                        i for i, f in zip(v.frac, v.frac_values)
                        if self.beat + f < self.length and (previous is None or self.beat + f > previous)
                    ]
                ] = True  # fmt: skip
                if previous is not None and self.beat <= previous:
                    return mask  # 前の位置と同じ拍なので分数が要る
            self._allow_items(mask, remaining)
        elif stage in ("after_sv", "after_cross"):
            if stage == "after_sv":
                mask[CROSS] = True
            mask[[i for i, q in zip(v.dur, v.dur_quarters) if self.position + q <= self.length]] = True
        elif stage == "after_dur":
            if remaining > 1:  # 音高の分を残す
                if self.stops == 0 and self.starts == 0 and self.last_art < v.first_tremolo:
                    mask[v.art[v.art > self.last_art]] = True
                if self.starts == 0 and self.stops < self.open_slurs:
                    mask[SLUR_STOP] = True
                if self.starts < MAX_SLUR_STARTS:
                    mask[SLUR_START] = True
            mask[v.pitch] = True
        elif stage in ("after_pitch", "after_tie", "after_gliss"):
            if stage == "after_pitch":
                mask[TIE] = True
            if stage != "after_gliss":
                mask[GLISS] = True
            mask[v.pitch[v.pitch_midi > self.last_pitch]] = True
            self._allow_groups(mask, remaining)
            self._allow_next_onset(mask, remaining)
            mask[[EOM, EOS]] = True
        return mask

    def _allow_next_onset(self, mask: torch.Tensor, remaining: int) -> None:
        """次の位置の BEAT。今の位置より後で、小節の長さより前に置けるものだけ"""
        if remaining < NEW_ONSET_TOKENS:
            return
        current = self.position
        for beat, token in enumerate(self.v.beat):
            if beat >= self.length:
                break
            if current is None or beat > current:
                mask[token] = True
            elif beat == int(current) and any(current < beat + f < self.length for f in self.v.frac_values):
                mask[token] = True  # 同じ拍の中の後ろの位置 (分数が要る)

    def _allow_items(self, mask: torch.Tensor, remaining: int) -> None:
        """位置を決めた後: 指示 (番号の昇順、状態を変えるものだけ)、塊、次の位置、小節の終わり"""
        v = self.v
        for token in v.item[v.item > self.last_item]:
            name, value = _state_of(self.tok, int(token))
            if name == "direction":
                mask[token] = True
            # 状態 (音部記号・オクターブ記号・スイング) は変えるときだけ、1 つの位置で 1 回。小節の頭の状態はヘッダに置く
            elif self.states.get(name) != value and name not in self.set_here and self.position > 0:
                mask[token] = True
        self._allow_groups(mask, remaining)
        if self.items_here > 0:  # 位置には指示か塊が 1 つ以上要る
            self._allow_next_onset(mask, remaining)
            mask[[EOM, EOS]] = True

    def _allow_groups(self, mask: torch.Tensor, remaining: int) -> None:
        if remaining < NEW_GROUP_TOKENS:
            return
        # 開いている塊 (装飾音でないもの) の声部は、その音が鳴っている間なので同じ位置では続けられない
        open_main = self.sv if self.stage in ("after_pitch", "after_tie", "after_gliss") and not self.grace else -1
        for token in self.v.sv:
            token = int(token)
            if token == open_main:
                continue
            # 段・声部の順。同じ声部は、前の塊が装飾音のときだけ続けられる (本体の後は鳴っている間なので busy で弾かれる)
            if token >= self.last_sv and self.busy.get(token, Fraction(0)) <= self.position:
                mask[token] = True

    # ------------------------------------------------------------------
    # 出したトークンで状態を進める
    # ------------------------------------------------------------------
    def update(self, token: int) -> None:
        if self.finished:
            return
        self.count += 1
        tok = self.tok
        kind = tok.kinds[token]
        if self.stage in ("after_pitch", "after_tie", "after_gliss") and token not in (TIE, GLISS) and kind != "pitch":
            self._close_group()
        if token in (EOM, EOS):
            self.finished = True
            self.song_end = token == EOS
        elif kind == "mtime":
            self.stage = "ts"
        elif kind == "ts":
            beats, beat_type = tok.values[token]
            self.nominal = self.length = Fraction(4 * beats, beat_type)
            self.stage = "key"
        elif kind == "key":
            self.stage = "clef1"
        elif kind == "clef" and self.stage in ("clef1", "clef2"):
            self.states[f"clef{tok.values[token][0]}"] = tok.values[token][1]
            self.stage = "clef2" if self.stage == "clef1" else "header"
        elif token == MLEN:
            self.mlen_used = True
            self.last_item = CLOSED
            self.stage = "mlen_beat"
        elif kind == "beat" and self.stage == "mlen_beat":
            self.mlen_beat = tok.values[token]
            self.length = Fraction(self.mlen_beat)
            self.stage = "mlen_frac"
        elif kind == "frac" and self.stage == "mlen_frac":
            self.length = self.mlen_beat + tok.values[token]
            self.stage = "header"
        elif kind == "beat":
            self.previous_position = self.position
            self.beat = tok.values[token]
            self.position = Fraction(self.beat)
            self.items_here, self.last_item, self.last_sv = 0, -1, -1
            self.set_here = set()
            self.stage = "after_beat"
        elif kind == "frac":
            self.position = self.beat + tok.values[token]
            self.stage = "items"
        elif kind in ("clef", "ottava", "swing", "direction"):
            name, value = _state_of(tok, token)
            if name in self.states or name.startswith("clef"):
                self.states[name] = value
                self.set_here.add(name)
            self.last_item = token
            if self.stage != "header":
                self.items_here += 1
                self.stage = "items"
        elif kind == "sv":
            self.sv = token
            self.last_sv = token  # 以降の塊はこの段・声部より後
            self.items_here += 1
            self.last_item = CLOSED  # 塊の後に指示は置けない
            self.last_art, self.stops, self.starts, self.last_pitch = -1, 0, 0, -1
            self.stage = "after_sv"
        elif token == CROSS:
            self.stage = "after_cross"
        elif kind == "dur":
            duration = tok.values[token]
            self.duration = duration.quarters
            self.grace = duration.grace is not None
            self.stage = "after_dur"
        elif kind == "art":
            self.last_art = token
        elif token == SLUR_STOP:
            self.stops += 1
        elif token == SLUR_START:
            self.starts += 1
        elif kind == "pitch":
            self.last_pitch = tok.values[token][0]
            self.stage = "after_pitch"
        elif token == TIE:
            self.stage = "after_tie"
        elif token == GLISS:
            self.stage = "after_gliss"

    def _close_group(self) -> None:
        if not self.grace:
            self.busy[self.sv] = self.position + self.duration
        self.open_slurs += self.starts - self.stops
        self.stage = "items"
