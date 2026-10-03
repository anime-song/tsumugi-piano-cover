"""楽譜の位置のトークン (MTIME・BEAT・FRAC) の予測に、その位置に対応する時刻の近くにある演奏の打鍵の量を直接足す。
piano_cover.model.OnsetHead の楽譜版。

cross-attention で演奏を見ていても、生成では小節線や拍の位置が少しずつ演奏からずれ、音が隣の小節に入っていく。
ここでは候補ごとに演奏の時刻を見積もり、その近くの打鍵の量を種類 (CHANNELS) と幅 (WIDTHS) ごとに数えた特徴を作って、
Local の出力 h との内積をその候補の logit に足す。どの種類の打鍵にどれだけ寄せるかは h (いまの文脈) で決まる。
    MTIME  候補の値 v ごとに、小節の開始 = 前の小節の開始 + v (曲の最初の小節は 最初の音 - v)
    BEAT   候補の拍 b ごとに、小節の開始 + b x 4 分音符の長さの見積もり
    FRAC   直前の拍 b と候補の分数 f ごとに、小節の開始 + (b + f) x 4 分音符の長さの見積もり
各種類のトークン全体の確率は元に戻し、その中の配分だけを変える (MTIME を出すか BEAT を出すかなどは変えない)。
query を 0 で初期化するので、足した直後は足す前と同じ出力になる (学習済みのモデルに後から足して続きから学習できる)。

打鍵の量は、小節の基準の時刻から local_back 前〜 local_ahead 後をフレームごとのヒストグラムにして、ガウス窓で
ぼかしてから候補の時刻で読む (候補がいくつあっても、読むだけで済む)。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from piano_score.tokenizer import MLEN, ScoreTokenizer

from .data import FRAMES_PER_SECOND, PerformanceVocab

# 打鍵の種類: すべての音・低い音 (中央ハより下)・強い音 (ベロシティ 72 以上)・ペダルの踏み込み
CHANNELS = ("note", "low", "loud", "pedal_down")
LOW_PITCH = 60
LOUD_VELOCITY = 72


def row_channels(features: Tensor, vocab: PerformanceVocab) -> Tensor:
    """演奏の行の特徴 [..., 4] -> 打鍵の種類ごとの重み [..., len(CHANNELS)]"""
    o = vocab.offset
    kind = features[..., 0] - o["type"]
    note = kind == 0
    pitch = features[..., 1] - o["pitch"]
    velocity = (features[..., 2] - o["velocity"]) * 128 / vocab.velocity_bins
    return torch.stack(
        (note, note & (pitch < LOW_PITCH), note & (velocity >= LOUD_VELOCITY), kind == 1), dim=-1
    ).float()


class ScoreOnsetHead(nn.Module):
    WIDTHS = (2.0, 6.0)  # フレーム

    def __init__(self, dim: int, hidden: int, tokenizer: ScoreTokenizer, back: int, ahead: int) -> None:
        super().__init__()
        self.back = back
        self.span = back + ahead + 1
        radius = int(3 * max(self.WIDTHS))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernels = torch.stack([torch.exp(-0.5 * (x / w) ** 2) for w in self.WIDTHS])[:, None]
        self.register_buffer("kernels", kernels, persistent=False)
        features = len(CHANNELS) * len(self.WIDTHS)
        self.candidate = nn.Sequential(nn.Linear(features, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.query = nn.Linear(dim, hidden, bias=False)

        def token_range(kind: str) -> tuple[slice, Tensor]:
            ids = [i for i, k in enumerate(tokenizer.kinds) if k == kind]
            assert ids == list(range(ids[0], ids[-1] + 1)), f"{kind} のトークンの番号が連続していない"
            return slice(ids[0], ids[-1] + 1), torch.tensor([float(tokenizer.values[i]) for i in ids])

        self.mtime_slice, _ = token_range("mtime")
        self.beat_slice, beat_values = token_range("beat")
        self.frac_slice, frac_values = token_range("frac")
        self.register_buffer("mtime_values", torch.from_numpy(tokenizer.mtime_values).float() * FRAMES_PER_SECOND,
                             persistent=False)  # fmt: skip
        self.register_buffer("beat_values", beat_values, persistent=False)
        self.register_buffer("frac_values", frac_values, persistent=False)

    # ------------------------------------------------------------------
    # 特徴
    # ------------------------------------------------------------------
    @torch.no_grad()
    def smoothed(self, pos: Tensor, valid: Tensor, channels: Tensor) -> Tensor:
        """打鍵の時刻 pos [N, M] (基準からの相対フレーム)・有効 [N, M]・種類 [N, M, C] -> ぼかしたヒストグラム [N, C*W, span]"""
        N = pos.shape[0]
        C = channels.shape[-1]
        index = (pos.round() + self.back).long()
        inside = valid & (index >= 0) & (index < self.span)
        weights = (channels * inside[..., None]).transpose(1, 2).float()  # [N, C, M]
        hist = torch.zeros(N, C, self.span, device=pos.device)
        hist.scatter_add_(2, index.clamp(0, self.span - 1)[:, None, :].expand(-1, C, -1), weights)
        out = F.conv1d(hist.reshape(N * C, 1, self.span), self.kernels, padding=self.kernels.shape[-1] // 2)
        return torch.log1p(out.reshape(N, C * len(self.WIDTHS), self.span))

    def read(self, smoothed: Tensor, times: Tensor) -> Tensor:
        """ぼかしたヒストグラム [N, F, span] を候補の時刻 times [N, K] (相対フレーム) で読む -> 候補の埋め込み [N, K, hidden]"""
        index = (times.round() + self.back).long()
        inside = (index >= 0) & (index < self.span)
        values = smoothed.gather(2, index.clamp(0, self.span - 1)[:, None, :].expand(-1, smoothed.shape[1], -1))
        features = (values * inside[:, None, :]).transpose(1, 2)
        with torch.autocast(features.device.type, enabled=False):
            return self.candidate(features.float())

    def candidates(self, smoothed: Tensor, start: Tensor, spq: Tensor, first: Tensor) -> tuple[Tensor, Tensor]:
        """小節ごとの MTIME と BEAT の候補の埋め込み ([N, 128, hidden], [N, 33, hidden])。
        start / spq [N] は小節の開始 (基準からの相対) と 4 分音符の長さ、first [N] は曲の最初の小節か"""
        mtime_times = torch.where(first[:, None], -self.mtime_values[None], self.mtime_values[None])
        beat_times = start[:, None] + self.beat_values[None] * spq[:, None]
        return self.read(smoothed, mtime_times), self.read(smoothed, beat_times)

    def frac_candidates(self, smoothed: Tensor, start: Tensor, spq: Tensor, beat: Tensor) -> Tensor:
        """直前の拍 beat [N] のあとの FRAC の候補の埋め込み [N, 81, hidden]"""
        times = start[:, None] + (beat[:, None] + self.frac_values[None]) * spq[:, None]
        return self.read(smoothed, times)

    # ------------------------------------------------------------------
    # logit に足す
    # ------------------------------------------------------------------
    @staticmethod
    def _shift(logits: Tensor, bias: Tensor) -> Tensor:
        """logits [..., K] に bias を足して、K 全体の確率は元に戻す。すべて -inf (出せない) の所はそのまま"""
        before = logits.logsumexp(-1, keepdim=True)
        shifted = logits + bias
        return torch.where(torch.isfinite(before), shifted - shifted.logsumexp(-1, keepdim=True) + before, logits)

    def forward(
        self,
        h: Tensor,
        logits: Tensor,
        mtime: Tensor | None,
        beat: Tensor | None,
        frac: Tensor | None = None,
        frac_where: tuple[Tensor, Tensor] | None = None,
        mtime_rows: Tensor | None = None,
    ) -> Tensor:
        """h [N, T, D], logits [N, T, V]。mtime / beat [N, K, hidden] は小節ごとの候補 (None なら足さない)。
        MTIME は位置 0 だけ (mtime_rows [N] が False の行は足さない)、BEAT はすべての位置に足す。
        frac [P, 81, hidden] と frac_where (行, 位置) は FRAC を足す位置とその候補"""
        with torch.autocast(h.device.type, enabled=False):
            q = self.query(h.float())
            logits = logits.float()
            # 種類ごとに足してからつなぎ直す (その場で書き換えると逆伝播できない)
            pieces = {}
            if mtime is not None:
                s = self.mtime_slice
                bias = torch.einsum("nh,nkh->nk", q[:, 0], mtime)
                if mtime_rows is not None:
                    bias = bias * mtime_rows[:, None]
                first = self._shift(logits[:, :1, s], bias[:, None])
                pieces["mtime"] = torch.cat((first, logits[:, 1:, s]), dim=1)
            if beat is not None:
                s = self.beat_slice
                pieces["beat"] = self._shift(logits[..., s], torch.einsum("nth,nkh->ntk", q, beat))
            if frac is not None and frac_where is not None and len(frac_where[0]):
                n, t = frac_where
                s = self.frac_slice
                bias = torch.zeros_like(logits[..., s]).index_put((n, t), torch.einsum("ph,pkh->pk", q[n, t], frac))
                pieces["frac"] = self._shift(logits[..., s], bias)
            parts, cursor = [], 0
            for name, s in (("mtime", self.mtime_slice), ("beat", self.beat_slice), ("frac", self.frac_slice)):
                if name in pieces:
                    parts += [logits[..., cursor : s.start], pieces[name]]
                    cursor = s.stop
            parts.append(logits[..., cursor:])
            return torch.cat(parts, dim=-1)


def frac_positions(prefix: Tensor, beat_value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """FRAC を予測しうる位置 (直前の入力が小節の長さでない BEAT の位置) の (行, 位置, その拍) [P]。
    位置は Local の出力の位置 (BOS + prefix の t 番目 = prefix[t-1] の次)"""
    if prefix.shape[1] == 0:
        empty = prefix.new_zeros(0)
        return empty, empty, empty.float()
    beat = beat_value[prefix]
    previous = F.pad(prefix, (1, 0), value=-1)[:, :-1]
    is_beat = (beat >= 0) & (previous != MLEN)
    n, i = is_beat.nonzero(as_tuple=True)
    return n, i + 1, beat[n, i]

