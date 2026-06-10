from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from src_v2.config import SourceEncoderConfig
from src_v2.models.transformer import Transformer


class SourceNoteChunkEncoder(nn.Module):
    # チャンク内の音符シーケンスを固定次元ベクトルにエンコード
    def __init__(self, config: SourceEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.continuous_projection = nn.Linear(config.note_feature_dim, config.d_model)
        self.program_embedding = nn.Embedding(config.num_programs, config.d_model)
        self.drum_embedding = nn.Embedding(2, config.d_model)
        self.track_role_embedding = nn.Embedding(config.num_source_track_roles, config.d_model)
        # 複数 query token でチャンク内の情報を読み出す
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_chunk_queries, config.d_model))
        self.query_projection = nn.Linear(config.d_model * config.num_chunk_queries, config.d_model)

        self.note_encoder = Transformer(
            input_dim=config.d_model,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_layers=config.chunk_encoder_layers,
            ffn_hidden_size_factor=config.ff_multiplier,
            dropout=config.dropout,
            output_norm=True,
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        source_features: torch.Tensor,
        source_programs: torch.Tensor,
        source_drums: torch.Tensor,
        source_track_roles: torch.Tensor,
        source_note_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_chunks, max_notes, _ = source_features.shape

        # 特徴量の射影と埋め込みの加算
        note_states = self.continuous_projection(source_features)
        note_states = (
            note_states
            + self.program_embedding(source_programs)
            + self.drum_embedding(source_drums)
            + self.track_role_embedding(source_track_roles)
        )
        note_states = rearrange(note_states, "b c n d -> (b c) n d")

        # ステート（クエリ + 音符）の結合
        query_tokens = self.query_tokens.expand(batch_size * num_chunks, -1, -1)
        note_states = torch.cat([query_tokens, note_states], dim=1)

        # マスク（クエリ用 + 音符用）の結合
        query_valid = torch.ones(
            (batch_size * num_chunks, self.config.num_chunk_queries),
            dtype=torch.bool,
            device=source_note_mask.device,
        )
        valid_notes = rearrange(source_note_mask, "b c n -> (b c) n")
        attention_mask = torch.cat([query_valid, valid_notes], dim=1)

        encoded = self.note_encoder(note_states, attention_mask=attention_mask)
        # 複数 query token の出力を 1 本のチャンク表現に圧縮
        queried = rearrange(encoded[:, : self.config.num_chunk_queries], "bc q d -> bc (q d)")
        chunk_embeddings = self.output_norm(self.query_projection(queried))
        return rearrange(chunk_embeddings, "(b c) d -> b c d", b=batch_size)


class SongEncoder(nn.Module):
    # チャンク表現に時間情報を付与し、曲全体としてエンコード
    def __init__(self, config: SourceEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.time_projection = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.encoder = Transformer(
            input_dim=config.d_model,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_layers=config.song_encoder_layers,
            ffn_hidden_size_factor=config.ff_multiplier,
            dropout=config.dropout,
            output_norm=True,
        )

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        source_chunk_mask: torch.Tensor,
        source_chunk_times: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # 時間情報の埋め込みを加算
        time_embeddings = self.time_projection(source_chunk_times.unsqueeze(-1))
        hidden = chunk_embeddings + time_embeddings
        encoded, layer_outputs = self.encoder(hidden, attention_mask=source_chunk_mask, return_hiddens=True)
        return encoded, layer_outputs
