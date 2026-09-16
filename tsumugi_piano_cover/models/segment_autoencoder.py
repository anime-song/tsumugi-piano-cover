from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from tsumugi_piano_cover.config import SegmentAutoencoderConfig, TargetRollConfig
from tsumugi_piano_cover.data.segment import segments_to_roll
from tsumugi_piano_cover.models.transformer import Transformer


class SegmentEncoder(nn.Module):
    # Encode each segment into 4 query-latent tokens.
    def __init__(self, config: SegmentAutoencoderConfig, roll_config: TargetRollConfig) -> None:
        super().__init__()
        self.config = config
        self.roll_config = roll_config
        self.frame_projection = nn.Linear(roll_config.feature_dim, config.d_model)
        self.frame_positions = nn.Parameter(torch.zeros(1, roll_config.frames_per_phrase, config.d_model))
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_latent_queries, config.d_model))
        self.encoder = Transformer(
            input_dim=config.d_model,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_layers=config.encoder_layers,
            ffn_hidden_size_factor=config.ff_multiplier,
            dropout=config.dropout,
            output_norm=True,
        )
        bottleneck_dim = config.d_model * config.num_latent_queries
        self.to_mu = nn.Linear(bottleneck_dim, config.latent_dim)
        self.to_logvar = nn.Linear(bottleneck_dim, config.latent_dim)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.zeros_(self.to_logvar.bias)

    def forward(self, segments: torch.Tensor) -> torch.Tensor:
        # segments: [batch_size, num_segments, num_frames, feature_dim]
        batch_size, num_segments, num_frames, _ = segments.shape
        hidden = self.frame_projection(rearrange(segments, "b s f d -> (b s) f d"))
        hidden = hidden + self.frame_positions[:, :num_frames]

        queries = self.query_tokens.expand(batch_size * num_segments, -1, -1)
        hidden = torch.cat([queries, hidden], dim=1)
        encoded = self.encoder(hidden)

        # [batch_size, num_segments, num_latent_queries, d_model]
        return rearrange(
            encoded[:, : self.config.num_latent_queries],
            "(b s) q d -> b s q d",
            b=batch_size,
        )

    def posterior(self, prefix_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Compress 4 latent tokens into a single bottleneck vector.
        flattened = rearrange(prefix_latents, "b s q d -> b s (q d)")
        mu = self.to_mu(flattened)
        logvar = self.to_logvar(flattened)
        return mu, logvar


class SegmentDecoder(nn.Module):
    # Decode a segment from 4 prefix latent tokens.
    def __init__(self, config: SegmentAutoencoderConfig, roll_config: TargetRollConfig) -> None:
        super().__init__()
        self.config = config
        self.roll_config = roll_config
        self.prefix_positions = nn.Parameter(torch.zeros(1, config.num_latent_queries, config.d_model))
        self.frame_queries = nn.Parameter(torch.zeros(1, roll_config.frames_per_phrase, config.d_model))
        self.decoder = Transformer(
            input_dim=config.d_model,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_layers=config.decoder_layers,
            ffn_hidden_size_factor=config.ff_multiplier,
            dropout=config.dropout,
            use_cross_attention=False,
            output_norm=True,
        )
        self.frame_head = nn.Linear(config.d_model, roll_config.feature_dim)

    def forward(self, prefix_latents: torch.Tensor) -> dict[str, torch.Tensor]:
        # prefix_latents: [batch_size, num_segments, num_latent_queries, d_model]
        batch_size, num_segments, num_prefix, d_model = prefix_latents.shape
        if num_prefix != self.config.num_latent_queries or d_model != self.config.d_model:
            raise ValueError(
                "unexpected prefix latent shape: "
                f"{tuple(prefix_latents.shape)}, expected (*, *, {self.config.num_latent_queries}, {self.config.d_model})"
            )

        prefix_tokens = rearrange(prefix_latents, "b s q d -> (b s) q d")
        prefix_tokens = prefix_tokens + self.prefix_positions[:, :num_prefix]

        frame_queries = self.frame_queries.expand(batch_size * num_segments, -1, -1)
        decoder_input = torch.cat([prefix_tokens, frame_queries], dim=1)
        decoded = self.decoder(decoder_input)

        frame_hidden = decoded[:, self.config.num_latent_queries :]
        frame_logits = rearrange(self.frame_head(frame_hidden), "(b s) f d -> b s f d", b=batch_size)
        return {"frame_logits": frame_logits}


class SegmentLatentAutoencoder(nn.Module):
    # Two-stage phrase VAE:
    # 1) AE stage uses 4 encoder latents directly as decoder prefix.
    # 2) VAE stage compresses them to latent_dim, then expands back to 4 prefix tokens.
    def __init__(self, config: SegmentAutoencoderConfig, roll_config: TargetRollConfig) -> None:
        super().__init__()
        self.config = config
        self.roll_config = roll_config
        self.encoder = SegmentEncoder(config, roll_config)
        self.decoder = SegmentDecoder(config, roll_config)
        self.latent_to_prefix = nn.Sequential(
            nn.Linear(config.latent_dim, config.d_model * 2),
            nn.SiLU(),
            nn.Linear(config.d_model * 2, config.d_model * config.num_latent_queries),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _latent_vector_to_prefix(self, latents: torch.Tensor) -> torch.Tensor:
        return rearrange(
            self.latent_to_prefix(latents),
            "b s (q d) -> b s q d",
            q=self.config.num_latent_queries,
            d=self.config.d_model,
        )

    def encode(
        self,
        segments: torch.Tensor,
        sample_posterior: bool = True,
        variational: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_latents = self.encoder(segments)
        if variational:
            mu, logvar = self.encoder.posterior(prefix_latents)
            latents = self.reparameterize(mu, logvar) if sample_posterior else mu
        else:
            batch_size, num_segments = prefix_latents.shape[:2]
            mu = torch.zeros(
                (batch_size, num_segments, self.config.latent_dim),
                dtype=prefix_latents.dtype,
                device=prefix_latents.device,
            )
            logvar = torch.zeros_like(mu)
            latents = prefix_latents
        return latents, mu, logvar

    def decode(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        if latents.dim() == 4:
            prefix_latents = latents
        elif latents.dim() == 3:
            prefix_latents = self._latent_vector_to_prefix(latents)
        else:
            raise ValueError(f"unexpected latent rank: {latents.dim()}, expected 3 or 4")

        decoded = self.decoder(prefix_latents)
        decoded["prefix_latents"] = prefix_latents
        return decoded

    @torch.no_grad()
    def reconstruct_roll(
        self,
        outputs: dict[str, torch.Tensor],
        segment_times: torch.Tensor,
        segment_valid_lengths: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        del segment_valid_lengths

        if int(outputs["frame_logits"].shape[0]) != 1:
            raise ValueError("reconstruct_roll expects outputs for a single song batch")

        segment_rolls = torch.sigmoid(outputs["frame_logits"][0].detach())
        return segments_to_roll(
            segment_rolls=segment_rolls,
            segment_times=segment_times.to(segment_rolls.device),
            num_frames=int(num_frames),
            config=self.roll_config,
        )

    def forward(
        self,
        segments: torch.Tensor,
        sample_posterior: bool = True,
        variational: bool = True,
    ) -> dict[str, torch.Tensor]:
        latents, mu, logvar = self.encode(
            segments,
            sample_posterior=sample_posterior,
            variational=variational,
        )
        recon = self.decode(latents)
        recon["latents"] = latents
        recon["mu"] = mu
        recon["logvar"] = logvar
        recon["variational"] = variational
        return recon
