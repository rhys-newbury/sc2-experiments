import torch
from torch import nn, Tensor
from konductor.models import MODEL_REGISTRY


@MODEL_REGISTRY.register_module("transformer-v1")
class TransformerDecoderV1(nn.Module):
    is_logit_output = True

    def __init__(
        self,
        in_ch: int,
        max_time: float,
        hidden_size: int = 32,
        num_layers: int = 2,
        num_time_freq: int = 8,
        max_time_freq: int = 8,
        num_latents: int = 8,
    ):
        """Max time in minutes, will be converted internally to gameloops"""
        super().__init__()
        self.squeeze = nn.Linear(in_ch, hidden_size)
        self.decode = nn.Linear(num_latents * hidden_size, 1)

        self.max_time = max_time / (22.4 * 60)
        self.num_freq = num_time_freq
        self.max_freq = max_time_freq

        self.transformer = nn.Transformer(
            d_model=hidden_size + self.num_freq * 2,
            num_encoder_layers=num_layers,
            num_decoder_layers=1,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
        )

        self.latents = nn.Parameter(
            torch.empty(num_latents, hidden_size + self.num_freq * 2)
        )
        with torch.no_grad():
            self.latents.normal_(0.0, 0.5).clamp_(-2.0, 2.0)

    @torch.no_grad()
    def create_time_embeddings(self, timepoints: Tensor):
        """Create position embeddings based on timepoints relative to maximum time"""
        frequencies = torch.linspace(
            1.0, self.max_freq / 2.0, self.num_freq, device=timepoints.device
        )
        timepoints_norm = timepoints / self.max_time
        frequency_grid = timepoints_norm[..., None] * frequencies[None, ...]
        encodings = [
            torch.sin(torch.pi * frequency_grid),
            torch.cos(torch.pi * frequency_grid),
        ]
        return torch.cat(encodings, dim=-1)

    def forward(self, inputs: Tensor, timesteps: Tensor):
        """Input is a [B, T, C] tensor, timesteps is [B, T] in gamestep units, returns [B, T]"""
        time_embed = self.create_time_embeddings(timesteps)
        cat_features = torch.cat([self.squeeze(inputs), time_embed], dim=-1)
        latents = self.latents.unsqueeze(0).expand(inputs.shape[0], -1, -1)
        mask = self.transformer.generate_square_subsequent_mask(
            cat_features.shape[1], device=cat_features.device
        )
        out: Tensor = self.transformer(
            cat_features, latents, src_mask=mask, src_is_causal=True
        )
        decoded: Tensor = self.decode(
            out.view(inputs.shape[0], -1, self.latents.nelement())
        )

        return decoded.squeeze(-1)
