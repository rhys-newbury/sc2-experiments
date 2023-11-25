import torch
from torch import nn, Tensor


class BasicPredictor(nn.Module):
    def __init__(
        self, image_enc: nn.Module, scalar_enc: nn.Module, decoder: nn.Module
    ) -> None:
        super().__init__()
        self.image_enc = image_enc
        self.scalar_enc = scalar_enc
        self.decoder = decoder

    def forward(self, step_data) -> Tensor:
        image_feats = self.image_enc(step_data["minimap_features"])
        scalar_feats = self.scalar_enc(step_data["scalar_features"])

        all_feats = torch.cat([image_feats, scalar_feats], dim=-1)
        result = self.decoder(all_feats)
        return result
