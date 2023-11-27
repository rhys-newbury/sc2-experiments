from dataclasses import dataclass
from typing import Any

import torch
from torch import nn, Tensor
from konductor.data import get_dataset_properties
from konductor.init import ModuleInitConfig
from konductor.models import MODEL_REGISTRY, ExperimentInitConfig
from konductor.models._pytorch import TorchModelConfig


class BasicPredictor(nn.Module):
    """Make basic prediction of game outcome based on current observation"""

    def __init__(
        self, image_enc: nn.Module, scalar_enc: nn.Module, decoder: nn.Module
    ) -> None:
        super().__init__()
        self.image_enc = image_enc
        self.scalar_enc = scalar_enc
        self.decoder = decoder

    def forward(self, step_data: dict[str, Tensor]) -> Tensor:
        image_feats = self.image_enc(step_data["minimap_features"].flatten(0, 1))
        scalar_feats = self.scalar_enc(step_data["scalar_features"].flatten(0, 1))
        all_feats = torch.cat([image_feats, scalar_feats], dim=-1)
        result = self.decoder(all_feats).reshape(step_data["win"].shape[0], -1, 1)
        return result


@dataclass
@MODEL_REGISTRY.register_module("win-pred-basic")
class BasicPredictorConfig(TorchModelConfig):
    image_enc: ModuleInitConfig = ModuleInitConfig("image-v1", {})
    scalar_enc: ModuleInitConfig = ModuleInitConfig("scalar-v1", {})
    decoder: ModuleInitConfig = ModuleInitConfig("scalar-v1", {})

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0) -> Any:
        props = get_dataset_properties(config)
        model_cfg = config.model[idx].args
        model_cfg["image_enc"]["args"]["in_ch"] = props["image_ch"]
        model_cfg["scalar_enc"]["args"]["in_ch"] = props["scalar_ch"]
        return super().from_config(config, idx)

    def __post_init__(self):
        if isinstance(self.image_enc, dict):
            self.image_enc = ModuleInitConfig(**self.image_enc)
        if isinstance(self.scalar_enc, dict):
            self.scalar_enc = ModuleInitConfig(**self.scalar_enc)
        if isinstance(self.decoder, dict):
            self.decoder = ModuleInitConfig(**self.decoder)

    def get_instance(self, *args, **kwargs) -> Any:
        image_enc = MODEL_REGISTRY[self.image_enc.type](**self.image_enc.args)
        scalar_enc = MODEL_REGISTRY[self.scalar_enc.type](**self.scalar_enc.args)
        decoder = MODEL_REGISTRY[self.decoder.type](
            in_ch=image_enc.out_ch + scalar_enc.out_ch, **self.decoder.args
        )
        return self._apply_extra(BasicPredictor(image_enc, scalar_enc, decoder))
