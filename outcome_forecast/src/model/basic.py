from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn, Tensor
from konductor.data import get_dataset_properties
from konductor.init import ModuleInitConfig
from konductor.models import MODEL_REGISTRY, ExperimentInitConfig
from konductor.models._pytorch import TorchModelConfig


@dataclass
class BaseConfig(TorchModelConfig):
    image_enc: ModuleInitConfig = ModuleInitConfig("image-v1", {})
    scalar_enc: ModuleInitConfig = ModuleInitConfig("scalar-v1", {})
    decoder: ModuleInitConfig = ModuleInitConfig("scalar-v1", {})

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0) -> Any:
        props = get_dataset_properties(config)
        model_cfg = config.model[idx].args
        model_cfg["image_enc"]["args"]["in_ch"] = props["image_ch"]
        model_cfg["scalar_enc"]["args"]["in_ch"] = props["scalar_ch"]

        if model_cfg["scalar_enc"]["type"] == "scalar-v2":
            model_cfg["scalar_enc"]["args"]["timerange"] = props["timepoints"]

        return super().from_config(config, idx)

    def __post_init__(self):
        if isinstance(self.image_enc, dict):
            self.image_enc = ModuleInitConfig(**self.image_enc)
        if isinstance(self.scalar_enc, dict):
            self.scalar_enc = ModuleInitConfig(**self.scalar_enc)
        if isinstance(self.decoder, dict):
            self.decoder = ModuleInitConfig(**self.decoder)


class SnapshotPredictor(nn.Module):
    """Make basic prediction of game outcome based on single snapshot observation"""

    def __init__(
        self, image_enc: nn.Module, scalar_enc: nn.Module, decoder: nn.Module
    ) -> None:
        super().__init__()
        self.image_enc = image_enc
        self.scalar_enc = scalar_enc
        self.decoder = decoder

    def forward(self, step_data: dict[str, Tensor]) -> Tensor:
        """Step data features should be [B,T,...]"""
        # Merge batch and time dimension for minimaps
        image_feats = self.image_enc(step_data["minimap_features"].flatten(0, 1))

        # Process scalar features per timestep
        scalar_feats = []
        for tidx in range(step_data["scalar_features"].shape[1]):
            scalar_feats.append(self.scalar_enc(step_data["scalar_features"][:, tidx]))
        # Make same shape as image feats
        scalar_feats = torch.stack(scalar_feats, dim=1).flatten(0, 1)

        # Stack image and scalar features, decode, then reshape to [B, T, 1]
        all_feats = torch.cat([image_feats, scalar_feats], dim=-1)
        result = self.decoder(all_feats).reshape(step_data["win"].shape[0], -1)
        return result


@dataclass
@MODEL_REGISTRY.register_module("snapshot-prediction")
class SnapshotConfig(BaseConfig):
    """Basic snapshot model configuration"""

    def get_instance(self, *args, **kwargs) -> Any:
        image_enc = MODEL_REGISTRY[self.image_enc.type](**self.image_enc.args)
        scalar_enc = MODEL_REGISTRY[self.scalar_enc.type](**self.scalar_enc.args)
        decoder = MODEL_REGISTRY[self.decoder.type](
            in_ch=image_enc.out_ch + scalar_enc.out_ch, **self.decoder.args
        )
        return self._apply_extra(SnapshotPredictor(image_enc, scalar_enc, decoder))


class SequencePredictor(nn.Module):
    """
    Predict the outcome of a game depending based on a small sequence of observations.
    The decoder should handle unraveling a set of stacked features
    """

    def __init__(
        self, image_enc: nn.Module, scalar_enc: nn.Module, decoder: nn.Module
    ) -> None:
        super().__init__()
        self.image_enc = image_enc
        self.scalar_enc = scalar_enc
        self.decoder = decoder

    def forward(self, step_data: dict[str, Tensor]) -> Tensor:
        """"""
        batch_sz, n_timestep = step_data["scalar_features"].shape[:2]
        # Merge batch and time dimension for minimaps
        image_feats = self.image_enc(step_data["minimap_features"].flatten(0, 1))
        image_feats = image_feats.reshape(batch_sz, n_timestep, -1)

        # Process scalar features per timestep
        scalar_feats = []
        for tidx in range(n_timestep):
            scalar_feats.append(self.scalar_enc(step_data["scalar_features"][:, tidx]))
        # Make same shape as image feats
        scalar_feats = torch.stack(scalar_feats, dim=1)

        # Stack image and scalar features, decode, then reshape to [B, T, 1]
        all_feats = torch.cat([image_feats, scalar_feats], dim=-1)

        return self.decoder(all_feats)


@dataclass
@MODEL_REGISTRY.register_module("sequence-prediction")
class SequenceConfig(BaseConfig):
    """Basic snapshot model configuration"""

    def get_instance(self, *args, **kwargs) -> Any:
        image_enc = MODEL_REGISTRY[self.image_enc.type](**self.image_enc.args)
        scalar_enc = MODEL_REGISTRY[self.scalar_enc.type](**self.scalar_enc.args)
        decoder = MODEL_REGISTRY[self.decoder.type](
            in_ch=image_enc.out_ch + scalar_enc.out_ch, **self.decoder.args
        )
        return self._apply_extra(SequencePredictor(image_enc, scalar_enc, decoder))
