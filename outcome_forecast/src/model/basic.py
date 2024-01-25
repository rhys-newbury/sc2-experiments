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
    image_enc: ModuleInitConfig | None = None
    scalar_enc: ModuleInitConfig | None = None
    decoder: ModuleInitConfig = field(kw_only=True)

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0) -> Any:
        props = get_dataset_properties(config)
        model_cfg = config.model[idx].args

        if image_enc := model_cfg.get("image_enc", None):
            image_enc["args"]["in_ch"] = props["image_ch"]

        if scalar_enc := model_cfg.get("scalar_enc", None):
            scalar_enc["args"]["in_ch"] = props["scalar_ch"]
            if scalar_enc["type"] == "scalar-v2":
                scalar_enc["args"]["timerange"] = props["timepoints"]

        assert image_enc or scalar_enc, "At least Image or Scalar encoder required"

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
        self,
        image_enc: nn.Module | None,
        scalar_enc: nn.Module | None,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.image_enc = image_enc
        self.scalar_enc = scalar_enc
        self.decoder = decoder

    def forward(self, step_data: dict[str, Tensor]) -> Tensor:
        """Step data features should be [B,T,...]"""
        # Merge batch and time dimension for minimaps
        feats: list[Tensor] = []
        if self.image_enc is not None:
            feats.append(self.image_enc(step_data["minimap_features"].flatten(0, 1)))

        if self.scalar_enc is not None:
            # Process scalar features per timestep
            scalar_feats = []
            for tidx in range(step_data["scalar_features"].shape[1]):
                scalar_feats.append(
                    self.scalar_enc(step_data["scalar_features"][:, tidx])
                )
            # Make same shape as image feats
            feats.append(torch.stack(scalar_feats, dim=1).flatten(0, 1))

        # Cat feats if more than 1, or remove list dimension
        all_feats = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]

        result = self.decoder(all_feats).reshape(step_data["win"].shape[0], -1)

        return result


@dataclass
@MODEL_REGISTRY.register_module("snapshot-prediction")
class SnapshotConfig(BaseConfig):
    """Basic snapshot model configuration"""

    def get_instance(self, *args, **kwargs) -> Any:
        def get_enc_ch(conf: ModuleInitConfig | None):
            """Get encoder model and channels if not None"""
            if conf is None:
                return None, 0
            model = MODEL_REGISTRY[conf.type](**conf.args)
            return model, model.out_ch

        image_enc, image_ch = get_enc_ch(self.image_enc)
        scalar_enc, scalar_ch = get_enc_ch(self.scalar_enc)

        decoder = MODEL_REGISTRY[self.decoder.type](
            in_ch=image_ch + scalar_ch, **self.decoder.args
        )
        return self._apply_extra(SnapshotPredictor(image_enc, scalar_enc, decoder))


class SequencePredictor(nn.Module):
    """
    Predict the outcome of a game depending based on a small sequence of observations.
    The decoder should handle unraveling a set of stacked features
    """

    def __init__(
        self,
        image_enc: nn.Module | None,
        scalar_enc: nn.Module | None,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.image_enc = image_enc
        self.scalar_enc = scalar_enc
        self.decoder = decoder

    def forward(self, step_data: dict[str, Tensor]) -> Tensor:
        """"""
        batch_sz, n_timestep = step_data["scalar_features"].shape[:2]
        feats: list[Tensor] = []
        if self.image_enc is not None:
            # Merge batch and time dimension for minimaps
            image_feats = self.image_enc(step_data["minimap_features"].flatten(0, 1))
            feats.append(image_feats.reshape(batch_sz, n_timestep, -1))

        if self.scalar_enc is not None:
            # Process scalar features per timestep
            scalar_feats = []
            for tidx in range(n_timestep):
                scalar_feats.append(
                    self.scalar_enc(step_data["scalar_features"][:, tidx])
                )
            # Make same shape as image feats
            feats.append(torch.stack(scalar_feats, dim=1))

        # Stack image and scalar features, decode, then reshape to [B, T, 1]
        all_feats = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]

        return self.decoder(all_feats)


@dataclass
@MODEL_REGISTRY.register_module("sequence-prediction")
class SequenceConfig(BaseConfig):
    """Basic snapshot model configuration"""

    def get_instance(self, *args, **kwargs) -> Any:
        def get_enc_ch(conf: ModuleInitConfig | None):
            """Get encoder model and channels if not None"""
            if conf is None:
                return None, 0
            model = MODEL_REGISTRY[conf.type](**conf.args)
            return model, model.out_ch

        image_enc, image_ch = get_enc_ch(self.image_enc)
        scalar_enc, scalar_ch = get_enc_ch(self.scalar_enc)

        decoder = MODEL_REGISTRY[self.decoder.type](
            in_ch=image_ch + scalar_ch, **self.decoder.args
        )
        return self._apply_extra(SequencePredictor(image_enc, scalar_enc, decoder))
