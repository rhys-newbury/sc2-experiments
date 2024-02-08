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
    encoder: ModuleInitConfig
    decoder: ModuleInitConfig
    history_len: int = 4

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0) -> Any:
        props = get_dataset_properties(config)
        model_cfg = config.model[idx].args
        image_enc = model_cfg["image_enc"]
        image_enc["in_ch"] = props["image_ch"] * model_cfg.get(
            "history_len", BaseConfig.history_len
        )
        return super().from_config(config, idx)

    def __post_init__(self):
        if isinstance(self.encoder, dict):
            self.encoder = ModuleInitConfig(**self.encoder)
        if isinstance(self.decoder, dict):
            self.decoder = ModuleInitConfig(**self.decoder)


@dataclass
@MODEL_REGISTRY.register_module("conv-forecast")
class ConvConfig(BaseConfig):
    def get_instance(self, *args, **kwargs) -> Any:
        encoder = MODEL_REGISTRY[self.encoder.type](**self.encoder.args)
        decoder = MODEL_REGISTRY[self.decoder.type](**self.decoder.args)
        return ConvForecaster(encoder, decoder, self.history_len)


class ConvForecaster(nn.Module):
    """Given a stack of old minimaps, predict the next one"""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, history_len: int):
        self.encoder = encoder
        self.decoder = decoder
        self.history_len = history_len

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """"""
        minimaps = inputs["minimap_features"]
        ntime = minimaps.shape[1]
        preds: list[Tensor] = []
        for start_idx in range(ntime - self.history_len):
            end_idx = start_idx + self.history_len + 1
            feats = self.encoder(minimaps[:, start_idx:end_idx])
            preds.append(self.decoder(feats))

        return torch.stack(preds, dim=1)
