from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from konductor.data import get_dataset_properties
from konductor.init import ModuleInitConfig
from konductor.models import MODEL_REGISTRY, ExperimentInitConfig
from konductor.models._pytorch import TorchModelConfig


@MODEL_REGISTRY.register_module("temporal-conv")
class TemporalConv(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        out_ch: int,
        n_timesteps: int,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__(
            nn.Conv3d(
                in_ch, hidden_ch, (n_timesteps // 2 + 1, 3, 3), padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(hidden_ch),
            activation(),
            nn.Conv3d(hidden_ch, out_ch, (n_timesteps // 2, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_ch),
            activation(),
        )
        self.out_ch = out_ch


@dataclass
class BaseConfig(TorchModelConfig):
    encoder: ModuleInitConfig
    decoder: ModuleInitConfig
    history_len: int = 8

    def __post_init__(self):
        if isinstance(self.encoder, dict):
            self.encoder = ModuleInitConfig(**self.encoder)
        if isinstance(self.decoder, dict):
            self.decoder = ModuleInitConfig(**self.decoder)


@dataclass
@MODEL_REGISTRY.register_module("conv-forecast-v1")
class ConvV1Config(BaseConfig):
    temporal: ModuleInitConfig = field(kw_only=True)

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0) -> Any:
        props = get_dataset_properties(config)
        model_cfg = config.model[idx].args
        model_cfg["encoder"]["args"]["in_ch"] = props["image_ch"]
        return super().from_config(config, idx)

    def __post_init__(self):
        super().__post_init__()
        assert self.history_len % 2 == 0, "history_len must be even"
        if isinstance(self.temporal, dict):
            self.temporal = ModuleInitConfig(**self.temporal)
        self.temporal.args["n_timesteps"] = self.history_len

    def get_instance(self, *args, **kwargs) -> Any:
        """Construct modules and return conv forecaster"""
        encoder = MODEL_REGISTRY[self.encoder.type](**self.encoder.args)

        self.temporal.args["in_ch"] = encoder.out_ch[-1]
        temporal = MODEL_REGISTRY[self.temporal.type](**self.temporal.args)

        self.decoder.args["in_ch"] = temporal.out_ch + encoder.out_ch[0]
        decoder = MODEL_REGISTRY[self.decoder.type](**self.decoder.args)
        return ConvForecaster(encoder, temporal, decoder, self.history_len)


class ConvForecaster(nn.Module):
    """
    Use Siamese ConvNet to Extract features
    3dConv Along Features
    Upsample and Concatenate with Last Image
    Conv Decode TemporalFeatures+Last Image for Final Output
    """

    is_logit_output = True

    def __init__(
        self,
        encoder: nn.Module,
        temporal: nn.Module,
        decoder: nn.Module,
        history_len: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.temporal_conv = temporal
        self.decoder = decoder
        self.history_len = history_len

    def forward_sequence(self, inputs: Tensor) -> Tensor:
        minimap_low: list[Tensor] = []
        minimap_high: list[Tensor] = []

        for t in range(inputs.shape[1]):
            enc = self.encoder(inputs[:, t])
            minimap_low.append(enc[0])
            minimap_high.append(enc[-1])

        stacked_feats = torch.stack(minimap_low, dim=2)
        temporal_feats = self.temporal_conv(stacked_feats)
        temporal_feats = F.interpolate(
            temporal_feats.squeeze(2),
            size=minimap_high[-1].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        cat_features = torch.cat([temporal_feats, minimap_high[-1]], dim=1)
        decoded = self.decoder(cat_features)
        return decoded

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """"""
        minimaps = inputs["minimap_features"]
        ntime = minimaps.shape[1]
        preds: list[Tensor] = []
        for start_idx in range(ntime - self.history_len):
            end_idx = start_idx + self.history_len
            pred = self.forward_sequence(minimaps[:, start_idx:end_idx])
            pred = F.interpolate(
                pred, mode="bilinear", size=minimaps.shape[-2:], align_corners=True
            )
            preds.append(pred)

        out = torch.stack(preds, dim=1)
        return out
