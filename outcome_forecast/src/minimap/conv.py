from dataclasses import dataclass, field
from typing import Any

import torch
from konductor.init import ModuleInitConfig
from konductor.models import MODEL_REGISTRY
from torch import Tensor, nn
from torch.nn import functional as F

from .common import BaseConfig, MinimapTarget


@MODEL_REGISTRY.register_module("temporal-conv")
class TemporalConv(nn.Sequential):
    """Steps temporal dimension in half and then to one"""

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


@MODEL_REGISTRY.register_module("temporal-conv-v2")
class TemporalConv2(nn.Sequential):
    """Convolve over adjacent frames twice before temporally downsampling"""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        out_ch: int,
        n_timesteps: int,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__(
            nn.Conv3d(in_ch, hidden_ch, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_ch),
            activation(),
            nn.Conv3d(hidden_ch, hidden_ch, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_ch),
            activation(),
            nn.Conv3d(
                hidden_ch, hidden_ch, (n_timesteps // 2 + 1, 3, 3), padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(hidden_ch),
            activation(),
            nn.Conv3d(hidden_ch, out_ch, (n_timesteps // 2, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_ch),
            activation(),
        )
        self.out_ch = out_ch


@MODEL_REGISTRY.register_module("temporal-conv-multi-out")
class TemporalConv2Multi(nn.Sequential):
    """Convolve over volume a few times"""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        out_ch: int,
        n_timesteps: int,
        out_timesteps: int,
        n_layers: int = 2,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        self.out_ch = out_ch
        reduce_dim = n_timesteps - out_timesteps + 1

        def make_layer(in_ch_: int):
            return (
                nn.Conv3d(in_ch_, hidden_ch, (3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(hidden_ch),
                activation(),
            )

        layers = [*make_layer(in_ch)]
        for _ in range(n_layers - 1):
            layers.extend(make_layer(hidden_ch))

        super().__init__(
            *layers,
            nn.Conv3d(hidden_ch, out_ch, (reduce_dim, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_ch),
            activation(),
        )


@MODEL_REGISTRY.register_module("temporal-conv-decoder")
class TemporalConv2MultiDecoder(TemporalConv2Multi):
    """Additional point-wise linear layer with final output channels"""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        out_ch: int,
        n_timesteps: int,
        out_timesteps: int,
        n_layers: int = 2,
    ) -> None:
        super().__init__(
            in_ch, hidden_ch, hidden_ch, n_timesteps, out_timesteps, n_layers
        )
        # Add last pixel-wise module without ReLU
        self.add_module(
            str(len(self._modules)), nn.Conv3d(hidden_ch, out_ch, (1, 1, 1))
        )


@dataclass
@MODEL_REGISTRY.register_module("conv-forecast-v1")
class ConvV1Config(BaseConfig):
    """
    Next frame convolution forecasting v1
    """

    def __post_init__(self):
        super().__post_init__()
        assert self.history_len % 2 == 0, "history_len must be even"
        self.temporal.args["n_timesteps"] = self.history_len

    def get_instance(self, *args, **kwargs) -> Any:
        """Construct modules and return conv forecaster"""
        encoder = MODEL_REGISTRY[self.encoder.type](**self.encoder.args)

        self.temporal.args["in_ch"] = encoder.out_ch[-1]
        temporal = MODEL_REGISTRY[self.temporal.type](**self.temporal.args)

        self.decoder.args["in_ch"] = temporal.out_ch + encoder.out_ch[0]
        self.decoder.args["out_ch"] = len(MinimapTarget.names(self.target))
        decoder = MODEL_REGISTRY[self.decoder.type](**self.decoder.args)
        return ConvForecast(encoder, temporal, decoder, self.history_len)


class ConvForecast(nn.Module):
    """
    Use Siamese ConvNet to Extract features
    3dConv Along Features
    Upsample and Concatenate with Last Image
    Conv Decode TemporalFeatures+Last Image for Final Output
    """

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

    @property
    def future_len(self):
        return 1

    @property
    def is_logit_output(self):
        return True

    def forward_sequence(self, inputs: Tensor) -> Tensor:
        minimap_low: list[Tensor] = []
        for t in range(inputs.shape[1]):
            enc: list[Tensor] = self.encoder(inputs[:, t])
            minimap_low.append(enc[0])
        last_minimap_high = enc[-1]

        stacked_feats = torch.stack(minimap_low, dim=2)
        temporal_feats: Tensor = self.temporal_conv(stacked_feats)
        temporal_feats = F.interpolate(
            temporal_feats.squeeze(2),
            size=last_minimap_high.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        cat_features = torch.cat([temporal_feats, last_minimap_high], dim=1)
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


@dataclass
@MODEL_REGISTRY.register_module("conv-forecast-v2")
class ConvV2Config(BaseConfig):
    """
    Next frame convolution forecasting v2
    Adds separate high res last frame context encoder
    """

    last_frame_encoder: ModuleInitConfig = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        assert self.history_len % 2 == 0, "history_len must be even"
        self.temporal.args["n_timesteps"] = self.history_len
        if isinstance(self.last_frame_encoder, dict):
            self.last_frame_encoder = ModuleInitConfig(**self.last_frame_encoder)

    def get_instance(self, *args, **kwargs) -> Any:
        """Construct modules and return conv forecaster"""
        encoder = MODEL_REGISTRY[self.encoder.type](**self.encoder.args)

        self.temporal.args["in_ch"] = encoder.out_ch
        temporal = MODEL_REGISTRY[self.temporal.type](**self.temporal.args)

        self.last_frame_encoder.args["in_ch"] = self.encoder.args["in_ch"]
        last_frame_encoder = MODEL_REGISTRY[self.last_frame_encoder.type](
            **self.last_frame_encoder.args
        )

        self.decoder.args["in_ch"] = temporal.out_ch + last_frame_encoder.out_ch
        self.decoder.args["out_ch"] = len(MinimapTarget.names(self.target))
        decoder = MODEL_REGISTRY[self.decoder.type](**self.decoder.args)

        return ConvForecastV2(
            encoder, temporal, decoder, last_frame_encoder, self.history_len
        )


class ConvForecastV2(nn.Module):
    """
    V2 uses another feature extractor for high resolution last frame
    features as to not add confounding factor to temporal feature inputs.
    """

    def __init__(
        self,
        encoder: nn.Module,
        temporal: nn.Module,
        decoder: nn.Module,
        last_frame_encoder: nn.Module,
        history_len: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.temporal_conv = temporal
        self.decoder = decoder
        self.last_frame_encoder = last_frame_encoder
        self.history_len = history_len

    @property
    def future_len(self):
        return 1

    @property
    def is_logit_output(self):
        return True

    def forward_sequence(self, inputs: Tensor) -> Tensor:
        # Squeeze and unsqueeze time dimension for siamese encoder
        img_feats: Tensor = self.encoder(inputs.reshape(-1, *inputs.shape[2:]))
        img_feats = img_feats.reshape(*inputs.shape[:2], *img_feats.shape[1:])
        img_feats = img_feats.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W] -> [B,C,T,H,W]

        last_feats: Tensor = self.last_frame_encoder(inputs[:, -1])
        temporal_feats: Tensor = self.temporal_conv(img_feats)
        temporal_feats = F.interpolate(
            temporal_feats.squeeze(2),
            size=last_feats.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        cat_features = torch.cat([temporal_feats, last_feats], dim=1)
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


@dataclass
@MODEL_REGISTRY.register_module("conv-forecast-v2-multiframe")
class ConvV2MultiConfig(ConvV2Config):
    """
    Predicts three frames into the future rather than one
    """

    last_frame_encoder: ModuleInitConfig = field(kw_only=True)

    @property
    def future_len(self) -> int:
        return 3

    def __post_init__(self):
        # Subtrack two off history length as we will predict 3 frames instead
        self.history_len -= 2
        super().__post_init__()

    def get_instance(self, *args, **kwargs) -> Any:
        """Construct modules and return conv forecaster"""
        encoder = MODEL_REGISTRY[self.encoder.type](**self.encoder.args)

        self.temporal.args["in_ch"] = encoder.out_ch
        self.temporal.args["out_timesteps"] = self.future_len
        temporal = MODEL_REGISTRY[self.temporal.type](**self.temporal.args)

        self.last_frame_encoder.args["in_ch"] = self.encoder.args["in_ch"]
        last_frame_encoder = MODEL_REGISTRY[self.last_frame_encoder.type](
            **self.last_frame_encoder.args
        )

        assert temporal.out_ch == last_frame_encoder.out_ch
        self.decoder.args["in_ch"] = temporal.out_ch
        self.decoder.args["out_ch"] = len(MinimapTarget.names(self.target))
        decoder = MODEL_REGISTRY[self.decoder.type](**self.decoder.args)

        return ConvForecastV2Multi(
            encoder, temporal, decoder, last_frame_encoder, self.history_len
        )


class ConvForecastV2Multi(ConvForecastV2):
    """ConvV2 Minimap Forecast but 3 frames output"""

    @property
    def future_len(self):
        return 3

    @property
    def is_logit_output(self):
        return True

    def forward_sequence(self, inputs: Tensor):
        # Squeeze and unsqueeze time dimension for siamese encoder
        img_feats: Tensor = self.encoder(inputs.reshape(-1, *inputs.shape[2:]))
        img_feats = img_feats.reshape(*inputs.shape[:2], *img_feats.shape[1:])
        img_feats = img_feats.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W] -> [B,C,T,H,W]

        last_feats: Tensor = self.last_frame_encoder(inputs[:, -1])
        # Add time dim
        last_feats = last_feats.unsqueeze(2)

        temporal_feats: Tensor = self.temporal_conv(img_feats)
        temporal_feats = F.interpolate(
            temporal_feats,
            size=[self.future_len, *last_feats.shape[-2:]],
            mode="trilinear",
            align_corners=True,
        )

        cat_features = torch.cat([last_feats, temporal_feats], dim=2)
        decoded: Tensor = self.decoder(cat_features)
        decoded = decoded.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W] -> [B,T,C,H,W]
        return decoded

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """"""
        minimaps = inputs["minimap_features"]

        pred = self.forward_sequence(minimaps[:, : self.history_len])
        out: Tensor = F.interpolate(
            pred.reshape(-1, *pred.shape[-3:]),
            mode="bilinear",
            size=minimaps.shape[-2:],
            align_corners=True,
        )
        out = out.reshape(*pred.shape[:3], *minimaps.shape[-2:])
        return out
