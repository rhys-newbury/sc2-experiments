from dataclasses import dataclass
from typing import Any

import torch
from konductor.models import MODEL_REGISTRY
from konductor.models._pytorch import TorchModelConfig
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock

from .common import MinimapTarget


class ResiduleConvV1(nn.Module):
    """
    Ultra barebones temporal model that channel-wise stacks self/enemy
    and predicts the residual from the previous frame and always stays at
    high resolution to ensure fidelity.
    """

    @property
    def future_len(self):
        return 1

    @property
    def is_logit_output(self):
        return False

    def __init__(
        self,
        history_len: int,
        hidden_ch: list[int],
        kernel_size: list[int],
        in_layers: MinimapTarget,
        out_layers: MinimapTarget,
    ) -> None:
        super().__init__()
        assert len(hidden_ch) == len(kernel_size)
        self.in_layers = in_layers
        self.out_layers = out_layers
        self.history_len = history_len

        modules = []
        in_ch = history_len * len(MinimapTarget.indices(in_layers))
        for out_ch, kernel in zip(hidden_ch, kernel_size):
            modules.append(self._make_layer(in_ch, out_ch, kernel))
            in_ch = out_ch
        modules.append(nn.Conv2d(in_ch, len(MinimapTarget.indices(out_layers)), 1))
        self.model = nn.Sequential(*modules)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, kernel_size: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, inputs: dict[str, Tensor]):
        """"""
        ch = inputs["minimap_features"].shape[2]
        in_ch = MinimapTarget.indices(self.in_layers)
        in_ch = [i + ch for i in in_ch]
        minimaps = inputs["minimap_features"][:, : self.history_len, in_ch]
        minimap_stack = minimaps.reshape(
            -1, self.history_len * len(in_ch), *minimaps.shape[-2:]
        )
        residule = self.model(minimap_stack)
        residule = F.tanh(residule)

        out_ch = MinimapTarget.indices(self.out_layers)
        out_ch = [i + ch for i in out_ch]
        last_minimap = inputs["minimap_features"][:, self.history_len - 1, out_ch]
        # Ensure prediction between 0 and 1 and unsqueeze time dimension
        prediction = torch.clamp(last_minimap + residule, 0, 1).unsqueeze(1)
        return prediction


@dataclass
@MODEL_REGISTRY.register_module("residule-conv-v1")
class ResiduleConvV1Cfg(TorchModelConfig):
    hidden_chs: list[int]
    kernel_sizes: list[int]
    in_layers: MinimapTarget
    history_len: int = 8
    target: MinimapTarget = MinimapTarget.BOTH

    def __post_init__(self):
        if isinstance(self.target, str):
            self.target = MinimapTarget[self.target.upper()]
        if isinstance(self.in_layers, str):
            self.in_layers = MinimapTarget[self.in_layers.upper()]

    @property
    def future_len(self) -> int:
        return 1

    @property
    def is_logit_output(self):
        return False

    def get_instance(self, *args, **kwargs) -> Any:
        return ResiduleConvV1(
            self.history_len,
            self.hidden_chs,
            self.kernel_sizes,
            self.in_layers,
            self.target,
        )


# Backcompat
MODEL_REGISTRY.register_module("conv-forecast-v3", ResiduleConvV1Cfg)


class ResiduleConvV2(nn.Module):

    @property
    def future_len(self):
        return 9 - self.history_len

    @property
    def is_logit_output(self):
        return False

    def __init__(
        self,
        history_len: int,
        hidden_chs: list[int],
        num_blocks: list[int],
        in_layers: MinimapTarget,
        target: MinimapTarget,
    ) -> None:
        super().__init__()
        assert len(hidden_chs) - 1 == len(num_blocks)
        self.in_layers = in_layers
        self.out_layers = target
        self.history_len = history_len

        in_ch = history_len * len(MinimapTarget.indices(in_layers))
        modules: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_chs[0], 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(hidden_chs[0]),
                nn.ReLU(),
                nn.MaxPool2d(3, 2, 1),
            )
        ]
        for in_ch, out_ch, num_block in zip(hidden_chs, hidden_chs[1:], num_blocks):
            modules.append(self._make_block(in_ch, out_ch, num_block))

        # Upsample to original resolution and apply final layer as depthwise-separable
        modules.append(
            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv2d(out_ch, out_ch, 5, padding=2, groups=out_ch),
                nn.Conv2d(
                    out_ch, self.future_len * len(MinimapTarget.indices(target)), 1
                ),
            )
        )
        self.blocks = nn.ModuleList(modules)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)  # Make identiy

    @staticmethod
    def _make_block(in_ch: int, out_ch: int, num_block: int):
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False), nn.BatchNorm2d(out_ch)
        )
        block = [BasicBlock(in_ch, out_ch, downsample=downsample)]
        for _ in range(1, num_block):
            block.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*block)

    def forward(self, inputs: dict[str, Tensor]):
        ch = inputs["minimap_features"].shape[2]

        in_ch = MinimapTarget.indices(self.in_layers)
        in_ch = [i + ch for i in in_ch]
        minimaps = inputs["minimap_features"][:, : self.history_len, in_ch]
        size = minimaps.shape[-2:]
        residule: Tensor = minimaps.reshape(-1, self.history_len * len(in_ch), *size)
        for block in self.blocks:
            residule = block(residule)
        residule = F.tanh(residule)

        out_ch = MinimapTarget.indices(self.out_layers)
        out_ch = [i + ch for i in out_ch]
        last_minimap = inputs["minimap_features"][:, self.history_len - 1, None, out_ch]
        residule = residule.reshape(-1, self.future_len, len(out_ch), *size)
        # Ensure prediction between 0 and 1
        prediction = torch.clamp(last_minimap + residule, 0, 1)
        return prediction


@dataclass
@MODEL_REGISTRY.register_module("residule-conv-v2")
class ResiduleConvV2Cfg(TorchModelConfig):
    in_layers: MinimapTarget
    hidden_chs: list[int]
    num_blocks: list[int]
    history_len: int = 8
    target: MinimapTarget = MinimapTarget.BOTH

    @property
    def future_len(self) -> int:
        return 9 - self.history_len

    @property
    def is_logit_output(self):
        return False

    def __post_init__(self):
        if isinstance(self.in_layers, str):
            self.in_layers = MinimapTarget[self.in_layers.upper()]
        if isinstance(self.target, str):
            self.target = MinimapTarget[self.target.upper()]
        assert len(self.hidden_chs) - 1 == len(self.num_blocks)

    def get_instance(self, *args, **kwargs) -> Any:
        return self.init_auto_filter(ResiduleConvV2)


def _make_conv_block(in_channels: int, out_channels: int):
    """Double pump conv->norm->relu"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 2, 1),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(),
    )


class UpsampleBlock(nn.Module):
    def __init__(self, up_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.transpose_conv = nn.ConvTranspose2d(up_channels, skip_channels, 2, 2)
        self.module = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, 3, 1, 1),
            nn.InstanceNorm2d(skip_channels),
            nn.LeakyReLU(),
            nn.Conv2d(skip_channels, skip_channels, 3, 1, 1),
            nn.InstanceNorm2d(skip_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: Tensor, x_skip: Tensor) -> Tensor:
        x = self.transpose_conv(x)
        x = self.module(x + x_skip)
        return x


class ResiduleUnet(nn.Module):
    """Based on nnUnet https://arxiv.org/pdf/2110.03352v2.pdf"""

    @property
    def future_len(self):
        return 9 - self.history_len

    @property
    def is_logit_output(self):
        return False

    def __init__(
        self,
        history_len: int,
        in_layers: MinimapTarget,
        target: MinimapTarget,
        hidden_chs: list[int],
        deep_supervision: bool,
        include_heightmap: bool,
    ) -> None:
        super().__init__()
        self.in_layers = in_layers
        self.out_layers = target
        self.history_len = history_len
        self.deep_supervision = deep_supervision
        self.include_heightmap = include_heightmap

        in_ch = history_len * len(MinimapTarget.indices(in_layers))
        if include_heightmap:
            in_ch += 1
        self.input_module = nn.Sequential(
            nn.Conv2d(in_ch, hidden_chs[0], 3, padding=1, bias=False),
            nn.Conv2d(hidden_chs[0], hidden_chs[0], 3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_chs[0]),
        )
        self.downsamples = nn.ModuleList(
            _make_conv_block(*args) for args in zip(hidden_chs[:-2], hidden_chs[1:])
        )
        self.bottleneck = _make_conv_block(hidden_chs[-2], hidden_chs[-1])
        self.upsamples = nn.ModuleList(
            UpsampleBlock(*args)
            for args in zip(
                reversed(hidden_chs[1:]),
                reversed(hidden_chs[:-1]),
            )
        )
        out_ch = self.future_len * len(MinimapTarget.indices(target))
        out_mods = [nn.Conv2d(hidden_chs[0], out_ch, 1)]
        if deep_supervision:
            out_mods.append(nn.Conv2d(hidden_chs[1], out_ch, 1))
            out_mods.append(nn.Conv2d(hidden_chs[2], out_ch, 1))
        self.linear_proj = nn.ModuleList(out_mods)

    def forward_aux(self, inputs: Tensor) -> list[Tensor]:
        """"""
        out = self.input_module(inputs)
        encoder_outputs = [out]
        for downsample in self.downsamples:
            out = downsample(out)
            encoder_outputs.append(out)
        out = self.bottleneck(out)

        decoder_outputs = []
        for upsample, skip in zip(self.upsamples, reversed(encoder_outputs)):
            out = upsample(out, skip)
            decoder_outputs.append(out)

        out = [self.linear_proj[0](out)]
        if self.training and self.deep_supervision:
            for i, decoder_out in enumerate(decoder_outputs[-3:-1][::-1], 1):
                out.append(self.linear_proj[i](decoder_out))

        out = [F.tanh(o) for o in out]
        return out

    def forward(self, inputs: dict[str, Tensor]):
        ch = inputs["minimap_features"].shape[2]

        in_ch = MinimapTarget.indices(self.in_layers)
        in_ch = [i + ch for i in in_ch]
        minimaps = inputs["minimap_features"][:, : self.history_len, in_ch]
        # Flatten to B[TC]HW
        minimaps = minimaps.reshape(
            -1, self.history_len * len(in_ch), *minimaps.shape[-2:]
        )
        if self.include_heightmap:
            heightmap = (inputs["minimap_features"][:, 0, [0]] - 127) / 128
            minimaps = torch.cat([minimaps, heightmap], dim=1)
        residules = self.forward_aux(minimaps)

        out_ch = MinimapTarget.indices(self.out_layers)
        out_ch = [i + ch for i in out_ch]
        last_minimap = inputs["minimap_features"][:, self.history_len - 1, out_ch]

        preds: list[Tensor] = []
        for residule in residules:
            residule = residule.reshape(
                -1, self.future_len, len(out_ch), *residule.shape[-2:]
            )
            last_resize = F.interpolate(
                last_minimap, size=residule.shape[-2:]
            ).unsqueeze(1)
            # Ensure prediction between 0 and 1
            preds.append(torch.clamp(last_resize + residule, 0, 1))

        if len(preds) == 1:  # remove list dim
            return preds[0]

        return preds


@dataclass
@MODEL_REGISTRY.register_module("residule-unet")
class ResiduleUnetCfg(TorchModelConfig):
    in_layers: MinimapTarget
    hidden_chs: list[int]
    history_len: int = 8
    target: MinimapTarget = MinimapTarget.BOTH
    deep_supervision: bool = False
    include_heightmap: bool = False

    @property
    def future_len(self) -> int:
        return 9 - self.history_len

    @property
    def is_logit_output(self):
        return False

    def __post_init__(self):
        if isinstance(self.in_layers, str):
            self.in_layers = MinimapTarget[self.in_layers.upper()]
        if isinstance(self.target, str):
            self.target = MinimapTarget[self.target.upper()]

    def get_instance(self, *args, **kwargs) -> Any:
        return self.init_auto_filter(ResiduleUnet)
