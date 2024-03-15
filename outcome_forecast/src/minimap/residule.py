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
        return 1

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
                nn.Conv2d(out_ch, len(MinimapTarget.indices(target)), 1),
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
        residule = minimaps.reshape(
            -1, self.history_len * len(in_ch), *minimaps.shape[-2:]
        )
        for block in self.blocks:
            residule = block(residule)
        residule = F.tanh(residule)

        out_ch = MinimapTarget.indices(self.out_layers)
        out_ch = [i + ch for i in out_ch]
        last_minimap = inputs["minimap_features"][:, self.history_len - 1, out_ch]
        # Ensure prediction between 0 and 1 and unsqueeze time dimension
        prediction = torch.clamp(last_minimap + residule, 0, 1).unsqueeze(1)
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
        return 1

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
