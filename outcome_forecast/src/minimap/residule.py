from dataclasses import dataclass
from typing import Any

import torch
from konductor.models import MODEL_REGISTRY
from konductor.models._pytorch import TorchModelConfig
from torch import Tensor, nn
from torch.nn import functional as F

from .common import MinimapTarget


class TemporalConvV3(nn.Module):
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
@MODEL_REGISTRY.register_module("conv-forecast-v3")
class ConvForecasterV3(TorchModelConfig):
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
        return TemporalConvV3(
            self.history_len,
            self.hidden_chs,
            self.kernel_sizes,
            self.in_layers,
            self.target,
        )
