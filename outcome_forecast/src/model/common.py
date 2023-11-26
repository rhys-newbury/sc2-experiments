from typing import Literal

from konductor.models import MODEL_REGISTRY
from torch import nn, Tensor


@MODEL_REGISTRY.register_module("image-v1")
class ImageEncV1(nn.Module):
    """Simple conv->flatten encoder for images"""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 32,
        out_ch: int = 32,
        n_layers: int = 3,
        pooling: Literal["max", "avg"] = "max",
    ) -> None:
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(),
            ]
        )
        for idx in range(1, n_layers):
            _hidden_ch = hidden_ch if idx == 1 else hidden_ch * 2
            self.encoder.extend(
                [
                    nn.Conv2d(
                        _hidden_ch, 2 * hidden_ch, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(2 * hidden_ch),
                    nn.ReLU(),
                ]
            )
        match pooling:
            case "avg":
                pool_t = nn.AdaptiveAvgPool2d
            case "max":
                pool_t = nn.AdaptiveMaxPool2d
            case _:
                raise KeyError("Invalid Pooling Type, should be [avg|max]")
        self.encoder.extend(
            [
                pool_t(9),
                nn.Conv2d(2 * hidden_ch, hidden_ch, kernel_size=3, stride=3),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(hidden_ch * 3 * 3, out_ch),
                nn.ReLU(),
            ]
        )
        self._out_ch = out_ch

    @property
    def out_ch(self):
        return self._out_ch

    def forward(self, x: Tensor) -> Tensor:
        for mod in self.encoder:
            x = mod(x)
        return x


@MODEL_REGISTRY.register_module("scalar-v1")
class ScalarEncoderV1(nn.Module):
    """Simple set of linear layers to encode feature vector"""

    def __init__(
        self, in_ch: int, hidden_ch: int = 32, out_ch: int = 32, n_layers: int = 2
    ) -> None:
        super().__init__()
        self.encoder = nn.ModuleList(
            [nn.Linear(in_ch, hidden_ch), nn.LayerNorm(hidden_ch), nn.LeakyReLU()]
        )
        for idx in range(1, n_layers):
            _hidden_ch = hidden_ch if idx == 1 else hidden_ch * 2
            self.encoder.extend(
                [
                    nn.Linear(_hidden_ch, 2 * hidden_ch),
                    nn.LayerNorm(2 * hidden_ch),
                    nn.LeakyReLU(),
                ]
            )
        self.encoder.extend(
            [
                nn.Linear(2 * hidden_ch, out_ch),
                nn.LayerNorm(out_ch),
                nn.ReLU(),
            ]
        )
        self._out_ch = out_ch

    @property
    def out_ch(self):
        return self._out_ch

    def forward(self, x: Tensor) -> Tensor:
        for mod in self.encoder:
            x = mod(x)
        return x
