from typing import Literal

from torch import nn, Tensor


class ImageEncV1(nn.Module):
    def __init__(
        self,
        in_ch: int = 10,
        hidden_dim: int = 32,
        n_layers: int = 3,
        pooling: Literal["max", "avg"] = "max",
    ) -> None:
        super().__init__()

        self.encoder = nn.ModuleList(
            [
                nn.Conv2d(in_ch, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
            ]
        )
        for _ in range(1, n_layers):
            self.encoder.extend(
                [
                    nn.Conv2d(
                        hidden_dim, 2 * hidden_dim, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(2 * hidden_dim),
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
                nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for mod in self.encoder:
            x = mod(x)
        return x
