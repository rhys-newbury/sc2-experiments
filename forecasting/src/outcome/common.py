import enum
from copy import deepcopy
from typing import Literal

import torch
from konductor.models import MODEL_REGISTRY
from torch import nn, Tensor

from ..utils import TimeRange
import torchvision.models


@MODEL_REGISTRY.register_module("pretrained-v1")
class ResNet18(nn.Module):
    is_logit_output = True

    def __init__(
        self,
        in_ch: int,
        out_ch: int = 32,
        pretrained: bool = True,
        dropout: float = 0.0,
        model_name: str = "resnet18",
        freeze_until_idx: int
        | None = None,  # parameter to specify index until which layers are frozen
    ) -> None:
        super().__init__()

        self.encoder = getattr(torchvision.models, model_name)(pretrained=pretrained)

        # Modify the first convolutional layer to accept in_ch channels
        self.encoder.conv1 = nn.Conv2d(
            in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify the final fully connected layer to output out_ch channels
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, out_ch)

        # Freeze layers by index if specified
        if freeze_until_idx is not None:
            self.freeze_until_index(freeze_until_idx)

        # Ensure that the modified layers (conv1 and fc) are trainable
        self.unfreeze_layer(self.encoder.conv1)
        self.unfreeze_layer(self.encoder.fc)

        self.dropout = nn.Dropout(dropout)

        self._out_ch = out_ch

    @property
    def out_ch(self):
        return self._out_ch

    def freeze_until_index(self, freeze_until_idx: int) -> None:
        for idx, param in enumerate(self.encoder.named_parameters()):
            if idx < freeze_until_idx:
                param[1].requires_grad = False
            else:
                param[1].requires_grad = True

    def unfreeze_layer(self, layer) -> None:
        for param in layer.parameters():
            param.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.encoder(x))


@MODEL_REGISTRY.register_module("image-v1")
class ImageEncV1(nn.Module):
    """Simple conv->flatten encoder for images"""

    is_logit_output = True

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 32,
        out_ch: int = 32,
        n_layers: int = 3,
        pooling: Literal["max", "avg"] = "max",
        dropout: float = 0.0,
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
                    nn.Dropout2d(dropout),
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
                nn.Dropout(dropout),
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


@MODEL_REGISTRY.register_module("image-fpn")
class ImageFPN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden_chs: list[int],
        strides: list[int],
        paddings: list[int],
        output_lvl: int | None = None,
        disable_fpn: bool = False,
    ) -> None:
        super().__init__()
        if not disable_fpn:
            assert output_lvl is not None
        if output_lvl is None:
            assert disable_fpn

        self.enc = nn.ModuleList()
        chs = [in_ch] + hidden_chs
        for in_ch, out_ch, stride, padding in zip(chs[:-1], chs[1:], strides, paddings):
            self.enc.append(ImageFPN.make_conv(in_ch, out_ch, stride, padding))

        if output_lvl is None:
            output_lvl = 1
        self.out_ch = hidden_chs[-output_lvl:]
        self.disable_fpn = disable_fpn
        if self.disable_fpn:
            self.out_ch = self.out_ch[0]
        self._output_idx = len(self.enc) - output_lvl - 1

    @staticmethod
    def make_conv(in_ch, out_ch, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor) -> list[Tensor] | Tensor:
        """Output FPN sorted by low to high spatial resolution"""
        if inputs.ndim == 5:
            b, t, c, h, w = inputs.shape
            feat = inputs.reshape(b, t * c, h, w)
        else:
            feat = inputs

        out: list[Tensor] = []
        for idx, mod in enumerate(self.enc):
            feat = mod(feat)
            if idx > self._output_idx:
                out.append(feat)

        return out[0] if self.disable_fpn else out[::-1]


def make_basic_encoder(
    in_ch: int,
    hidden_ch: int,
    out_ch: int,
    n_layers: int,
    norm: type[nn.Module] = nn.LayerNorm,
    activation: type[nn.Module] = nn.LeakyReLU,
    dropout: float = 0.0,
):
    """
    Create basic linear encoder with linear->norm->activation loop
    """
    encoder = [nn.Linear(in_ch, hidden_ch), norm(hidden_ch), activation()]

    for idx in range(1, n_layers):
        _hidden_ch = hidden_ch if idx == 1 else hidden_ch * 2
        encoder.extend(
            [
                nn.Linear(_hidden_ch, 2 * hidden_ch),
                norm(2 * hidden_ch),
                activation(),
                nn.Dropout(dropout),
            ]
        )
    encoder.append(nn.Linear(2 * hidden_ch, out_ch))

    return nn.Sequential(*encoder)


@MODEL_REGISTRY.register_module("scalar-v1")
class ScalarEncoderV1(nn.Module):
    """Simple set of linear layers to encode feature vector"""

    is_logit_output = True

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int = 32,
        out_ch: int = 32,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = make_basic_encoder(
            in_ch, hidden_ch, out_ch, n_layers, dropout=dropout
        )
        self._out_ch = out_ch

    @property
    def out_ch(self):
        return self._out_ch

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


@MODEL_REGISTRY.register_module("scalar-v2")
class ScalarEncoderV2(nn.Module):
    """
    Encoder that tests the game step and either normalizes in a particular
        range or selects a set of parameters.
    Assuming from sc2_serializer/include/replay_parsing.hpp
        that gameStep is the last scalar feature.
    """

    is_logit_output = True

    class Strategy(enum.Enum):
        """
        Strategy for normalising scalar data
        batch_norm - different input batch norm layer per timestep
        weights - different set of weights per timestep
        both - use both stratergies
        """

        batch_norm = enum.auto()
        weights = enum.auto()
        both = enum.auto()

    def __init__(
        self,
        in_ch: int,
        strategy: str | Strategy,
        timerange: TimeRange,
        time_idx: int = -1,
        hidden_ch: int = 32,
        out_ch: int = 32,
        n_layers: int = 2,
        dropout: float = 0.0,
        use_other_player: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(strategy, str):
            strategy = ScalarEncoderV2.Strategy[strategy]
        in_ch -= 1  # We are not processing the gameStep feature
        in_ch *= 1 + use_other_player
        self.use_other_player = use_other_player
        self.timerange: Tensor
        self.register_buffer("timerange", timerange.arange(), persistent=False)
        self.time_idx = time_idx
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(in_ch)])
        self.encoders = nn.ModuleList(
            [make_basic_encoder(in_ch, hidden_ch, out_ch, n_layers, dropout=dropout)]
        )
        self.strategy = strategy
        self._out_ch = out_ch

        # Check to duplicate batch norm
        if strategy in {
            ScalarEncoderV2.Strategy.batch_norm,
            ScalarEncoderV2.Strategy.both,
        }:
            for _ in range(1, len(self.timerange)):
                self.batch_norms.append(deepcopy(self.batch_norms[0]))

        # Check to duplicate params
        if strategy in {
            ScalarEncoderV2.Strategy.weights,
            ScalarEncoderV2.Strategy.both,
        }:
            for _ in range(1, len(self.timerange)):
                self.encoders.append(deepcopy(self.encoders[0]))

    @property
    def out_ch(self):
        return self._out_ch

    def forward(self, inputs: Tensor) -> Tensor:
        """Inputs are assumed to all be of the same timestep which is the gameStep"""
        with torch.no_grad():
            timestep = inputs[0, -1] / (22.4 * 60)
            mod_idx = int(torch.argmin((timestep - self.timerange).abs()))

        with torch.autocast("cuda", enabled=False):
            norm_idx = mod_idx if len(self.batch_norms) > 1 else 0
            norm_feats = self.batch_norms[norm_idx](
                inputs[..., : -(1 + self.use_other_player)]
            )

        params_idx = mod_idx if len(self.encoders) > 1 else 0
        feats = self.encoders[params_idx](norm_feats)

        return feats

@MODEL_REGISTRY.register_module("vit-v2")
class VIT2(nn.Module):
    is_logit_output = True
    def __init__(
        self,
        in_ch: int,
        out_ch: int = 32,
        pretrained: bool = True,
        dropout: float = 0.0,
        freeze: bool = False
    ) -> None:
        super().__init__()
        self.encoder = models.vit_b_16(pretrained=pretrained)
        if freeze:
            # Freeze all layers except the final layer
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.encoder.heads.head = torch.nn.Linear(self.encoder.heads.head.in_features, out_ch)
        if freeze:
            # Ensure the final layer's parameters are trainable
            for param in self.encoder.heads.head.parameters():
                param.requires_grad = True
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.color_list = torch.Tensor([
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [0, 255, 255],  # Cyan
            [255, 0, 255],  # Magenta
            [128, 0, 0],    # Maroon
            [0, 128, 0],    # Dark Green
            [0, 0, 128],    # Navy
            [128, 128, 0],  # Olive
            [128, 0, 128],  # Purple
            [0, 128, 128]   # Teal
        ]).cuda() // 6
        self._out_ch = out_ch
    @property
    def out_ch(self):
        return self._out_ch
    def forward(self, x: Tensor) -> Tensor:
        p_image = torch.zeros_like(x[:, 0:1, :, :]).repeat(1,3,1,1)
        for idx, l in enumerate(self.color_list[:6]):
            image = x[:, idx:idx+1, :, :].repeat(1,3,1,1) * l.view(1,3,1,1)
            p_image += image
        for idx, l in enumerate(self.color_list[6:], 6):
            image = x[:, idx, :, :][:, None, :, :].repeat(1,3,1,1) * l.view(1,3,1,1)
            p_image += image
        with torch.no_grad():
            x = self.transform(p_image)
        return self.encoder(x)
    
@MODEL_REGISTRY.register_module("vit-v1")
class VIT(nn.Module):
    is_logit_output = True
    def __init__(
        self,
        in_ch: int,
        out_ch: int = 32,
        pretrained: bool = True,
        dropout: float = 0.0,
        freeze: bool = False
    ) -> None:
        super().__init__()
        self.encoder = models.vit_b_16(pretrained=pretrained)
        if freeze:
            # Freeze all layers except the final layer
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.encoder.heads.head = torch.nn.Linear(self.encoder.heads.head.in_features, out_ch)
        if freeze:
            # Ensure the final layer's parameters are trainable
            for param in self.encoder.heads.head.parameters():
                param.requires_grad = True
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self._out_ch = out_ch
    @property
    def out_ch(self):
        return self._out_ch
    def forward(self, x: Tensor) -> Tensor:
        rows = []
        for i in range(0, x.shape[1], 3):
            image0 = x[:, i, :, :]
            image1 = x[:, i+1, :, :]
            image2 = x[:, i+2, :, :]
            rows.append(torch.concat((image0, image1, image2), dim=1))
        x = torch.concat(rows, dim=-1)[:, None, :, :].repeat(1,3,1,1)
        with torch.no_grad():
            x = self.transform(x)
        return self.encoder(x)