"""Losses"""

from dataclasses import dataclass

from konductor.init import ExperimentInitConfig
from konductor.models import get_model_config
from konductor.losses import LossConfig, REGISTRY
from torch import nn, Tensor

from .utils import get_valid_sequence_mask


class WinBCELogits(nn.BCEWithLogitsLoss):
    def __init__(self) -> None:
        super().__init__(reduction="none")

    def forward(self, pred: Tensor, data: dict[str, Tensor]) -> dict[str, Tensor]:
        loss = super().forward(pred, data["win"].unsqueeze(-1).repeat(1, pred.shape[1]))
        loss *= data["valid"]
        return {"win-bce": loss.mean()}


class WinBCE(nn.BCELoss):
    def __init__(self) -> None:
        super().__init__(reduction="none")

    def forward(self, pred: Tensor, data: dict[str, Tensor]) -> dict[str, Tensor]:
        loss = super().forward(
            pred.unsqueeze(-1),
            data["win"].unsqueeze(-1).repeat(1, pred.shape[1]).unsqueeze(-1),
        )
        loss *= data["valid"].unsqueeze(-1)
        return {"win-bce": loss.mean()}


@dataclass
@REGISTRY.register_module("win-bce")
class WinBCECfg(LossConfig):
    model_output_logits: bool = True

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int, **kwargs):
        model = get_model_config(config=config).get_instance()
        if model.is_logit_output:
            config.criterion[idx].args["model_output_logits"] = model.is_logit_output
        return super().from_config(config, idx, **kwargs)

    def get_instance(self, *args, **kwargs):
        return WinBCELogits() if self.model_output_logits else WinBCE()


class MinimapForecast(nn.BCEWithLogitsLoss):
    def __init__(self, history_len: int):
        super().__init__(reduction="none")
        self.history_len = history_len

    def forward(self, pred: Tensor, target: dict[str, Tensor]) -> dict[str, Tensor]:
        """Only apply loss where all images in the sequence are valid"""
        minimaps_player = target["minimap_features"][:, :, [-4, -1]]
        next_minimap = minimaps_player[:, self.history_len :]
        loss = super().forward(pred, next_minimap).mean(dim=(-1, -2, -3))

        valid_seq = get_valid_sequence_mask(target["valid"], self.history_len)
        loss *= valid_seq

        return {"minimap-bce": loss.mean()}


@dataclass
@REGISTRY.register_module("minimap-bce")
class MinimapForecastBCE(LossConfig):
    history_len: int

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int, **kwargs):
        model_cfg = get_model_config(config=config)
        config.criterion[idx].args["history_len"] = model_cfg.history_len
        return super().from_config(config, idx, **kwargs)

    def get_instance(self, *args, **kwargs):
        return MinimapForecast(self.history_len)
