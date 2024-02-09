"""Losses"""

from dataclasses import dataclass

import torch
from konductor.init import ExperimentInitConfig
from konductor.models import get_model_config
from konductor.losses import LossConfig, REGISTRY
from torch import nn, Tensor


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

    def get_valid_mask(self, valid: Tensor) -> Tensor:
        is_valid: list[Tensor] = []
        n_time = valid.shape[1]
        for start_idx in range(n_time - self.history_len):
            end_idx = start_idx + self.history_len
            is_valid.append(valid[:, start_idx:end_idx].all(dim=1))
        return torch.stack(is_valid, dim=1)

    def get_next_minimap_truth(self, minimaps: Tensor) -> Tensor:
        next_minimaps: list[Tensor] = []
        n_time = minimaps.shape[1]
        for start_idx in range(n_time - self.history_len):
            end_idx = start_idx + self.history_len
            next_minimaps.append(minimaps[:, end_idx])
        return torch.stack(next_minimaps, dim=1)

    def forward(self, pred: Tensor, target: dict[str, Tensor]) -> dict[str, Tensor]:
        """Only apply loss where all images in the sequence are valid"""
        minimaps_player = target["minimap_features"][:, :, [-4, -1]]
        next_minimap = self.get_next_minimap_truth(minimaps_player)
        loss = super().forward(pred, next_minimap).mean(dim=(-1, -2, -3))

        valid_seq = self.get_valid_mask(target["valid"])
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
