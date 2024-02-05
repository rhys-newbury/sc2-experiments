"""Losses"""

from dataclasses import dataclass

from konductor.init import ExperimentInitConfig
from konductor.models import get_model
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
        model = get_model(config=config)
        if model.is_logit_output:
            config.criterion[idx].args["model_output_logits"] = model.is_logit_output
        return super().from_config(config, idx, **kwargs)

    def get_instance(self, *args, **kwargs):
        return WinBCELogits() if self.model_output_logits else WinBCE()
