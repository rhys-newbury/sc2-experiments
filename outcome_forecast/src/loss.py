"""Losses"""
from konductor.init import ExperimentInitConfig
from konductor.losses import LossConfig, REGISTRY
from torch import nn, Tensor


class WinBCE(nn.BCEWithLogitsLoss):
    def __init__(self) -> None:
        super().__init__(reduction="none")

    def forward(self, pred: Tensor, data: dict[str, Tensor]) -> dict[str, Tensor]:
        win = data["win"]
        if len(win.shape) == 1:
            win = win.unsqueeze(-1)
        loss = super().forward(pred, win.repeat(1, pred.shape[1]))
        loss *= data["valid"]
        return {"win-bce": loss.mean()}


@REGISTRY.register_module("win-bce")
class WinBCECfg(LossConfig):
    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int, **kwargs):
        return super().from_config(config, idx, **kwargs)

    def get_instance(self, *args, **kwargs):
        return WinBCE()
