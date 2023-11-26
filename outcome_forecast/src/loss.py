"""Losses"""

from konductor.init import ExperimentInitConfig
from konductor.losses import LossConfig, REGISTRY
from torch import nn, Tensor


class WinBCE(nn.BCEWithLogitsLoss):
    def forward(self, pred: Tensor, data: dict[str, Tensor]) -> dict[str, Tensor]:
        return {"win-bce": super().forward(pred, data["win"])}


@REGISTRY.register_module("win-bce")
class WinBCECfg(LossConfig):
    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int, **kwargs):
        return super().from_config(config, idx, **kwargs)

    def get_instance(self, *args, **kwargs):
        return WinBCE()
