"""Losses"""

from dataclasses import dataclass

import torch
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


class MinimapBCE(nn.BCEWithLogitsLoss):
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
class MinimapBCECfg(LossConfig):
    history_len: int

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int, **kwargs):
        model_cfg = get_model_config(config=config)
        config.criterion[idx].args["history_len"] = model_cfg.history_len
        return super().from_config(config, idx, **kwargs)

    def get_instance(self, *args, **kwargs):
        return MinimapBCE(self.history_len)


class MinimapFocal(nn.BCEWithLogitsLoss):
    def __init__(
        self,
        history_len: int,
        alpha: float = 0.75,
        gamma: float = 2,
        pos_weight: float = 2,
    ) -> None:
        pos_weight_tensor = torch.tensor(pos_weight)
        if torch.cuda.is_available():
            pos_weight_tensor.cuda()
        super().__init__(reduction="none", pos_weight=pos_weight_tensor)
        self.alpha = alpha
        self.gamma = gamma
        self.history_len = history_len

    def _focal_loss(self, pred: Tensor, target: Tensor):
        prob = pred.sigmoid()
        bce_loss = super().forward(pred, target)
        p_t = prob * target + (1 - prob) * (1 - target)
        loss = bce_loss * ((1 - p_t + torch.finfo(prob.dtype).eps) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        return loss

    def forward(
        self, predictions: Tensor, targets: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        minimaps_player = targets["minimap_features"][:, :, [-4, -1]]
        next_minimap = minimaps_player[:, self.history_len :]
        loss_mask = self._focal_loss(predictions, next_minimap)
        loss_sequence = loss_mask.mean(dim=(-1, -2, -3))

        if "valid" in targets:
            valid_seq = get_valid_sequence_mask(targets["valid"], self.history_len + 1)
            loss_sequence *= valid_seq

        return {"minimap-focal": loss_sequence.mean()}


@dataclass
@REGISTRY.register_module("minimap-focal")
class MinimapFocalCfg(LossConfig):
    history_len: int
    alpha: float = 0.75
    gamma: float = 0.2
    pos_weight: float = 1.0

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int, **kwargs):
        model_cfg = get_model_config(config=config)
        config.criterion[idx].args["history_len"] = model_cfg.history_len
        return super().from_config(config, idx, **kwargs)

    def get_instance(self, *args, **kwargs):
        return self.init_auto_filter(MinimapFocal)


class MinimapFocalMotion(MinimapFocal):
    def __init__(
        self,
        history_len: int,
        motion_weight: float = 1.5,
        alpha: float = 0.75,
        gamma: float = 2,
        pos_weight: float = 2,
    ) -> None:
        super().__init__(history_len, alpha, gamma, pos_weight)
        self.motion_weight = motion_weight

    @torch.no_grad()
    def _calc_motion(self, prev: Tensor, nxt: Tensor) -> Tensor:
        """Calculate pixel-wise motion mask with weighting factor"""
        mask = (prev != nxt).to(torch.float32) * self.motion_weight
        return mask

    def forward(
        self, predictions: Tensor, targets: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        minimaps_player = targets["minimap_features"][:, :, [-4, -1]]
        next_minimap = minimaps_player[:, self.history_len :]
        loss_pixels = self._focal_loss(predictions, next_minimap)
        prev_minimap = minimaps_player[:, self.history_len - 1 : -1]
        loss_pixels = loss_pixels * self._calc_motion(prev_minimap, next_minimap)
        loss_sequence = loss_pixels.mean(dim=(-1, -2, -3))

        if "valid" in targets:
            valid_seq = get_valid_sequence_mask(targets["valid"], self.history_len + 1)
            loss_sequence *= valid_seq

        return {"minimap-focal": loss_sequence.mean()}


@dataclass
@REGISTRY.register_module("minimap-focal-motion")
class MinimapFocalMotionCfg(LossConfig):
    history_len: int
    alpha: float = 0.75
    gamma: float = 0.2
    pos_weight: float = 1.0
    motion_weight: float = 1.5

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int, **kwargs):
        model_cfg = get_model_config(config=config)
        config.criterion[idx].args["history_len"] = model_cfg.history_len
        return super().from_config(config, idx, **kwargs)

    def get_instance(self, *args, **kwargs):
        return self.init_auto_filter(MinimapFocal)
