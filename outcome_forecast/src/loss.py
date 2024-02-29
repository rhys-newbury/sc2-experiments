"""Losses"""

from dataclasses import dataclass

import torch
from konductor.init import ExperimentInitConfig
from konductor.models import get_model_config
from konductor.losses import LossConfig, REGISTRY
from torch import nn, Tensor
from torch.nn import functional as F

from .utils import get_valid_sequence_mask


class WinBCE(nn.Module):
    def __init__(self, pred_is_logit: bool) -> None:
        super().__init__()
        self.loss_fn = (
            F.binary_cross_entropy_with_logits
            if pred_is_logit
            else F.binary_cross_entropy
        )

    def forward(self, pred: Tensor, data: dict[str, Tensor]) -> dict[str, Tensor]:
        win_repeat = data["win"].unsqueeze(-1).repeat(1, pred.shape[1])
        loss = self.loss_fn(pred, win_repeat, reduction="none")
        loss *= data["valid"]
        return {"win-bce": loss.mean()}


@dataclass
@REGISTRY.register_module("win-bce")
class WinBCECfg(LossConfig):
    model_output_logits: bool = True

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int, **kwargs):
        model = get_model_config(config=config).get_instance()
        if model.is_logit_output:  # check Truth and None
            config.criterion[idx].args["model_output_logits"] = model.is_logit_output
        return super().from_config(config, idx, **kwargs)

    def get_instance(self, *args, **kwargs):
        return WinBCE(self.model_output_logits)


class MinimapLoss(nn.Module):
    """
    Base minimap loss that deals with masking and motion weighting, pixel-wise
    loss must be defined and the name of loss assigned in the derived class
    """

    def __init__(self, history_len: int, motion_weight: float | None) -> None:
        super().__init__()
        self.history_len = history_len
        self.motion_weight = motion_weight
        if self.motion_weight is not None:
            assert self.motion_weight > 1, f"{motion_weight=}"

    @property
    def _key(self) -> str:
        """Name of the loss function"""
        raise NotImplementedError()

    def _loss_fn(self, preds: Tensor, target: Tensor) -> Tensor:
        """Returns pixel-wise loss between preds and target"""
        raise NotImplementedError()

    def _get_motion_weight(self, prev: Tensor, nxt: Tensor) -> Tensor:
        """Calculate pixel-wise motion weighting factor to emphasise loss"""
        assert self.motion_weight is not None
        mask = torch.ones_like(prev)
        mask[prev != nxt] = self.motion_weight
        return mask

    def forward(
        self, predictions: Tensor, targets: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        minimaps_player = targets["minimap_features"][:, :, [-4, -1]]
        next_minimap = minimaps_player[:, self.history_len :]
        loss_mask = self._loss_fn(predictions, next_minimap)

        if self.motion_weight is not None:
            prev_minimap = minimaps_player[:, self.history_len - 1 : -1]
            loss_mask *= self._get_motion_weight(prev_minimap, next_minimap)

        loss_sequence = loss_mask.mean(dim=(-1, -2, -3))

        if "valid" in targets:
            valid_seq = get_valid_sequence_mask(targets["valid"], self.history_len + 1)
            loss_sequence *= valid_seq

        return {self._key: loss_sequence.mean()}


class MinimapBCE(MinimapLoss):
    def __init__(self, history_len: int, motion_weight: float | None = None):
        super().__init__(history_len, motion_weight)

    @property
    def _key(self):
        return "minimap-bce"

    def _loss_fn(self, preds: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(preds, target, reduction="none")


@dataclass
@REGISTRY.register_module("minimap-bce")
class MinimapBCECfg(LossConfig):
    history_len: int
    motion_weight: float | None = None

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int, **kwargs):
        model_cfg = get_model_config(config=config)
        config.criterion[idx].args["history_len"] = model_cfg.history_len
        return super().from_config(config, idx, **kwargs)

    def get_instance(self, *args, **kwargs):
        return self.init_auto_filter(MinimapBCE)


def focal_loss(pred: Tensor, target: Tensor, eps: float, gamma: float, alpha: float):
    prob = pred.sigmoid()
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    loss = bce_loss * ((1 - p_t + eps) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss

    return loss


focal_loss_jit: torch.ScriptFunction = torch.jit.script(focal_loss)


class MinimapFocal(MinimapLoss):
    def __init__(
        self,
        history_len: int,
        alpha: float = 0.75,
        gamma: float = 2,
        motion_weight: float | None = None,
    ) -> None:
        super().__init__(history_len, motion_weight)
        self.alpha = alpha
        self.gamma = gamma

    @property
    def _key(self):
        return "minimap-focal"

    def _loss_fn(self, pred: Tensor, target: Tensor):
        return focal_loss_jit(
            pred, target, torch.finfo(pred.dtype).eps, self.gamma, self.alpha
        )


@dataclass
@REGISTRY.register_module("minimap-focal")
class MinimapFocalCfg(LossConfig):
    history_len: int
    alpha: float = 0.75
    gamma: float = 0.2
    motion_weight: float | None = None

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int, **kwargs):
        model_cfg = get_model_config(config=config)
        config.criterion[idx].args["history_len"] = model_cfg.history_len
        return super().from_config(config, idx, **kwargs)

    def get_instance(self, *args, **kwargs):
        return self.init_auto_filter(MinimapFocal)
