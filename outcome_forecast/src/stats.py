"""
Statistics for game outcome prediction
"""
from typing import Sequence

from torch import Tensor
from konductor.init import ExperimentInitConfig
from konductor.metadata.base_statistic import Statistic, STATISTICS_REGISTRY


@STATISTICS_REGISTRY.register_module("win_auc")
class WinAUC(Statistic):
    """AUC of win prediction performance at several timepoints (min) in the game"""

    @classmethod
    def from_config(cls, cfg: ExperimentInitConfig, **extras):
        return cls(**extras)

    def get_keys(self) -> list[str]:
        return [f"auc_{t}" for t in self.timepoints]

    def __init__(self, timepoints: Sequence[int] | None = None) -> None:
        """Initialize the win prediciton statistics calculator

        Args:
            timepoints (Sequence[int] | None, optional): Sequence of timepoints
            in minutes where to sample the accuracy of the model. If None given
            defauts to every 2min up to 30min.
        """
        self.timepoints = list(range(2, 32, 2)) if timepoints is None else timepoints
        self._game_loops = [int(t * 60 * 22.4) for t in self.timepoints]  # Game stp

    def __call__(
        self, predictions: dict[str, Tensor], targets: dict[str, Tensor]
    ) -> dict[str, float]:
        """Calculate AUC of the win prediction closest to the previously specified timepoints

        Args:
            predictions (dict[str, Tensor]): Outputs of the model
            targets (dict[str, Tensor]): Loaded data, should contain win/loss

        Returns:
            dict[str, float]: Dictionary of AUC at each timepoint (until end of replay)
        """
        return {}
