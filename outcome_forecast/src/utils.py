"""Utility things I guess"""

from dataclasses import dataclass
from math import ceil

import torch


@dataclass
class TimeRange:
    """Holds parameters for creating a time range"""

    min: float
    max: float
    step: float

    def __post_init__(self):
        assert self.min < self.max

    def __len__(self) -> int:
        return max(int(ceil((self.max - self.min) / self.step)), 0)

    def arange(self):
        return torch.arange(self.min, self.max, self.step)
