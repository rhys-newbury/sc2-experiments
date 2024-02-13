"""Utility things I guess"""

from dataclasses import dataclass
from math import ceil

import torch
from torch import Tensor


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


@torch.no_grad()
def get_valid_sequence_mask(valid: Tensor, sequence_len: int):
    """
    For a validity mask of the observations in a sequence, find
    create a new mask that corresponds to when they are all true
    for a given subsequence window.
    """
    is_valid: list[Tensor] = []
    n_time = valid.shape[1]
    for start_idx in range(n_time - sequence_len + 1):
        end_idx = start_idx + sequence_len
        is_valid.append(valid[:, start_idx:end_idx].all(dim=1))
    return torch.stack(is_valid, dim=1)
