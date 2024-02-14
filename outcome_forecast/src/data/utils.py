import sqlite3
from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from nvidia.dali.types import BatchInfo, SampleInfo
from torch import Tensor


def upper_bound(x: Tensor, value: float) -> int:
    """
    Find the index of the last element which is less or equal to value
    """
    return int(torch.argwhere(torch.le(x, value))[-1].item())


def find_closest_indicies(options: Sequence[int], targets: Sequence[int]):
    """
    Find the closest option corresponding to a target, if there is no match, place -1
    TODO Convert this to cpp
    """
    tgt_idx = 0
    nearest = torch.full([len(targets)], -1, dtype=torch.int32)
    for idx, (prv, nxt) in enumerate(zip(options, options[1:])):
        if prv > targets[tgt_idx]:  # not in between, skip
            tgt_idx += 1
            if tgt_idx == nearest.nelement():
                break
            continue
        if prv <= targets[tgt_idx] <= nxt:
            nearest[tgt_idx] = idx
            tgt_idx += 1
            if tgt_idx == nearest.nelement():
                break
    return nearest


def gen_val_query(database: Path, sql_filters: list[str] | None):
    """Transform list of sql filters to valid query and test that it works"""
    sql_filter_string = (
        ""
        if sql_filters is None or len(sql_filters) == 0
        else (" WHERE " + " AND ".join(sql_filters))
    )
    sql_query = "SELECT * FROM game_data" + sql_filter_string + ";"
    assert sqlite3.complete_statement(sql_query), "Incomplete SQL Statement"
    with sqlite3.connect(database) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(sql_query)
        except sqlite3.OperationalError as e:
            raise AssertionError("Invalid SQL Syntax") from e
    return sql_query


class BaseDALIDataset(ABC):
    """External Iterator for DALI"""

    def __init__(
        self,
        batch_size: int,
        shard_id: int,
        num_shards: int,
        random_shuffle: bool,
        yields_batch: bool = False,
    ) -> None:
        self.logger = getLogger(type(self).__name__)
        self.shard_id = shard_id
        self.batch_size = batch_size
        self.num_shards = num_shards
        self.random_shuffle = random_shuffle
        self.idx_samples: np.ndarray = np.zeros(0, dtype=np.int64)
        self.last_seen_epoch = -1
        self.yields_batch = yields_batch

    def _initialize(self):
        self.idx_samples = np.arange(len(self), dtype=np.int64)

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples in dataset"""

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._initialize()

    @property
    def num_iterations(self):
        _num_iter = len(self) // self.num_shards
        if not self.yields_batch:
            _num_iter //= self.batch_size
        return _num_iter

    def resample_indicies(self, epoch_idx: int):
        self.last_seen_epoch = epoch_idx
        self.idx_samples = np.random.default_rng(seed=42 + epoch_idx).permutation(
            len(self)
        )

    def __call__(self, yield_info: SampleInfo | BatchInfo) -> int:
        if len(self) == 0:
            self._initialize()

        if yield_info.iteration >= self.num_iterations:
            raise StopIteration

        if self.random_shuffle and yield_info.epoch_idx != self.last_seen_epoch:
            self.resample_indicies(yield_info.epoch_idx)

        idx = self.shard_id
        idx += (
            yield_info.idx_in_epoch
            if isinstance(yield_info, SampleInfo)
            else yield_info.iteration
        )

        return self.idx_samples[idx].item()
