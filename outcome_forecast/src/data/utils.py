import sqlite3
from logging import getLogger
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from nvidia.dali import types
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
        if prv > targets[tgt_idx]:  # not inbetween, skip
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


class BaseDALIDataset:
    """External Iterator for DALI"""

    def __init__(
        self,
        batch_size: int,
        shard_id: int,
        num_shards: int,
        random_shuffle: bool,
    ) -> None:
        self.logger = getLogger(type(self).__name__)
        self.shard_id = shard_id
        self.batch_size = batch_size
        self.num_shards = num_shards
        self.random_shuffle = random_shuffle
        self.idx_samples: np.ndarray = np.zeros(0, dtype=np.int64)
        self.last_seen_epoch = -1

    def _initialize(self):
        self.idx_samples = np.arange(len(self), dtype=np.int64)

    def __len__(self):
        raise NotImplementedError

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._initialize()

    @property
    def num_iterations(self):
        return len(self) // self.num_shards // self.batch_size

    def resample_indicies(self, sample_info: types.SampleInfo):
        self.last_seen_epoch = sample_info.epoch_idx
        self.idx_samples = np.random.default_rng(
            seed=42 + sample_info.epoch_idx
        ).permutation(len(self))

    def __call__(self, sample_info: types.SampleInfo) -> int:
        if len(self) == 0:
            self._initialize()

        if sample_info.iteration >= self.num_iterations:
            raise StopIteration

        if self.random_shuffle and sample_info.epoch_idx != self.last_seen_epoch:
            self.resample_indicies(sample_info)

        idx = sample_info.idx_in_epoch + self.shard_id
        return self.idx_samples[idx].item()
