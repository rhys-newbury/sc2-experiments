from typing import Sequence, List
from pathlib import Path
import sqlite3
import torch
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
        if prv <= targets[tgt_idx] and nxt >= targets[tgt_idx]:
            nearest[tgt_idx] = idx
            tgt_idx += 1
            if tgt_idx == nearest.nelement():
                break
    return nearest


def gen_val_query(database: Path, sql_filters: List[str] | None):
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
