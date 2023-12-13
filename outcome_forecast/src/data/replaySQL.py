from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from konductor.data import DATASET_REGISTRY, DatasetConfig, ExperimentInitConfig, Split
from konductor.data._pytorch.dataloader import DataloaderV1Config
from sc2_replay_reader import (
    GAME_INFO_FILE,
    ReplayDatabase,
    ReplayParser,
    Result,
    setReplayDBLoggingLevel,
    spdlog_lvl,
)
from torch.utils.data import Dataset
import sqlite3
from .utils import find_closest_indicies


@dataclass
class TimeRange:
    min: float
    max: float
    step: float

    def __post_init__(self):
        assert self.min < self.max

    def arange(self):
        return torch.arange(self.min, self.max, self.step)


class SC2SQLReplay(Dataset):
    def __init__(
        self,
        basepath: Path,
        split: Split,
        train_ratio: float,
        features: set[str] | None,
        timepoints: TimeRange,
        min_game_time: float,
        sql_query: str,
        database: Path,
    ) -> None:
        super().__init__()
        self.features = features
        self.db_handle = ReplayDatabase()
        self.parser = ReplayParser(GAME_INFO_FILE)
        self.basepath = basepath      
        self.sql_query = sql_query

        setReplayDBLoggingLevel(spdlog_lvl.warn)

        self.conn = sqlite3.connect(database)
        self.cursor = self.conn.cursor()     
        self.cursor.execute(sql_query.replace(" * ", " COUNT(*) "))

        self.n_replays = self.cursor.fetchone()[0]
        self.n_replays *= train_ratio if split is Split.TRAIN else 1 - train_ratio
        self.n_replays = int(self.n_replays)
        assert self.n_replays > 0, "No replays in dataset"

        # Extract and print column names
        self.cursor.execute("PRAGMA table_info('game_data');")
        # Fetch all rows containing column information
        columns_info = self.cursor.fetchall()
        self.column_names = [column[1] for column in columns_info]
        self.file_name_idx = self.column_names.index("partition") 
        self.idx_idx = self.column_names.index("idx") 


        _loop_per_min = 22.4 * 60
        self._target_game_loops = (timepoints.arange() * _loop_per_min).to(torch.int)
        difference_array = torch.absolute(timepoints.arange() - min_game_time)
        self.min_index = difference_array.argmin()

    def __len__(self) -> int:
        return self.n_replays

    # @profile
    def __getitem__(self, index: int):
        
        squery = self.sql_query[:-1] + f" LIMIT 1 OFFSET {index};"
        self.cursor.execute(squery)        
        result = self.cursor.fetchone()

        self.db_handle.open(self.basepath / result[self.file_name_idx])
        self.parser.parse_replay(self.db_handle.getEntry(result[self.idx_idx]))

        outputs_list = self.parser.sample(0)
        if self.features is not None:
            outputs_list = {k: [outputs_list[k]] for k in self.features}
        else:
            outputs_list = {k: [outputs_list[k]] for k in outputs_list}

        sample_indicies = find_closest_indicies(
            self.parser.data.gameStep, self._target_game_loops[1:]
        )
        for idx in sample_indicies:
            if idx == -1:
                sample = {k: np.zeros_like(outputs_list[k][-1]) for k in outputs_list}
            else:
                sample = self.parser.sample(int(idx.item()))
            for k in outputs_list:
                outputs_list[k].append(sample[k])

        outputs = {
            "win": torch.as_tensor(
                self.parser.data.playerResult == Result.Win, dtype=torch.float32
            ),
            "valid": torch.cat([torch.tensor([True]), sample_indicies != -1]),
        }
        for k in outputs_list:
            outputs[k] = torch.stack([torch.as_tensor(o) for o in outputs_list[k]])

        return outputs


@dataclass
@DATASET_REGISTRY.register_module("sc2-sql-replay")
class SC2ReplayConfig(DatasetConfig):
    # Dataloader type we want to use
    train_loader: DataloaderV1Config
    val_loader: DataloaderV1Config

    features: set[str] | None = None
    train_ratio: float = 0.8  # Portion of all data to use for training
    timepoints: TimeRange = TimeRange(0, 30, 2)  # Minutes
    min_game_time: float = 5.0  # Minutes
    database: Path = Path("./")
    sql_filters: List[str] | None = None
    sql_query: str = ""

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0):
        return super().from_config(config, idx)

    def __post_init__(self):
        assert 0 < self.train_ratio < 1, f"Failed: 0<{self.train_ratio=}<1"
        # If features is not None, ensure that it is a set
        if self.features is not None and not isinstance(self.features, set):
            self.features = set(self.features)
        if isinstance(self.timepoints, dict):
            self.timepoints = TimeRange(**self.timepoints)

        sql_filter_string = (
            ""
            if self.sql_filters is None or len(self.sql_filters) == 0
            else (" " + " AND ".join(self.sql_filters))
        )
        self.sql_query = "SELECT * FROM game_data" + sql_filter_string + ";"
        assert sqlite3.complete_statement(self.sql_query), "Invalid SQL filters"

    @property
    def properties(self) -> Dict[str, Any]:
        ret = {"image_ch": 10, "scalar_ch": 28}
        ret.update(self.__dict__)
        return ret

    def get_dataloader(self, split: Split) -> Any:
        known_unused = {"train_loader", "val_loader", "basepath", "sql_filters"}
        dataset = self.init_auto_filter(
            SC2SQLReplay, known_unused=known_unused, split=split
        )
        match split:
            case Split.TRAIN:
                return self.train_loader.get_instance(dataset)
            case Split.VAL | Split.TEST:
                return self.train_loader.get_instance(dataset)
            case _:
                raise RuntimeError(f"How did I get here with {split=}")
