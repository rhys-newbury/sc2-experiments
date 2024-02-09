import abc
import os
import sqlite3
from pathlib import Path

import torch
from konductor.data import Split, Registry
from sc2_replay_reader import ReplayDataScalarOnlyDatabase

from .utils import upper_bound, gen_val_query

SAMPLER_REGISTRY = Registry("replay-sampler")


class ReplaySampler(abc.ABC):
    """Abstract interface to sample from set of replay databases"""

    def __init__(self, train_ratio: float, split: Split) -> None:
        self.train_ratio = train_ratio
        self.split = split

    @abc.abstractmethod
    def sample(self, index: int) -> tuple[Path, int]:
        """Return path to database and index to sample"""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of replays in dataset"""

    def get_split_params(self, len_full_dataset: int):
        """Return start_idx and n_replays based on train or test|val"""
        train_size = int(len_full_dataset * self.train_ratio)

        if self.split is not Split.TRAIN:
            start_idx = train_size
            n_replays = len_full_dataset - train_size
        else:
            start_idx = 0
            n_replays = train_size

        assert n_replays > 0, f"n_replays is not greater than zero: {n_replays}"

        return start_idx, n_replays


@SAMPLER_REGISTRY.register_module("basic")
class BasicSampler(ReplaySampler):
    """Sample replays from single file or folder of replay files"""

    def __init__(self, replays_path: Path, train_ratio: float, split: Split) -> None:
        """Sample replays from a single database or folder of databases

        Args:
            path (Path): Path to individual or folder of .SC2Replays file(s)
            train_ratio (float): Proportion of all data used for training
            split (Split): whether to sample from train or test|val split

        Raises:
            FileNotFoundError: path doesn't exist
            AssertionError: no .SC2Replay files found in folder
        """
        super().__init__(train_ratio, split)
        if not replays_path.exists():
            raise FileNotFoundError(
                f"Replay dataset or folder doesn't exist: {replays_path}"
            )

        if replays_path.is_file():
            self.replays = [replays_path]
        else:
            self.replays = list(replays_path.glob("*.SC2Replays"))
            assert len(self.replays) > 0, f"No .SC2Replays found in {replays_path}"

        replays_per_file = torch.empty([len(self.replays) + 1], dtype=torch.int)
        replays_per_file[0] = 0

        db_interface = ReplayDataScalarOnlyDatabase()

        for idx, replay in enumerate(self.replays, start=1):
            db_interface.open(replay)
            replays_per_file[idx] = db_interface.size()

        self._accumulated_replays = torch.cumsum(replays_per_file, 0)
        self.start_idx, self.n_replays = self.get_split_params(
            int(self._accumulated_replays[-1].item())
        )

    def __len__(self) -> int:
        return self.n_replays

    def sample(self, index: int) -> tuple[Path, int]:
        file_index = upper_bound(self._accumulated_replays, self.start_idx + index)
        db_index = index - int(self._accumulated_replays[file_index].item())
        return self.replays[file_index], db_index


@SAMPLER_REGISTRY.register_module("sql")
class SQLSampler(ReplaySampler):
    """Use SQLite3 database to yield from a folder of replay databases with filters"""

    def __init__(
        self,
        database: str,
        replays_path: Path,
        filter_query: str | list[str],
        train_ratio: float,
        split: Split,
    ) -> None:
        """Filter sampled replays from a folder of replays

        Args:
            database (str): Path to sqlite3 database with replay info,
                            prefix '$ENV:' will be prefixed with DATAPATH
            replays_folder (Path): Path to folder of .SC2Replays file(s)
            filter_query (str): SQL query to filter sampled replays
            train_ratio (float): Proportion of all data used for training
            split (Split): whether to sample from train or test|val split

        Raises:
            AssertionError: replays_folder doesn't exist
            AssertionError: no .SC2Replay files found in folder
        """
        super().__init__(train_ratio, split)
        if database.startswith("$ENV:"):
            database_pth = Path(
                os.environ.get("DATAPATH", "/data")
            ) / database.removeprefix("$ENV:")
        else:
            database_pth = Path(database)
        assert database_pth.is_file(), f"Missing db {database_pth}"
        self.database = sqlite3.connect(database_pth)
        if isinstance(filter_query, list):
            filter_query = gen_val_query(database_pth, filter_query)
        self.filter_query = filter_query
        assert replays_path.exists(), f"replays_path not found: {replays_path}"
        self.replays_folder = replays_path

        cursor = self.database.cursor()
        cursor.execute(filter_query.replace(" * ", " COUNT (*) "))
        self.start_idx, self.n_replays = self.get_split_params(cursor.fetchone()[0])

        cursor.execute("PRAGMA table_info('game_data');")
        column_names = [col_info[1] for col_info in cursor.fetchall()]
        self.filename_col: int = column_names.index("partition")
        self.index_col: int = column_names.index("idx")

    def __len__(self) -> int:
        return self.n_replays

    def sample(self, index: int) -> tuple[Path, int]:
        query = self.filter_query[:-1] + f" LIMIT 1 OFFSET {self.start_idx + index};"
        cursor = self.database.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        return self.replays_folder / result[self.filename_col], result[self.index_col]
