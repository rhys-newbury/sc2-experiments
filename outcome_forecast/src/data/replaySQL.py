import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List

from konductor.data import DATASET_REGISTRY, Split

from .baseDataset import SC2ReplayBase, SC2ReplayConfigBase, TimeRange
from .utils import gen_val_query


class SC2SQLReplay(SC2ReplayBase):
    """Filter dataset sampling based on sql queries"""

    def __init__(
        self,
        basepath: Path,
        split: Split,
        train_ratio: float,
        features: set[str] | None,
        timepoints: TimeRange,
        sql_query: str,
        database: Path,
        minimap_layers: list[str] | None = None,
    ) -> None:
        self.sql_query = sql_query
        self.database = database

        super().__init__(
            basepath, split, train_ratio, features, timepoints, minimap_layers
        )

        # Extract and print column names
        self.cursor.execute("PRAGMA table_info('game_data');")
        # Fetch all rows containing column information
        columns_info = self.cursor.fetchall()
        self.column_names = [column[1] for column in columns_info]
        self.file_name_idx = self.column_names.index("partition")
        self.idx_idx = self.column_names.index("idx")

    def load_files(self, basepath):
        self.conn = sqlite3.connect(self.database)
        self.cursor = self.conn.cursor()
        self.cursor.execute(self.sql_query.replace(" * ", " COUNT(*) "))

        self.init_split_params(self.cursor.fetchone()[0])

    def __getitem__(self, index: int):
        squery = self.sql_query[:-1] + f" LIMIT 1 OFFSET {self.start_idx + index};"
        self.cursor.execute(squery)
        result = self.cursor.fetchone()

        self.load_to_parser(
            self.basepath / result[self.file_name_idx], result[self.idx_idx]
        )

        return self.process_replay()


@dataclass
@DATASET_REGISTRY.register_module("sc2-sql-replay")
class SC2ReplayConfig(SC2ReplayConfigBase):
    database: Path = Path("./")
    sql_filters: List[str] | None = None
    sql_query: str = ""

    def get_class(self):
        return SC2SQLReplay

    def _known_unused(self):
        return {"train_loader", "val_loader", "basepath", "sql_filters"}

    def __post_init__(self):
        super().__post_init__()
        self.sql_query = gen_val_query(self.database, self.sql_filters)
