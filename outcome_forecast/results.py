#!/usr/bin/env python3
"""Tool for gathering and formatting results"""
from pathlib import Path
import math

import typer
import numpy as np
import pandas as pd
from pyarrow import parquet as pq
from typing_extensions import Annotated
from konductor.metadata.database.sqlite import SQLiteDB, DEFAULT_FILENAME

app = typer.Typer()


class TimePoint:
    __slots__ = "value"

    def as_db_key(self):
        intergral = int(self.value)
        return f"{intergral}_{self.value - intergral}"

    def as_pq_key(self):
        return str(self.value)

    def as_float(self):
        return float(self.value)


@app.command()
def gather_parquets(workspace: Annotated[Path, typer.Option()] = Path.cwd()):
    """"""
    db_handle = SQLiteDB(workspace / DEFAULT_FILENAME)
    time_points = np.arange(0, 20, 0.5)
    table_name = "binary_accuracy"
    db_handle.create_table(table_name, [f"t_{t}" for t in time_points])

    for exp_run in filter(lambda x: x.is_dir(), workspace.iterdir()):
        data: pd.DataFrame = pq.read_table(
            exp_run / "val_binary-acc.parquet"
        ).to_pandas()
        last_iter = data["iteration"].max()
        # transform to dict[time_point, float]
        # db_handle.write(table_name, exp_run.parent.name, DATAHERE)


if __name__ == "__main__":
    app()
