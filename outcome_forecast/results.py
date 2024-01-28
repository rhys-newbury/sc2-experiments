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

    def __init__(self, value: float):
        self.value = value

    def as_db_key(self):
        return "t_" + str(self.value).replace(".", "_")

    def as_pq_key(self):
        return "binary_acc_" + str(self.value)

    def as_float(self):
        return float(self.value)


def transform_latest_to_db_format(
    data: pd.DataFrame, time_points: list[TimePoint]
) -> dict[str, float]:
    """Grab the last iteration from the parquet
    data and trasform to database dictionary input format"""
    iteration = data["iteration"].max()
    average = data.query(f"iteration == {iteration}").mean()
    transformed = {
        t.as_db_key(): float(average[t.as_pq_key()])
        for t in time_points
        if t.as_pq_key() in data.columns
    }
    transformed["iteration"] = int(iteration)
    return transformed


@app.command()
def gather_ml_binary_accuracy(workspace: Annotated[Path, typer.Option()] = Path.cwd()):
    """"""
    db_handle = SQLiteDB(workspace / DEFAULT_FILENAME)
    time_points = [TimePoint(t) for t in np.arange(0, 20, 0.5)]
    table_name = "binary_accuracy"
    db_format = {"iteration": "INTEGER"}
    db_format.update({t.as_db_key(): "FLOAT" for t in time_points})
    db_handle.create_table(table_name, db_format)

    for exp_run in filter(lambda x: x.is_dir(), workspace.iterdir()):
        parquet_filename = exp_run / "val_binary-acc.parquet"
        if not parquet_filename.exists():
            continue
        data: pd.DataFrame = pq.read_table(parquet_filename).to_pandas()
        results = transform_latest_to_db_format(data, time_points)
        db_handle.write(table_name, exp_run.name, results)

    db_handle.commit()
    db_handle.con.close()


if __name__ == "__main__":
    app()
