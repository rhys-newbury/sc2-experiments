#!/usr/bin/env python3
"""Tool for gathering and formatting results"""
import os
from pathlib import Path

import typer
import torch
from torch import nn
import numpy as np
import pandas as pd
from pyarrow import parquet as pq
from typing_extensions import Annotated
from konductor.metadata.database.sqlite import SQLiteDB, DEFAULT_FILENAME
from konductor.utilities.metadata import update_database
from konductor.init import ExperimentInitConfig
from konductor.models import get_model
from konductor.data import get_dataset_config, Split
from konductor.metadata.loggers import ParquetLogger
from konductor.metadata.database import Metadata
from konductor.utilities.pbar import LivePbar

from src.stats import BinaryAcc

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
    """Add Binary Accuracy to database table (and update metadata at the same time)"""
    update_database(
        workspace, "sqlite", f'{{"path": "{workspace / DEFAULT_FILENAME}"}}'
    )

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


# -------------------------------------------------------------------
# ------- Re-Running evaluation and saving to a separate file -------
# -------------------------------------------------------------------


@app.command()
def evaluate(
    run_path: Annotated[Path, typer.Option()],
    datapath: Annotated[Path, typer.Option()],
    outdir: Annotated[str, typer.Option()],
):
    """Run validation and save results new subdirectory"""
    if not datapath.exists():
        raise FileNotFoundError(datapath)
    os.environ["DATAPATH"] = str(datapath)

    exp_config = ExperimentInitConfig.from_run(run_path)

    # AMP isn't enabled during eval
    if "amp" in exp_config.trainer:
        del exp_config.trainer["amp"]

    model: nn.Module = get_model(exp_config)
    ckpt = torch.load(run_path / "latest.pt")["model"]
    model.load_state_dict(ckpt)
    model = model.eval().cuda()
    dataset = get_dataset_config(exp_config)
    dataloader = dataset.get_dataloader(Split.VAL)
    binary_acc = BinaryAcc.from_config(exp_config)

    outpath = run_path / outdir
    outpath.mkdir(exist_ok=True)
    logger = ParquetLogger(outpath)

    meta = Metadata.from_yaml(run_path / "metadata.yaml")

    with LivePbar(total=len(dataloader)) as pbar, torch.inference_mode():
        for sample in dataloader:
            if isinstance(sample, list):
                sample = sample[0]
            preds = model(sample)
            results = binary_acc(preds, sample)
            logger(Split.VAL, meta.iteration, results)
            pbar.update(1)

    logger.flush()


@app.command()
def evaluate_all(
    workspace: Annotated[Path, typer.Option()],
    datapath: Annotated[Path, typer.Option()],
    outdir: Annotated[str, typer.Option()],
):
    """Run validaiton and save to a subdirectory"""
    if not datapath.exists():
        raise FileNotFoundError(datapath)

    def is_valid_run(path: Path):
        return path.is_dir() and (path / "latest.pt").exists()

    for run in filter(is_valid_run, workspace.iterdir()):
        print(f"Doing {run}")
        evaluate(run, datapath, outdir)


if __name__ == "__main__":
    app()
