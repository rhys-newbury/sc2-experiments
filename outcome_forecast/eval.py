#!/usr/bin/env python3
"""Tool for gathering and formatting results"""
import os
from pathlib import Path
from typing import Optional
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
from konductor.utilities.pbar import IntervalPbar
import sqlite3

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
    data and transform to database dictionary input format"""
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
    batch_size: Annotated[Optional[int], typer.Option()] = None,
):
    """Run validation and save results new subdirectory"""
    if not datapath.exists():
        raise FileNotFoundError(datapath)
    os.environ["DATAPATH"] = str(datapath)

    exp_config = ExperimentInitConfig.from_run(run_path)
    if batch_size is not None:
        exp_config.set_batch_size(batch_size, Split.VAL)

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

    with IntervalPbar(
        total=len(dataloader), fraction=0.1
    ) as pbar, torch.inference_mode():
        for sample in dataloader:
            if isinstance(sample, list):
                sample = sample[0]
            preds = model(sample)
            results = binary_acc(preds, sample)
            logger(Split.VAL, meta.iteration, results)
            pbar.update(1)

    logger.flush()


@app.command()
def evaluate_percent(
    run_path: Annotated[Path, typer.Option()],
    datapath: Annotated[Path, typer.Option()],
    outdir: Annotated[str, typer.Option()],
    database: Annotated[Path, typer.Option()],
    num_buckets: Annotated[int, typer.Option()] = 50,
    batch_size: Annotated[Optional[int], typer.Option()] = None,
    workers: Annotated[Optional[int], typer.Option()] = None,
):
    """Run validation and save results new subdirectory"""
    if not datapath.exists():
        raise FileNotFoundError(datapath)

    os.environ["DATAPATH"] = str(datapath)

    conn = sqlite3.connect(str(database))
    cursor = conn.cursor()

    exp_config = ExperimentInitConfig.from_run(run_path)
    if batch_size is not None:
        exp_config.set_batch_size(batch_size, Split.VAL)
    if workers is not None:
        exp_config.set_workers(workers)

    # AMP isn't enabled during eval
    if "amp" in exp_config.trainer:
        del exp_config.trainer["amp"]

    model: nn.Module = get_model(exp_config)
    ckpt = torch.load(run_path / "latest.pt")["model"]
    model.load_state_dict(ckpt)
    model = model.eval().cuda()
    # import pdb; pdb.set_trace()
    # exp_config.data[0].val_loader.keys.append("")
    exp_config.data[0].dataset.args["keys"].append("metadata")
    dataset = get_dataset_config(exp_config)

    dataloader = dataset.get_dataloader(Split.VAL)

    binary_acc = BinaryAcc.from_config(exp_config)
    binary_acc.keep_batch = True

    outpath = run_path / outdir
    outpath.mkdir(exist_ok=True)

    total_results = torch.zeros((2, num_buckets), device="cuda")
    interval = 100 / num_buckets

    with IntervalPbar(
        total=len(dataloader), fraction=0.1
    ) as pbar, torch.inference_mode():
        for sample in dataloader:
            if isinstance(sample, list):
                sample = sample[0]
            preds = model(sample)
            results = binary_acc(preds, sample)

            metadata = [
                "".join([chr(x) for x in sublist])
                for sublist in sample["metadata"].cpu().numpy().tolist()
            ]

            replayHash, playerId = [x[:-1] for x in metadata], [x[-1] for x in metadata]
            query = "SELECT game_length FROM 'game_data' where " + " OR ".join(
                [
                    f'(replayHash = "{rh}" AND playerId = {pId})'
                    for rh, pId in zip(replayHash, playerId)
                ]
            )
            cursor.execute(query)
            game_length = torch.tensor(
                [x[0] for x in cursor.fetchall()], device=preds.device
            )
            game_length_mins = game_length / 22.4 / 60

            for idx, k in enumerate(results.keys()):
                time_point = float(k.split("_")[-1])
                percent = 100 * time_point / game_length_mins

                mask = torch.logical_and(
                    sample["valid"][:, idx].type(torch.bool), percent < 100
                )

                corrects = results[k][mask]

                if corrects.sum() > 0:
                    idx_mask = (percent // interval).type(torch.int64)

                    values, counts = torch.unique(
                        idx_mask[mask][corrects.type(torch.bool)], return_counts=True
                    )

                    total_results[0, values] += counts

                    values, counts = torch.unique(idx_mask[mask], return_counts=True)
                    total_results[1, values] += counts

            pbar.update(1)

    np.savetxt(
        outpath / f"game_length_results_{num_buckets}",
        total_results.cpu().numpy(),
        delimiter=",",
    )


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


@app.command()
def evaluate_all_percent(
    workspace: Annotated[Path, typer.Option()],
    datapath: Annotated[Path, typer.Option()],
    outdir: Annotated[str, typer.Option()],
    database: Annotated[Path, typer.Option()],
    num_buckets: Annotated[int, typer.Option()] = 50,
    batch_size: Annotated[Optional[int], typer.Option()] = None,
    workers: Annotated[Optional[int], typer.Option()] = None,
):
    """Run validaiton and save to a subdirectory"""
    if not datapath.exists():
        raise FileNotFoundError(datapath)

    def is_valid_run(path: Path):
        return path.is_dir() and (path / "latest.pt").exists()

    for run in filter(is_valid_run, workspace.iterdir()):
        print(f"Doing {run}")
        evaluate_percent(
            run, datapath, outdir, database, num_buckets, batch_size, workers
        )


if __name__ == "__main__":
    app()
