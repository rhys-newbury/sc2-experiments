#!/usr/bin/env python3
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import yaml
from konductor.data import (
    DatasetInitConfig,
    Split,
    make_from_init_config,
    DatasetConfig,
)
from konductor.utilities.pbar import LivePbar
from konductor.metadata.database.sqlite import SQLiteDB, DEFAULT_FILENAME, Metadata

from ..stats import MinimapSoftIoU, MinimapTarget

app = typer.Typer()


def get_dataset(config_path: Path):
    """
    From configuration file create dataloader.
    Configuration file can be of a full experiment format, or just the data configuration.
    ```
    dataset:
      - type: foo
        args: {}
    model:
       ...
    ```
    *OR*
    ```
    type: foo
    args: {}
    ```
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "dataset" in config:
        config = config["dataset"][0]

    init_cfg = DatasetInitConfig.from_dict(config)
    dataset_cfg = make_from_init_config(init_cfg)

    return dataset_cfg


def evaluate_trivial_prediction(dataset: DatasetConfig) -> dict[str, float]:
    """Trivially predict the next frame with the previous frame"""
    evaluator = MinimapSoftIoU(
        dataset.properties["clip_len"], MinimapTarget.BOTH, should_sigmoid=False
    )

    dataset.val_loader.workers = 8
    dataset.val_loader.dali_py_workers = 6
    dataloader = dataset.get_dataloader(Split.VAL)
    results: dict[str, list[float]] = {k: [] for k in evaluator.get_keys()}
    with LivePbar(total=len(dataloader)) as pbar:
        for sample in dataloader:
            # Unwrap dali pipe list
            if isinstance(sample, list):
                sample = sample[0]

            # Prediction is second-to-last minimap
            prediction = sample["minimap_features"][
                :, -2, None, MinimapTarget.indices(MinimapTarget.BOTH)
            ]
            result = evaluator(prediction, sample)
            for k, v in result.items():
                results[k].append(v)
            pbar.update(1)

    # Calculate average and remove soft_iou_ from key
    return {
        k.replace("soft_iou_", ""): np.nansum(v) / np.isfinite(v).sum()
        for k, v in results.items()
    }


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="Dataloader configuration to use")],
    workspace: Annotated[Path, typer.Option(help="Workspace with Results Database")],
):
    """Basic minimap baseline where the previous frame is used to predict the next frame"""
    dataset = get_dataset(config)
    results = evaluate_trivial_prediction(dataset)
    results["iteration"] = 0  # Add dummy iteration

    print(f"Writing results: {results}")

    results_db = SQLiteDB(workspace / DEFAULT_FILENAME)
    dummy_hash = "baseline_method"
    results_db.update_metadata(
        dummy_hash, Metadata(Path(), brief="current frame predicts next")
    )
    results_db.write("next_frame_soft_iou", dummy_hash, results)
    results_db.commit()


if __name__ == "__main__":
    app()
