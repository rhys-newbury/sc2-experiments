"""
Data transform utilities
"""
from pathlib import Path

import numpy as np
import typer
import yaml
from konductor.init import DatasetInitConfig
from konductor.data import make_from_init_config
from src.data.base_dataset import Split
from sc2_replay_reader import set_replay_database_logger_level, spdlog_lvl
from typing_extensions import Annotated
from konductor.utilities.pbar import LivePbar, IntervalPbar

app = typer.Typer()


def convert_split(outfolder: Path, dataloader, live: bool):
    """Run conversion and write to outfolder"""
    pbar_type = LivePbar if live else IntervalPbar
    pbar_kwargs = {"total": len(dataloader), "desc": outfolder.stem}
    if not live:
        pbar_kwargs["fraction"] = 0.1
    with pbar_type(**pbar_kwargs) as pbar:
        for sample in dataloader:
            # Unwrap batch dim and change torch tensor to numpy array
            formatted: dict[str, str | np.ndarray] = {}
            for k, v in sample.items():
                v = v[0]
                formatted[k] = v if isinstance(v, str) else v.numpy()
            np.savez_compressed(outfolder / formatted["metadata"], **formatted)
            pbar.update(1)


@app.command()
def make_numpy_subset(
    config: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    workers: Annotated[int, typer.Option()] = 4,
    live: Annotated[bool, typer.Option(help="Use live pbar")] = False,
):
    """
    Make a subfolder dataset of numpy files from configuration
    for less expensive dataloading for training. When training with
    a small amount of data from a full replay, it is probably worth
    it to make a small subset of the required data, rather than loading
    all data and discarding the vast majority of it.

    Args:
        config (Path): Path to data configuration yaml
        output (Path): Root folder to save new dataset
        workers (int): Number of dataloader workers to use
    """
    with open(config, "r", encoding="utf-8") as f:
        loaded_dict = yaml.safe_load(f)
        if "dataset" in loaded_dict:  # Unwrap if normal experiment config
            loaded_dict = loaded_dict["dataset"]

    init_config = DatasetInitConfig.from_dict(loaded_dict)
    dataset_cfg = make_from_init_config(init_config)

    # Return metadata to save as filename
    dataset_cfg.metadata = True
    # Set dataloader workers
    dataset_cfg.train_loader.workers = workers
    dataset_cfg.val_loader.workers = workers
    # Enforce batch size of 1
    dataset_cfg.train_loader.batch_size = 1
    dataset_cfg.val_loader.batch_size = 1

    # Create output root and copy configuration for traceability
    output.mkdir(exist_ok=True)
    with open(output / "generation-config.yml", "w", encoding="utf-8") as f:
        yaml.dump(loaded_dict, f)

    for split in [Split.TRAIN, Split.VAL]:
        dataloader = dataset_cfg.get_dataloader(split)
        outsubfolder = output / split.name.lower()
        outsubfolder.mkdir(exist_ok=True)
        convert_split(outsubfolder, dataloader, live)


if __name__ == "__main__":
    set_replay_database_logger_level(spdlog_lvl.warn)
    app()
