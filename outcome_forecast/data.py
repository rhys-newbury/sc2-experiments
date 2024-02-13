"""
Data transform utilities
"""

from pathlib import Path
from typing import Callable

import numpy as np
import typer
import yaml
from konductor.data import make_from_init_config
from konductor.init import DatasetInitConfig
from konductor.registry import Registry
from konductor.utilities.pbar import IntervalPbar, LivePbar
from sc2_replay_reader import set_replay_database_logger_level, spdlog_lvl
from src.data.base_dataset import Split
from src.utils import StrEnum
from torch import Tensor
from typing_extensions import Annotated

app = typer.Typer()


def get_dataset_config_for_data_transform(config_file: Path, workers: int):
    """Returns configured dataset and original configuration dictionary"""
    with open(config_file, "r", encoding="utf-8") as f:
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

    return dataset_cfg, loaded_dict


def save_dataset_configuration(config_dict: dict[str, object], outfolder: Path):
    """Saves configuration to outfolder for traceability"""
    with open(outfolder / "generation-config.yml", "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)


def make_pbar(dataloader, desc: str, live: bool):
    """Make pbar live or fraction depending on flag"""
    pbar_type = LivePbar if live else IntervalPbar
    pbar_kwargs = {"total": len(dataloader), "desc": desc}
    if not live:
        pbar_kwargs["fraction"] = 0.1
    return pbar_type(**pbar_kwargs)


def convert_to_numpy_files(outfolder: Path, dataloader, live: bool):
    """Run conversion and write to outfolder"""
    with make_pbar(dataloader, outfolder.stem, live) as pbar:
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
    dataset_cfg, loaded_dict = get_dataset_config_for_data_transform(config, workers)

    # Create output root and copy configuration for traceability
    output.mkdir(exist_ok=True)
    save_dataset_configuration(loaded_dict, output)

    for split in [Split.TRAIN, Split.VAL]:
        dataloader = dataset_cfg.get_dataloader(split)
        outsubfolder = output / split.name.lower()
        outsubfolder.mkdir(exist_ok=True)
        convert_to_numpy_files(outsubfolder, dataloader, live)


WriterFn = Callable[[Tensor, Path], None]


def convert_minimaps_to_videos(
    outfolder: Path, dataloader, live: bool, writer: WriterFn
):
    """Write each minimap sequence as a video file"""
    with make_pbar(dataloader, outfolder.stem, live) as pbar:
        for sample_ in dataloader:
            sample = sample_[0] if isinstance(sample_, list) else sample_
            outpath = outfolder / sample["metadata"]
            writer(sample["minimap_features"], outpath)
            pbar.update(1)


WRITER_REGISTRY = Registry("video-writer")


@WRITER_REGISTRY.register_module()
def self_enemy_heightmap(minimap_seq: Tensor, writepath: Path):
    """Write video with three channels [self, enemy, heightmap]"""
    pass


WriterType = StrEnum("WriterType", list(WRITER_REGISTRY.module_dict))


@app.command()
def make_minimap_videos(
    config: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    writer: Annotated[WriterType, typer.Option()],
    workers: Annotated[int, typer.Option()] = 4,
    live: Annotated[bool, typer.Option(help="Use live pbar")] = False,
):
    """
    Reading a few frames from a SC2Replay is a bit wasteful, perhaps
    using the native DALI videoreader could perhaps be faster.
    """
    dataset_cfg, loaded_dict = get_dataset_config_for_data_transform(config, workers)

    output.mkdir(exist_ok=True)
    save_dataset_configuration(loaded_dict, output)

    writer_func: WriterFn = WRITER_REGISTRY[writer]

    for split in [Split.TRAIN, Split.VAL]:
        dataloader = dataset_cfg.get_dataloader(split)
        outsubfolder = output / split.name.lower()
        outsubfolder.mkdir(exist_ok=True)
        convert_minimaps_to_videos(outsubfolder, dataloader, live, writer_func)


if __name__ == "__main__":
    set_replay_database_logger_level(spdlog_lvl.warn)
    app()
