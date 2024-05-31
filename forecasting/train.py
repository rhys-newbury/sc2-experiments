"""Main training entrypoint"""

import logging
import shutil
from functools import partial
from pathlib import Path
from typing import Optional

import src
import torch
import typer
import yaml
from konductor.data import get_dataset_config
from konductor.init import ExperimentInitConfig, ModuleInitConfig
from konductor.metadata import DataManager, Statistic
from konductor.metadata.loggers import MultiWriter, ParquetLogger, TBLogger, WandBLogger
from konductor.trainer.pytorch import (
    PyTorchTrainer,
    PyTorchTrainerConfig,
    PyTorchTrainerModules,
)
from konductor.utilities import comm
from konductor.utilities.pbar import PbarType, pbar_wrapper
from sc2_serializer import set_replay_database_logger_level, spdlog_lvl
from torch import Tensor
from typing_extensions import Annotated
from src.data.base_dataset import SC2FolderCfg


class Trainer(PyTorchTrainer):
    """Specialize for prediciton"""

    min_index: int | None = None

    def data_transform(
        self, data: dict[str, Tensor] | list[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        if isinstance(data, list):
            data = data[0]  # Unwrap dali pipeline list, should already be on gpu
        elif torch.cuda.is_available():
            stream = torch.cuda.Stream()

            def filter(x: Tensor | list):
                return x.cuda(non_blocking=True) if isinstance(x, Tensor) else x

            with torch.cuda.stream(stream):
                data = {k: filter(d) for k, d in data.items()}

            stream.synchronize()

        if self.min_index is None:
            return data

        mask = data["valid"].sum(axis=1) > self.min_index

        return {k: d[mask, ...] for k, d in data.items()}


def get_statistics(exp_cfg: ExperimentInitConfig) -> dict[str, Statistic]:
    """Determine what statistics to track depending on losses used"""
    loss_types = [loss.type for loss in exp_cfg.criterion]
    stats: dict[str, Statistic] = {}
    if any(loss.startswith("win") for loss in loss_types):
        stats.update(
            {
                "win-auc": src.stats.WinAUC.from_config(exp_cfg),
                "binary-acc": src.stats.BinaryAcc.from_config(exp_cfg),
            }
        )

    if any(loss.startswith("minimap") for loss in loss_types):
        stats["minimap-soft-iou"] = src.stats.MinimapSoftIoU.from_config(exp_cfg)

    if len(stats) == 0:
        raise NotImplementedError(f"no statistics for {loss_types=}")

    return stats


def apply_dali_pipe_kwargs(
    exp_cfg: ExperimentInitConfig,
    py_workers: int,
    pipe_prefetch: int,
    source_prefetch: int,
):
    """Add dali specific pipeline args"""
    exp_cfg.data[0].train_loader.args["py_num_workers"] = py_workers
    exp_cfg.data[0].val_loader.args["py_num_workers"] = py_workers
    exp_cfg.data[0].train_loader.args["prefetch_queue_depth"] = pipe_prefetch
    exp_cfg.data[0].val_loader.args["prefetch_queue_depth"] = pipe_prefetch
    exp_cfg.data[0].dataset.args["prefetch_queue_depth"] = source_prefetch
    exp_cfg.data[0].dataset.args["prefetch_queue_depth"] = source_prefetch


app = typer.Typer()


@app.command()
def main(
    workspace: Annotated[Path, typer.Option()],
    epoch: Annotated[Optional[int], typer.Option()] = None,
    iteration: Annotated[Optional[int], typer.Option()] = None,
    remote: Annotated[Optional[Path], typer.Option()] = None,
    run_hash: Annotated[Optional[str], typer.Option()] = None,
    config_file: Annotated[Optional[Path], typer.Option()] = None,
    workers: Annotated[int, typer.Option()] = 4,
    dali_py_workers: Annotated[int, typer.Option()] = 2,
    dali_external_prefetch: Annotated[int, typer.Option()] = 2,
    dali_pipe_prefetch: Annotated[int, typer.Option()] = 2,
    pbar: Annotated[bool, typer.Option()] = False,
    brief: Annotated[Optional[str], typer.Option()] = None,
):
    """Run training"""
    if config_file is not None:
        assert run_hash is None, "config-file or run-hash should be exclusively set"
        exp_cfg = ExperimentInitConfig.from_config(workspace, config_file)
    elif run_hash is not None:
        exp_cfg = ExperimentInitConfig.from_run(workspace / run_hash)
    else:
        raise RuntimeError("Either config-file or run-hash should be set")
    exp_cfg.set_workers(workers)

    if exp_cfg.data[0].train_loader.type == "DALI":
        apply_dali_pipe_kwargs(
            exp_cfg, dali_py_workers, dali_pipe_prefetch, dali_external_prefetch
        )

    if remote is not None:
        with open(remote, "r", encoding="utf-8") as file:
            remote_cfg = yaml.safe_load(file)
        exp_cfg.remote_sync = ModuleInitConfig(**remote_cfg)

    train_modules = PyTorchTrainerModules.from_config(exp_cfg)

    wb_writer = []
    if "wandb" in exp_cfg.logger:
        import wandb

        wandb.init(**exp_cfg.logger.get("wandb", {}))
        wb_writer = [WandBLogger()]

    data_manager = DataManager.default_build(
        exp_cfg,
        train_modules.get_checkpointables(),
        get_statistics(exp_cfg),
        MultiWriter(
            [ParquetLogger(exp_cfg.exp_path), TBLogger(exp_cfg.exp_path)] + wb_writer
        ),
    )

    if brief is not None:
        data_manager.metadata.brief = brief

    trainer_cfg = PyTorchTrainerConfig(**exp_cfg.trainer)
    if pbar and comm.get_local_rank() == 0:
        trainer_cfg.pbar = partial(pbar_wrapper, pbar_type=PbarType.LIVE)
    elif comm.get_local_rank() == 0:
        trainer_cfg.pbar = partial(
            pbar_wrapper, pbar_type=PbarType.INTERVAL, fraction=0.1
        )

    trainer = Trainer(trainer_cfg, train_modules, data_manager)

    data_cfg = get_dataset_config(exp_cfg)
    if isinstance(data_cfg, SC2FolderCfg):
        shutil.copyfile(
            data_cfg.generation_config_path,
            exp_cfg.exp_path / data_cfg.generation_config_path.name,
        )
    trainer.min_index = data_cfg.properties.get("min_index", None)  # Set min index?

    trainer.train(epoch=epoch, iteration=iteration)
    # from konductor.trainer.profiler import profile_function

    # profile_function(trainer._train, Path.cwd())


if __name__ == "__main__":
    set_replay_database_logger_level(spdlog_lvl.warn)
    if torch.cuda.is_available():
        comm.initialize()
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(
        format=(
            f"%(asctime)s-RANK:{comm.get_local_rank()}-"
            "%(levelname)s-%(name)s: %(message)s"
        ),
        level=logging.INFO,
        force=True,
    )
    app()
