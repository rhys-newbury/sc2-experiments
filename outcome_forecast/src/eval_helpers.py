"""Common helper functions for evaluation"""

from pathlib import Path

import torch
import pandas as pd
from torch import nn, Tensor
from konductor.data import Split, get_dataset_config
from konductor.init import ExperimentInitConfig
from konductor.models import get_model


def metadata_to_str(metadata: Tensor) -> list[str]:
    return ["".join(chr(x) for x in sublist) for sublist in metadata.cpu()]


def load_model_checkpoint(
    exp_config: ExperimentInitConfig, filename: str = "latest.pt"
):
    """Load model from checkpoint in experiment directory"""
    model: nn.Module = get_model(exp_config)
    ckpt = torch.load(exp_config.exp_path / filename)["model"]
    model.load_state_dict(ckpt)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


def get_dataloader_with_metadata(
    exp_config: ExperimentInitConfig, split: Split = Split.VAL
):
    """Get dataloader that also returns metadata (unique id associated with sample)"""
    dataset_cfg = get_dataset_config(exp_config)
    if hasattr(dataset_cfg, "keys"):
        if "metadata" not in dataset_cfg.keys:
            dataset_cfg.keys.append("metadata")
    else:
        dataset_cfg.metadata = True  # Need to add metadata list of keys to yield
    return dataset_cfg.get_dataloader(split)


def setup_eval_model_and_dataloader(
    run_path: Path,
    split: Split = Split.VAL,
    batch_size: int | None = None,
    workers: int | None = None,
):
    """Read experiment config from run path and create model and dataloader"""
    exp_config = ExperimentInitConfig.from_run(run_path)

    # AMP isn't enabled during eval
    if "amp" in exp_config.trainer:
        del exp_config.trainer["amp"]

    if batch_size is not None:
        exp_config.set_batch_size(batch_size, split)
    if workers is not None:
        exp_config.set_workers(workers)

    model = load_model_checkpoint(exp_config)
    dataloader = get_dataloader_with_metadata(exp_config, split)

    return exp_config, model, dataloader


def write_outcome_prediction(
    data: dict[str, Tensor], preds: Tensor, gidx: int, df: pd.DataFrame
):
    """Write the predicted outcome of the game over its duration"""
    replay_names = metadata_to_str(data["metadata"])
    for bidx in range(preds.shape[0]):
        df_idx = gidx + bidx
        if df_idx >= df.size:
            break

        row = df.iloc[gidx + bidx]
        row["replay"] = replay_names[bidx][:-1]
        row["playerId"] = int(replay_names[bidx][-1])
        row["outcome"] = bool(data["win"][bidx].item())
        for tidx in range(preds.shape[1]):
            if not data["valid"][bidx, tidx].item():
                continue
            col = df.columns[tidx + 3]
            row[col] = preds[bidx, tidx].sigmoid().item()
