"""Common helper functions for evaluation"""

from pathlib import Path

import torch
from torch import nn, Tensor
from konductor.data import Split, get_dataset_config
from konductor.init import ExperimentInitConfig
from konductor.models import get_model
from ..src.utils import get_valid_sequence_mask


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
    # Need to add metadata list of keys to yield
    yield_keys: list[str] = exp_config.data[0].dataset.args["keys"]
    if "metadata" not in yield_keys:
        yield_keys.append("metadata")
    dataloader = get_dataset_config(exp_config).get_dataloader(split)
    return dataloader


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
    dataset = get_dataset_config(exp_config)
    dataloader = dataset.get_dataloader(split)

    return exp_config, model, dataloader


def write_minimap_forecast_results(
    preds: Tensor, data: dict[str, Tensor], outdir: Path
):
    """
    Write minimap forecast predictions for konduct review image viewer.
    The history sequence length and thus association between prediction and
    ground truth time indicies is infered from the time dimension of the
    prediction which is full_sequence - history_length.
    """
    predFolder = outdir / "pred"
    predFolder.mkdir(exist_ok=True)
    dataFolder = outdir / "data"
    dataFolder.mkdir(exist_ok=True)

    targets = data["minimap_features"][:, :, [-4, -1]]

    metadata = metadata_to_str(data["metadata"])

    sequence_len = preds.shape[1] - targets.shape[1]
    valid_seq = get_valid_sequence_mask(data["valid"], sequence_len)

    for bidx in range(preds.shape[0]):
        prefix = metadata[bidx]
