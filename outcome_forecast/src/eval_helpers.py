"""Common helper functions for evaluation"""

import random
from pathlib import Path

import cv2
import torch
from torch import nn, Tensor
from konductor.data import Split, get_dataset_config
from konductor.init import ExperimentInitConfig
from konductor.models import get_model
from .utils import get_valid_sequence_mask, TimeRange


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


def create_score_frame(pred: Tensor, target: Tensor) -> Tensor:
    """Create an rgb frame showing the ground truth and predicted frames"""
    ctor_kwargs = {"device": pred.device, "dtype": torch.uint8}
    bgr_frame = torch.full([*pred.shape, 3], 255, **ctor_kwargs)

    # Red for false negatives
    bgr_frame[target == 1] = torch.tensor((0, 0, 200), **ctor_kwargs)

    mask = target == 0
    rg = (255 * pred).to(torch.uint8)[mask]
    b = torch.zeros_like(rg)
    # subtract rg from prediction
    bgr_frame[mask] -= torch.stack([b, rg, rg], dim=-1)

    # Blue for false positives
    bgr_frame[pred > 0.5] = torch.tensor((255, 0, 0), **ctor_kwargs)

    # Green for true positives
    mask = (pred > 0.5) & (target == 1)
    rb = ((1 - pred) * 255).to(torch.uint8)[mask]
    g = 200 * torch.ones_like(rb)
    bgr_frame[mask] = torch.stack([rb, g, rb], dim=-1)

    return bgr_frame


def write_minimaps(
    pred: Tensor, target: Tensor, timepoint: float, folder: Path, prefix: str
):
    """Write visualization results to disk"""
    predFolder = folder / "pred"
    dataFolder = folder / "data"

    for idx, name in enumerate(["self", "enemy"]):
        cv2.imwrite(
            str(predFolder / f"{prefix}_{timepoint}_{name}.png"),
            (255 * (1 - pred[idx])).to(torch.uint8).cpu().numpy(),
        )
        cv2.imwrite(
            str(dataFolder / f"{prefix}_{timepoint}_{name}.png"),
            (255 * (1 - target[idx])).to(torch.uint8).cpu().numpy(),
        )

        cv2.imwrite(
            str(predFolder / f"{prefix}_{timepoint}_diff_{name}.png"),
            create_score_frame(pred[idx], target[idx]).cpu().numpy(),
        )


def write_minimap_forecast_results(
    preds: Tensor,
    data: dict[str, Tensor],
    outdir: Path,
    timepoints: Tensor,
    n_time: int,
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

    metadata = metadata_to_str(data["metadata"])

    targets = data["minimap_features"][:, :, [-4, -1]]
    sequence_len = targets.shape[1] - preds.shape[1]
    valid_seq = get_valid_sequence_mask(data["valid"], sequence_len)
    pred_sig = preds.sigmoid()
    timepoints = timepoints.to(preds.device)

    for bidx in range(preds.shape[0]):
        prefix = metadata[bidx]
        valid_mask = valid_seq[bidx]
        valid_time = timepoints[sequence_len:][valid_mask]
        valid_pred = pred_sig[bidx][valid_mask]
        valid_tgt = targets[bidx, sequence_len:][valid_mask]

        # Only get a few random samples
        rand_idx = list(range(valid_mask.sum()))
        random.shuffle(rand_idx)
        rand_idx = rand_idx[:n_time]

        for idx in rand_idx:
            write_minimaps(
                valid_pred[idx], valid_tgt[idx], valid_time[idx].item(), outdir, prefix
            )

    with open(outdir / "samples.txt", "a") as f:
        f.writelines(m + "\n" for m in metadata)
