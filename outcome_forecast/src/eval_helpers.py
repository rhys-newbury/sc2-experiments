"""Common helper functions for evaluation"""

import math
import random
from pathlib import Path

import cv2
import torch
import pandas as pd
from torch import nn, Tensor
from torch.nn import functional as F
from konductor.data import Split, get_dataset_config
from konductor.init import ExperimentInitConfig
from konductor.models import get_model
from .utils import get_valid_sequence_mask


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


def write_minimaps(pred: Tensor, target: Tensor, folder: Path, prefix: str):
    """Write visualization results to disk"""
    predFolder = folder / "pred"
    dataFolder = folder / "data"

    for idx, name in enumerate(["self", "enemy"]):
        cv2.imwrite(
            str(predFolder / f"{prefix}_{name}.png"),
            (255 * (1 - pred[idx])).to(torch.uint8).cpu().numpy(),
        )
        cv2.imwrite(
            str(dataFolder / f"{prefix}_{name}.png"),
            (255 * (1 - target[idx])).to(torch.uint8).cpu().numpy(),
        )

        cv2.imwrite(
            str(predFolder / f"{prefix}_diff_{name}.png"),
            create_score_frame(pred[idx], target[idx]).cpu().numpy(),
        )


def get_upper_left_coord(idx: int, grid_size: int, tile_size: int):
    """
    Get pixel coordinates of upper left of tile at index
    Returns (h,w)
    """
    row, col = divmod(idx, grid_size)
    assert row <= grid_size and col <= grid_size, f"Tile {idx=} out of bounds"
    return row * tile_size, col * tile_size


def write_tiled_sequence(
    data: Tensor, end_idx: int, seq_len: int, folder: Path, prefix: str
):
    """Write the historicalsequence and target frame"""
    dataFolder = folder / "data"
    im_size = 1024
    grid_size = int(math.ceil(math.sqrt(seq_len)))
    tile_size = im_size // grid_size
    data = (255 * (1 - data)).to(torch.uint8)
    for ch_idx, name in enumerate(["self", "enemy"]):
        base_image = torch.full(
            [im_size, im_size], 255, dtype=torch.uint8, device=data.device
        )
        for idx, t_idx in enumerate(range(end_idx + 1 - seq_len, end_idx + 1)):
            data_time: Tensor = F.interpolate(
                data[t_idx, ch_idx, None, None],
                size=(tile_size, tile_size),
                mode="nearest",
            )
            px_y, px_x = get_upper_left_coord(idx, grid_size, tile_size)
            base_image[px_y : px_y + tile_size, px_x : px_x + tile_size] = data_time[
                0, 0
            ]
        cv2.imwrite(
            str(dataFolder / f"{prefix}_{name}_seq.png"),
            base_image.cpu().numpy(),
        )


def write_gradient_sequence(
    data: Tensor, end_idx: int, seq_len: int, folder: Path, prefix: str
):
    """Display a sequence of frames as a ghost trail to indicate motion of units over time"""
    px_vals = torch.linspace(200, 0, seq_len, dtype=torch.uint8, device=data.device)
    dataFolder = folder / "data"
    data = data.to(torch.bool)
    for ch_idx, name in enumerate(["self", "enemy"]):
        base_image = torch.full(
            data.shape[-2:], 255, dtype=torch.uint8, device=data.device
        )
        for px_val, t_idx in zip(px_vals, range(end_idx - seq_len + 1, end_idx + 1)):
            base_image[data[t_idx, ch_idx]] = px_val
        cv2.imwrite(
            str(dataFolder / f"{prefix}_{name}_seq.png"), base_image.cpu().numpy()
        )


def write_minimap_forecast_results(
    preds: Tensor,
    data: dict[str, Tensor],
    outdir: Path,
    timepoints: list[float] | None,
    n_time: int,
):
    """
    Write minimap forecast predictions for konduct review image viewer.
    The history sequence length and thus association between prediction and
    ground truth time indices is inferred from the time dimension of the
    prediction which is full_sequence - history_length.
    """
    predFolder = outdir / "pred"
    predFolder.mkdir(exist_ok=True)
    dataFolder = outdir / "data"
    dataFolder.mkdir(exist_ok=True)

    metadata = metadata_to_str(data["metadata"])
    if all(m == metadata[0] for m in metadata):  # If batch from  same replay append idx
        metadata = [m + str(i) for i, m in enumerate(metadata)]
    pred_sig = preds.sigmoid()

    targets = data["minimap_features"][:, :, [-4, -1]]
    sequence_len = targets.shape[1] - preds.shape[1]
    if "valid" in data:
        valid_seq = get_valid_sequence_mask(data["valid"], sequence_len + 1)
    else:
        valid_seq = torch.ones(targets.shape[0], 1, dtype=torch.bool)
    indices = torch.arange(valid_seq.shape[1], device=valid_seq.device) + sequence_len

    for bidx in range(preds.shape[0]):
        prefix = metadata[bidx]
        # Only get a few random samples
        rand_idxs: list[int] = [i.item() for i in indices[valid_seq[bidx]].cpu()]
        random.shuffle(rand_idxs)
        rand_idxs = rand_idxs[:n_time]

        for idx in rand_idxs:
            prefix_ = prefix
            if timepoints is not None:
                prefix_ += f"_{timepoints[idx]}"

            write_minimaps(
                pred_sig[bidx, idx - sequence_len], targets[bidx, idx], outdir, prefix
            )
            write_gradient_sequence(
                targets[bidx], idx, sequence_len + 1, outdir, prefix
            )

    with open(outdir / "samples.txt", "a") as f:
        f.writelines(m + "\n" for m in metadata)


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
        row["replay"] = replay_names[bidx]
        row["outcome"] = bool(data["win"][bidx].item())
        for tidx in range(preds.shape[1]):
            if not data["valid"][bidx, tidx].item():
                continue
            col = df.columns[tidx + 2]
            row[col] = preds[bidx, tidx].sigmoid().item()
