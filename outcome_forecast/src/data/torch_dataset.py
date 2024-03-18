import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
import torch
from konductor.data import DATASET_REGISTRY, Split
from konductor.data._pytorch.dataloader import DataloaderV1Config
from sc2_replay_reader import Result, get_database_and_parser
from torch import Tensor
from torch.utils.data import Dataset

from ..utils import TimeRange
from .base_dataset import SC2FolderCfg, SC2SamplerCfg
from .replay_sampler import SAMPLER_REGISTRY, ReplaySampler
from .utils import find_closest_indices


class SC2ReplayBase(Dataset):
    def __init__(
        self,
        sampler: ReplaySampler,
        features: list[str],
        minimap_layers: list[str] | None = None,
        metadata: bool = False,
    ):
        super().__init__()
        self.sampler = sampler
        self.features = features
        self.metadata = metadata
        self.logger = logging.getLogger("replay-dataset")
        if features is not None:
            self.db_handle, self.parser = get_database_and_parser(
                parse_units="unit_features" in features,
                parse_minimaps="minimap_features" in features,
            )
        else:
            self.db_handle, self.parser = get_database_and_parser(
                parse_units=True, parse_minimaps=True
            )

        if minimap_layers is not None:
            self.parser.setMinimapFeatures(minimap_layers)

    def __len__(self) -> int:
        return len(self.sampler)

    def process_replay(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        replay_file, replay_idx = self.sampler.sample(index)
        if not self.db_handle.load(replay_file):
            raise FileNotFoundError(replay_file)
        replay_data = self.db_handle.getEntry(replay_idx)
        self.parser.parse_replay(replay_data)
        outputs = self.process_replay()

        if self.metadata:
            outputs["metadata"] = self.parser.info.replayHash + str(
                self.parser.info.playerId
            )

        return outputs


class SC2ReplayOutcome(SC2ReplayBase):
    def __init__(
        self,
        sampler: ReplaySampler,
        timepoints: TimeRange,
        features: list[str],
        metadata: bool = False,
        minimap_layers: list[str] | None = None,
        min_game_time: float | None = None,
    ) -> None:
        super().__init__(sampler, features, minimap_layers, metadata)

        _loop_per_min = 22.4 * 60
        self._target_game_loops = (timepoints.arange() * _loop_per_min).to(torch.int)

        if min_game_time is None:
            self.min_index = None
        else:
            difference_array = torch.absolute(timepoints.arange() - min_game_time)
            self.min_index = difference_array.argmin()

    def process_replay(self):
        """Process replay data currently in parser into dictionary of
        features and game outcome"""
        try:
            test_sample: dict[str, Any] = self.parser.sample(0)
        except (RuntimeError, IndexError) as err:
            raise RuntimeError(f"Parse failure for {self.parser.info}") from err

        # Find the indices to sample at based on recorded gamesteps
        sample_indices = find_closest_indices(
            self.parser.data.gameStep, self._target_game_loops
        )

        if (sample_indices == -1).all():
            print(f"No valid samples in {self.parser.info.replayHash}")

        outputs_list = {
            k: []
            for k in (test_sample.keys() if self.features is None else self.features)
        }

        for idx in sample_indices:
            if idx == -1:
                sample = {k: np.zeros_like(test_sample[k]) for k in outputs_list}
            else:
                sample = self.parser.sample(int(idx.item()))
            for k in outputs_list:
                outputs_list[k].append(sample[k])

        outputs = {
            "win": torch.as_tensor(
                self.parser.info.playerResult == Result.Win, dtype=torch.float32
            ),
            "valid": (sample_indices != -1).to(torch.bool),
        }
        for k in outputs_list:
            outputs[k] = torch.stack([torch.as_tensor(o) for o in outputs_list[k]])

        return outputs


class SC2MinimapSequence(SC2ReplayBase):
    def __init__(
        self,
        sampler: ReplaySampler,
        timediff_sec: float,
        features: list[str],
        minimap_layers: list[str] | None = None,
        metadata: bool = False,
    ):
        super().__init__(sampler, features, minimap_layers, metadata)
        self.timediff_sec = timediff_sec

    def process_replay(self):
        tgt_indices: list[int] = [0]
        last_step = self.parser.data.gameStep[0]
        for idx, step in enumerate(self.parser.data.gameStep[1:], 1):
            last_sec = (step - last_step) * 22.4
            if last_sec > self.timediff_sec:
                prev_step = self.parser.data.gameStep[idx - 1]
                last_last_sec = (prev_step - last_step) * 22.4
                if last_sec < -last_last_sec:
                    tgt_indices.append(idx)
                    last_step = step
                else:
                    tgt_indices.append(idx - 1)
                    last_step = prev_step

        minimaps: list[Tensor] = []
        for idx in tgt_indices:
            sample = self.parser.sample(idx)["minimap_features"]
            minimaps.append(torch.as_tensor(sample))
        return {"minimap_features": torch.stack(minimaps)}


@dataclass
class SC2ReplayBaseConfig(SC2SamplerCfg):
    # Dataloader type we want to use
    train_loader: DataloaderV1Config
    val_loader: DataloaderV1Config

    def get_cls(self):
        raise NotImplementedError

    def get_dataloader(self, split: Split) -> Any:

        sampler = SAMPLER_REGISTRY[self.sampler_cfg.type](
            split=split,
            train_ratio=self.train_ratio,
            replays_path=self.basepath,
            **self.sampler_cfg.args,
        )
        dataset = self.init_auto_filter(
            self.get_cls(), known_unused=SC2SamplerCfg._known_unused, sampler=sampler
        )
        match split:
            case Split.TRAIN:
                return self.train_loader.get_instance(dataset)
            case Split.VAL | Split.TEST:
                return self.val_loader.get_instance(dataset)
            case _:
                raise RuntimeError(f"How did I get here with {split=}")


@dataclass
@DATASET_REGISTRY.register_module("sc2-replay-outcome")
class SC2ReplayConfig(SC2ReplayBaseConfig):
    timepoints: TimeRange = TimeRange(0, 30, 2)  # Minutes
    min_game_time: float = 5.0  # Minutes

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.timepoints, dict):
            self.timepoints = TimeRange(**self.timepoints)

    def get_cls(self):
        return SC2ReplayOutcome


@dataclass
@DATASET_REGISTRY.register_module("minimap-sequence")
class MinimapSequence(SC2ReplayBaseConfig):
    timediff_sec: float = field(kw_only=True)

    def get_cls(self):
        return SC2MinimapSequence


class FolderDataset(Dataset):
    """
    Basic folder dataset (basepath/split/sample.npz) which contains numpy
    files that contain the expected format to train on directly
    """

    def __init__(self, basepath: Path, split: Split) -> None:
        super().__init__()
        self.folder = basepath / split.name.lower()
        assert self.folder.exists(), f"Root folder doesn't exist: {basepath}"
        self.files = [f.name for f in self.folder.iterdir()]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        data = np.load(self.folder / self.files[index], allow_pickle=True)

        def transform(x: np.ndarray):
            return str(x) if "str" in x.dtype.name else torch.tensor(x)

        try:
            out_data = {k: transform(v) for k, v in data.items()}
        except BadZipFile as e:
            raise RuntimeError(f"Got bad data from {self.files[index]}") from e

        return out_data


@dataclass
@DATASET_REGISTRY.register_module("folder-dataset")
class FolderDatasetConfig(SC2FolderCfg):
    # Dataloader type we want to use
    train_loader: DataloaderV1Config
    val_loader: DataloaderV1Config

    def get_dataloader(self, split: Split) -> Any:
        dataset = FolderDataset(self.basepath, split)
        match split:
            case Split.TRAIN:
                return self.train_loader.get_instance(dataset)
            case Split.VAL | Split.TEST:
                return self.val_loader.get_instance(dataset)
            case _:
                raise RuntimeError(f"How did I get here with {split=}")
