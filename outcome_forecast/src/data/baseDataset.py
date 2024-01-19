from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from konductor.data import DatasetConfig, ExperimentInitConfig, Split
from konductor.data._pytorch.dataloader import DataloaderV1Config
from sc2_replay_reader import get_database_and_parser, Result
from torch.utils.data import Dataset

from .utils import find_closest_indicies
from abc import abstractmethod


@dataclass
class TimeRange:
    min: float
    max: float
    step: float

    def __post_init__(self):
        assert self.min < self.max

    def arange(self):
        return torch.arange(self.min, self.max, self.step)


class SC2ReplayBase(Dataset):
    def __init__(
        self,
        basepath: Path,
        split: Split,
        train_ratio: float,
        features: set[str] | None,
        timepoints: TimeRange,
        minimap_layers: list[str] | None = None,
        min_game_time: float | None = None,
    ) -> None:
        super().__init__()
        self.features = features
        if self.features is None or "unit_features" in self.features:
            self.db_handle, self.parser = get_database_and_parser(
                parse_units=True, parse_minimaps=True
            )
        elif "minimap_features" in self.features:
            self.db_handle, self.parser = get_database_and_parser(
                parse_units=False, parse_minimaps=True
            )
        else:
            self.db_handle, self.parser = get_database_and_parser(
                parse_units=False, parse_minimaps=False
            )

        if minimap_layers is not None:
            self.parser.setMinimapFeatures(minimap_layers)

        self.basepath = basepath
        self.split = split
        self.train_ratio = train_ratio
        self.n_replays = 0
        self.start_idx = 0

        self.load_files(basepath)

        _loop_per_min = 22.4 * 60
        self._target_game_loops = (timepoints.arange() * _loop_per_min).to(torch.int)

        if min_game_time is None:
            self.min_index = None
        else:
            difference_array = torch.absolute(timepoints.arange() - min_game_time)
            self.min_index = difference_array.argmin()

    @abstractmethod
    def load_files(self, basepath: Path):
        pass

    def __len__(self) -> int:
        assert hasattr(self, "n_replays"), "n_replays must be set before this"
        return self.n_replays

    def init_split_params(self, len_full_dataset: int):
        """Set n_replays and start_idx based on train or test|val"""
        assert hasattr(self, "n_replays"), "n_replays must be set before this"
        train_size = int(len_full_dataset * self.train_ratio)

        if self.split is not Split.TRAIN:
            self.start_idx = train_size
            self.n_replays = len_full_dataset - train_size
        else:
            self.start_idx = 0
            self.n_replays = train_size

        assert self.n_replays > 0, "No replays in dataset"

    def load_to_parser(self, path: Path, index: int):
        """Load replay data from database to parser"""
        self.db_handle.open(path)
        self.parser.parse_replay(self.db_handle.getEntry(index))

    def process_replay(self):
        """Process replay data currently in parser into dictonary of features and game outcome"""
        outputs_list = self.parser.sample(0)
        if self.features is not None:
            outputs_list = {k: [outputs_list[k]] for k in self.features}
        else:
            outputs_list = {k: [outputs_list[k]] for k in outputs_list}

        sample_indicies = find_closest_indicies(
            self.parser.data.gameStep, self._target_game_loops[1:]
        )
        for idx in sample_indicies:
            if idx == -1:
                sample = {k: np.zeros_like(outputs_list[k][-1]) for k in outputs_list}
            else:
                sample = self.parser.sample(int(idx.item()))
            for k in outputs_list:
                outputs_list[k].append(sample[k])

        outputs = {
            "win": torch.as_tensor(
                self.parser.info.playerResult == Result.Win, dtype=torch.float32
            ),
            "valid": torch.cat([torch.tensor([True]), sample_indicies != -1]),
        }
        for k in outputs_list:
            outputs[k] = torch.stack([torch.as_tensor(o) for o in outputs_list[k]])

        return outputs


@dataclass
class SC2ReplayConfigBase(DatasetConfig):
    # Dataloader type we want to use
    train_loader: DataloaderV1Config
    val_loader: DataloaderV1Config

    features: set[str] | None = None
    minimap_layers: list[str] | None = field(
        default_factory=lambda: ["player_relative", "visibility", "creep"]
    )
    train_ratio: float = 0.8  # Portion of all data to use for training
    timepoints: TimeRange = TimeRange(0, 30, 2)  # Minutes
    min_game_time: float = 5.0  # Minutes

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0):
        return super().from_config(config, idx)

    def __post_init__(self):
        assert 0 < self.train_ratio < 1, f"Failed: 0<{self.train_ratio=}<1"
        # If features is not None, ensure that it is a set
        if self.features is not None and not isinstance(self.features, set):
            self.features = set(self.features)
        if isinstance(self.timepoints, dict):
            self.timepoints = TimeRange(**self.timepoints)

    @property
    def properties(self) -> Dict[str, Any]:
        if self.minimap_layers is not None:
            image_ch = len(self.minimap_layers)
            if "player_relative" in self.minimap_layers:
                image_ch += 3
        else:
            image_ch = 10
        ret = {"image_ch": image_ch, "scalar_ch": 28}
        ret.update(self.__dict__)
        return ret

    def _known_unused(self):
        return {"train_loader", "val_loader", "basepath", "sql_filters"}

    @abstractmethod
    def get_class(self):
        pass

    def get_dataloader(self, split: Split) -> Any:
        known_unused = self._known_unused()
        dataset = self.init_auto_filter(
            self.get_class(), known_unused=known_unused, split=split
        )
        match split:
            case Split.TRAIN:
                return self.train_loader.get_instance(dataset)
            case Split.VAL | Split.TEST:
                return self.train_loader.get_instance(dataset)
            case _:
                raise RuntimeError(f"How did I get here with {split=}")
