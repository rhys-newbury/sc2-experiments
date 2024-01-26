from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
import torch
import yaml
from konductor.data import (
    DATASET_REGISTRY,
    DatasetConfig,
    ExperimentInitConfig,
    ModuleInitConfig,
    Split,
    DatasetInitConfig,
    make_from_init_config,
)
from konductor.data._pytorch.dataloader import DataloaderV1Config
from sc2_replay_reader import Result, get_database_and_parser
from torch.utils.data import Dataset

from ..utils import TimeRange
from .replay_sampler import SAMPLER_REGISTRY, ReplaySampler
from .utils import find_closest_indicies


class SC2ReplayOutcome(Dataset):
    def __init__(
        self,
        sampler: ReplaySampler,
        timepoints: TimeRange,
        features: set[str] | None,
        metadata: bool = False,
        minimap_layers: list[str] | None = None,
        min_game_time: float | None = None,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.features = features
        self.metadata = metadata
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

        _loop_per_min = 22.4 * 60
        self._target_game_loops = (timepoints.arange() * _loop_per_min).to(torch.int)

        if min_game_time is None:
            self.min_index = None
        else:
            difference_array = torch.absolute(timepoints.arange() - min_game_time)
            self.min_index = difference_array.argmin()

    def __len__(self) -> int:
        return len(self.sampler)

    def process_replay(self):
        """Process replay data currently in parser into dictonary of features and game outcome"""
        try:
            outputs_list = self.parser.sample(0)
        except (RuntimeError, IndexError) as err:
            raise RuntimeError(f"Parse failure for {self.parser.info}") from err

        # Find the indicies to sample at based on recorded gamesteps
        sample_indicies = find_closest_indicies(
            self.parser.data.gameStep, self._target_game_loops
        )

        # Determine the features available my running the parser at the first index
        outputs_list = self.parser.sample(int(sample_indicies[0].item()))

        if self.features is not None:
            outputs_list = {k: [outputs_list[k]] for k in self.features}
        else:
            outputs_list = {k: [outputs_list[k]] for k in outputs_list}

        for idx in sample_indicies[1:]:
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
            "valid": (sample_indicies != -1).to(torch.bool),
        }
        for k in outputs_list:
            outputs[k] = torch.stack([torch.as_tensor(o) for o in outputs_list[k]])

        if self.metadata:
            outputs["metadata"] = self.parser.info.replayHash + str(
                self.parser.info.playerId
            )

        return outputs

    def __getitem__(self, index: int):
        replay_file, replay_idx = self.sampler.sample(index)
        if not self.db_handle.load(replay_file):
            raise FileNotFoundError(replay_file)
        replay_data = self.db_handle.getEntry(replay_idx)
        self.parser.parse_replay(replay_data)
        outputs = self.process_replay()
        return outputs


@dataclass
@DATASET_REGISTRY.register_module("sc2-replay-outcome")
class SC2ReplayConfig(DatasetConfig):
    # Dataloader type we want to use
    train_loader: DataloaderV1Config
    val_loader: DataloaderV1Config

    sampler_cfg: ModuleInitConfig
    features: set[str] | None = None
    minimap_layers: list[str] | None = field(
        default_factory=lambda: ["player_relative", "visibility", "creep"]
    )
    train_ratio: float = 0.8  # Portion of all data to use for training
    timepoints: TimeRange = TimeRange(0, 30, 2)  # Minutes
    min_game_time: float = 5.0  # Minutes
    metadata: bool = False  # Return replayHash-playerId data

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
        if not isinstance(self.sampler_cfg, ModuleInitConfig):
            self.sampler_cfg = ModuleInitConfig(**self.sampler_cfg)

    @property
    def properties(self) -> dict[str, Any]:
        if self.minimap_layers is not None:
            image_ch = len(self.minimap_layers)
            if "player_relative" in self.minimap_layers:
                image_ch += 3
        else:
            image_ch = 10
        ret = {"image_ch": image_ch, "scalar_ch": 28}
        ret.update(self.__dict__)
        return ret

    def get_dataloader(self, split: Split) -> Any:
        known_unused = {
            "train_loader",
            "val_loader",
            "basepath",
            "sampler_cfg",
            "train_ratio",
        }
        sampler = SAMPLER_REGISTRY[self.sampler_cfg.type](
            split=split,
            train_ratio=self.train_ratio,
            replays_path=self.basepath,
            **self.sampler_cfg.args,
        )
        dataset = self.init_auto_filter(
            SC2ReplayOutcome, known_unused=known_unused, sampler=sampler
        )
        match split:
            case Split.TRAIN:
                return self.train_loader.get_instance(dataset)
            case Split.VAL | Split.TEST:
                return self.train_loader.get_instance(dataset)
            case _:
                raise RuntimeError(f"How did I get here with {split=}")


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
class FolderDatasetConfig(DatasetConfig):
    # Dataloader type we want to use
    train_loader: DataloaderV1Config
    val_loader: DataloaderV1Config

    _gen_properties: dict[str, Any] = field(init=False)

    def __post_init__(self):
        # Get the original dataset properties
        with open(self.generation_config_path, "r", encoding="utf-8") as f:
            gen_config = yaml.safe_load(f)
        init_config = DatasetInitConfig.from_dict(gen_config)
        dataset_cfg = make_from_init_config(init_config)
        self._gen_properties = dataset_cfg.properties

    @property
    def generation_config_path(self):
        return self.basepath / "generation-config.yml"

    @property
    def properties(self) -> dict[str, Any]:
        """Get properties from original generated version"""
        return self._gen_properties

    def get_dataloader(self, split: Split) -> Any:
        dataset = FolderDataset(self.basepath, split)
        match split:
            case Split.TRAIN:
                return self.train_loader.get_instance(dataset)
            case Split.VAL | Split.TEST:
                return self.train_loader.get_instance(dataset)
            case _:
                raise RuntimeError(f"How did I get here with {split=}")
