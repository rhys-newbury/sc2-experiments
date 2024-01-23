from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import torch
from konductor.data import (
    DATASET_REGISTRY,
    DatasetConfig,
    ExperimentInitConfig,
    ModuleInitConfig,
    Split,
)
from konductor.data._pytorch.dataloader import DataloaderV1Config
from sc2_replay_reader import Result, get_database_and_parser
from torch.utils.data import Dataset

from .replay_sampler import SAMPLER_REGISTRY, ReplaySampler
from .utils import find_closest_indicies
from ..utils import TimeRange


class SC2ReplayOutcome(Dataset):
    def __init__(
        self,
        sampler: ReplaySampler,
        timepoints: TimeRange,
        features: set[str] | None,
        minimap_layers: list[str] | None = None,
        min_game_time: float | None = None,
    ) -> None:
        super().__init__()
        self.sampler = sampler
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

        return outputs

    def __getitem__(self, index: int):
        replay_file, replay_idx = self.sampler.sample(index)
        self.db_handle.open(replay_file)
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

    def get_dataloader(self, split: Split) -> Any:
        known_unused = {"train_loader", "val_loader", "basepath"}
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
