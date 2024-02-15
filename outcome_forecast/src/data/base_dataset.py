import logging
import random
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from zipfile import BadZipFile

import numpy as np
import torch
import yaml
from konductor.data import (
    DATASET_REGISTRY,
    DatasetConfig,
    DatasetInitConfig,
    ExperimentInitConfig,
    ModuleInitConfig,
    Split,
    make_from_init_config,
)
from konductor.data._pytorch.dataloader import DataloaderV1Config
from konductor.data.dali import DaliLoaderConfig, DALI_AUGMENTATIONS
from nvidia.dali import fn
from nvidia.dali.data_node import DataNode
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.types import DALIDataType, SampleInfo, BatchInfo
from sc2_replay_reader import (
    ReplayDatabase,
    ReplayParser,
    Result,
    get_database_and_parser,
    set_replay_database_logger_level,
    spdlog_lvl,
)
from torch import Tensor
from torch.utils.data import Dataset

from ..utils import TimeRange
from .replay_sampler import SAMPLER_REGISTRY, ReplaySampler
from .utils import BaseDALIDataset, find_closest_indicies


def _min_to_game_step(t: float):
    """Convert time (min) to game step"""
    return int(t * 22.4 * 60)


class SC2ReplayBase(Dataset):
    def __init__(
        self,
        sampler: ReplaySampler,
        features: set[str] | None,
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
        features: set[str] | None,
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
        sample_indicies = find_closest_indicies(
            self.parser.data.gameStep, self._target_game_loops
        )

        if (sample_indicies == -1).all():
            print(f"No valid samples in {self.parser.info.replayHash}")

        outputs_list = {
            k: []
            for k in (test_sample.keys() if self.features is None else self.features)
        }

        for idx in sample_indicies:
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
            "valid": (sample_indicies != -1).to(torch.bool),
        }
        for k in outputs_list:
            outputs[k] = torch.stack([torch.as_tensor(o) for o in outputs_list[k]])

        return outputs


class SC2MinimapSequence(SC2ReplayBase):

    def __init__(
        self,
        sampler: ReplaySampler,
        timediff_sec: float,
        features: set[str] | None,
        minimap_layers: list[str] | None = None,
        metadata: bool = False,
    ):
        super().__init__(sampler, features, minimap_layers, metadata)
        self.timediff_sec = timediff_sec

    def process_replay(self):
        tgt_indicies: list[int] = [0]
        last_step = self.parser.data.gameStep[0]
        for idx, step in enumerate(self.parser.data.gameStep[1:], 1):
            last_sec = (step - last_step) * 22.4
            if last_sec > self.timediff_sec:
                prev_step = self.parser.data.gameStep[idx - 1]
                last_last_sec = (prev_step - last_step) * 22.4
                if last_sec < -last_last_sec:
                    tgt_indicies.append(idx)
                    last_step = step
                else:
                    tgt_indicies.append(idx - 1)
                    last_step = prev_step

        minimaps: list[Tensor] = []
        for idx in tgt_indicies:
            sample = self.parser.sample(idx)["minimap_features"]
            minimaps.append(torch.as_tensor(sample))
        return {"minimap_features": torch.stack(minimaps)}


@dataclass
class SC2ReplayBaseConfig(DatasetConfig):
    # Dataloader type we want to use
    train_loader: DataloaderV1Config
    val_loader: DataloaderV1Config

    sampler_cfg: ModuleInitConfig
    features: set[str] | None = None
    minimap_layers: list[str] | None = field(
        default_factory=lambda: ["player_relative", "visibility", "creep"]
    )
    train_ratio: float = 0.8  # Portion of all data to use for training
    metadata: bool = False  # Return replayHash-playerId data

    def __post_init__(self):
        assert 0 < self.train_ratio < 1, f"Failed: 0<{self.train_ratio=}<1"
        # If features is not None, ensure that it is a set
        if self.features is not None and not isinstance(self.features, set):
            self.features = set(self.features)
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

    def get_cls(self):
        raise NotImplementedError

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
            self.get_cls(), known_unused=known_unused, sampler=sampler
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
                return self.val_loader.get_instance(dataset)
            case _:
                raise RuntimeError(f"How did I get here with {split=}")


class DaliFolderDataset(BaseDALIDataset):
    def __init__(
        self,
        path: Path,
        split: Split,
        keys: list[str],
        batch_size: int,
        shard_id: int,
        num_shards: int,
        random_shuffle: bool,
    ) -> None:
        super().__init__(batch_size, shard_id, num_shards, random_shuffle)
        self.split = split
        self.keys = keys
        self.folder = path / split.name.lower()

        if not self.folder.exists():
            raise FileNotFoundError(self.folder)

        self.files: list[str] = []

    def _initialize(self):
        file_list = self.folder.parent / f"{self.split.lower()}-list.txt"
        with open(file_list, "r", encoding="utf-8") as f:
            self.files = [s.strip() for s in f.readlines()]
        return super()._initialize()

    def __len__(self) -> int:
        return len(self.files)

    def __call__(self, sample_info: SampleInfo):
        sample_idx = super().__call__(sample_info)

        data = np.load(self.folder / self.files[sample_idx], allow_pickle=True)

        try:
            out_data = tuple(
                (
                    np.array(
                        [ord(c) for c in data["metadata"].tolist()], dtype=np.uint8
                    )
                    if k == "metadata"
                    else data[k]
                )
                for k in self.keys
            )
        except BadZipFile as e:
            raise RuntimeError(f"Got bad data from {self.files[sample_idx]}") from e

        return out_data


@dataclass
@DATASET_REGISTRY.register_module("dali-folder")
class DaliFolderDatasetConfig(FolderDatasetConfig):
    train_loader: DaliLoaderConfig
    val_loader: DaliLoaderConfig
    keys: list[str]  # List of items to read
    fp16_out: bool = False

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0):
        if "amp" in config.trainer:
            config.data[idx].dataset.args["fp16_out"] = True
        return super().from_config(config, idx)

    def _make_source(self, split: Split) -> DaliFolderDataset:
        loader = self.train_loader if split is Split.TRAIN else self.val_loader
        source = self.init_auto_filter(
            DaliFolderDataset,
            path=self.basepath,
            split=split,
            **loader.pipe_kwargs(),
        )
        return source

    def _get_size(self, split: Split):
        inst = self._make_source(split)
        inst._initialize()
        return inst.num_iterations * inst.batch_size

    def get_dataloader(self, split: Split) -> Any:
        loader = self.train_loader if split is Split.TRAIN else self.val_loader
        pipeline = sc2_data_pipeline(
            source=self._make_source(split),
            keys=self.keys,
            fp16_out=self.fp16_out,
            **loader.pipe_kwargs(),
        )
        size = self._get_size(split)
        return loader.get_instance(pipeline, out_map=self.keys, size=size)


class DaliReplayClipDataset(BaseDALIDataset):
    def __init__(
        self,
        sampler_cfg: ModuleInitConfig,
        start_min: float,
        end_min: float,
        step_sec: float,
        clip_len: int,
        features: list[str],
        batch_size: int,
        shard_id: int,
        num_shards: int,
        random_shuffle: bool,
        yields_batch: bool = False,
        metadata: bool = False,
        minimap_layers: list[str] | None = None,
    ) -> None:
        super().__init__(batch_size, shard_id, num_shards, random_shuffle)
        self.sampler_cfg = sampler_cfg
        self.sampler: ReplaySampler | None = None
        self.start_step = _min_to_game_step(start_min)
        self.end_step = _min_to_game_step(end_min)
        self.step_size = int(step_sec * 22.4)
        self.clip_len = clip_len + 1  # Need to yield frame after history
        self.metadata = metadata
        self.features = features

        self.db_handle: ReplayDatabase | None = None
        self.parser: ReplayParser | None = None
        self.minimap_layers = minimap_layers
        self.yields_batch = yields_batch

    def _initialize(self):
        self.sampler = SAMPLER_REGISTRY[self.sampler_cfg.type](**self.sampler_cfg.args)
        self.db_handle, self.parser = get_database_and_parser(
            parse_units="unit_features" in self.features,
            parse_minimaps="minimap_features" in self.features,
        )
        if self.minimap_layers is not None:
            self.parser.setMinimapFeatures(self.minimap_layers)
        set_replay_database_logger_level(spdlog_lvl.warn)
        return super()._initialize()

    def __len__(self) -> int:
        assert self.sampler is not None, "Must be _initialized()"
        return len(self.sampler)

    def sample_start_time(self, game_steps: list[int]) -> int:
        """Sample a start time that is within self.timepoints and the current replay"""
        start = max(self.start_step, game_steps[0])
        end = min(self.end_step, game_steps[-1] - self.clip_len)
        return random.randint(start, end)

    def process_replay(self):
        assert self.parser is not None
        try:
            test_sample: dict[str, Any] = self.parser.sample(0)
        except (RuntimeError, IndexError) as err:
            raise RuntimeError(f"Parse failure for {self.parser.info}") from err

        outputs_list = {
            k: []
            for k in (test_sample.keys() if self.features is None else self.features)
        }

        sample_indicies = torch.full([self.clip_len], -1, dtype=torch.int32)
        attempts = 0
        while (sample_indicies == -1).any():
            start_idx = self.sample_start_time(self.parser.data.gameStep)
            end_idx = start_idx + self.clip_len * self.step_size
            sample_indicies = find_closest_indicies(
                self.parser.data.gameStep,
                range(start_idx, end_idx, self.step_size),
            )
            attempts += 1
            if attempts > 50:
                raise RuntimeError("Maximum iteration attempt exceeded")

        for idx in sample_indicies:
            sample = self.parser.sample(int(idx.item()))
            for k in outputs_list:
                outputs_list[k].append(sample[k])

        outputs = [np.stack(outputs_list[k]) for k in outputs_list]

        if self.metadata:
            _str = self.parser.info.replayHash + str(self.parser.info.playerId)
            outputs.append(np.array([ord(c) for c in _str], dtype=np.uint8))

        return outputs

    def __call__(self, sample_info: SampleInfo | BatchInfo):
        assert self.sampler is not None
        assert self.db_handle is not None
        assert self.parser is not None
        # LSP doesn't recognise 'assert all(m is not None for m in ...)'
        sample_idx = super().__call__(sample_info)
        replay_file, replay_idx = self.sampler.sample(sample_idx)
        if not self.db_handle.load(replay_file):
            raise FileNotFoundError(replay_file)
        replay_data = self.db_handle.getEntry(replay_idx)
        self.parser.parse_replay(replay_data)
        if self.yields_batch:
            samples = [self.process_replay() for _ in range(self.batch_size)]
            outputs = []
            for idx in range(len(samples[0])):
                outputs.append(np.stack([s[idx] for s in samples]))
        else:
            outputs = self.process_replay()
        return outputs


@dataclass
@DATASET_REGISTRY.register_module("dali-replay-clip")
class DaliReplayClipConfig(DatasetConfig):
    train_loader: DaliLoaderConfig
    val_loader: DaliLoaderConfig
    sampler_cfg: ModuleInitConfig
    start_min: float
    end_min: float
    step_sec: float
    clip_len: int
    features: list[str] = field(
        default_factory=lambda: ["scalar_features", "minimap_features"]
    )
    minimap_layers: list[str] | None = field(
        default_factory=lambda: ["player_relative", "visibility", "creep"]
    )
    train_ratio: float = 0.8  # Portion of all data to use for training
    fp16_out: bool = False
    metadata: bool = False
    yields_batch: bool = False

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0):
        if "amp" in config.trainer:
            config.data[idx].dataset.args["fp16_out"] = True
        return super().from_config(config, idx)

    def __post_init__(self):
        assert len(self.features) == len(
            set(self.features)
        ), f"Duplicate keys in features: {self.features}"
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

    def _make_source(self, split: Split) -> DaliReplayClipDataset:
        loader = self.train_loader if split is Split.TRAIN else self.val_loader
        sampler_cfg = deepcopy(self.sampler_cfg)
        sampler_cfg.args["split"] = split
        sampler_cfg.args["train_ratio"] = self.train_ratio
        sampler_cfg.args["replays_path"] = self.basepath
        source = self.init_auto_filter(
            DaliReplayClipDataset, sampler_cfg=sampler_cfg, **loader.pipe_kwargs()
        )
        return source

    def _get_size(self, split: Split):
        inst = self._make_source(split)
        inst._initialize()
        return inst.num_iterations * inst.batch_size

    def get_dataloader(self, split: Split) -> Any:
        loader = self.train_loader if split is Split.TRAIN else self.val_loader
        pipeline = sc2_data_pipeline(
            source=self._make_source(split),
            keys=self.features,
            fp16_out=self.fp16_out,
            **loader.pipe_kwargs(),
        )
        size = self._get_size(split)
        out_map = deepcopy(self.features)
        if self.metadata:
            out_map.append("metadata")
        return loader.get_instance(pipeline, out_map=out_map, size=size)


@dataclass
class FeatureType:
    dtype: DALIDataType
    ndim: int
    layout: str
    should_cast: bool


_DTYPES = {
    "win": FeatureType(DALIDataType.FLOAT, 0, "", False),
    "valid": FeatureType(DALIDataType.BOOL, 1, "", False),
    "metadata": FeatureType(DALIDataType.UINT8, 1, "", False),
    "scalar_features": FeatureType(DALIDataType.FLOAT, 2, "", False),
    "minimap_features": FeatureType(DALIDataType.FLOAT, 4, "FCHW", True),
}


def apply_minimap_augs(minimaps: DataNode, augs: list[ModuleInitConfig]):
    """Apply list of augmentations to minimaps"""
    for aug in augs:
        minimaps = DALI_AUGMENTATIONS[aug.type](minimaps, **aug.args)
    return minimaps


@pipeline_def(
    py_start_method="spawn",
    prefetch_queue_depth=2,
    enable_conditionals=True,
    py_num_workers=4,
)
def sc2_data_pipeline(
    shard_id: int,
    num_shards: int,
    random_shuffle: bool,
    source: BaseDALIDataset,
    keys: list[str],
    fp16_out: bool,
    augmentations: list[ModuleInitConfig],
):
    """Create pipeline"""
    outputs = fn.external_source(
        source=source,
        num_outputs=len(keys),
        parallel=True,
        batch=source.yields_batch,
        batch_info=source.yields_batch,
        dtype=[_DTYPES[k].dtype for k in keys],
        ndim=[_DTYPES[k].ndim for k in keys],
        layout=[_DTYPES[k].layout for k in keys],
        prefetch_queue_depth=3,
    )

    def transform(data: DataNode, key: str):
        """Move data to gpu and cast to fp16 if enabled"""
        data = data.gpu()
        if (
            _DTYPES[key].dtype == DALIDataType.FLOAT
            and _DTYPES[key].should_cast
            and fp16_out
        ):
            return fn.cast(data, dtype=DALIDataType.FLOAT16)
        if key == "metadata":
            return fn.pad(data, fill_value=0)
        return data

    if len(augmentations) != 0:
        minimap_idx = keys.index("minimap_features")
        outputs[minimap_idx] = apply_minimap_augs(
            outputs[minimap_idx].gpu(), augmentations
        )

    outputs = [transform(o, k) for o, k in zip(outputs, keys)]

    return tuple(outputs)
