from dataclasses import dataclass, field
from typing import Any

import yaml
from konductor.data import (
    DatasetConfig,
    DatasetInitConfig,
    ModuleInitConfig,
    make_from_init_config,
)
from sc2_replay_reader import get_database_and_parser


@dataclass
class SC2DatasetCfg(DatasetConfig):
    """Basic Dataset Configuration"""

    features: list[str] = field(
        default_factory=lambda: ["scalar_features", "minimap_features"]
    )
    minimap_layers: list[str] | None = field(
        default_factory=lambda: ["player_relative", "visibility", "creep"]
    )
    metadata: bool = False  # Return replayHash-playerId data
    minimap_ch_names: list[str] | None = field(init=False)

    def __post_init__(self):
        assert len(self.features) == len(
            set(self.features)
        ), f"Duplicate keys in features: {self.features}"

        if "minimap_features" in self.features:
            _, parser = get_database_and_parser(
                parse_units="unit_features" in self.features,
                parse_minimaps="minimap_features" in self.features,
            )
            if self.minimap_layers is not None:
                parser.setMinimapFeatures(self.minimap_layers)
            self.minimap_ch_names = parser.getMinimapFeatures()

    @property
    def properties(self) -> dict[str, Any]:
        ret = {"scalar_ch": 28}
        if "minimap_features" in self.features:
            assert self.minimap_ch_names is not None
            ret["image_ch"] = len(self.minimap_ch_names)
        ret.update(self.__dict__)
        return ret


@dataclass
class SC2SamplerCfg(SC2DatasetCfg):
    """Dataset that uses Sampler Method"""

    sampler_cfg: ModuleInitConfig = field(kw_only=True)
    train_ratio: float = 0.8  # Portion of all data to use for training

    _known_unused = {
        "train_loader",
        "val_loader",
        "basepath",
        "sampler_cfg",
        "train_ratio",
    }

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.train_ratio < 1, f"Failed: 0<{self.train_ratio=}<1"
        if not isinstance(self.sampler_cfg, ModuleInitConfig):
            self.sampler_cfg = ModuleInitConfig(**self.sampler_cfg)


@dataclass
class SC2FolderCfg(SC2DatasetCfg):
    """Dataset that consists of preprocessed numpy files of data"""

    _gen_properties: dict[str, Any] = field(init=False)

    @property
    def generation_config_path(self):
        return self.basepath / "generation-config.yml"

    def __post_init__(self):
        super().__post_init__()
        # Get the original dataset properties
        with open(self.generation_config_path, "r", encoding="utf-8") as f:
            gen_config = yaml.safe_load(f)
        init_config = DatasetInitConfig.from_dict(gen_config)
        dataset_cfg = make_from_init_config(init_config)
        self._gen_properties = dataset_cfg.properties

    @property
    def properties(self) -> dict[str, Any]:
        """Get properties from original generated version"""
        return self._gen_properties
