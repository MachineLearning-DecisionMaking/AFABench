from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


@dataclass
class SplitRatioConfig:
    train: float
    val: float
    test: float


@dataclass
class DatasetConfig:
    class_name: str
    kwargs: dict[str, Any]


@dataclass
class DatasetGenerationConfig:
    save_path: str
    instance_indices: list[int]
    seeds: list[int]
    split_ratio: SplitRatioConfig
    dataset: DatasetConfig
    fixed_test_seed: int | None = None
    stratify: bool = False
    max_train_size: int | None = None


cs.store(name="dataset_generation", node=DatasetGenerationConfig)
