from typing import TYPE_CHECKING, cast

import torch

from afabench.core.bundle_system.bundle import load_bundle
from afabench.datasets.config import SplitRatioConfig
from afabench.datasets.datasets import CubeDataset
from scripts.dataset_generation.generate_dataset import generate_and_save_split

if TYPE_CHECKING:
    from pathlib import Path

    from afabench.core.types import AFADataset


def _features(path: "Path", split: str) -> torch.Tensor:
    dataset, _ = load_bundle(path / f"{split}.bundle")
    return cast("AFADataset", cast("object", dataset)).get_all_data()[0]


def _generate(path: "Path", max_train_size: int | None) -> None:
    generate_and_save_split(
        dataset_class=CubeDataset,
        split_ratio=SplitRatioConfig(train=0.6, val=0.2, test=0.2),
        seed_for_split=3,
        save_path=path,
        dataset_kwargs={"n_samples": 200, "seed": 0},
        metadata_to_save={},
        max_train_size=max_train_size,
    )


def test_max_train_size_subsamples_only_the_training_split(
    tmp_path: "Path",
) -> None:
    _generate(tmp_path / "full", None)
    _generate(tmp_path / "capped", 50)

    full_train = _features(tmp_path / "full", "train")
    capped_train = _features(tmp_path / "capped", "train")

    assert len(full_train) == 120
    assert len(capped_train) == 50
    for row in capped_train:
        assert (full_train == row).all(dim=1).any()
    for split in ("val", "test"):
        assert torch.equal(
            _features(tmp_path / "full", split),
            _features(tmp_path / "capped", split),
        )
