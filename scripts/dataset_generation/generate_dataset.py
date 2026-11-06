"""Generate multiple instances of a dataset, see dataset_generation.md."""

import logging
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from afabench.core.bundle_system.bundle import save_bundle
from afabench.core.registry import get_class
from afabench.core.types import AFADataset
from afabench.datasets.config import DatasetGenerationConfig, SplitRatioConfig

log = logging.getLogger(__name__)


def generate_and_save_split(
    dataset_class: type[AFADataset],
    split_ratio: SplitRatioConfig,
    seed_for_split: int,
    save_path: Path,
    dataset_kwargs: dict[str, Any],
    metadata_to_save: dict[str, Any],
    *,
    fixed_test_seed: int | None = None,
    stratify: bool = False,
    max_train_size: int | None = None,
) -> None:
    """
    Generate and save a single train/val/test split.

    Args:
        dataset_class: The dataset class to instantiate.
        split_ratio: The ratio for splitting the dataset into train/val/test.
        seed_for_split: Seed used during splitting.
        save_path: Path to save the generated dataset splits. Will create separate folders for each split instance.
        dataset_kwargs: Keyword arguments to pass to the dataset class constructor.
        metadata_to_save: Additional metadata to save alongside the dataset.
    """
    # Generate full dataset
    dataset = dataset_class(**dataset_kwargs)

    total_size = len(dataset)
    all_indices = np.arange(total_size)
    labels = None
    if stratify:
        _, encoded_labels = dataset.get_all_data()
        labels_array = encoded_labels.detach().cpu().numpy()
        labels = (
            labels_array.argmax(axis=1)
            if labels_array.ndim > 1
            else labels_array
        )
    if fixed_test_seed is None and not stratify:
        shuffled = all_indices.tolist()
        random.Random(seed_for_split).shuffle(shuffled)
        train_size = int(split_ratio.train * total_size)
        val_size = int(split_ratio.val * total_size)
        train_indices = shuffled[:train_size]
        val_indices = shuffled[train_size : train_size + val_size]
        test_indices = shuffled[train_size + val_size :]
    else:
        test_seed = (
            fixed_test_seed if fixed_test_seed is not None else seed_for_split
        )
        train_val_indices, test_indices = train_test_split(
            all_indices,
            test_size=split_ratio.test,
            random_state=test_seed,
            stratify=labels,
        )
        train_val_labels = (
            None if labels is None else labels[train_val_indices]
        )
        relative_val_size = split_ratio.val / (
            split_ratio.train + split_ratio.val
        )
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=relative_val_size,
            random_state=seed_for_split,
            stratify=train_val_labels,
        )
    if max_train_size is not None and len(train_indices) > max_train_size:
        train_indices = random.Random(seed_for_split).sample(
            list(train_indices), max_train_size
        )
    train_dataset, val_dataset, test_dataset = dataset.create_splits(
        train_indices,
        val_indices,
        test_indices,
    )

    # Save splits
    save_path.mkdir(parents=True, exist_ok=True)
    train_path = save_path / "train.bundle"
    val_path = save_path / "val.bundle"
    test_path = save_path / "test.bundle"

    for obj, path in zip(
        [train_dataset, val_dataset, test_dataset],
        [train_path, val_path, test_path],
        strict=True,
    ):
        save_bundle(
            obj=obj,
            path=path,
            metadata=metadata_to_save
            | {
                "seed_for_split": seed_for_split,
                "fixed_test_seed": fixed_test_seed,
                "stratified": stratify,
                "generated_at": datetime.now(UTC).isoformat(),
                "kwargs": dataset_kwargs,
            },
        )

    # # Prepare metadata
    # metadata_to_save = metadata_to_save | {
    #     "seed_for_split": seed_for_split,
    #     "generated_at": datetime.now(UTC).isoformat(),
    # }
    # json_data = metadata_to_save | {
    #     "kwargs": dataset_kwargs,
    # }
    # # Save metadata
    # with (save_path / "metadata.json").open("w") as f:
    #     json.dump(json_data, f)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/dataset_generation",
    config_name="config",
)
def main(cfg: DatasetGenerationConfig) -> None:
    cfg = cast("DatasetGenerationConfig", OmegaConf.to_object(cfg))
    log.info(f"Generating {cfg.dataset.class_name} to {cfg.save_path}")
    for instance_idx, seed in zip(
        cfg.instance_indices, cfg.seeds, strict=True
    ):
        dataset_class = cast(
            "type[AFADataset]", get_class(cfg.dataset.class_name)
        )
        if dataset_class.accepts_seed():
            dataset_kwargs = dict(cfg.dataset.kwargs) | {"seed": seed}
        else:
            dataset_kwargs = dict(cfg.dataset.kwargs)
        generate_and_save_split(
            dataset_class=dataset_class,
            split_ratio=cfg.split_ratio,
            # use same instance for splitting as for data generation
            seed_for_split=seed,
            save_path=Path(cfg.save_path) / str(instance_idx),
            dataset_kwargs=dataset_kwargs,
            metadata_to_save={
                "instance_idx": instance_idx,
            },
            fixed_test_seed=cfg.fixed_test_seed,
            stratify=cfg.stratify,
            max_train_size=cfg.max_train_size,
        )
    log.info(
        f"Generated {len(cfg.instance_indices)} dataset instances to {
            cfg.save_path
        }"
    )


if __name__ == "__main__":
    main()
