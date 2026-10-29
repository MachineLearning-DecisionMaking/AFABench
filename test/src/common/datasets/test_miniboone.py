"""MiniBooNE drops its placeholder and outlier rows and maps the rest to source rows (issue #129, ADR 0004)."""

import hashlib
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest
import torch

from afabench.core.bundle_system.bundle import load_bundle
from afabench.datasets import MiniBooNEDataset
from afabench.datasets.config import SplitRatioConfig
from scripts.dataset_generation.generate_dataset import generate_and_save_split

N_ROWS = 30
N_FEATURES = 50
# Source rows that are all -999, or have an extreme Feature 20
PLACEHOLDER_ROWS = (3, 17)
OUTLIER_ROWS = {8: -17256.7, 24: 16000900.0}
DROPPED_ROWS = sorted([*PLACEHOLDER_ROWS, *OUTLIER_ROWS])


def write_source(path: Path) -> Path:
    """Write a MiniBooNE-like CSV whose first feature is the source row number."""
    rng = np.random.default_rng(0)
    frame = pl.DataFrame(
        rng.normal(size=(N_ROWS, N_FEATURES)),
        schema=[f"Feature {i + 1}" for i in range(N_FEATURES)],
    )
    frame = frame.with_columns(
        pl.Series("Feature 1", np.arange(N_ROWS, dtype=float))
    )
    frame = frame.with_columns(
        pl.when(~pl.int_range(pl.len()).is_in(PLACEHOLDER_ROWS))
        .then(pl.all())
        .otherwise(-999.0)
    )
    for row, value in OUTLIER_ROWS.items():
        frame[row, "Feature 20"] = value
    frame = frame.with_columns(
        pl.Series("Outcome", [i % 2 for i in range(N_ROWS)])
    )
    frame.write_csv(path)
    return path


def miniboone(path: Path) -> MiniBooNEDataset:
    source = write_source(path)
    sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    return MiniBooNEDataset(root=str(source), sha256=sha256)


def kept_rows() -> list[int]:
    return [i for i in range(N_ROWS) if i not in DROPPED_ROWS]


def test_placeholder_and_outlier_rows_are_dropped(tmp_path: Path) -> None:
    dataset = miniboone(tmp_path / "m.csv")

    features, labels = dataset.get_all_data()

    assert len(dataset) == N_ROWS - len(DROPPED_ROWS)
    # z-normalized standard normal features stay small once extremes are gone
    assert float(features.abs().max()) < 5
    assert labels.argmax(-1).tolist() == [i % 2 for i in kept_rows()]


def test_source_rows_locate_instances_in_the_source(tmp_path: Path) -> None:
    dataset = miniboone(tmp_path / "m.csv")

    subset = dataset.create_subset([5, 0, 2])

    assert dataset.get_source_rows().tolist() == kept_rows()
    assert subset.get_source_rows().tolist() == [
        kept_rows()[i] for i in (5, 0, 2)
    ]


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_split_bundles_keep_source_rows(tmp_path: Path, split: str) -> None:
    source = write_source(tmp_path / "m.csv")
    generate_and_save_split(
        dataset_class=MiniBooNEDataset,
        split_ratio=SplitRatioConfig(train=0.6, val=0.2, test=0.2),
        seed_for_split=5,
        save_path=tmp_path / "out",
        dataset_kwargs={
            "root": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        metadata_to_save={},
        dataset_key="miniboone",
        dataset_realization_index=0,
        resolved_config={"dataset": {"class_name": "MiniBooNEDataset"}},
        feature_costs_file=tmp_path / "no_feature_costs.csv",
    )

    bundle = cast(
        "MiniBooNEDataset",
        load_bundle(tmp_path / "out" / f"{split}.bundle").obj,
    )

    source_rows = bundle.get_source_rows()
    expected = torch.tensor(kept_rows())[bundle.get_generation_indices()]
    assert source_rows.tolist() == expected.tolist()
    assert not set(source_rows.tolist()) & set(DROPPED_ROWS)


def test_missing_non_outlier_feature_keeps_its_source_row(
    tmp_path: Path,
) -> None:
    source = write_source(tmp_path / "m.csv")
    frame = pl.read_csv(source)
    frame[2, "Feature 2"] = None
    frame.write_csv(source)
    dataset = MiniBooNEDataset(
        root=str(source),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )

    assert dataset.get_source_rows().tolist() == kept_rows()
    assert float(dataset.features[2, 1]) == pytest.approx(0, abs=1e-6)
