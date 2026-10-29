"""The MiniBooNE subset is MiniBooNE restricted to the L2M paper's features."""

import hashlib
from pathlib import Path

import numpy as np
import polars as pl
import torch

from afabench.datasets import MiniBooNEDataset, MiniBooNESubsetDataset

PAPER_FEATURES = [
    "Feature 1",
    "Feature 2",
    "Feature 3",
    "Feature 4",
    "Feature 12",
    "Feature 17",
    "Feature 23",
    "Feature 25",
    "Feature 27",
    "Feature 32",
]


def test_subset_is_miniboone_restricted_to_the_paper_features(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(0)
    source = pl.DataFrame(
        rng.normal(size=(12, 50)),
        schema=[f"Feature {i}" for i in range(1, 51)],
    )
    # A placeholder row, and an outlier in Feature 20, which the subset
    # does not keep
    source = source.with_columns(
        pl.when(pl.int_range(pl.len()) != 3).then(pl.all()).otherwise(-999.0)
    )
    source[7, "Feature 20"] = -17256.7
    source = source.with_columns(
        pl.Series("Outcome", [i % 2 for i in range(12)])
    )
    path = tmp_path / "miniboone.csv"
    source.write_csv(path)

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    subset = MiniBooNESubsetDataset(root=str(path), sha256=sha256)
    full = MiniBooNEDataset(root=str(path), sha256=sha256)

    assert subset.feature_names == PAPER_FEATURES
    columns = [full.feature_names.index(name) for name in PAPER_FEATURES]
    torch.testing.assert_close(subset.features, full.features[:, columns])
    assert torch.equal(subset.labels, full.labels)
    assert torch.equal(subset.get_source_rows(), full.get_source_rows())
    assert len(subset.features) == 10
