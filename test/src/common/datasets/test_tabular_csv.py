"""Prepared CSV generation keeps category order and population statistics."""

import hashlib
from pathlib import Path

import pytest
import torch

from afabench.datasets import BankMarketingDataset, CKDDataset


def test_prepared_csv_preserves_categories_missing_values_and_source_rows(
    tmp_path: Path,
) -> None:
    content = (
        b"category,numeric,constant,target\nz,1,7,0\na,,7,1\n,3,7,0\nz,5,7,1\n"
    )
    path = tmp_path / "source.csv"
    path.write_bytes(content)
    dataset = CKDDataset(
        path=str(path), sha256=hashlib.sha256(content).hexdigest()
    )

    expected = torch.tensor(
        [
            [0.816496580927726, -1.414213562373095, 0],
            [-1.632993161855452, 0, 0],
            [0, 0, 0],
            [0.816496580927726, 1.414213562373095, 0],
        ]
    )
    torch.testing.assert_close(dataset.features, expected)
    assert dataset.labels.argmax(-1).tolist() == [0, 1, 0, 1]
    assert dataset.feature_names == ["category", "numeric", "constant"]
    assert dataset.get_source_rows().tolist() == [0, 1, 2, 3]


def test_bank_missing_category_is_encoded_as_nan_string(
    tmp_path: Path,
) -> None:
    content = b"job;age;y\nz;1;yes\na;2;no\n;3;yes\nz;4;no\n"
    path = tmp_path / "bank.csv"
    path.write_bytes(content)

    dataset = BankMarketingDataset(
        path=str(path), sha256=hashlib.sha256(content).hexdigest()
    )

    # Sorted categories are a, nan, z, including missing as a category.
    expected = torch.tensor([2, 0, 1, 2], dtype=torch.float64)
    expected = (expected - expected.mean()) / expected.std(correction=0)
    torch.testing.assert_close(dataset.features[:, 0], expected.float())
    assert dataset.labels.argmax(-1).tolist() == [1, 0, 1, 0]


def test_all_missing_feature_is_rejected(tmp_path: Path) -> None:
    content = b"missing,target\n,0\n,1\n"
    path = tmp_path / "empty.csv"
    path.write_bytes(content)

    with pytest.raises(ValueError, match="column has no values"):
        CKDDataset(path=str(path), sha256=hashlib.sha256(content).hexdigest())
