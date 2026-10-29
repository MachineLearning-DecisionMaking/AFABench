"""UCI's external pandas frames cross into Polars before preparation."""

from types import SimpleNamespace

import pandas as pd
import pytest

from afabench.datasets import ACTG175Dataset, BankMarketingDataset, CKDDataset


@pytest.mark.parametrize(
    ("dataset_class", "targets", "expected"),
    [
        (
            ACTG175Dataset,
            {"cid": [1.0, 0.0]},
            b"numeric,category,target\n1.0,z,1\n,a,0\n",
        ),
        (
            CKDDataset,
            {"classification": [" CKD\t", "notckd"]},
            b"numeric,category,target\n1.0,z,1\n,a,0\n",
        ),
        (
            BankMarketingDataset,
            {"y": ["yes", "no"]},
            b"numeric;category;y\n1.0;z;yes\n;a;no\n",
        ),
    ],
)
def test_preparation_preserves_external_values_and_csv_bytes(
    monkeypatch: pytest.MonkeyPatch,
    dataset_class: type[ACTG175Dataset | CKDDataset | BankMarketingDataset],
    targets: dict[str, list[float] | list[str]],
    expected: bytes,
) -> None:
    # pandas is the ucimlrepo interface, not a maintained table operation.
    source = SimpleNamespace(
        data=SimpleNamespace(
            features=pd.DataFrame(
                {"numeric": [1.0, None], "category": ["z", "a"]}
            ),
            targets=pd.DataFrame(targets),
        )
    )
    monkeypatch.setattr(
        f"{dataset_class.__module__}.fetch_ucirepo", lambda **_: source
    )

    assert dataset_class.make_prepared_source() == expected
