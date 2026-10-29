from typing import ClassVar, final, override

import polars as pl
from ucimlrepo import fetch_ucirepo

from afabench.datasets.uci import UCIDataset


@final
class CKDDataset(UCIDataset):
    """Chronic Kidney Disease (UCI 336), labelled 1 for chronic kidney disease."""

    SHA256: ClassVar[str] = (
        "45fb1b3b76b1a1a0f3bf1eab70796c76137ecc922e0f7c4c338141ae6e258f5b"
    )

    @classmethod
    @override
    def _fetch_uci(cls) -> pl.DataFrame:
        ckd_data = fetch_ucirepo(id=336)
        assert ckd_data is not None
        assert ckd_data.data is not None
        features = pl.from_pandas(ckd_data.data.features)
        target = pl.from_pandas(ckd_data.data.targets).to_series(0)
        target = (
            target.cast(pl.String)
            .str.strip_chars()
            .str.to_lowercase()
            .replace_strict({"ckd": 1, "notckd": 0}, default=None)
        )
        return features.with_columns(target.alias("target"))
