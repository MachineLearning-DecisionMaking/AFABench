from typing import ClassVar, final, override

import polars as pl
from ucimlrepo import fetch_ucirepo

from afabench.datasets.uci import UCIDataset


@final
class ACTG175Dataset(UCIDataset):
    """AIDS Clinical Trials Group Study 175 (UCI 890), labelled with the failure indicator `cid`."""

    SHA256: ClassVar[str] = (
        "71493efd4ac3d8edf9edb43df83f0cd91b4cfe73bc9f96278fa8b536e8131cda"
    )
    # UCI links to ClinicalTrials.gov for the license, which gives none
    SOURCE_COMMITTED: ClassVar[bool] = False

    @classmethod
    @override
    def _fetch_uci(cls) -> pl.DataFrame:
        actg_data = fetch_ucirepo(id=890)
        assert actg_data is not None
        assert actg_data.data is not None
        features = pl.from_pandas(actg_data.data.features)
        target = pl.from_pandas(actg_data.data.targets).to_series(0)
        return features.with_columns(target.cast(pl.Int64).alias("target"))
