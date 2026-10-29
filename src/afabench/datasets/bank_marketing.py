from typing import ClassVar, final, override

import polars as pl
from ucimlrepo import fetch_ucirepo

from afabench.datasets.tabular_csv import (
    encode_categories,
    prepared_csv_bytes,
    read_source_csv,
)
from afabench.datasets.uci import UCIDataset


@final
class BankMarketingDataset(UCIDataset):
    """Bank Marketing (UCI 222, `bank-full`), labelled 1 when the client subscribed a term deposit."""

    SHA256: ClassVar[str] = (
        "7e3c9e305e7871ef398254dd01c1e3a1ff2b66d6c3e36666367cf07fdb0f8f03"
    )

    @override
    def _read_source(self) -> pl.DataFrame:
        frame = read_source_csv(self.source, separator=";")
        # Missing categorical values are a category of their own.
        return frame.with_columns(
            pl.col("y").replace_strict({"yes": 1, "no": 0}),
            *(
                encode_categories(pl.col(name).fill_null("nan")).alias(name)
                for name, dtype in frame.schema.items()
                if dtype == pl.String and name != "y"
            ),
        )

    @classmethod
    @override
    def make_prepared_source(cls) -> bytes:
        return prepared_csv_bytes(cls._fetch_uci(), separator=";")

    @classmethod
    @override
    def _fetch_uci(cls) -> pl.DataFrame:
        bank_data = fetch_ucirepo(id=222)
        assert bank_data is not None
        assert bank_data.data is not None
        return pl.concat(
            [
                pl.from_pandas(bank_data.data.features),
                pl.from_pandas(bank_data.data.targets),
            ],
            how="horizontal",
        )
