"""Loading result identities must not round unsigned integer values."""

from pathlib import Path

import polars as pl
import pytest

from afabench.evaluation.tables import load_plotting_ready_table


@pytest.mark.parametrize("dtype", [pl.UInt64, pl.String])
def test_loader_preserves_large_unsigned_identities(
    tmp_path: Path, dtype: pl.DataType
) -> None:
    values = [2**53 + 1, 2**64 - 1, None]
    path = tmp_path / "results.parquet"
    pl.DataFrame(
        {"eval_seed": pl.Series(values, dtype=pl.UInt64).cast(dtype)}
    ).write_parquet(path)
    assert load_plotting_ready_table(path)["eval_seed"].to_list() == values


@pytest.mark.parametrize(
    "value", ["1.5", "-1", "18446744073709551616", "18446744073709551615.5"]
)
def test_loader_rejects_invalid_unsigned_identity_strings(
    tmp_path: Path, value: str
) -> None:
    path = tmp_path / "results.parquet"
    pl.DataFrame({"eval_seed": [value]}).write_parquet(path)
    with pytest.raises(TypeError, match="eval_seed"):
        load_plotting_ready_table(path)


def test_loader_parses_integral_decimal_strings_without_rounding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.parquet"
    pl.DataFrame(
        {"eval_seed": ["18446744073709551615.0", "2.0", "None"]}
    ).write_parquet(path)
    assert load_plotting_ready_table(path)["eval_seed"].to_list() == [
        2**64 - 1,
        2,
        None,
    ]
