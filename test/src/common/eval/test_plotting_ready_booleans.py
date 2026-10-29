"""Boolean restoration preserves null and numeric boolean Parquet columns."""

from pathlib import Path

import polars as pl
import pytest

from afabench.evaluation.tables import load_plotting_ready_table


@pytest.mark.parametrize(
    "values", [[None], [0, 1, None], [0.0, 1.0, None], [False, True, None]]
)
def test_loader_restores_nullable_boolean_columns(
    tmp_path: Path, values: list[object]
) -> None:
    path = tmp_path / "results.parquet"
    pl.DataFrame({"forced_stop": values}).write_parquet(path)
    loaded = load_plotting_ready_table(path)
    assert loaded["forced_stop"].dtype == pl.Boolean
    assert loaded["forced_stop"].to_list() == [
        None if value is None else bool(value) for value in values
    ]


@pytest.mark.parametrize("values", [["false"], [0, 2], [0.0, 0.5]])
def test_loader_reports_invalid_boolean_values(
    tmp_path: Path, values: list[object]
) -> None:
    path = tmp_path / "results.parquet"
    pl.DataFrame({"forced_stop": values}).write_parquet(path)
    with pytest.raises(TypeError, match=repr(values[-1])):
        load_plotting_ready_table(path)
