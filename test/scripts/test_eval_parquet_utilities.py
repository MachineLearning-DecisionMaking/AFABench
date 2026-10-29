"""Miscellaneous table scripts preserve nullable Parquet values and types."""

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from scripts.misc import (
    add_dataset_split_col,
    combine_soft_budget_results,
    generate_mock_evaluation_data,
)


def test_add_dataset_split_preserves_parquet_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.parquet"
    output = tmp_path / "nested/output.parquet"
    original = pl.DataFrame(
        {"prediction": pl.Series([1, None], dtype=pl.Int64)}
    )
    original.write_parquet(source)
    monkeypatch.setattr(
        "sys.argv", ["add_dataset_split_col.py", str(source), str(output), "3"]
    )
    add_dataset_split_col.main()
    assert_frame_equal(
        pl.read_parquet(output),
        original.with_columns(
            pl.lit(3, dtype=pl.Int64).alias("dataset_split")
        ),
    )


def test_combine_soft_budget_results_preserves_parquet_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = pl.DataFrame(
        {
            "method": ["a", "b"],
            "training_seed": [1, 2],
            "cost_parameter": [0.1, 0.2],
            "dataset": ["example", "example"],
            "features_chosen": [1, 2],
            "predicted_label_builtin": pl.Series([1, None], dtype=pl.Int64),
            "predicted_label_external": [0, 1],
            "true_label": [1, 1],
        }
    )
    paths = [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    for row, path in enumerate(paths):
        original.slice(row, 1).write_parquet(path)
    output = tmp_path / "nested/combined.parquet"
    monkeypatch.setattr(
        "sys.argv",
        [
            "combine_soft_budget_results.py",
            *(str(p) for p in paths),
            str(output),
        ],
    )
    combine_soft_budget_results.main()
    assert_frame_equal(pl.read_parquet(output), original)


def test_combine_soft_budget_results_validates_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "invalid.parquet"
    output = tmp_path / "combined.parquet"
    pl.DataFrame({"method": ["a"]}).write_parquet(source)
    monkeypatch.setattr(
        "sys.argv",
        ["combine_soft_budget_results.py", str(source), str(output)],
    )
    with pytest.raises(ValueError, match="missing columns"):
        combine_soft_budget_results.main()
    assert not output.exists()


def test_mock_evaluation_data_saves_nullable_integers_as_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = generate_mock_evaluation_data.generate_mock_data(
        n_methods=2,
        n_seeds=1,
        n_cost_params=1,
        n_datasets=1,
        n_samples=2,
        n_splits=1,
    )
    monkeypatch.setattr(
        generate_mock_evaluation_data, "generate_mock_data", lambda: original
    )
    monkeypatch.chdir(tmp_path)
    generate_mock_evaluation_data.main()
    saved = pl.read_parquet(tmp_path / "eval_results.parquet")
    assert_frame_equal(saved, original)
    assert saved["predicted_label_builtin"].dtype == pl.Int64
    assert saved["predicted_label_builtin"].is_null().any()
    assert not (tmp_path / "eval_results.csv").exists()
