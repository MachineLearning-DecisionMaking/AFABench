"""The transformation CLI accepts only complete current evaluation tables."""

from pathlib import Path

import polars as pl
import pytest
import typer
from pandera.errors import SchemaError, SchemaErrors
from typer.testing import CliRunner

from afabench.evaluation.provenance import (
    evaluation_table_provenance,
    save_evaluation_table,
)
from afabench.evaluation.schemas import (
    IDENTITY_DTYPES,
    PLOTTING_READY_DTYPES,
    identity_column,
)
from afabench.testing.provenance import placeholder_provenance
from scripts.misc.transform_eval_data_pipeline import (
    IdentityArgumentMismatchError,
    transform,
)


@pytest.fixture
def raw_path(tmp_path: Path) -> Path:
    frame = pl.DataFrame(
        {
            "episode_id": [0, 0, 1],
            "generation_index": [4, 4, 9],
            "split_index": [1, 1, 0],
            "step": [0, 1, 0],
            "action_performed": [1, 0, 0],
            "builtin_predicted_probabilities": pl.Series(
                [None] * 3, dtype=pl.List(pl.Float64)
            ),
            "external_predicted_probabilities": [
                [0.1, 0.2, 0.7],
                [0.2, 0.6, 0.2],
                [0.8, 0.1, 0.1],
            ],
            "true_class": [1, 1, 0],
            "accumulated_cost": [1.0, 1.0, 0.0],
            "forced_stop": [False, True, False],
        }
    )
    identity = dict.fromkeys(IDENTITY_DTYPES)
    identity.update(
        afa_method="method_a",
        dataset="dataset_a",
        dataset_realization_index=3,
        eval_split="val",
        initializer="cold",
        train_seed=2,
        train_soft_budget_param=0.5,
        eval_seed=7,
    )
    frame = frame.with_columns(
        identity_column(k, v, len(frame)) for k, v in identity.items()
    )
    path = tmp_path / "raw.parquet"
    save_evaluation_table(
        frame, path, provenance=placeholder_provenance("evaluation", seed=7)
    )
    return path


def test_transform_preserves_identity_probabilities_and_provenance(
    raw_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / "transformed.parquet"
    app = typer.Typer()
    app.command()(transform)
    result = CliRunner().invoke(
        app, ["--input_path", str(raw_path), "--output_path", str(output)]
    )
    assert result.exit_code == 0, result.output
    table = pl.read_parquet(output)
    assert dict(table.schema) == PLOTTING_READY_DTYPES
    assert table["n_selections_performed"].to_list() == [0, 1, 0, 0, 1, 0]
    assert table["predicted_probabilities"].to_list() == [
        None,
        None,
        None,
        [0.1, 0.2, 0.7],
        [0.2, 0.6, 0.2],
        [0.8, 0.1, 0.1],
    ]
    assert table["generation_index"].to_list() == [4, 4, 9] * 2
    raw = pl.read_parquet(raw_path)
    for column in IDENTITY_DTYPES:
        assert table[column].to_list() == raw[column].to_list() * 2
    assert evaluation_table_provenance(output) == evaluation_table_provenance(
        raw_path
    )


@pytest.mark.parametrize(
    "removed",
    [
        ["episode_id", "step"],
        list(IDENTITY_DTYPES),
        ["generation_index", "split_index"],
    ],
)
def test_transform_rejects_older_shapes(
    raw_path: Path, tmp_path: Path, removed: list[str]
) -> None:
    pl.read_parquet(raw_path).drop(removed).write_parquet(raw_path)
    with pytest.raises((SchemaError, SchemaErrors)):
        transform(raw_path, tmp_path / "output.parquet")
    assert not (tmp_path / "output.parquet").exists()


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("method", "method_b"),
        ("dataset", "dataset_b"),
        ("initializer", "warm"),
        ("train_seed", "3"),
        ("train_soft_budget_param", "null"),
    ],
)
def test_transform_rejects_disagreeing_arguments(
    raw_path: Path, tmp_path: Path, argument: str, value: str
) -> None:
    with pytest.raises(IdentityArgumentMismatchError, match=value):
        transform(raw_path, tmp_path / "output.parquet", **{argument: value})


def test_transform_fills_null_identity(raw_path: Path, tmp_path: Path) -> None:
    pl.read_parquet(raw_path).with_columns(
        pl.lit(None, dtype=pl.UInt64).alias("train_seed")
    ).write_parquet(raw_path)
    output = tmp_path / "output.parquet"
    transform(raw_path, output, train_seed="5")
    assert pl.read_parquet(output)["train_seed"].to_list() == [5] * 6
