"""Evaluation contracts validate Polars frames without coercing dtypes."""

from pathlib import Path

import polars as pl
import pytest
from pandera.errors import SchemaError, SchemaErrors
from polars.testing import assert_frame_equal

from afabench.evaluation.schemas import (
    IDENTITY_DTYPES,
    EvaluationSchema,
    SavedEvaluationSchema,
    identity_column,
)


@pytest.fixture
def evaluation_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "episode_id": [0, 0],
            "generation_index": [5, 5],
            "split_index": [2, 2],
            "step": [0, 1],
            "action_performed": [1, 0],
            "builtin_predicted_probabilities": pl.Series(
                [None, None], dtype=pl.List(pl.Float64)
            ),
            "external_predicted_probabilities": [[0.25, 0.75], [0.75, 0.25]],
            "true_class": [1, 1],
            "accumulated_cost": [1.0, 1.0],
            "forced_stop": [False, True],
        }
    )


def test_evaluation_schema_preserves_dtypes(
    evaluation_frame: pl.DataFrame,
) -> None:
    assert_frame_equal(
        EvaluationSchema.validate(evaluation_frame), evaluation_frame
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("action_performed", -1),
        ("true_class", -1),
        ("episode_id", -1),
        ("generation_index", -1),
        ("split_index", -1),
        ("step", -1),
        ("accumulated_cost", -0.5),
        ("forced_stop", "false"),
        ("step", "0"),
    ],
)
def test_evaluation_schema_rejects_invalid_values(
    evaluation_frame: pl.DataFrame, column: str, value: object
) -> None:
    frame = evaluation_frame.with_columns(pl.Series(column, [value, value]))
    with pytest.raises((SchemaError, SchemaErrors)):
        EvaluationSchema.validate(frame)


@pytest.mark.parametrize("column", list(EvaluationSchema.to_schema().columns))
def test_evaluation_schema_requires_all_columns(
    evaluation_frame: pl.DataFrame, column: str
) -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        EvaluationSchema.validate(evaluation_frame.drop(column))


def test_evaluation_schema_rejects_incomplete_episodes(
    evaluation_frame: pl.DataFrame,
) -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        EvaluationSchema.validate(evaluation_frame.slice(1))


@pytest.mark.parametrize("column", ["generation_index", "split_index"])
def test_evaluation_schema_rejects_episode_spanning_instances(
    evaluation_frame: pl.DataFrame, column: str
) -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        EvaluationSchema.validate(
            evaluation_frame.with_columns(pl.Series(column, [0, 1]))
        )


def test_evaluation_schema_rejects_extra_columns(
    evaluation_frame: pl.DataFrame,
) -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        EvaluationSchema.validate(
            evaluation_frame.with_columns(pl.lit(0).alias("unexpected"))
        )


def identified(frame: pl.DataFrame) -> pl.DataFrame:
    identity = dict.fromkeys(IDENTITY_DTYPES)
    identity.update(
        afa_method="gdfs",
        dataset="cube",
        dataset_realization_index=2,
        eval_split="test",
        initializer="cold",
        eval_seed=8,
        train_hard_budget=3,
        eval_soft_budget_param=0.5,
    )
    return frame.with_columns(
        identity_column(name, value, len(frame))
        for name, value in identity.items()
    )


def test_saved_schema_round_trips_identity_dtypes_through_parquet(
    evaluation_frame: pl.DataFrame, tmp_path: Path
) -> None:
    frame = identified(evaluation_frame)
    path = tmp_path / "eval.parquet"
    SavedEvaluationSchema.validate(frame).write_parquet(path)
    loaded = pl.read_parquet(path)
    assert_frame_equal(loaded, frame)
    SavedEvaluationSchema.validate(loaded)


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("dataset_realization_index", pl.Series([2, 2])),
        ("eval_seed", pl.Series([None, None], dtype=pl.UInt64)),
        ("initializer", pl.Series([None, None], dtype=pl.String)),
        ("eval_split", pl.Series(["validation"] * 2)),
        ("eval_hard_budget", pl.Series([-1.0, -1.0])),
        ("dataset", pl.Series(["cube", "mnist"])),
        ("train_seed", pl.Series([1, None], dtype=pl.UInt64)),
    ],
)
def test_saved_schema_rejects_invalid_identity(
    evaluation_frame: pl.DataFrame, column: str, values: pl.Series
) -> None:
    frame = identified(evaluation_frame).with_columns(values.alias(column))
    with pytest.raises((SchemaError, SchemaErrors)):
        SavedEvaluationSchema.validate(frame)
