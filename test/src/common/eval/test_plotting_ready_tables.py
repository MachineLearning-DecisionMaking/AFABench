"""Public Polars evaluation-table loading, filtering and metric contracts."""

from pathlib import Path

import polars as pl
import pytest

from afabench.evaluation.schemas import PLOTTING_READY_DTYPES
from afabench.evaluation.tables import (
    apply_exclusions,
    combine_soft_budget_params,
    keep_largest_hard_budget,
    keep_smallest_soft_budget_param,
    load_plotting_ready_table,
    metrics_at_every_step,
    metrics_at_stop,
    variance_over_seeds_at_every_step,
    variance_over_seeds_at_stop,
)


def test_loader_restores_known_dtypes_and_probability_nulls(
    tmp_path: Path,
) -> None:
    path = tmp_path / "table.parquet"
    pl.DataFrame(
        {
            "action_performed": [0, 2],
            "true_class": [1, 0],
            "accumulated_cost": [1.0, 2.0],
            "forced_stop": [False, True],
            "n_selections_performed": [1, 2],
            "predicted_probabilities": [[0.25, 0.75], None],
            "afa_method": ["jafa"] * 2,
            "dataset": ["cube"] * 2,
            "initializer": ["cold"] * 2,
            "train_seed": [0] * 2,
            "eval_seed": [0] * 2,
            "eval_hard_budget": ["2.0", "None"],
            "eval_soft_budget_param": [None] * 2,
        }
    ).write_parquet(path)
    table = load_plotting_ready_table(path)
    assert table.schema == {
        name: PLOTTING_READY_DTYPES[name] for name in table.columns
    }
    assert table["eval_hard_budget"].to_list() == [2.0, None]
    assert table["predicted_probabilities"].to_list() == [[0.25, 0.75], None]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("action_performed", 1.5),
        ("action_performed", -1),
        ("forced_stop", "false"),
    ],
)
def test_loader_rejects_unrepresentable_values(
    tmp_path: Path, column: str, value: object
) -> None:
    path = tmp_path / "table.parquet"
    pl.DataFrame({column: [value]}).write_parquet(path)
    with pytest.raises(TypeError, match=column):
        load_plotting_ready_table(path)


def budget_rows(
    rows: list[tuple[str, str, float | None, float | None]],
) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "afa_method": pl.String,
            "dataset": pl.String,
            "eval_hard_budget": pl.Float64,
            "soft_budget_param": pl.Float64,
        },
        orient="row",
    )


def test_budget_filters_select_per_dataset_and_method() -> None:
    table = budget_rows(
        [
            ("jafa", "cube", 1.0, None),
            ("gdfs", "cube", 3.0, None),
            ("jafa", "mnist", 2.0, None),
            ("jafa", "mnist", None, 0.1),
        ]
    )
    assert keep_largest_hard_budget(table).rows() == [
        ("gdfs", "cube", 3.0, None),
        ("jafa", "mnist", 2.0, None),
    ]
    table = budget_rows(
        [
            ("jafa", "cube", None, 0.1),
            ("jafa", "cube", None, 0.01),
            ("gdfs", "cube", None, 0.5),
            ("jafa", "mnist", None, 1.0),
            ("jafa", "mnist", 2.0, None),
        ]
    )
    assert keep_smallest_soft_budget_param(table).rows() == [
        ("jafa", "cube", None, 0.01),
        ("gdfs", "cube", None, 0.5),
        ("jafa", "mnist", None, 1.0),
    ]


def test_soft_budget_columns_combine_or_reject_conflicts() -> None:
    table = pl.DataFrame(
        {
            "train_soft_budget_param": [0.1, None, None],
            "eval_soft_budget_param": [None, 0.5, None],
        }
    )
    combined = combine_soft_budget_params(table)
    assert combined.columns == ["soft_budget_param"]
    assert combined["soft_budget_param"].to_list() == [0.1, 0.5, None]
    with pytest.raises(ValueError, match=r"train_soft_budget_param=0\.2"):
        combine_soft_budget_params(
            table.with_columns(
                pl.Series("train_soft_budget_param", [0.1, 0.2, None])
            )
        )


def test_exclusions_depend_on_exact_method_set() -> None:
    exclusions = {
        ("odin_model_based", "diabetes"): frozenset(
            [frozenset(["odin_model_based", "odin_model_free"])]
        )
    }
    pair = budget_rows(
        [
            ("odin_model_based", "diabetes", 1.0, None),
            ("odin_model_free", "diabetes", 1.0, None),
        ]
    )
    other = budget_rows(
        [
            ("odin_model_based", "diabetes", 1.0, None),
            ("jafa", "diabetes", 1.0, None),
            ("odin_model_based", "cube", 1.0, None),
        ]
    )
    assert apply_exclusions(pair, exclusions).equals(pair)
    assert apply_exclusions(other, exclusions).select(
        "afa_method", "dataset"
    ).rows() == [("jafa", "diabetes"), ("odin_model_based", "cube")]


def classification_rows(
    true_classes: list[int], probabilities: list[list[float]]
) -> pl.DataFrame:
    n = len(true_classes)
    return pl.DataFrame(
        {
            "afa_method": ["jafa"] * n,
            "dataset": ["cube"] * n,
            "train_seed": [0] * n,
            "eval_seed": [0] * n,
            "eval_hard_budget": [2.0] * n,
            "soft_budget_param": pl.Series([None] * n, dtype=pl.Float64),
            "n_selections_performed": [0] * n,
            "true_class": true_classes,
            "predicted_probabilities": probabilities,
            "accumulated_cost": [1.0] * n,
        }
    )


@pytest.mark.parametrize(
    ("classes", "probabilities", "expected"),
    [
        ([1, 0, 0], [[0.4, 0.6], [0.7, 0.3], [0.5, 0.5]], (1.0, 1.0, 1.0)),
        (
            [0, 0, 1, 1],
            [[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]],
            (0.75, 11 / 15, 0.75),
        ),
        (
            [0, 1, 2],
            [[0.4, 0.5, 0.1], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1]],
            (1 / 3, 1 / 6, 2 / 3),
        ),
        (
            [0, 1, 2],
            [
                [0.4, 0.5, 0.0999895],
                [0.3, 0.6, 0.0999895],
                [0.2, 0.7, 0.0999895],
            ],
            (1 / 3, 1 / 6, 2 / 3),
        ),
    ],
)
def test_metrics_use_argmax_and_macro_one_vs_rest_auroc(
    classes: list[int],
    probabilities: list[list[float]],
    expected: tuple[float, float, float],
    tmp_path: Path,
) -> None:
    path = tmp_path / "table.parquet"
    classification_rows(classes, probabilities).write_parquet(path)
    metrics = metrics_at_stop(load_plotting_ready_table(path))
    assert metrics.select("accuracy", "f_score", "auroc").row(
        0
    ) == pytest.approx(expected)
    assert metrics["avg_accumulated_cost"].to_list() == [1.0]


@pytest.mark.parametrize(
    ("classes", "probabilities"),
    [
        ([0, 0], [[0.9, 0.1], [0.8, 0.2]]),
        ([1, 1], [[0.1, 0.9], [0.2, 0.8]]),
        ([0, 1], [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]),
        ([1, 2], [[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]),
    ],
)
def test_missing_classes_have_undefined_auroc(
    classes: list[int], probabilities: list[list[float]]
) -> None:
    metrics = metrics_at_stop(classification_rows(classes, probabilities))
    assert metrics["auroc"].is_nan().all()
    assert metrics["accuracy"].to_list() == [1.0]
    assert metrics["f_score"].to_list() == [1.0]


def test_metrics_group_by_seed_budget_and_step() -> None:
    initial = classification_rows(
        [0, 0, 1, 1], [[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]]
    )
    later = classification_rows([1, 1], [[0.1, 0.9], [0.2, 0.8]]).with_columns(
        pl.lit(1).cast(pl.Int64).alias("n_selections_performed")
    )
    other = classification_rows([0, 1], [[0.1, 0.9], [0.9, 0.1]]).with_columns(
        pl.lit(1).cast(pl.Int64).alias("train_seed")
    )
    metrics = metrics_at_every_step(pl.concat([initial, later, other]))
    assert metrics.select("train_seed", "n_selections_performed").rows() == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert metrics["auroc"].to_list() == pytest.approx(
        [0.75, float("nan"), 0.0], nan_ok=True
    )
    assert metrics["accuracy"].to_list() == [0.75, 1.0, 0.0]


@pytest.mark.parametrize("every_step", [False, True])
@pytest.mark.parametrize("undefined", [False, True])
def test_seed_summaries_average_run_scores_and_propagate_undefined(
    every_step: bool, undefined: bool
) -> None:
    first = classification_rows(
        [0, 0, 1, 1], [[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]]
    )
    second = (
        classification_rows([1, 1], [[0.1, 0.9], [0.2, 0.8]])
        if undefined
        else classification_rows(
            [0, 0, 1, 1], [[0.1, 0.9], [0.4, 0.6], [0.35, 0.65], [0.8, 0.2]]
        )
    )
    second = second.with_columns(
        pl.lit(1).cast(pl.Int64).alias("train_seed"),
        pl.lit(1).cast(pl.Int64).alias("eval_seed"),
    )
    table = pl.concat([first, second])
    summary = (
        variance_over_seeds_at_every_step(metrics_at_every_step(table))
        if every_step
        else variance_over_seeds_at_stop(metrics_at_stop(table))
    )
    if undefined:
        assert summary["mean_auroc"].is_nan().all()
        assert summary["std_auroc"].is_nan().all()
        assert summary["mean_accuracy"].to_list() == [0.875]
    else:
        assert summary["mean_auroc"].to_list() == [0.5]
        assert summary["std_auroc"].to_list() == [
            pytest.approx(0.3535533905932738)
        ]
        assert summary["mean_accuracy"].to_list() == [0.5]


@pytest.mark.parametrize("every_step", [False, True])
@pytest.mark.parametrize("defined", [False, True])
def test_single_seed_has_unavailable_standard_deviation(
    every_step: bool, defined: bool
) -> None:
    table = classification_rows(
        [0, 1] if defined else [1, 1], [[0.8, 0.2], [0.2, 0.8]]
    )
    summary = (
        variance_over_seeds_at_every_step(metrics_at_every_step(table))
        if every_step
        else variance_over_seeds_at_stop(metrics_at_stop(table))
    )
    assert summary["std_auroc"].is_null().all()
    if defined:
        assert summary["mean_auroc"].to_list() == [1.0]
    else:
        assert summary["mean_auroc"].is_nan().all()


@pytest.mark.parametrize(
    ("probabilities", "error", "message"),
    [
        ([None, [0.2, 0.8]], TypeError, "vector"),
        ([0.5, [0.2, 0.8]], TypeError, "vector"),
        ([[1.0], [1.0]], ValueError, "vector"),
        ([[], []], ValueError, "vector"),
        ([[[0.8, 0.2]], [[0.2, 0.8]]], ValueError, "vector"),
        ([[0.8, 0.2], [0.2, 0.3, 0.5]], ValueError, "vector"),
        ([[1, 0], [0.2, 0.8]], TypeError, "floating"),
        ([["0.8", "0.2"], [0.2, 0.8]], TypeError, "floating"),
        ([[True, False], [0.2, 0.8]], TypeError, "floating"),
        ([[float("nan"), 0.2], [0.2, 0.8]], ValueError, "finite"),
        ([[float("inf"), 0.2], [0.2, 0.8]], ValueError, "finite"),
        ([[-float("inf"), 0.2], [0.2, 0.8]], ValueError, "finite"),
        ([[-0.1, 1.1], [0.2, 0.8]], ValueError, r"\[0, 1\]"),
        ([[1.1, -0.1], [0.2, 0.8]], ValueError, r"\[0, 1\]"),
        ([[0.6, 0.2], [0.2, 0.8]], ValueError, "sum to 1"),
    ],
)
def test_metrics_reject_malformed_probabilities(
    probabilities: list[object], error: type[Exception], message: str
) -> None:
    table = classification_rows([0, 1], [[0.8, 0.2], [0.2, 0.8]]).with_columns(
        pl.Series("predicted_probabilities", probabilities, dtype=pl.Object)
    )
    with pytest.raises(error, match=f"predicted_probabilities.*{message}"):
        metrics_at_stop(table)
