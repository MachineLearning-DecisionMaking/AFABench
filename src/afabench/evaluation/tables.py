"""
Load plotting-ready evaluation tables and prepare them for plots.

A plotting-ready table is the output of the transformation stage, merged
over runs and usually split by classifier type
(`docs/reference/evaluation_dataframes.md`). Every visualization job reads
it through `load_plotting_ready_table`, so all plots see the same dtypes.
"""

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from afabench.evaluation.schemas import PLOTTING_READY_DTYPES


def load_plotting_ready_table(path: Path) -> pl.DataFrame:
    """
    Read a plotting-ready table, restoring the known nullable column types.

    Non-numeric strings in numeric columns become null; out-of-range or
    fractional unsigned integers and non-boolean boolean values raise.
    Absent columns stay absent.
    """
    table = pl.read_parquet(path)
    columns: list[pl.Series] = []
    for name, dtype in PLOTTING_READY_DTYPES.items():
        if name not in table.columns:
            continue
        values = table[name]
        if dtype == pl.UInt64:
            if values.dtype.is_integer():
                if (values < 0).any():
                    msg = f"{name} contains negative UInt64 values: {values.filter(values < 0).to_list()}"
                    raise TypeError(msg)
            else:
                values = pl.Series(
                    name,
                    [_unsigned_value(name, value) for value in values],
                    dtype=pl.UInt64,
                )
        elif dtype == pl.Float64:
            values = values.cast(pl.Float64, strict=False).fill_nan(None)
        elif dtype == pl.Boolean:
            values = _boolean_column(values)
        columns.append(values.cast(dtype))
    return table.with_columns(columns)


def _boolean_column(values: pl.Series) -> pl.Series:
    if values.dtype in (pl.Boolean(), pl.Null()):
        return values.cast(pl.Boolean)
    if values.dtype.is_numeric():
        values = values.fill_nan(None)
        invalid = values.filter((values != 0) & (values != 1))
    else:
        invalid = values.drop_nulls()
        if not len(invalid):
            return pl.Series(
                values.name, [None] * len(values), dtype=pl.Boolean
            )
    if len(invalid):
        msg = (
            f"{values.name} requires boolean values, got {values.dtype}: "
            f"{invalid.head(5).to_list()!r}"
        )
        raise TypeError(msg)
    return values.cast(pl.Boolean)


def _unsigned_value(name: str, value: object) -> int | None:
    if value is None:
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    if number.is_nan():
        return None
    if (
        not number.is_finite()
        or number < 0
        or number > 2**64 - 1
        or number != number.to_integral_value()
    ):
        msg = f"{name} contains a value not representable as UInt64: {value!r}"
        raise TypeError(msg)
    return int(number)


class SoftBudgetParamConflictError(ValueError):
    """Raised when a row sets both the training and evaluation parameter."""


def combine_soft_budget_params(table: pl.DataFrame) -> pl.DataFrame:
    """
    Replace the two soft-budget parameter columns by `soft_budget_param`.

    A row may set the training or the evaluation soft-budget parameter, never
    both; `soft_budget_param` holds whichever is set.
    """
    train = table["train_soft_budget_param"]
    evaluation = table["eval_soft_budget_param"]
    both_set = train.is_not_null() & evaluation.is_not_null()
    if both_set.any():
        row = table.filter(both_set).row(0, named=True)
        msg = (
            "A row sets both soft-budget parameters, "
            f"train_soft_budget_param={row['train_soft_budget_param']} and "
            f"eval_soft_budget_param={row['eval_soft_budget_param']}; "
            "expected at most one."
        )
        raise SoftBudgetParamConflictError(msg)
    return table.with_columns(
        pl.coalesce("train_soft_budget_param", "eval_soft_budget_param").alias(
            "soft_budget_param"
        )
    ).drop("train_soft_budget_param", "eval_soft_budget_param")


def keep_largest_hard_budget(table: pl.DataFrame) -> pl.DataFrame:
    """For each dataset, keep only the rows of its largest hard budget."""
    return table.filter(
        pl.col("dataset").is_not_null()
        & (
            pl.col("eval_hard_budget")
            == pl.col("eval_hard_budget").max().over("dataset")
        )
    )


def keep_smallest_soft_budget_param(table: pl.DataFrame) -> pl.DataFrame:
    """For each method and dataset, keep only its smallest soft-budget parameter."""
    return table.filter(
        pl.col("afa_method").is_not_null()
        & pl.col("dataset").is_not_null()
        & (
            pl.col("soft_budget_param")
            == pl.col("soft_budget_param").min().over("afa_method", "dataset")
        )
    )


def _probability_matrix(series: pl.Series) -> NDArray[np.float64]:
    vectors: list[NDArray[np.float64]] = []
    for value in cast("list[object]", series.to_list()):
        if not isinstance(value, (list, np.ndarray)):
            msg = (
                f"predicted_probabilities must contain vectors, got {value!r}"
            )
            raise TypeError(msg)
        vector = np.asarray(cast("ArrayLike", value))
        if (
            vector.ndim != 1
            or vector.size < 2
            or (vectors and vector.shape != vectors[0].shape)
        ):
            msg = (
                "predicted_probabilities must contain equally sized class "
                f"vectors with at least two classes, got {value!r}"
            )
            raise ValueError(msg)
        if not np.issubdtype(vector.dtype, np.floating):
            msg = (
                "predicted_probabilities must contain floating values, "
                f"got {value!r} with dtype {vector.dtype}"
            )
            raise TypeError(msg)
        if not np.isfinite(vector).all():
            msg = (
                "predicted_probabilities must contain finite values, "
                f"got {value!r}"
            )
            raise ValueError(msg)
        if ((vector < 0) | (vector > 1)).any():
            msg = f"predicted_probabilities must be in [0, 1], got {value!r}"
            raise ValueError(msg)
        if not np.isclose(vector.sum(), 1.0, atol=1e-6, rtol=1e-5):
            msg = (
                "predicted_probabilities must sum to 1, "
                f"got {value!r} with sum {vector.sum()}"
            )
            raise ValueError(msg)
        vectors.append(vector.astype(np.float64))
    return np.stack(vectors)


def _classification_metrics(group: pl.DataFrame) -> dict[str, float]:
    true_class = group["true_class"].to_numpy().astype(np.int64)
    probabilities = _probability_matrix(group["predicted_probabilities"])
    # Accuracy and F1 decide the most probable class, the lowest on ties.
    predicted_class = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(true_class, predicted_class)),
        "f_score": float(
            f1_score(true_class, predicted_class, average="macro")
        ),
        "auroc": float(
            roc_auc_score(true_class, probabilities[:, 1])
            if probabilities.shape[1] == 2
            # Binary OVR scores avoid sklearn's stricter multiclass
            # sum check; validation uses the evaluator's tolerance.
            else np.mean(
                [
                    cast(
                        "float",
                        roc_auc_score(
                            true_class == class_index,
                            probabilities[:, class_index],
                        ),
                    )
                    for class_index in range(probabilities.shape[1])
                ]
            )
        ),
    }


# A run: one method trained and evaluated with one pair of seeds and budgets
RUN_COLUMNS = [
    "afa_method",
    "dataset",
    "train_seed",
    "eval_seed",
    "eval_hard_budget",
    "soft_budget_param",
]


def metrics_at_stop(table: pl.DataFrame) -> pl.DataFrame:
    """
    Score each run on its rows, which should be its stop actions.

    Columns: the run columns, `accuracy`, `f_score` (macro F1), `auroc`
    and `avg_accumulated_cost`. Class probabilities use label index order.
    """
    return _metrics(table, RUN_COLUMNS, include_cost=True)


def metrics_at_every_step(table: pl.DataFrame) -> pl.DataFrame:
    """Score each run at each time step (`n_selections_performed`)."""
    return _metrics(
        table, [*RUN_COLUMNS, "n_selections_performed"], include_cost=False
    )


def _metrics(
    table: pl.DataFrame, keys: list[str], *, include_cost: bool
) -> pl.DataFrame:
    schema = {name: table.schema[name] for name in keys} | dict.fromkeys(
        ["accuracy", "f_score", "auroc"], pl.Float64
    )
    if include_cost:
        schema["avg_accumulated_cost"] = pl.Float64
    rows = []
    for group in table.sort(keys, nulls_last=True).partition_by(
        keys, maintain_order=True
    ):
        row = group.select(keys).row(0, named=True) | _classification_metrics(
            group
        )
        if include_cost:
            row["avg_accumulated_cost"] = group["accumulated_cost"].mean()
        rows.append(row)
    return pl.DataFrame(rows, schema=schema)


def variance_over_seeds_at_stop(metrics: pl.DataFrame) -> pl.DataFrame:
    """Summarize stop metrics over seeds; undefined AUROCs propagate."""
    return _variance(metrics, every_step=False)


def variance_over_seeds_at_every_step(metrics: pl.DataFrame) -> pl.DataFrame:
    """Summarize per-step metrics over seeds; undefined AUROCs propagate."""
    return _variance(metrics, every_step=True)


def _variance(metrics: pl.DataFrame, *, every_step: bool) -> pl.DataFrame:
    keys = ["afa_method", "dataset", "eval_hard_budget", "soft_budget_param"]
    columns = ["accuracy", "f_score", "auroc"]
    if every_step:
        keys.append("n_selections_performed")
    else:
        columns.append("avg_accumulated_cost")
    return (
        metrics.group_by(keys)
        .agg(
            expression
            for name in columns
            for expression in (
                pl.col(name).mean().alias(f"mean_{name}"),
                pl.col(name).std(ddof=1).alias(f"std_{name}"),
            )
        )
        .sort(keys, nulls_last=True)
    )


# (method, dataset) -> the sets of methods beside which the method is kept on
# that dataset; beside any other set of methods it is excluded
type Exclusions = Mapping[tuple[str, str], frozenset[frozenset[str]]]

EXCLUSION_MAPPING: Exclusions = {}


def apply_exclusions(
    table: pl.DataFrame, exclusions: Exclusions = EXCLUSION_MAPPING
) -> pl.DataFrame:
    """
    Drop each excluded method from a dataset unless allowed by its methods.

    For example, the exclusion

        ("odin_model_based", "diabetes"): frozenset(
            [frozenset(["odin_model_based", "odin_model_free"])]
        )

    keeps odin_model_based on diabetes only when the methods on diabetes are
    exactly odin_model_based and odin_model_free.
    """
    method_dataset_pairs = (
        table.select("afa_method", "dataset").unique().iter_rows()
    )
    for method, dataset in method_dataset_pairs:
        allowed_method_sets = exclusions.get((method, dataset))
        if allowed_method_sets is None:
            continue
        methods_on_dataset = frozenset(
            table.filter(pl.col("dataset") == dataset)["afa_method"].unique()
        )
        if methods_on_dataset not in allowed_method_sets:
            table = table.filter(
                ~(
                    (pl.col("afa_method") == method)
                    & (pl.col("dataset") == dataset)
                ).fill_null(False)
            )
    return table
