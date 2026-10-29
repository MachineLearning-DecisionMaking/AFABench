"""
Transform a current raw evaluation table into a plotting-ready table.

Identity arguments check the table's columns and fill explicitly null values.
The source provenance record is copied to the output's Parquet metadata.
"""

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from afabench.evaluation.provenance import (
    evaluation_table_provenance,
    save_evaluation_table,
)
from afabench.evaluation.schemas import (
    IDENTITY_DTYPES,
    PLOTTING_READY_DTYPES,
    SavedEvaluationSchema,
    identity_column,
)

IDENTITY_ARGUMENTS = {
    "afa_method": "method",
    "dataset": "dataset",
    "initializer": "initializer",
    "train_seed": "train_seed",
    "train_hard_budget": "train_hard_budget",
    "train_soft_budget_param": "train_soft_budget_param",
    "eval_soft_budget_param": "eval_soft_budget_param",
}


class IdentityArgumentMismatchError(ValueError):
    """An identity argument disagrees with the table's column."""


def check_or_fill_identity(
    df: pl.DataFrame, column: str, argument: str | None
) -> pl.DataFrame:
    """Check an identity argument; the string 'null' requests an explicit null."""
    if argument is None:
        return df
    expected = identity_column(
        column, None if argument == "null" else argument, len(df)
    )
    if df[column].null_count() == len(df):
        return df.with_columns(expected)
    disagreeing = (df[column] != expected).fill_null(True)
    if disagreeing.any():
        value = df.filter(disagreeing)[column][0]
        msg = (
            f"Argument --{IDENTITY_ARGUMENTS[column]}={argument} "
            f"disagrees with column {column!r}, which holds {value!r}."
        )
        raise IdentityArgumentMismatchError(msg)
    return df


def transform(
    input_path: Annotated[Path, typer.Option("--input_path")],
    output_path: Annotated[Path, typer.Option("--output_path")],
    method: Annotated[str | None, typer.Option()] = None,
    dataset: Annotated[str | None, typer.Option()] = None,
    initializer: Annotated[str | None, typer.Option()] = None,
    train_seed: Annotated[str | None, typer.Option("--train_seed")] = None,
    train_hard_budget: Annotated[
        str | None, typer.Option("--train_hard_budget")
    ] = None,
    train_soft_budget_param: Annotated[
        str | None, typer.Option("--train_soft_budget_param")
    ] = None,
    eval_soft_budget_param: Annotated[
        str | None, typer.Option("--eval_soft_budget_param")
    ] = None,
) -> None:
    df = SavedEvaluationSchema.validate(pl.read_parquet(input_path))
    arguments = {
        "afa_method": method,
        "dataset": dataset,
        "initializer": initializer,
        "train_seed": train_seed,
        "train_hard_budget": train_hard_budget,
        "train_soft_budget_param": train_soft_budget_param,
        "eval_soft_budget_param": eval_soft_budget_param,
    }
    for column, argument in arguments.items():
        df = check_or_fill_identity(df, column, argument)
    df = df.with_columns(pl.col("step").alias("n_selections_performed"))
    df = (
        df.rename(
            {
                "builtin_predicted_probabilities": "builtin",
                "external_predicted_probabilities": "external",
            }
        )
        .unpivot(
            index=[
                "generation_index",
                "split_index",
                "action_performed",
                "true_class",
                "accumulated_cost",
                "forced_stop",
                "n_selections_performed",
                *IDENTITY_DTYPES,
            ],
            on=["builtin", "external"],
            variable_name="classifier",
            value_name="predicted_probabilities",
        )
        .cast(pl.Schema(PLOTTING_READY_DTYPES))
    )
    save_evaluation_table(
        df, output_path, provenance=evaluation_table_provenance(input_path)
    )


def main() -> None:
    typer.run(transform)


if __name__ == "__main__":
    main()
