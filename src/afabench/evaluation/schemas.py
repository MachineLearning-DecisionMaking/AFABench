"""Polars contracts for evaluation rows and metadata-enriched Parquet results."""

from typing import Annotated, Literal

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series


def validate_episode_log(frame: pl.DataFrame) -> None:
    """Require complete episodes within one artifact, regardless of row order."""
    for column in ("episode_id", "step", "action_performed"):
        if column not in frame.columns:
            msg = f"Missing episode-log column: {column}"
            raise ValueError(msg)
        values = frame[column]
        if (
            not values.dtype.is_integer()
            or values.null_count()
            or (values < 0).any()
        ):
            msg = f"{column} must contain nonnegative integers"
            raise ValueError(msg)
    for episode in frame.sort("step").partition_by(
        "episode_id", maintain_order=True
    ):
        episode_id = episode["episode_id"][0]
        if episode["step"].to_list() != list(range(len(episode))):
            msg = f"Episode {episode_id} has missing or duplicate steps"
            raise ValueError(msg)
        actions = episode["action_performed"]
        if actions[-1] != 0 or (actions[:-1] == 0).any():
            msg = f"Episode {episode_id} must have exactly one terminal stop"
            raise ValueError(msg)


class BatchEvaluationSchema(pa.DataFrameModel):
    """One row per episode and zero-based time step of one evaluated batch."""

    episode_id: Series[pl.Int64] = pa.Field(ge=0)
    step: Series[pl.Int64] = pa.Field(ge=0)
    action_performed: Series[pl.Int64] = pa.Field(ge=0)
    builtin_predicted_probabilities: Series[Annotated[pl.List, pl.Float64]] = (
        pa.Field(nullable=True)
    )
    external_predicted_probabilities: Series[
        Annotated[pl.List, pl.Float64]
    ] = pa.Field(nullable=True)
    true_class: Series[pl.Int64] = pa.Field(ge=0)
    accumulated_cost: Series[pl.Float64] = pa.Field(ge=0)
    forced_stop: Series[pl.Boolean] = pa.Field()

    @pa.dataframe_check
    @classmethod
    def complete_episodes(cls, data: pa.PolarsData) -> bool:
        try:
            validate_episode_log(data.lazyframe.collect())
        except ValueError:
            return False
        return True

    class Config(pa.DataFrameModel.Config):
        strict: bool | Literal["filter"] = True


class EvaluationSchema(BatchEvaluationSchema):
    """Evaluation rows identifying each episode's instance (ADR 0004)."""

    generation_index: Series[pl.Int64] = pa.Field(ge=0)
    split_index: Series[pl.Int64] = pa.Field(ge=0)

    @pa.dataframe_check
    @classmethod
    def one_instance_per_episode(cls, data: pa.PolarsData) -> pl.LazyFrame:
        return (
            data.lazyframe.group_by("episode_id")
            .agg(
                pl.all_horizontal(
                    pl.col("generation_index").n_unique() <= 1,
                    pl.col("split_index").n_unique() <= 1,
                ).alias("valid")
            )
            .select("valid")
        )


IDENTITY_DTYPES = {
    "afa_method": pl.String,
    "dataset": pl.String,
    "dataset_realization_index": pl.UInt64,
    "eval_split": pl.String,
    "initializer": pl.String,
    "train_seed": pl.UInt64,
    "train_hard_budget": pl.Float64,
    "train_soft_budget_param": pl.Float64,
    "eval_seed": pl.UInt64,
    "eval_hard_budget": pl.Float64,
    "eval_soft_budget_param": pl.Float64,
}

PLOTTING_READY_DTYPES = {
    "classifier": pl.String,
    "generation_index": pl.UInt64,
    "split_index": pl.UInt64,
    "action_performed": pl.UInt64,
    "true_class": pl.UInt64,
    "accumulated_cost": pl.Float64,
    "forced_stop": pl.Boolean,
    "n_selections_performed": pl.UInt64,
    "predicted_probabilities": pl.List(pl.Float64),
    **IDENTITY_DTYPES,
}


def identity_column(name: str, value: object, length: int) -> pl.Series:
    """Return a typed identity column; None is an explicit null."""
    return pl.Series(name, [value] * length).cast(IDENTITY_DTYPES[name])


class SavedEvaluationSchema(EvaluationSchema):
    """Evaluation rows with constant, nullable identity columns (ADR 0002)."""

    afa_method: Series[pl.String] = pa.Field(nullable=True)
    dataset: Series[pl.String] = pa.Field(nullable=True)
    dataset_realization_index: Series[pl.UInt64] = pa.Field(nullable=True)
    eval_split: Series[pl.String] = pa.Field(
        nullable=True, isin=["train", "val", "test"]
    )
    initializer: Series[pl.String] = pa.Field()
    train_seed: Series[pl.UInt64] = pa.Field(nullable=True)
    train_hard_budget: Series[pl.Float64] = pa.Field(nullable=True, ge=0)
    train_soft_budget_param: Series[pl.Float64] = pa.Field(nullable=True)
    eval_seed: Series[pl.UInt64] = pa.Field()
    eval_hard_budget: Series[pl.Float64] = pa.Field(nullable=True, ge=0)
    eval_soft_budget_param: Series[pl.Float64] = pa.Field(nullable=True)

    @pa.dataframe_check
    @classmethod
    def constant_identity(cls, data: pa.PolarsData) -> bool:
        return (
            data.lazyframe.select(list(IDENTITY_DTYPES))
            .unique()
            .collect()
            .height
            <= 1
        )
