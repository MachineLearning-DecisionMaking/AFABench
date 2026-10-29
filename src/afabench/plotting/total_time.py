"""
Plots of the time each method's jobs take per pipeline stage.

The input is a job duration table (`docs/reference/job_records.md`). Only
completed jobs count, so failed and timed-out attempts are left out. A job
belongs to a method if its record sits in that method's folders of the
selected initializer and evaluation split: its pretraining (through the
pretrained model the method trains from), its method-specific classifier
training, its training, its evaluation and its transformation. Dataset
generation and shared classifiers belong to no method.

The averaged plot shows each stage's mean job duration over the datasets
every method has jobs on; the per-dataset plot stacks every job, so it
shows each stage's total.
"""

from collections.abc import Sequence
from pathlib import Path

import plotnine as p9
import polars as pl
from plotnine import (
    aes,
    coord_flip,
    element_text,
    facet_wrap,
    geom_bar,
    ggplot,
    labs,
    scale_fill_brewer,
    scale_x_discrete,
    theme,
)

from afabench.plotting.config import PlottingDisplayConfig

PLOT_FONT_SIZE = 12

# Pipeline stages a method's jobs run in, in pipeline order
STAGE_LABELS = {
    "classifier_training": "Classifier training",
    "pretraining": "Pretraining",
    "training": "Training",
    "evaluation": "Evaluation",
    "transformation": "Transformation",
}


def method_job_durations(
    table: pl.DataFrame,
    *,
    methods: list[str],
    pretrained_models: dict[str, str],
    initializer_tag: str,
    eval_dataset_split: str,
) -> pl.DataFrame:
    """
    Return the job duration of each completed job of each method.

    `pretrained_models` maps a method to the pretrained model it trains
    from. Columns: afa_method, dataset, stage, time (seconds); a pretraining
    job shared by several methods is a row of each.
    """
    completed = table.filter(pl.col("exit_status") == "completed")
    split_tag = f"eval_split-{eval_dataset_split}/{initializer_tag}"
    method_jobs = []
    for method in methods:
        folders = {
            "classifier_training": f"trained_classifiers/{initializer_tag}/method-{method}+",
            "training": f"trained_methods/{initializer_tag}/{method}/",
            "evaluation": f"eval_results/{split_tag}/{method}/",
            "transformation": f"eval_results_transformed/{split_tag}/{method}/",
        }
        if method in pretrained_models:
            folders["pretraining"] = (
                f"pretrained_models/{initializer_tag}/"
                f"{pretrained_models[method]}/"
            )
        for stage, folder in folders.items():
            jobs = completed.filter(
                (pl.col("stage") == stage)
                & pl.col("job_record_path").str.starts_with(folder)
            )
            method_jobs.append(
                jobs.select(
                    pl.lit(method).alias("afa_method"),
                    pl.col("dataset_key").alias("dataset"),
                    pl.lit(stage).alias("stage"),
                    pl.col("job_duration_seconds").alias("time"),
                )
            )
    return pl.concat(method_jobs)


def common_plot_operations(
    p: p9.ggplot, plotting_config: PlottingDisplayConfig
) -> p9.ggplot:
    return (
        p
        + coord_flip()
        + labs(x="Policy", y="Time (s)", fill="Stage")
        + theme(
            text=element_text(
                family=plotting_config.plot_font_family,
                size=PLOT_FONT_SIZE,
            )
        )
        + scale_x_discrete()
        + scale_fill_brewer(
            type="qual",
            palette=plotting_config.color_palette_name,
            labels=STAGE_LABELS,
            breaks=list(STAGE_LABELS),
        )
    )


def filter_common_datasets(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter dataframe to only include datasets present in all methods.

    This ensures that the averaging plot is fair - methods that only train on
    a subset of datasets won't be penalized by missing data from larger datasets.
    """
    common_datasets = (
        df.group_by("dataset")
        .agg(pl.col("afa_method").n_unique().alias("methods"))
        .filter(pl.col("methods") == df["afa_method"].n_unique())
    )
    return df.filter(
        pl.col("dataset").is_in(
            common_datasets["dataset"].implode(), nulls_equal=True
        )
    )


def get_plots(
    df: pl.DataFrame, plotting_config: PlottingDisplayConfig
) -> tuple[p9.ggplot, p9.ggplot]:
    # Apply name transforms
    df = df.with_columns(
        pl.col("dataset").replace(plotting_config.dataset_name_mapping),
        pl.col("afa_method").replace(plotting_config.method_name_mapping),
    )

    # For averaging plot, only use datasets present in all methods
    df_common = filter_common_datasets(df)

    # Get display method order based on METHOD_NAME_MAPPING order.
    # After name transformation, values are already display names.
    method_order = [
        plotting_config.method_name_mapping.get(m, m)
        for m in plotting_config.method_name_mapping
    ]
    # Reverse the order to match the desired display.
    method_order.reverse()

    # Filter to only mapped methods present in the data.
    available_methods = set(df["afa_method"].unique().to_list())
    method_order_filtered = [m for m in method_order if m in available_methods]
    # Include methods that are present in data but not in METHOD_NAME_MAPPING.
    # This keeps new/experimental methods from becoming nulls in the cast.
    unknown_methods = sorted(available_methods - set(method_order_filtered))
    method_order_filtered.extend(unknown_methods)

    # Cast to categorical with the correct order. Stages stack in reverse
    # pipeline order, the last stage at the bottom.
    dtypes = pl.Schema(
        {
            "afa_method": pl.Enum(method_order_filtered),
            "stage": pl.Enum(list(reversed(STAGE_LABELS))),
        }
    )
    df_common = df_common.cast(dtypes)
    df = df.cast(dtypes)

    # One plot averaged over datasets (using only common datasets)
    averaged_plot = ggplot(
        df_common.group_by(["afa_method", "stage"])
        .agg(pl.col("time").mean())
        .sort(["afa_method", "stage"])
        .to_pandas()
    ) + geom_bar(
        aes(x="afa_method", y="time", fill="stage"),
        stat="identity",
    )
    averaged_plot = common_plot_operations(averaged_plot, plotting_config)

    # Another one faceted over datasets (showing all datasets)
    dataset_plot = (
        ggplot(df.to_pandas())
        + geom_bar(
            aes(x="afa_method", y="time", fill="stage"), stat="identity"
        )
        + facet_wrap("dataset", nrow=2, scales="free_x")
    )
    dataset_plot = common_plot_operations(dataset_plot, plotting_config)
    return averaged_plot, dataset_plot


def save_time_plots(
    durations: pl.DataFrame,
    output_folder: Path,
    plotting_config: PlottingDisplayConfig,
    formats: Sequence[str] = ("pdf",),
) -> None:
    """Save the averaged and per-dataset plots of `method_job_durations`."""
    averaged_plot, dataset_plot = get_plots(
        df=durations, plotting_config=plotting_config
    )

    output_folder.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        averaged_plot.save(
            output_folder / f"average_time.{fmt}",
            width=10,
            height=3,
            verbose=False,
        )
        dataset_plot.save(
            output_folder / f"dataset_time.{fmt}",
            width=20,
            height=5,
            verbose=False,
        )
