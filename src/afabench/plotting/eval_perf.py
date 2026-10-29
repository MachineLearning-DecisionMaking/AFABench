"""
Evaluation performance plots of a plotting-ready table.

Plots each method's metric against its hard budget, its number of
selections and, in the soft-budget setting, its accumulated cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

import plotnine as p9
import polars as pl
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_errorbar,
    geom_errorbarh,
    geom_line,
    geom_point,
    geom_ribbon,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    theme,
    theme_bw,
)

from afabench.evaluation.tables import (
    apply_exclusions,
    combine_soft_budget_params,
    keep_largest_hard_budget,
    metrics_at_every_step,
    metrics_at_stop,
    variance_over_seeds_at_every_step,
    variance_over_seeds_at_stop,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from afabench.plotting.config import PlottingDisplayConfig

SUBPLOT_HEIGHT = 2.0
PLOT_FONT_SIZE = 12
SUBPLOT_TITLE_FONT_SIZE = 10

METRIC_COLUMNS = {"accuracy": "accuracy", "f1": "f_score", "auroc": "auroc"}
METRIC_LABELS = {"accuracy": "Accuracy", "f1": "F1", "auroc": "AUROC"}


NON_MYOPIC_METHODS = frozenset(
    {
        "aaco",
        "aaco_nn",
        "jafa",
        "odin_model_based",
        "odin_model_free",
        "ol_with_mask",
        "ol_without_mask",
    }
)


def get_method_color_mapping(
    plotting_config: PlottingDisplayConfig,
) -> dict[str, str]:
    family_colors = plotting_config.method_family_color_schemes[
        plotting_config.active_method_color_scheme
    ]
    return {
        method: family_colors[family]
        for method, family in plotting_config.method_policy_family_mapping.items()
    }


def get_dataset_name_mapping_including_metric(
    plotting_config: PlottingDisplayConfig,
) -> dict[str, str]:
    return {
        dataset: f"{name} ({METRIC_LABELS[plotting_config.dataset_metrics.get(dataset, plotting_config.default_metric)]})"
        for dataset, name in plotting_config.dataset_name_mapping.items()
    }


def get_plot(
    df: pl.DataFrame,
    x_col: str,
    x_label: str,
    plotting_config: PlottingDisplayConfig,
    x_error_min: str | None = None,
    x_error_max: str | None = None,
    *,
    use_line: bool = True,
    per_dataset: bool = False,
    method_order: list[str] | None = None,
    figure_width: float | None = None,
    figure_height: float | None = None,
) -> p9.ggplot:
    """
    Create a plot with metrics vs cost/budget.

    Args:
        df: Input dataframe
        x_col: Column name for x-axis
        x_label: Label for x-axis
        x_error_min: Column name for x-axis error minimum
        x_error_max: Column name for x-axis error maximum
        use_line: Whether to use line plot (True) or point plot (False)
        method_order: List of method names in desired legend order
        figure_width: Figure width in inches (default: PLOT_WIDTH)
        figure_height: Figure height in inches (default: PLOT_HEIGHT)
    """
    method_color_mapping = get_method_color_mapping(plotting_config)
    dataset_name_mapping_including_metric = (
        get_dataset_name_mapping_including_metric(plotting_config)
    )

    numeric_columns = [x_col, "mean_metric", "low_metric", "high_metric"]
    if x_error_min and x_error_max:
        numeric_columns.extend([x_error_min, x_error_max])
    # Inferred all-null columns need a numeric type for plotnine's scales.
    processed_df = df.with_columns(
        pl.col(column).cast(pl.Float64)
        for column in numeric_columns
        if df.schema[column] == pl.Null
    ).with_columns(
        pl.col("dataset").replace(dataset_name_mapping_including_metric),
        pl.when(pl.col("afa_method").is_in(NON_MYOPIC_METHODS))
        .then(pl.lit("Non-myopic"))
        .otherwise(pl.lit("Myopic"))
        .alias("policy_type"),
        pl.col("afa_method").replace(plotting_config.method_name_mapping),
    )

    # Get method order for legend
    if method_order is None:
        method_order = list(plotting_config.method_name_mapping.keys())

    # Get display names in order, filtering to only methods in the data
    available_methods = processed_df["afa_method"].unique().to_list()
    ordered_display_names = [
        plotting_config.method_name_mapping.get(orig, orig)
        for orig in method_order
        if plotting_config.method_name_mapping.get(orig, orig)
        in available_methods
    ]
    ordered_color_values = {
        plotting_config.method_name_mapping.get(
            orig, orig
        ): method_color_mapping[orig]
        for orig in method_order
        if plotting_config.method_name_mapping.get(orig, orig)
        in available_methods
    }
    # Use provided dimensions or defaults
    if figure_width is None:
        figure_width = plotting_config.plot_width
    if figure_height is None:
        figure_height = plotting_config.plot_height

    plot = (
        ggplot(
            processed_df,
            aes(
                x=x_col,
                y="mean_metric",
                color="afa_method",
                fill="afa_method",
            ),
        )
        + (
            facet_wrap(["classifier", "metric"], scales="free_y", ncol=3)
            if per_dataset
            else facet_wrap("dataset", scales="free", ncol=4)
        )
        + labs(
            color="Policy",
            fill="Policy",
            x=x_label,
            y="Metric",
            title=(
                plotting_config.dataset_name_mapping.get(
                    str(df["dataset"][0]), str(df["dataset"][0])
                )
                if per_dataset
                else None
            ),
        )
        + theme_bw()
        + theme(
            figure_size=(figure_width, figure_height),
            text=element_text(
                family=plotting_config.plot_font_family,
                size=PLOT_FONT_SIZE,
            ),
            strip_text=element_text(size=SUBPLOT_TITLE_FONT_SIZE),
        )
        + scale_color_manual(
            values=ordered_color_values,
            breaks=ordered_display_names,
        )
        + scale_fill_manual(
            values=ordered_color_values,
            breaks=ordered_display_names,
        )
    )

    if plotting_config.caption is not None:
        plot += labs(caption=plotting_config.caption)

    if use_line:
        plot += geom_point()
        plot += geom_ribbon(
            aes(ymin="low_metric", ymax="high_metric"),
            alpha=0.1,
            size=0.0,
        )
        plot += geom_line(aes(linetype="policy_type"))
        plot += scale_linetype_manual(
            name="Policy type",
            values={"Myopic": "solid", "Non-myopic": "dotted"},
            breaks=["Myopic", "Non-myopic"],
        )
    else:
        plot += geom_point()
        plot += geom_errorbar(
            aes(ymin="low_metric", ymax="high_metric"), width=0
        )
        if x_error_min and x_error_max:
            plot += geom_errorbarh(
                aes(xmin=x_error_min, xmax=x_error_max),
                height=0,
            )

    return plot


def get_per_dataset_plot(
    df: pl.DataFrame,
    x_col: str,
    x_label: str,
    plotting_config: PlottingDisplayConfig,
    *,
    use_line: bool = True,
    x_error_min: str | None = None,
    x_error_max: str | None = None,
) -> p9.ggplot:
    """Facet one dataset's classifier-separated seed summaries by metric."""
    metric_frames = []
    for metric, column in METRIC_COLUMNS.items():
        frame = df.with_columns(
            pl.lit(
                "Macro F1" if metric == "f1" else METRIC_LABELS[metric]
            ).alias("metric"),
            pl.col(f"mean_{column}").alias("mean_metric"),
            pl.col(f"std_{column}").alias("std_metric"),
        )
        metric_frames.append(frame)
    long = pl.concat(metric_frames).with_columns(
        (pl.col("mean_metric") - pl.col("std_metric")).alias("low_metric"),
        (pl.col("mean_metric") + pl.col("std_metric")).alias("high_metric"),
        pl.col("classifier")
        .replace(
            {
                "external": "External classifier",
                "builtin": "Built-in classifier",
            }
        )
        .cast(pl.Enum(["External classifier", "Built-in classifier"])),
        pl.col("metric").cast(pl.Enum(["Accuracy", "Macro F1", "AUROC"])),
    )
    return get_plot(
        long,
        x_col,
        x_label,
        plotting_config,
        x_error_min=x_error_min,
        x_error_max=x_error_max,
        use_line=use_line,
        per_dataset=True,
        figure_width=plotting_config.plot_width,
        figure_height=SUBPLOT_HEIGHT * df["classifier"].n_unique(),
    )


def get_normal_hard_budget_plot(
    df: pl.DataFrame,
    plotting_config: PlottingDisplayConfig,
    method_order: list[str] | None = None,
    figure_width: float | None = None,
    figure_height: float | None = None,
) -> p9.ggplot:
    return get_plot(
        df,
        x_col="eval_hard_budget",
        x_label="Hard budget",
        plotting_config=plotting_config,
        use_line=True,
        method_order=method_order,
        figure_width=figure_width,
        figure_height=figure_height,
    )


def get_traj_hard_budget_plot(
    df: pl.DataFrame,
    plotting_config: PlottingDisplayConfig,
    method_order: list[str] | None = None,
    figure_width: float | None = None,
    figure_height: float | None = None,
) -> p9.ggplot:
    return get_plot(
        df,
        x_col="n_selections_performed",
        x_label="Number of selections",
        plotting_config=plotting_config,
        use_line=True,
        method_order=method_order,
        figure_width=figure_width,
        figure_height=figure_height,
    )


def get_soft_budget_plot(
    df: pl.DataFrame,
    mode: str,
    plotting_config: PlottingDisplayConfig,
    method_order: list[str] | None = None,
    figure_width: float | None = None,
    figure_height: float | None = None,
) -> p9.ggplot:
    if mode == "2d_errors":
        return get_plot(
            df,
            x_col="mean_avg_accumulated_cost",
            x_label="Average cost accumulated per episode",
            plotting_config=plotting_config,
            x_error_min="low_avg_accumulated_cost",
            x_error_max="high_avg_accumulated_cost",
            use_line=False,
            method_order=method_order,
            figure_width=figure_width,
            figure_height=figure_height,
        )
    if mode == "lines":
        return get_plot(
            df,
            x_col="mean_avg_accumulated_cost",
            x_label="Average cost accumulated per episode",
            plotting_config=plotting_config,
            x_error_min=None,
            x_error_max=None,
            use_line=True,
            method_order=method_order,
            figure_width=figure_width,
            figure_height=figure_height,
        )
    msg = f"Expected either '2d_errors' or 'lines' as mode, got '{mode}'"
    raise ValueError(msg)


def calculate_figure_dimensions(
    num_datasets: int,
    plot_width: float,
    ncol: int = 4,
    subplot_height: float = 3.0,
) -> tuple[float, float]:
    """
    Calculate figure dimensions based on number of subplots.

    Args:
        num_datasets: Number of datasets (subplots)
        ncol: Number of columns in facet layout
        subplot_height: Height of each individual subplot in inches

    Returns:
        Tuple of (width, height) for the figure
    """
    num_rows = (num_datasets + ncol - 1) // ncol
    figure_width = plot_width
    figure_height = subplot_height * num_rows
    return figure_width, figure_height


def add_metric_column(
    df: pl.DataFrame, plotting_config: PlottingDisplayConfig
) -> pl.DataFrame:
    metrics = pl.col("dataset").replace_strict(
        plotting_config.dataset_metrics,
        default=plotting_config.default_metric,
    )
    return df.with_columns(
        pl.coalesce(
            [
                pl.when(metrics == metric).then(pl.col(f"{stat}_{column}"))
                for metric, column in METRIC_COLUMNS.items()
            ]
        ).alias(f"{stat}_metric")
        for stat in ("mean", "std")
    )


def process_df_only_stop_action_and_valid_prediction(
    df: pl.DataFrame,
    plotting_config: PlottingDisplayConfig,
) -> pl.DataFrame | None:
    df_only_stop_action = df.filter(
        (pl.col("action_performed") == 0)
        & pl.col("predicted_probabilities").is_not_null()
    )

    if df_only_stop_action.is_empty():
        return None

    metric_df_only_stop_action = metrics_at_stop(df_only_stop_action)

    # Variance of metrics across seeds
    var_metric_df_only_stop_action = variance_over_seeds_at_stop(
        metric_df_only_stop_action
    )

    # Datasets use different metrics
    var_metric_df_only_stop_action = add_metric_column(
        var_metric_df_only_stop_action, plotting_config
    )

    # Add "low" and "high" versions of metrics to enable plotting of ranges
    var_metric_df_only_stop_action = (
        var_metric_df_only_stop_action.with_columns(
            (pl.col("mean_metric") - pl.col("std_metric")).alias("low_metric"),
            (pl.col("mean_metric") + pl.col("std_metric")).alias(
                "high_metric"
            ),
            (
                pl.col("mean_avg_accumulated_cost")
                - pl.col("std_avg_accumulated_cost")
            ).alias("low_avg_accumulated_cost"),
            (
                pl.col("mean_avg_accumulated_cost")
                + pl.col("std_avg_accumulated_cost")
            ).alias("high_avg_accumulated_cost"),
        )
    )

    return var_metric_df_only_stop_action


def process_df_every_action(
    df: pl.DataFrame, plotting_config: PlottingDisplayConfig
) -> pl.DataFrame | None:
    # Just like in process_df_only_stop_action, filter out null predictions
    df = df.filter(pl.col("predicted_probabilities").is_not_null())

    if df.is_empty():
        return None

    # When considering performance up to some budget, we only look at the case when the largest budget is used
    df = keep_largest_hard_budget(df)

    metric_df_every_action = metrics_at_every_step(df)

    # Variance of metrics across seeds
    var_metric_df_every_action = variance_over_seeds_at_every_step(
        metric_df_every_action
    )

    # Datasets use different metrics
    var_metric_df_every_action = add_metric_column(
        var_metric_df_every_action, plotting_config
    )

    # Add "low" and "high" versions of metrics to enable plotting of ranges
    var_metric_df_every_action = var_metric_df_every_action.with_columns(
        (pl.col("mean_metric") - pl.col("std_metric")).alias("low_metric"),
        (pl.col("mean_metric") + pl.col("std_metric")).alias("high_metric"),
    )

    return var_metric_df_every_action


@final
class EvaluationPlotter:
    """Prepare a plotting-ready table and save its performance plots."""

    output_folder: Path
    plotting_config: PlottingDisplayConfig
    df_stop_action: pl.DataFrame | None
    df_traj: pl.DataFrame | None

    def __init__(
        self,
        table: pl.DataFrame,
        output_folder: Path,
        plotting_config: PlottingDisplayConfig,
        formats: Sequence[str] = ("pdf",),
    ) -> None:
        """
        Prepare the stop-action and per-time-step metrics of `table`.

        Args:
            table: A plotting-ready table, from `load_plotting_ready_table`
            output_folder: Root output directory for plots
            formats: Output formats (e.g. ("pdf", "svg")). Default: ("pdf",)
        """
        self.output_folder = output_folder
        self.plotting_config = plotting_config
        self.formats = formats
        table = combine_soft_budget_params(table)
        self.df_stop_action = process_df_only_stop_action_and_valid_prediction(
            table, plotting_config
        )
        self.df_traj = process_df_every_action(table, plotting_config)

    def produce_stop_action_plots(
        self,
        dataset_set: set[str],
        subfolder: Path,
    ) -> None:
        """Generate stop action plots (hard and soft budget)."""
        if self.df_stop_action is None:
            return

        df_stop_action_filtered = self.df_stop_action.filter(
            pl.col("dataset").is_in(dataset_set)
        )

        df_stop_action_hard_budget = df_stop_action_filtered.filter(
            pl.col("eval_hard_budget").is_not_null()
        )
        df_stop_action_hard_budget = apply_exclusions(
            df_stop_action_hard_budget
        )
        if not df_stop_action_hard_budget.is_empty():
            # Calculate figure dimensions based on number of unique datasets
            num_datasets = df_stop_action_hard_budget["dataset"].n_unique()
            fig_width, fig_height = calculate_figure_dimensions(
                num_datasets,
                plot_width=self.plotting_config.plot_width,
                subplot_height=SUBPLOT_HEIGHT,
            )
            normal_hard_budget_plot = get_normal_hard_budget_plot(
                df_stop_action_hard_budget,
                plotting_config=self.plotting_config,
                figure_width=fig_width,
                figure_height=fig_height,
            )
            for fmt in self.formats:
                self.save_plot(
                    normal_hard_budget_plot,
                    subfolder / f"hard_budget_normal.{fmt}",
                    fig_width,
                    fig_height,
                )
        df_stop_action_soft_budget = df_stop_action_filtered.filter(
            pl.col("soft_budget_param").is_not_null()
        )
        df_stop_action_soft_budget = apply_exclusions(
            df_stop_action_soft_budget
        )
        if not df_stop_action_soft_budget.is_empty():
            # Calculate figure dimensions based on number of unique datasets
            num_datasets = df_stop_action_soft_budget["dataset"].n_unique()
            fig_width, fig_height = calculate_figure_dimensions(
                num_datasets,
                plot_width=self.plotting_config.plot_width,
                subplot_height=SUBPLOT_HEIGHT,
            )
            soft_budget_plot_2d_errors = get_soft_budget_plot(
                df_stop_action_soft_budget,
                mode="2d_errors",
                plotting_config=self.plotting_config,
                figure_width=fig_width,
                figure_height=fig_height,
            )
            soft_budget_plot_lines = get_soft_budget_plot(
                df_stop_action_soft_budget,
                mode="lines",
                plotting_config=self.plotting_config,
                figure_width=fig_width,
                figure_height=fig_height,
            )
            for fmt in self.formats:
                self.save_plot(
                    soft_budget_plot_2d_errors,
                    subfolder / f"soft_budget_2d_errors.{fmt}",
                    fig_width,
                    fig_height,
                )
                self.save_plot(
                    soft_budget_plot_lines,
                    subfolder / f"soft_budget_lines.{fmt}",
                    fig_width,
                    fig_height,
                )

    def produce_trajectory_plots(
        self,
        dataset_set: set[str],
        subfolder: Path,
    ) -> None:
        """Generate trajectory plots (hard budget)."""
        if self.df_traj is None:
            return

        df_traj_filtered = self.df_traj.filter(
            pl.col("dataset").is_in(dataset_set)
        )
        df_traj_hard_budget = df_traj_filtered.filter(
            pl.col("eval_hard_budget").is_not_null()
        )
        df_traj_hard_budget = apply_exclusions(df_traj_hard_budget)
        if not df_traj_hard_budget.is_empty():
            # Calculate figure dimensions based on number of unique datasets
            num_datasets = df_traj_hard_budget["dataset"].n_unique()
            fig_width, fig_height = calculate_figure_dimensions(
                num_datasets,
                plot_width=self.plotting_config.plot_width,
                subplot_height=SUBPLOT_HEIGHT,
            )
            traj_hard_budget_plot = get_traj_hard_budget_plot(
                df_traj_hard_budget,
                plotting_config=self.plotting_config,
                figure_width=fig_width,
                figure_height=fig_height,
            )
            for fmt in self.formats:
                self.save_plot(
                    traj_hard_budget_plot,
                    subfolder / f"hard_budget_traj.{fmt}",
                    fig_width,
                    fig_height,
                )

    def generate_per_dataset_plots(
        self, table: pl.DataFrame, output_folder: Path
    ) -> None:
        """
        Save additional figures from an unsplit, single-method-set table.

        Classifiers are aggregated separately; missing predictions never
        contribute a built-in row. The output root must be independent of
        the usual plots so workflow directory outputs cannot overwrite it.
        """
        usual_root = self.output_folder.resolve()
        additional_root = output_folder.resolve()
        if additional_root.is_relative_to(
            usual_root
        ) or usual_root.is_relative_to(additional_root):
            msg = (
                f"Per-dataset output {output_folder!s} would overlap "
                f"usual output {self.output_folder!s}"
            )
            raise ValueError(msg)
        table = combine_soft_budget_params(table)
        stop_frames: list[pl.DataFrame] = []
        trajectory_frames: list[pl.DataFrame] = []
        for classifier in ("external", "builtin"):
            selected = table.filter(pl.col("classifier") == classifier)
            stop = process_df_only_stop_action_and_valid_prediction(
                selected, self.plotting_config
            )
            trajectory = process_df_every_action(
                selected, self.plotting_config
            )
            if stop is not None:
                stop_frames.append(
                    stop.with_columns(pl.lit(classifier).alias("classifier"))
                )
            if trajectory is not None:
                trajectory_frames.append(
                    trajectory.with_columns(
                        pl.lit(classifier).alias("classifier")
                    )
                )

        for frames, trajectory in (
            (stop_frames, False),
            (trajectory_frames, True),
        ):
            if not frames:
                continue
            self._save_per_dataset_variants(
                pl.concat(frames),
                output_folder,
                trajectory=trajectory,
            )

    def _save_per_dataset_variants(
        self,
        summaries: pl.DataFrame,
        output_folder: Path,
        *,
        trajectory: bool,
    ) -> None:
        for (dataset,), dataset_frame in summaries.group_by(
            "dataset", maintain_order=True
        ):
            hard = apply_exclusions(
                dataset_frame.filter(pl.col("eval_hard_budget").is_not_null())
            )
            soft = apply_exclusions(
                dataset_frame.filter(pl.col("soft_budget_param").is_not_null())
            )
            variants = [
                (
                    "hard_budget_traj" if trajectory else "hard_budget_normal",
                    hard,
                    "n_selections_performed"
                    if trajectory
                    else "eval_hard_budget",
                    "Number of selections" if trajectory else "Hard budget",
                    True,
                )
            ]
            if not trajectory:
                variants.extend(
                    [
                        (
                            "soft_budget_2d_errors",
                            soft,
                            "mean_avg_accumulated_cost",
                            "Average cost accumulated per episode",
                            False,
                        ),
                        (
                            "soft_budget_lines",
                            soft,
                            "mean_avg_accumulated_cost",
                            "Average cost accumulated per episode",
                            True,
                        ),
                    ]
                )
            for name, frame, x_col, x_label, use_line in variants:
                if frame.is_empty():
                    continue
                plot = get_per_dataset_plot(
                    frame,
                    x_col,
                    x_label,
                    self.plotting_config,
                    use_line=use_line,
                    x_error_min=(
                        "low_avg_accumulated_cost" if not use_line else None
                    ),
                    x_error_max=(
                        "high_avg_accumulated_cost" if not use_line else None
                    ),
                )
                height = SUBPLOT_HEIGHT * frame["classifier"].n_unique()
                for fmt in self.formats:
                    self.save_plot(
                        plot,
                        output_folder / f"dataset-{dataset}" / f"{name}.{fmt}",
                        self.plotting_config.plot_width,
                        height,
                    )

    @staticmethod
    def save_plot(
        plot: p9.ggplot, path: Path, width: float, height: float
    ) -> None:
        """Save a plot, creating its folder on the first save into it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        plot.save(path, width=width, height=height)

    def generate_all_plots(
        self, dataset_sets: Mapping[str, Sequence[str]]
    ) -> None:
        """
        Generate all plots for each dataset set.

        Each dataset set gets a `dataset_set-<name>` subfolder, created only
        if the set has evaluation results to plot.
        """
        for dataset_set_name, dataset_set in dataset_sets.items():
            subfolder = self.output_folder / f"dataset_set-{dataset_set_name}"
            self.produce_stop_action_plots(set(dataset_set), subfolder)
            self.produce_trajectory_plots(set(dataset_set), subfolder)
