"""Plot dataset-aware missing-training-data summary artifacts."""

from __future__ import annotations

import logging
import math
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotnine as p9
from matplotlib import font_manager
from matplotlib.patches import Patch
from omegaconf import OmegaConf

from afabench.plotting.config import (
    PlotMissingDataConfig,
    PlottingDisplayConfig,
)
from scripts.plotting.plot_eval_perf import (
    calculate_figure_dimensions,
    get_method_color_mapping,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

log = logging.getLogger(__name__)

PLOT_FONT_SIZE = 12
SUBPLOT_TITLE_FONT_SIZE = 10

MECHANISM_DISPLAY = {
    "mcar": "MCAR",
    "mar": "MAR",
    "mnar_logistic": "MNAR (logistic)",
    "mnar_self": "MNAR (self-masking)",
    "native": "Native",
}
MECHANISM_ORDER = list(MECHANISM_DISPLAY)
STRATEGY_ORDER = [
    "restricted",
    "mean_fill",
    "pvae_label_conditioned",
    "pvae_label_free",
    "pvae_stepwise",
    "pvae_oracle",
    "pvae_multi_draw",
    "true_completion",
    "zero_fill",
    "mice_fill",
    "mice_label_fill",
    "knn_fill",
]
STRATEGY_DISPLAY = {
    "complete": "Complete data",
    "restricted": "Restricted-action training",
    "mean_fill": "Mean completion",
    "pvae_label_conditioned": "Generative restoration",
    "pvae_label_free": "PVAE (label-free)",
    "pvae_stepwise": "Stepwise (PVAE)",
    "pvae_oracle": "PVAE (oracle)",
    "pvae_multi_draw": "PVAE (5 draws)",
    "true_completion": "True completion",
    "zero_fill": "Zero fill",
    "mice_fill": "MICE",
    "mice_label_fill": "MICE, Label-Conditioned",
    "knn_fill": "kNN",
}
# Distinct markers distinguish policy-state variants within each family.
METHOD_SHAPES = {
    "aaco": "o",
    "dime": "^",
    "gdfs": "h",
    "jafa": "<",
    "jafa_full_state": ">",
    "odin_model_based": "d",
    "odin_model_free": "8",
    "odin_model_free_full_state": "p",
    "ol_with_mask": "v",
    "ol_full_state": "P",
    "ol_without_mask": "X",
    "random_dummy": "*",
}
METHOD_LINETYPES = dict.fromkeys(METHOD_SHAPES, "solid")
METHOD_LINETYPES["random_dummy"] = "dotted"

INSTANCE_COLUMNS = {
    "dataset",
    "method",
    "mechanism",
    "p",
    "strategy",
    "instance",
    "strategy_display_name",
}
SUMMARY_COLUMNS = {
    "dataset",
    "method",
    "mechanism",
    "p",
    "strategy",
    "strategy_display_name",
    "accuracy_mean",
    "accuracy_sem",
    "f_score_mean",
    "f_score_sem",
    "accuracy_gap_to_complete_mean",
    "accuracy_gap_to_complete_sem",
    "f_score_gap_to_complete_mean",
    "f_score_gap_to_complete_sem",
}
ACTION_COLUMNS = INSTANCE_COLUMNS | {
    "selection",
    "acquisitions_per_sample",
}
RESTORATION_COLUMNS = {
    "dataset",
    "mechanism",
    "p",
    "instance",
    "strategy",
    "strategy_display_name",
    "split",
    "imputation_rmse",
}


def _read_csv(
    path: Path,
    required_columns: set[str],
    *,
    optional: bool = False,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        if optional:
            return pd.DataFrame(columns=pd.Index(sorted(required_columns)))
        raise
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        msg = f"{path} is missing required columns: {missing_columns}"
        raise ValueError(msg)
    return frame


def _ordered_present(
    values: Sequence[str], preferred_order: Sequence[str]
) -> list[str]:
    present = set(values)
    ordered = [value for value in preferred_order if value in present]
    return ordered + sorted(present - set(ordered))


def _strategy_labels(strategies: Sequence[str]) -> dict[str, str]:
    return {
        strategy: STRATEGY_DISPLAY.get(strategy, strategy)
        for strategy in strategies
    }


def _with_available_font(
    plotting_config: PlottingDisplayConfig,
) -> PlottingDisplayConfig:
    try:
        font_manager.findfont(
            plotting_config.plot_font_family,
            fallback_to_default=False,
        )
    except ValueError:
        fallback = "DejaVu Serif"
        log.warning(
            "Font %r is unavailable; using %r",
            plotting_config.plot_font_family,
            fallback,
        )
        return replace(plotting_config, plot_font_family=fallback)
    return plotting_config


def _dataset_display(
    dataset: str, plotting_config: PlottingDisplayConfig
) -> str:
    return plotting_config.dataset_name_mapping.get(dataset, dataset)


def _prepare_performance_frame(
    summary: pd.DataFrame,
    dataset: str,
    mechanism: str,
    metric: str,
    plotting_config: PlottingDisplayConfig,
) -> tuple[pd.DataFrame, list[str]]:
    mean_column = f"{metric}_mean"
    sem_column = f"{metric}_sem"
    frame = summary.loc[
        (summary["dataset"] == dataset)
        & (summary["mechanism"] == mechanism)
        & (summary["strategy"] != "complete")
    ].copy()
    if frame.empty:
        return frame, []

    frame[sem_column] = frame[sem_column].fillna(0.0)
    frame["mean_metric"] = frame[mean_column]
    frame["low_metric"] = frame[mean_column] - frame[sem_column]
    frame["high_metric"] = frame[mean_column] + frame[sem_column]
    if "gap_to_complete" not in metric:
        frame["low_metric"] = frame["low_metric"].clip(lower=0.0)
        frame["high_metric"] = frame["high_metric"].clip(upper=1.0)
    else:
        frame["low_metric"] = frame["low_metric"].clip(lower=-1.0)
        frame["high_metric"] = frame["high_metric"].clip(upper=1.0)

    strategies = _ordered_present(
        frame["strategy"].astype(str).tolist(), STRATEGY_ORDER
    )
    labels = _strategy_labels(strategies)
    frame["training_completion"] = pd.Categorical(
        frame["strategy"].map(labels),
        categories=[labels[strategy] for strategy in strategies],
        ordered=True,
    )
    frame["policy"] = frame["method"].map(
        lambda value: plotting_config.method_name_mapping.get(value, value)
    )
    return frame, strategies


def _performance_plot(
    frame: pd.DataFrame,
    strategies: Sequence[str],
    dataset: str,
    mechanism: str,
    y_label: str,
    plotting_config: PlottingDisplayConfig,
) -> tuple[p9.ggplot, float, float]:
    ncol = min(4, len(strategies))
    width, height = calculate_figure_dimensions(
        len(strategies),
        plot_width=plotting_config.plot_width,
        ncol=ncol,
        subplot_height=2.8,
    )
    display = plotting_config.method_name_mapping
    color_mapping = {
        display.get(method, method): color
        for method, color in get_method_color_mapping(plotting_config).items()
    }
    shape_mapping = {
        display.get(method, method): shape
        for method, shape in METHOD_SHAPES.items()
    }
    linetype_mapping = {
        display.get(method, method): linetype
        for method, linetype in METHOD_LINETYPES.items()
    }
    probabilities = sorted(frame["p"].unique())
    plot = (
        p9.ggplot(
            frame,
            p9.aes(
                x="p",
                y="mean_metric",
                color="policy",
                fill="policy",
                shape="policy",
                linetype="policy",
                group="policy",
            ),
        )
        + p9.geom_point(size=2.2)
        + p9.facet_wrap("training_completion", ncol=ncol, scales="fixed")
        + p9.scale_color_manual(values=color_mapping)
        + p9.scale_fill_manual(values=color_mapping)
        + p9.scale_shape_manual(values=shape_mapping)
        + p9.scale_linetype_manual(values=linetype_mapping)
        + p9.scale_x_continuous(
            breaks=probabilities,
            labels=[f"{value:g}" for value in probabilities],
            limits=(0.0, 1.0),
        )
        + p9.labs(
            title=(
                f"{_dataset_display(dataset, plotting_config)} — "
                f"{MECHANISM_DISPLAY.get(mechanism, mechanism)}"
            ),
            x="Training missingness probability",
            y=y_label,
            color="Policy",
            fill="Policy",
            shape="Policy",
            linetype="Policy",
        )
        + p9.theme_bw()
        + p9.theme(
            figure_size=(width, height),
            text=p9.element_text(
                family=plotting_config.plot_font_family,
                size=PLOT_FONT_SIZE,
            ),
            strip_text=p9.element_text(size=SUBPLOT_TITLE_FONT_SIZE),
        )
    )
    if len(probabilities) > 1:
        plot += p9.geom_ribbon(
            p9.aes(ymin="low_metric", ymax="high_metric"),
            alpha=0.1,
            color="none",
        )
        plot += p9.geom_line()
    else:
        plot += p9.geom_errorbar(
            p9.aes(ymin="low_metric", ymax="high_metric"), width=0.03
        )
    return plot, width, height


def _save_performance_plots(
    summary: pd.DataFrame,
    output_folder: Path,
    plotting_config: PlottingDisplayConfig,
    formats: Sequence[str],
) -> None:
    datasets = sorted(summary["dataset"].astype(str).unique())
    for dataset in datasets:
        primary = (
            "f_score"
            if dataset in plotting_config.datasets_with_f_score
            else "accuracy"
        )
        primary_label = "F1" if primary == "f_score" else "Accuracy"
        mechanisms = _ordered_present(
            summary.loc[
                (summary["dataset"] == dataset)
                & (summary["mechanism"] != "none"),
                "mechanism",
            ]
            .astype(str)
            .tolist(),
            MECHANISM_ORDER,
        )
        performance_folder = (
            output_folder / f"dataset-{dataset}" / "performance"
        )
        performance_folder.mkdir(parents=True, exist_ok=True)
        metrics = {
            primary: f"Evaluation {primary_label}",
            f"{primary}_gap_to_complete": (
                f"{primary_label} gap to complete training"
            ),
        }
        for mechanism in mechanisms:
            for metric, y_label in metrics.items():
                frame, strategies = _prepare_performance_frame(
                    summary,
                    dataset,
                    mechanism,
                    metric,
                    plotting_config,
                )
                if frame.empty:
                    continue
                plot, width, height = _performance_plot(
                    frame,
                    strategies,
                    dataset,
                    mechanism,
                    y_label,
                    plotting_config,
                )
                if "gap_to_complete" in metric:
                    plot += p9.geom_hline(
                        yintercept=0.0,
                        color="#666666",
                        linetype="dashed",
                        size=0.45,
                    )
                for fmt in formats:
                    plot.save(
                        performance_folder / f"{metric}_{mechanism}.{fmt}",
                        width=width,
                        height=height,
                        verbose=False,
                    )


def _aggregate_action_rates(
    instance_metrics: pd.DataFrame,
    action_rates: pd.DataFrame,
    dataset: str,
    mechanism: str,
    probability: float,
    selections: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_mask = (
        (instance_metrics["dataset"] == dataset)
        & (instance_metrics["mechanism"] == mechanism)
        & np.isclose(instance_metrics["p"], probability)
    )
    keys = ["dataset", "method", "mechanism", "p", "strategy", "instance"]
    conditions = instance_metrics.loc[condition_mask, keys].drop_duplicates()
    if conditions.empty or not selections:
        return pd.DataFrame(), conditions

    grid = conditions.merge(
        pd.DataFrame({"selection": selections}), how="cross"
    )
    action_mask = (
        (action_rates["dataset"] == dataset)
        & (action_rates["mechanism"] == mechanism)
        & np.isclose(action_rates["p"], probability)
    )
    observed = action_rates.loc[
        action_mask, [*keys, "selection", "acquisitions_per_sample"]
    ]
    merged = grid.merge(
        observed,
        on=[*keys, "selection"],
        how="left",
        validate="one_to_one",
    )
    merged["acquisitions_per_sample"] = merged[
        "acquisitions_per_sample"
    ].fillna(0.0)
    aggregated = (
        merged.groupby(["method", "strategy", "selection"], as_index=False)[
            "acquisitions_per_sample"
        ]
        .mean()
        .sort_values(["method", "strategy", "selection"])
    )
    return aggregated, conditions


def _action_matrix(
    aggregated: pd.DataFrame,
    evaluated: set[tuple[str, str]],
    method: str,
    strategies: Sequence[str],
    selections: Sequence[int],
) -> npt.NDArray[np.float64]:
    matrix = np.full((len(strategies), len(selections)), np.nan, dtype=float)
    for row, strategy in enumerate(strategies):
        if (method, strategy) not in evaluated:
            continue
        values = aggregated.loc[
            (aggregated["method"] == method)
            & (aggregated["strategy"] == strategy)
        ].set_index("selection")["acquisitions_per_sample"]
        matrix[row, :] = [
            float(values.get(selection, 0.0)) for selection in selections
        ]
    return matrix


def _format_action_axis(
    ax: Axes,
    method: str,
    strategies: Sequence[str],
    selections: Sequence[int],
    plotting_config: PlottingDisplayConfig,
    *,
    show_x_labels: bool,
    show_y_labels: bool,
) -> None:
    labels = _strategy_labels(strategies)
    method_name = plotting_config.method_name_mapping.get(method, method)
    ax.set_title(
        textwrap.fill(method_name, width=20),
        fontsize=SUBPLOT_TITLE_FONT_SIZE,
        fontweight="bold",
    )
    tick_step = max(1, math.ceil(len(selections) / 10))
    tick_positions = np.arange(0, len(selections), tick_step)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [str(selections[int(position)]) for position in tick_positions]
        if show_x_labels
        else []
    )
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_yticklabels(
        [labels[value] for value in strategies] if show_y_labels else []
    )
    ax.set_xlabel("Selection index" if show_x_labels else "")
    ax.set_ylabel("Training completion" if show_y_labels else "")
    ax.tick_params(axis="y", labelsize=9)


def _render_action_figure(
    aggregated: pd.DataFrame,
    conditions: pd.DataFrame,
    dataset: str,
    mechanism: str,
    probability: float,
    selections: Sequence[int],
    output_folder: Path,
    plotting_config: PlottingDisplayConfig,
    formats: Sequence[str],
) -> None:
    methods = _ordered_present(
        conditions["method"].astype(str).tolist(),
        list(plotting_config.method_name_mapping),
    )
    strategies = _ordered_present(
        conditions["strategy"].astype(str).tolist(), STRATEGY_ORDER
    )
    evaluated = set(
        conditions[["method", "strategy"]].itertuples(index=False, name=None)
    )
    ncols = min(5, len(methods))
    nrows = math.ceil(len(methods) / ncols)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#e0e0e0")
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.15 * ncols, 3.5 * nrows),
        squeeze=False,
    )
    image = None
    for index, method in enumerate(methods):
        ax = axes[index // ncols, index % ncols]
        image = ax.imshow(
            np.ma.masked_invalid(
                _action_matrix(
                    aggregated,
                    evaluated,
                    method,
                    strategies,
                    selections,
                )
            ),
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        _format_action_axis(
            ax,
            method,
            strategies,
            selections,
            plotting_config,
            show_x_labels=index // ncols == nrows - 1,
            show_y_labels=index % ncols == 0,
        )
    for index in range(len(methods), nrows * ncols):
        axes[index // ncols, index % ncols].set_visible(False)
    if image is not None:
        colorbar_axis = fig.add_axes((0.92, 0.19, 0.012, 0.58))
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Acquisitions per evaluation sample")
    fig.legend(
        handles=[Patch(facecolor="#e0e0e0", label="Not evaluated")],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
    )
    fig.suptitle(
        f"{_dataset_display(dataset, plotting_config)} — "
        f"{MECHANISM_DISPLAY.get(mechanism, mechanism)}, p={probability:g}",
        fontsize=PLOT_FONT_SIZE + 2,
    )
    fig.subplots_adjust(
        left=0.1,
        right=0.9,
        top=0.84,
        bottom=0.17,
        wspace=0.35,
        hspace=0.5,
    )
    for fmt in formats:
        fig.savefig(
            output_folder
            / f"acquisition_rate_{mechanism}_p-{probability:g}.{fmt}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


def _save_action_plots(
    instance_metrics: pd.DataFrame,
    action_rates: pd.DataFrame,
    output_folder: Path,
    plotting_config: PlottingDisplayConfig,
    formats: Sequence[str],
) -> None:
    if instance_metrics.empty or action_rates.empty:
        log.warning(
            "No action-rate rows available; skipping acquisition plots"
        )
        return
    missing_instances = instance_metrics.loc[
        instance_metrics["mechanism"] != "none"
    ]
    combinations = (
        missing_instances[["dataset", "mechanism", "p"]]
        .drop_duplicates()
        .sort_values(["dataset", "mechanism", "p"])
    )
    with plt.rc_context(
        {
            "font.family": plotting_config.plot_font_family,
            "font.size": PLOT_FONT_SIZE,
        }
    ):
        for combination in combinations.itertuples(index=False):
            dataset = str(combination.dataset)
            mechanism = str(combination.mechanism)
            probability = float(combination.p)
            dataset_actions = action_rates.loc[
                action_rates["dataset"] == dataset
            ]
            selections = sorted(
                dataset_actions["selection"].astype(int).unique()
            )
            aggregated, conditions = _aggregate_action_rates(
                instance_metrics,
                action_rates,
                dataset,
                mechanism,
                probability,
                selections,
            )
            if conditions.empty:
                continue
            action_folder = output_folder / f"dataset-{dataset}" / "actions"
            action_folder.mkdir(parents=True, exist_ok=True)
            _render_action_figure(
                aggregated,
                conditions,
                dataset,
                mechanism,
                probability,
                selections,
                action_folder,
                plotting_config,
                formats,
            )


def _prepare_restoration_frame(
    restoration: pd.DataFrame,
) -> pd.DataFrame:
    frame = restoration.dropna(subset=["imputation_rmse"]).copy()
    if frame.empty:
        return frame
    grouped = frame.groupby(
        [
            "dataset",
            "mechanism",
            "p",
            "strategy",
            "strategy_display_name",
            "split",
        ],
        as_index=False,
    )["imputation_rmse"]
    output = grouped.agg(["mean", "sem"]).reset_index()
    output["sem"] = output["sem"].fillna(0.0)
    output["low_rmse"] = (output["mean"] - output["sem"]).clip(lower=0.0)
    output["high_rmse"] = output["mean"] + output["sem"]
    output["mechanism_display"] = output["mechanism"].map(
        lambda value: MECHANISM_DISPLAY.get(value, value)
    )
    return output


def _save_restoration_plots(
    restoration: pd.DataFrame,
    output_folder: Path,
    plotting_config: PlottingDisplayConfig,
    formats: Sequence[str],
) -> None:
    prepared = _prepare_restoration_frame(restoration)
    if prepared.empty:
        log.warning(
            "No restoration-RMSE rows available; skipping restoration plots"
        )
        return
    for dataset, dataset_frame in prepared.groupby("dataset"):
        frame = dataset_frame.copy()
        mechanisms = _ordered_present(
            frame["mechanism"].astype(str).tolist(), MECHANISM_ORDER
        )
        mechanism_labels = [
            MECHANISM_DISPLAY.get(mechanism, mechanism)
            for mechanism in mechanisms
        ]
        strategies = _ordered_present(
            frame["strategy"].astype(str).tolist(), STRATEGY_ORDER
        )
        strategy_labels = _strategy_labels(strategies)
        frame["mechanism_display"] = pd.Categorical(
            frame["mechanism_display"],
            categories=mechanism_labels,
            ordered=True,
        )
        frame["strategy_display_name"] = pd.Categorical(
            frame["strategy"].map(
                lambda value, labels=strategy_labels: labels.get(value, value)
            ),
            categories=[strategy_labels[value] for value in strategies],
            ordered=True,
        )
        frame["series"] = (
            frame["strategy_display_name"].astype(str) + " / " + frame["split"]
        )
        ncol = min(4, len(mechanisms))
        width, height = calculate_figure_dimensions(
            len(mechanisms),
            plot_width=plotting_config.plot_width,
            ncol=ncol,
            subplot_height=3.0,
        )
        plot = (
            p9.ggplot(
                frame,
                p9.aes(
                    x="p",
                    y="mean",
                    color="strategy_display_name",
                    fill="strategy_display_name",
                    linetype="split",
                    group="series",
                ),
            )
            + p9.geom_point()
            + p9.facet_wrap("mechanism_display", ncol=ncol, scales="fixed")
            + p9.scale_color_brewer(
                type="qual", palette=plotting_config.color_palette_name
            )
            + p9.scale_fill_brewer(
                type="qual", palette=plotting_config.color_palette_name
            )
            + p9.scale_linetype_manual(
                values={"train": "dotted", "val": "solid"},
                breaks=["train", "val"],
                labels=["Train", "Validation"],
            )
            + p9.scale_x_continuous(
                breaks=sorted(frame["p"].unique()),
                labels=[f"{value:g}" for value in sorted(frame["p"].unique())],
                limits=(0.0, 1.0),
            )
            + p9.labs(
                title=_dataset_display(str(dataset), plotting_config),
                x="Training missingness probability",
                y="PVAE restoration RMSE",
                color="Restoration",
                fill="Restoration",
                linetype="Split",
            )
            + p9.theme_bw()
            + p9.theme(
                figure_size=(width, height),
                text=p9.element_text(
                    family=plotting_config.plot_font_family,
                    size=PLOT_FONT_SIZE,
                ),
                strip_text=p9.element_text(size=SUBPLOT_TITLE_FONT_SIZE),
            )
        )
        if len(frame["p"].unique()) > 1:
            plot += p9.geom_ribbon(
                p9.aes(ymin="low_rmse", ymax="high_rmse"),
                alpha=0.12,
                color="none",
            )
            plot += p9.geom_line()
        else:
            plot += p9.geom_errorbar(
                p9.aes(ymin="low_rmse", ymax="high_rmse"), width=0.03
            )
        restoration_folder = (
            output_folder / f"dataset-{dataset}" / "restoration"
        )
        restoration_folder.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            plot.save(
                restoration_folder / f"restoration_rmse.{fmt}",
                width=width,
                height=height,
                verbose=False,
            )


def generate_missing_data_plots(
    instance_metrics_path: Path,
    summary_path: Path,
    action_rates_path: Path,
    restoration_rmse_path: Path,
    output_folder: Path,
    plotting_config: PlottingDisplayConfig,
    formats: Sequence[str],
) -> None:
    plotting_config = _with_available_font(plotting_config)
    instance_metrics = _read_csv(
        instance_metrics_path, INSTANCE_COLUMNS, optional=True
    )
    summary = _read_csv(summary_path, SUMMARY_COLUMNS)
    action_rates = _read_csv(action_rates_path, ACTION_COLUMNS, optional=True)
    restoration = _read_csv(
        restoration_rmse_path, RESTORATION_COLUMNS, optional=True
    )
    if summary.empty:
        msg = f"No summarized experiments found in {summary_path}."
        raise ValueError(msg)

    output_folder.mkdir(parents=True, exist_ok=True)
    _save_performance_plots(summary, output_folder, plotting_config, formats)
    _save_action_plots(
        instance_metrics,
        action_rates,
        output_folder,
        plotting_config,
        formats,
    )
    _save_restoration_plots(
        restoration, output_folder, plotting_config, formats
    )


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/plotting/plot_missing_data",
    config_name="config",
)
def main(cfg: PlotMissingDataConfig) -> None:
    cfg = cast("PlotMissingDataConfig", OmegaConf.to_object(cfg))
    assert isinstance(cfg, PlotMissingDataConfig)
    generate_missing_data_plots(
        Path(cfg.instance_metrics),
        Path(cfg.summary),
        Path(cfg.action_rates),
        Path(cfg.restoration_rmse),
        Path(cfg.output_folder),
        cfg.plotting,
        cfg.formats,
    )


if __name__ == "__main__":
    main()
