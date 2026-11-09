"""Render restricted training against each restoration, synthetic and real."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from afabench.plotting.methods import (
    DATASET_LABELS_SHORT,
    GRID,
    INK,
    METHOD_LABELS_SHORT,
    PRIMARY_METHODS,
    SURFACE,
    TEXT_WIDTH_IN,
    apply_paper_style,
)
from scripts.plotting.plot_main_summary import (
    MAIN_MECHANISM,
    MAIN_RATE,
    ROW_GRAY,
    ROW_HATCH,
    ROW_HATCH_COLOR,
    _column,
    _interval,
    _largest_budget,
    _rows,
    _shade_rows,
    primary_metric,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

SYNTHETIC = ["cube_nm", "cube", "cube_nonuniform_costs"]
REAL = ["heart_disease", "actg", "diabetes", "nhanes_mortality", "miniboone"]

NAMESPACES = [
    "induced",
    "core_group_missingness_v2",
    "induced_nonuniform_missingness_v2",
    "induced_real_missingness_v2",
    "sklearn_imputers_core_group",
    "sklearn_imputers_induced_nonuniform",
    "sklearn_imputers_induced_real",
    "constant_fill_core_group",
    "constant_fill_induced_nonuniform",
    "constant_fill_induced_real",
    "mice_label_core_group",
    "mice_label_induced_nonuniform",
    "mice_label_induced_real",
]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
APPROACHES = {
    "restricted": ("Restricted-Action Training", INK, "o"),
    "mean_fill": ("Restoration: Mean-Fill", AQUA, "D"),
    "mice_label_fill": ("Restoration: MICE, Label-Conditioned", ORANGE, "v"),
    "pvae_label_conditioned": ("Restoration: PVAE", BLUE, "o"),
}
OFFSETS = dict(
    zip(APPROACHES, np.linspace(-0.3, 0.3, len(APPROACHES)), strict=True)
)
SEED = 0


def collect(summary_root: Path) -> pd.DataFrame:
    """Mean score and bootstrap interval per dataset, method and approach."""
    frames = []
    for namespace in NAMESPACES:
        path = summary_root / namespace / "instance_metrics.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["dataset", "method", "mechanism", "p", "strategy", "instance"]
        + ["eval_hard_budget"]
    )
    rng = np.random.default_rng(SEED)
    rows = []
    for dataset in SYNTHETIC + REAL:
        per_dataset = _largest_budget(frame, dataset)
        metric = primary_metric(dataset)
        for method in PRIMARY_METHODS:
            per_method = _rows(
                per_dataset, _column(per_dataset, "method") == method
            )
            for strategy in ["complete", *APPROACHES]:
                if strategy == "complete":
                    mask = _column(per_method, "strategy") == strategy
                else:
                    mask = (
                        (_column(per_method, "strategy") == strategy)
                        & (_column(per_method, "mechanism") == MAIN_MECHANISM)
                        & (_column(per_method, "p") == MAIN_RATE)
                    )
                values = (
                    _column(_rows(per_method, mask), metric)
                    .dropna()
                    .to_numpy(dtype=float)
                )
                if not len(values):
                    continue
                low, high = _interval(values, rng)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "strategy": strategy,
                        "score": float(values.mean()),
                        "low": low,
                        "high": high,
                        "n": len(values),
                    }
                )
    return pd.DataFrame(rows)


def _method_order(scores: pd.DataFrame) -> list[str]:
    """Most damaged first over all datasets, as in the main summary."""
    pivot = scores.pivot_table(
        index=["dataset", "method"], columns="strategy", values="score"
    )
    damage = (pivot["complete"] - pivot["restricted"]).groupby("method").mean()
    return [
        str(method) for method in damage.sort_values(ascending=False).index
    ]


def _limits(
    scores: pd.DataFrame, datasets: list[str]
) -> dict[str, tuple[float, float]]:
    """One span for every panel, centred on each dataset's own values."""
    bounds = {}
    for dataset in datasets:
        per_dataset = _rows(scores, _column(scores, "dataset") == dataset)
        values = per_dataset[["score", "low", "high"]].to_numpy(dtype=float)
        bounds[dataset] = (float(np.nanmin(values)), float(np.nanmax(values)))
    span = max(
        (high - low) + 2 * max(0.015, 0.06 * (high - low))
        for low, high in bounds.values()
    )
    limits = {}
    for dataset, (low, high) in bounds.items():
        start = (low + high - span) / 2
        start = min(max(start, 0.0), 1.0 - span) if span <= 1.0 else start
        limits[dataset] = (start, start + span)
    return limits


def _draw(
    axis: Axes, scores: pd.DataFrame, dataset: str, methods: list[str]
) -> None:
    per_dataset = _rows(scores, _column(scores, "dataset") == dataset)
    _shade_rows(axis, methods)
    for index, method in enumerate(methods):
        cells = _rows(
            per_dataset, _column(per_dataset, "method") == method
        ).set_index("strategy")
        if "complete" in cells.index:
            axis.plot(
                [cells.loc["complete", "score"]] * 2,
                [index - 0.42, index + 0.42],
                color=INK,
                linewidth=1.0,
                zorder=2,
            )
        for strategy, (_, color, marker) in APPROACHES.items():
            if strategy not in cells.index:
                continue
            row = index + OFFSETS[strategy]
            cell = cells.loc[strategy]
            if not np.isnan(cell["low"]):
                axis.plot(
                    [cell["low"], cell["high"]],
                    [row, row],
                    color=color,
                    linewidth=0.8,
                    alpha=0.35,
                    solid_capstyle="butt",
                    zorder=3,
                )
            restricted = strategy == "restricted"
            axis.scatter(
                [cell["score"]],
                [row],
                s=11 if marker in {"s", "D"} else 14,
                marker=marker,
                facecolor=SURFACE if restricted else color,
                edgecolor=color if restricted else SURFACE,
                linewidth=0.8 if restricted else 0.4,
                zorder=4,
            )
    axis.set_ylim(len(methods) - 0.5, -0.5)
    axis.set_yticks(range(len(methods)))
    axis.set_yticklabels(
        [METHOD_LABELS_SHORT[method] for method in methods], fontsize=6
    )
    axis.tick_params(axis="y", length=0)
    axis.set_title(DATASET_LABELS_SHORT.get(dataset, dataset), fontsize=7.5)
    axis.grid(True, axis="x", color=GRID, linewidth=0.4, alpha=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", labelsize=6)
    axis.locator_params(axis="x", nbins=4)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)


def _legend() -> list[Line2D | Patch]:
    handles: list[Line2D | Patch] = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="none",
            markersize=4.2,
            markerfacecolor=SURFACE if strategy == "restricted" else color,
            markeredgecolor=color if strategy == "restricted" else SURFACE,
            markeredgewidth=0.8 if strategy == "restricted" else 0.3,
            label=label,
        )
        for strategy, (label, color, marker) in APPROACHES.items()
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="|",
            linestyle="none",
            markersize=6,
            markeredgecolor=INK,
            markeredgewidth=1.1,
            label="Complete Data",
        ),
        Patch(facecolor=ROW_GRAY, edgecolor=INK, label="Myopic"),
        Patch(
            facecolor=SURFACE,
            edgecolor=ROW_HATCH_COLOR,
            hatch=ROW_HATCH,
            label="Non-Myopic, Intermediate Reward",
        ),
        Patch(
            facecolor=SURFACE,
            edgecolor=INK,
            label="Non-Myopic, Terminal Objective",
        ),
    ]
    return handles


def plot(
    scores: pd.DataFrame,
    datasets: list[str],
    methods: list[str],
    output: Path,
    xlabel: str,
    *,
    legend: bool,
) -> None:
    apply_paper_style()
    mpl.rcParams["hatch.linewidth"] = 0.35
    limits = _limits(scores, datasets)
    strip = 0.95 if legend else 0.32
    body = 0.3 + 0.28 * len(methods)
    height = strip + body
    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(TEXT_WIDTH_IN, height),
        squeeze=False,
        sharey=True,
    )
    for axis, dataset in zip(axes[0], datasets, strict=True):
        _draw(axis, scores, dataset, methods)
        axis.set_xlim(*limits[dataset])
    figure.supxlabel(xlabel, fontsize=8, y=(strip - 0.27) / height)
    if legend:
        figure.legend(
            handles=_legend(),
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=6.3,
            labelcolor=INK,
            columnspacing=1.0,
            handlelength=1.3,
            handleheight=1.0,
            labelspacing=0.45,
            bbox_to_anchor=(0.5, 0.0),
        )
    figure.subplots_adjust(
        left=0.12,
        right=0.99,
        top=1 - 0.2 / height,
        bottom=strip / height,
        wspace=0.12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("extra/output/missing_data/summary/val"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extra/output/paper/experiments/results"),
    )
    arguments = parser.parse_args()
    scores = collect(arguments.summary_root)
    methods = _method_order(scores)
    for name, datasets, xlabel in (
        ("synthetic", SYNTHETIC, "Accuracy"),
        ("real", REAL, "Macro-F1"),
    ):
        output = arguments.output_dir / f"restoration_{name}_mcar.pdf"
        plot(
            scores,
            datasets,
            methods,
            output,
            xlabel,
            legend=name == "synthetic",
        )
        scores[scores["dataset"].isin(datasets)].to_csv(
            output.with_suffix(".csv"), index=False
        )
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
