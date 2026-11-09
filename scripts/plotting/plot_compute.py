"""Plot score and compute for restricted and generatively restored training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.lines import Line2D

from afabench.plotting.methods import (
    DATASET_LABELS_SHORT,
    FAMILY_COLORS,
    GRID,
    INK,
    METHOD_FAMILIES,
    METHOD_MARKERS,
    PRIMARY_METHODS,
    SURFACE,
    TEXT_WIDTH_IN,
    apply_paper_style,
)
from scripts.plotting.family_summary import FAMILY_LABELS, FAMILY_MEMBERS

if TYPE_CHECKING:
    from matplotlib.axes import Axes

DATASET_MARKERS = {
    "cube": "o",
    "cube_nm": "s",
    "cube_nonuniform_costs": "D",
    "heart_disease": "P",
    "actg": "^",
    "diabetes": "v",
    "nhanes_mortality": "X",
    "miniboone": "*",
}
ACCURACY_DATASETS = {"cube", "cube_nm", "cube_nonuniform_costs"}
# Variants differ in state, not in compute, so nine series become six families.
FAMILY_ORDER = tuple(
    dict.fromkeys(METHOD_FAMILIES[method] for method in PRIMARY_METHODS)
)
FAMILY_MARKERS = {
    family: METHOD_MARKERS[FAMILY_MEMBERS[family][0]]
    for family in FAMILY_ORDER
}
# Strip under the axes, in inches: x label, family key, arm key.
STRIP_IN = 1.08
LABEL_IN, FAMILY_KEY_IN, ARM_KEY_IN = 0.74, 0.38, 0.02
RESTRICTED = "restricted"
GENERATIVE = "pvae_label_conditioned"
RESTORED = {"pvae_label_conditioned", "pvae_label_free", "pvae_stepwise"}
HARDWARE_FIELDS = [
    "device",
    "cores",
    "mem_mb",
    "gpu_workers",
    "mps",
    "architecture",
    "torch",
    "torch_cuda",
    "cuda_devices",
]
# `gpu_workers` is workflow concurrency, not hardware: keep it as provenance
# but never reject a pair over it.
PAIR_ENVIRONMENT_FIELDS = [
    field for field in HARDWARE_FIELDS if field != "gpu_workers"
]


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast("pd.Series", frame[name])


def _rows(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return cast("pd.DataFrame", frame[mask])


def _sum_cost(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return cast(
        "pd.DataFrame",
        frame.groupby(keys, dropna=False).agg(
            wall_seconds=("wall_seconds", "sum"),
            cpu_seconds=("cpu_seconds", "sum"),
            peak_rss_mb=("peak_rss_mb", "max"),
        ),
    )


def _component_totals(
    components: dict[str, Any],
    generator_share: int,
    restore_share: int,
) -> tuple[dict[str, float], float, float]:
    wall = {
        "train": float(components["train"]["wall_seconds"]),
        "pretrain": float(
            components.get("pretrain", {}).get("wall_seconds", 0.0)
        ),
        "eval": float(components.get("eval", {}).get("wall_seconds", 0.0)),
        "generator_share": float(
            components.get("generator", {}).get("wall_seconds", 0.0)
        )
        / generator_share,
        "restore_share": float(
            components.get("restore", {}).get("wall_seconds", 0.0)
        )
        / restore_share,
    }
    cpu_seconds = 0.0
    peak_rss_mb = 0.0
    for name, component in components.items():
        share = (
            generator_share
            if name == "generator"
            else restore_share
            if name == "restore"
            else 1
        )
        cpu_seconds += float(component["cpu_seconds"]) / share
        peak_rss_mb = max(peak_rss_mb, float(component["peak_rss_mb"]))
    return wall, cpu_seconds, peak_rss_mb


def attribute(compute: pd.DataFrame) -> pd.DataFrame:
    """Cost one restricted or restored result within each scientific cell."""
    cell = [
        "namespace",
        "dataset",
        "mechanism",
        "p",
        "strategy",
        "instance",
    ]
    provenance = ["hardware_signature", "git_commit", *HARDWARE_FIELDS]
    training = _rows(compute, _column(compute, "rule") == "train_method")
    training = _rows(
        training, _column(training, "method").isin(PRIMARY_METHODS)
    )
    pretraining = _rows(
        compute, _column(compute, "rule") == "pretrain_method"
    ).copy()
    pretraining["method"] = pretraining["pretrain_key"]
    evaluations = _rows(compute, _column(compute, "rule") == "eval_method")
    restoration = _rows(compute, _column(compute, "rule") == "restore_view")
    generators = _rows(
        compute,
        _column(compute, "rule") == "pretrain_restoration_pvae_incomplete",
    )

    method_cell = [*cell, "method"]
    method_keys = [*method_cell, *provenance]
    train_cost = _sum_cost(training, method_keys)
    eval_cost = _sum_cost(evaluations, method_cell)
    pretrain_cost = _sum_cost(pretraining, method_cell)
    generator_keys = [
        "namespace",
        "dataset",
        "mechanism",
        "p",
        "instance",
    ]
    generator_cost = _sum_cost(generators, generator_keys)
    restore_cost = _sum_cost(restoration, cell)
    consumers = (
        _rows(training, _column(training, "strategy").isin(RESTORED))
        .groupby(generator_keys, dropna=False)
        .size()
    )
    strategy_consumers = training.groupby(cell, dropna=False).size()
    generator_provenance = cast(
        "pd.DataFrame",
        generators.groupby(generator_keys, dropna=False).agg(
            generator_git_commit=(
                "git_commit",
                lambda values: ";".join(
                    sorted({str(value) for value in values})
                ),
            ),
            generator_hardware_signature=(
                "hardware_signature",
                lambda values: ";".join(
                    sorted({str(value) for value in values})
                ),
            ),
        ),
    )

    rows: list[dict[str, Any]] = []
    for key, train in cast("Any", train_cost).iterrows():
        record = dict(zip(method_keys, key, strict=True))
        strategy = record["strategy"]
        if strategy not in {RESTRICTED, GENERATIVE}:
            continue
        scientific_key = tuple(record[name] for name in method_cell)
        component_rows: dict[str, Any] = {"train": train}
        if scientific_key in pretrain_cost.index:
            component_rows["pretrain"] = pretrain_cost.loc[scientific_key]
        if scientific_key in eval_cost.index:
            component_rows["eval"] = eval_cost.loc[scientific_key]
        generator_share = 1
        restore_share = 1
        generator_key = tuple(record[name] for name in generator_keys)
        if strategy == GENERATIVE:
            if generator_key in generator_cost.index:
                generator_share = max(
                    int(cast("Any", consumers.get(generator_key, 1)) or 1),
                    1,
                )
                component_rows["generator"] = generator_cost.loc[generator_key]
            strategy_key = tuple(record[name] for name in cell)
            if strategy_key in restore_cost.index:
                restore_share = max(
                    int(
                        cast("Any", strategy_consumers.get(strategy_key, 1))
                        or 1
                    ),
                    1,
                )
                component_rows["restore"] = restore_cost.loc[strategy_key]
        wall_components, cpu_seconds, peak_rss_mb = _component_totals(
            component_rows, generator_share, restore_share
        )
        generator_meta = {
            "generator_git_commit": "",
            "generator_hardware_signature": "",
        }
        if (
            strategy == GENERATIVE
            and generator_key in generator_provenance.index
        ):
            generator_meta = cast(
                "dict[str, str]",
                generator_provenance.loc[generator_key].to_dict(),
            )
        rows.append(
            {
                **record,
                **generator_meta,
                "arm": (
                    "restricted" if strategy == RESTRICTED else "generative"
                ),
                "wall_seconds": sum(wall_components.values()),
                "cpu_seconds": cpu_seconds,
                "peak_rss_mb": peak_rss_mb,
                **{
                    f"{name}_wall_seconds": value
                    for name, value in wall_components.items()
                },
            }
        )
    return pd.DataFrame(rows)


def paired_costs(costs: pd.DataFrame, summary_root: Path) -> pd.DataFrame:
    scores = []
    for namespace in sorted(set(_column(costs, "namespace"))):
        path = summary_root / str(namespace) / "instance_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["namespace"] = namespace
        frame["arm"] = _column(frame, "strategy").map(
            {RESTRICTED: "restricted", GENERATIVE: "generative"}
        )
        for dataset in set(_column(frame, "dataset")):
            metric = "accuracy" if dataset in ACCURACY_DATASETS else "f_score"
            subset = _rows(frame, _column(frame, "dataset") == dataset).copy()
            subset["score"] = subset[metric]
            scores.append(subset)
    if not scores:
        return pd.DataFrame()
    score_frame = pd.concat(scores, ignore_index=True)
    keys = [
        "namespace",
        "dataset",
        "mechanism",
        "p",
        "instance",
        "method",
        "arm",
    ]
    joined = costs.merge(score_frame[[*keys, "score"]], on=keys, how="inner")
    pair_keys = [
        "namespace",
        "dataset",
        "mechanism",
        "p",
        "instance",
        "method",
        *PAIR_ENVIRONMENT_FIELDS,
    ]
    restricted = _rows(joined, _column(joined, "arm") == "restricted")
    generative = _rows(joined, _column(joined, "arm") == "generative")
    wide = restricted.merge(
        generative,
        on=pair_keys,
        how="inner",
        suffixes=("_restricted", "_generative"),
        validate="one_to_one",
    )
    required = [
        "wall_seconds_restricted",
        "wall_seconds_generative",
        "score_restricted",
        "score_generative",
    ]
    wide = wide.dropna(subset=required)
    wide["wall_time_ratio"] = (
        wide["wall_seconds_generative"] / wide["wall_seconds_restricted"]
    )
    wide["restoration_gain"] = (
        wide["score_generative"] - wide["score_restricted"]
    )
    wide["generator_git_commit"] = wide.pop("generator_git_commit_generative")
    wide["generator_hardware_signature"] = wide.pop(
        "generator_hardware_signature_generative"
    )
    wide = wide.drop(
        columns=[
            "generator_git_commit_restricted",
            "generator_hardware_signature_restricted",
        ]
    )
    return wide


def _thin_x_ticks(axis: Axes, keep: int = 3) -> None:
    """
    Drop every other label until at most ``keep`` remain.

    A 1-2-5 log locator gives a sensible number of ticks over half a decade and
    far too many over one and a half, which is the span between the cheapest and
    dearest method on Diabetes.
    """
    axis.figure.canvas.draw()
    lo, hi = axis.get_xlim()
    visible = [t for t in axis.get_xticks() if lo <= t <= hi]
    while len(visible) > keep:
        visible = visible[::2]
    axis.set_xticks(visible)


PANEL_MECHANISM, PANEL_RATE = "mcar", 0.5


def panel_cells(frame: pd.DataFrame) -> pd.DataFrame:
    cell = _rows(
        frame,
        (_column(frame, "mechanism") == PANEL_MECHANISM)
        & (_column(frame, "p") == PANEL_RATE),
    )
    expected = {
        (dataset, method)
        for dataset in DATASET_MARKERS
        for method in PRIMARY_METHODS
    }
    counts = cast("pd.Series", cell.groupby(["dataset", "method"]).size())
    observed = set(
        cell[["dataset", "method"]].itertuples(index=False, name=None)
    )
    if observed != expected or not (counts == 5).all():
        missing = sorted(expected - observed)
        incomplete = {
            key: value for key, value in counts.items() if value != 5
        }
        message = (
            "compute panel coverage mismatch: "
            f"missing={missing}; non-five-instance groups={incomplete}"
        )
        raise ValueError(message)
    columns = [
        "wall_seconds_restricted",
        "wall_seconds_generative",
        "score_restricted",
        "score_generative",
    ]
    per_method = cast(
        "pd.DataFrame",
        cell.groupby(["dataset", "method"], as_index=False)[columns].mean(),
    )
    per_method["family"] = _column(per_method, "method").map(METHOD_FAMILIES)
    return cast(
        "pd.DataFrame",
        per_method.groupby(["dataset", "family"], as_index=False)[
            columns
        ].mean(),
    )


def plot(frame: pd.DataFrame, output: Path) -> None:
    apply_paper_style()
    cells = panel_cells(frame)
    datasets = sorted(set(_column(cells, "dataset")))
    # Four columns past six datasets, so eight fills a 2x4 exactly.
    columns = 4 if len(datasets) > 6 else min(3, len(datasets))
    rows = -(-len(datasets) // columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(TEXT_WIDTH_IN, STRIP_IN + 0.35 + 1.55 * rows),
        squeeze=False,
    )
    for index, dataset in enumerate(datasets):
        axis = axes[index // columns][index % columns]
        per_dataset = _rows(cells, _column(cells, "dataset") == dataset)
        for family in FAMILY_ORDER:
            record = _rows(
                per_dataset, _column(per_dataset, "family") == family
            )
            if record.empty:
                continue
            row = cast("Any", record.iloc[0])
            color = FAMILY_COLORS[family]
            marker = FAMILY_MARKERS[family]
            axis.annotate(
                "",
                xy=(row["wall_seconds_generative"], row["score_generative"]),
                xytext=(
                    row["wall_seconds_restricted"],
                    row["score_restricted"],
                ),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 1.3,
                    "shrinkA": 2.0,
                    "shrinkB": 2.0,
                },
            )
            axis.scatter(
                [row["wall_seconds_restricted"]],
                [row["score_restricted"]],
                marker=marker,
                s=22,
                facecolor=SURFACE,
                edgecolor=color,
                linewidth=1.1,
                zorder=3,
            )
            axis.scatter(
                [row["wall_seconds_generative"]],
                [row["score_generative"]],
                marker=marker,
                s=22,
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
                zorder=3,
            )
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(
            mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12)
        )
        axis.xaxis.set_minor_locator(mticker.NullLocator())
        axis.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: f"{value:,.0f}")
        )
        # Family means narrow the range; cap the ticks a 1.2in axis can hold.
        axis.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        axis.tick_params(labelsize=7)
        # A 1.2in panel cannot carry three log labels without them touching.
        _thin_x_ticks(axis, keep=2 if columns > 3 else 3)
        axis.set_title(DATASET_LABELS_SHORT.get(dataset, dataset), fontsize=8)
        axis.grid(True, color=GRID, linewidth=0.4, alpha=0.55)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    for index in range(len(datasets), rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    height = STRIP_IN + 0.35 + 1.55 * rows
    figure.supxlabel(
        "Wall-Clock Time per Trained Method (s)",
        fontsize=8,
        y=LABEL_IN / height,
    )
    figure.supylabel(
        "Accuracy (CUBE) or Macro-F1 (Real-World)", fontsize=8, x=0.015
    )
    family_handles = [
        Line2D(
            [],
            [],
            color=FAMILY_COLORS[family],
            marker=FAMILY_MARKERS[family],
            markerfacecolor=FAMILY_COLORS[family],
            markersize=4.0,
            linewidth=1.0,
            label=FAMILY_LABELS[family],
        )
        for family in FAMILY_ORDER
    ]
    arm_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.5,
            markerfacecolor=SURFACE,
            markeredgecolor=INK,
            markeredgewidth=1.0,
            label="Restricted Training",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.5,
            markerfacecolor=INK,
            markeredgecolor=INK,
            label="Generative Restoration",
        ),
    ]
    family_legend = figure.legend(
        handles=family_handles,
        loc="lower center",
        ncol=len(FAMILY_ORDER),
        frameon=False,
        fontsize=7.0,
        labelcolor=INK,
        columnspacing=1.1,
        handlelength=1.4,
        bbox_to_anchor=(0.5, FAMILY_KEY_IN / height),
    )
    figure.add_artist(family_legend)
    figure.legend(
        handles=arm_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=7.0,
        labelcolor=INK,
        columnspacing=1.2,
        handlelength=1.4,
        bbox_to_anchor=(0.5, ARM_KEY_IN / height),
    )
    figure.subplots_adjust(
        left=0.11,
        right=0.985,
        top=0.955,
        bottom=STRIP_IN / height,
        hspace=0.45,
        wspace=0.32,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".svg"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", type=Path, required=True)
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("extra/output/missing_data/summary/val"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("extra/output/missing_data/analysis_figures/compute.pdf"),
    )
    parser.add_argument("--table", type=Path)
    arguments = parser.parse_args()

    costs = attribute(pd.read_csv(arguments.compute))
    frame = paired_costs(costs, arguments.summary_root)
    if frame.empty:
        message = "no complete paired compute cells"
        raise SystemExit(message)
    plot(frame, arguments.output)
    table = arguments.table or arguments.output.with_suffix(".paired.csv")
    if table.resolve() == arguments.compute.resolve():
        message = "paired table must not overwrite the raw compute input"
        raise ValueError(message)
    frame.to_csv(table, index=False)
    print(f"paired cells: {len(frame)}")
    print(f"wrote {arguments.output} and {table}")


if __name__ == "__main__":
    main()
