"""Render missingness damage and restoration error against dataset size."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from afabench.plotting.methods import (
    DATASET_LABELS_SHORT,
    GRID,
    INK,
    TEXT_WIDTH_IN,
    apply_paper_style,
)
from scripts.analysis.restoration_tables import (
    METHODS,
    RATE_P,
    SUMMARY_ROOT,
    _cell,
    _complete,
    _scores,
)

SIZES = (500, 1000, 4000, 16000)
DATASETS = ("cube", "cube_nm")
SERIES = {
    "damage": ("Missingness Damage", INK, "o"),
    "pvae_label_conditioned": ("Restoration Error: PVAE", "#2a78d6", "s"),
    "pvae_oracle": (
        "Restoration Error: PVAE Fit on Complete Data",
        "#b8860b",
        "D",
    ),
}


def collect() -> dict[str, dict[str, list[float]]]:
    """Mean damage and restoration errors over methods per dataset and size."""
    values = {d: {key: [] for key in SERIES} for d in DATASETS}
    for n in SIZES:
        scores = _scores([SUMMARY_ROOT / f"size_sweep_n{n}" / "summary.csv"])
        for dataset in DATASETS:
            cells = []
            for method in METHODS:
                cell = _cell(scores, dataset, method, "mcar", RATE_P)
                if "restricted" in cell:
                    cells.append((cell, _complete(scores, dataset, method)))
            values[dataset]["damage"].append(
                float(np.mean([c - cell["restricted"] for cell, c in cells]))
            )
            for strategy in ("pvae_label_conditioned", "pvae_oracle"):
                values[dataset][strategy].append(
                    float(np.mean([cell[strategy] - c for cell, c in cells]))
                )
    return values


def plot(values: dict[str, dict[str, list[float]]], output: Path) -> None:
    apply_paper_style()
    figure, axes = plt.subplots(
        1, len(DATASETS), figsize=(TEXT_WIDTH_IN, 2.1), sharey=True
    )
    for axis, dataset in zip(axes, DATASETS, strict=True):
        axis.axhline(0.0, color=INK, linewidth=0.6, zorder=1)
        for key, (label, color, marker) in SERIES.items():
            axis.plot(
                SIZES,
                values[dataset][key],
                color=color,
                marker=marker,
                markersize=3.5,
                linewidth=1.2,
                label=label,
                zorder=3,
            )
        axis.set_xscale("log")
        axis.set_xticks(SIZES, [str(n) for n in SIZES])
        axis.minorticks_off()
        axis.set_title(
            DATASET_LABELS_SHORT.get(dataset, dataset), fontsize=7.5
        )
        axis.set_xlabel("Training Instances", fontsize=7)
        axis.tick_params(labelsize=6)
        axis.grid(True, axis="y", color=GRID, linewidth=0.4, alpha=0.7)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("Accuracy Difference", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(SERIES),
        frameon=False,
        fontsize=6.3,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.subplots_adjust(
        left=0.1, right=0.99, top=0.9, bottom=0.3, wspace=0.1
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("extra/output/paper/experiments/results/size_sweep.pdf"),
    )
    arguments = parser.parse_args()
    plot(collect(), arguments.output)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
