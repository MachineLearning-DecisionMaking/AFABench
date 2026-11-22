"""
Plot method and category average ranks, plus a full-coverage rank heatmap.

Input: dataset_rankings.csv from scripts/misc/rank_hard_budget_auc.py.
Ranks are recomputed from mean_auc, with average ranks for exact ties.
Category means weight methods equally within each dataset, then datasets
equally, using only datasets shared by all included methods.

Usage:
    uv run python scripts/plotting/plot_method_rankings.py INPUT.csv OUTPUT_DIR
"""

from pathlib import Path
from typing import Annotated

import plotnine as p9
import polars as pl
import typer

# Display names from the recovered February 2026 evaluation snapshot.
METHOD_DISPLAY_NAMES = {
    "ma2018_external": "EDDI-GM",
    "covert2023": "GDFS-DM",
    "gadgil2023": "DIME-DM",
    "jafa": "JAFA-MFRL",
    "ol_with_mask": "OL-MFRL",
    "odin_model_based": "ODIN-MBRL",
    "aaco": "AACO",
    "cae": "CAE-S",
}


def prepare_rankings(
    table: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    classifiers = table["classifier"].unique().to_list()
    if len(classifiers) != 1 or classifiers[0] not in {"external", "builtin"}:
        msg = (
            f"Expected one classifier (external or builtin), got {classifiers}"
        )
        raise ValueError(msg)
    duplicates = table.group_by("dataset", "afa_method").len()
    if duplicates.filter(pl.col("len") > 1).height:
        msg = "Expected one row per dataset/method pair"
        raise ValueError(msg)
    if (
        table.is_empty()
        or table.filter(
            pl.col("mean_auc").is_null() | ~pl.col("mean_auc").is_finite()
        ).height
    ):
        msg = "Expected nonempty rankings with finite mean_auc values"
        raise ValueError(msg)
    ranked = table.with_columns(
        pl.col("mean_auc")
        .rank(method="average", descending=True)
        .over("dataset")
        .alias("rank")
    )
    n_methods = ranked["afa_method"].n_unique()
    shared = (
        ranked.group_by("dataset")
        .len()
        .filter(pl.col("len") == n_methods)
        .select("dataset")
    )
    common = ranked.join(shared, on="dataset", how="semi")
    if common.is_empty():
        msg = "No datasets contain every method; choose a method subset first"
        raise ValueError(msg)
    summary = (
        common.group_by("afa_method")
        .agg(
            pl.col("rank").mean().alias("average_rank"),
            pl.len().alias("n_datasets"),
        )
        .sort("average_rank", "afa_method")
    )
    return ranked, common, summary


def create_plots(
    ranked: pl.DataFrame, common: pl.DataFrame, summary: pl.DataFrame
) -> tuple[p9.ggplot, p9.ggplot]:
    classifier_label = (
        "Built-in predictor"
        if ranked["classifier"][0] == "builtin"
        else "External classifier"
    )
    methods: list[str] = summary["afa_method"].to_list()
    datasets = sorted(ranked["dataset"].unique().to_list())
    shared = sorted(common["dataset"].unique().to_list())
    positions = pl.DataFrame(
        {"afa_method": methods, "method_y": list(range(len(methods)))}
    )
    # Deterministic offsets separate overlapping dataset ranks without RNG.
    offsets = pl.DataFrame(
        {
            "dataset": shared,
            "offset": [
                0.24 * (i - (len(shared) - 1) / 2) / max(len(shared) - 1, 1)
                for i in range(len(shared))
            ],
        }
    )
    points = (
        common.join(positions, on="afa_method")
        .join(offsets, on="dataset")
        .with_columns((pl.col("method_y") + pl.col("offset")).alias("point_y"))
    )
    averages = summary.join(positions, on="afa_method").with_columns(
        pl.col("average_rank").round(2).cast(pl.String).alias("label")
    )
    average_plot = (
        p9.ggplot()
        + p9.geom_point(
            p9.aes("rank", "point_y"),
            data=points,
            color="#929da5",
            alpha=0.6,
            size=2,
        )
        + p9.geom_point(
            p9.aes("average_rank", "method_y"),
            data=averages,
            color="#007f86",
            size=4,
        )
        + p9.geom_text(
            p9.aes("average_rank", "method_y", label="label"),
            data=averages,
            nudge_y=-0.27,
            size=8,
            color="#00676c",
        )
        + p9.scale_y_reverse(
            breaks=list(range(len(methods))),
            labels=[
                METHOD_DISPLAY_NAMES.get(method, method) for method in methods
            ],
        )
        + p9.scale_x_continuous(
            breaks=list(range(1, len(methods) + 1)),
            limits=(0.7, len(methods) + 0.3),
        )
        + p9.labs(
            title="Method ranking across shared datasets",
            subtitle=(
                f"{classifier_label} · {len(methods)} methods · "
                f"{len(shared)} shared datasets"
            ),
            x="Average rank (1 = best)",
            y="",
            caption=(
                "Teal: average rank; gray: individual dataset ranks.\n"
                f"Equal dataset weights; {classifier_label.lower()}, "
                "mean hard-budget AUC."
            ),
        )
        + p9.theme_minimal(base_size=11)
        + p9.theme(figure_size=(8.5, 5.8))
    )
    grid = (
        pl.DataFrame({"afa_method": methods})
        .join(pl.DataFrame({"dataset": datasets}), how="cross")
        .join(
            ranked.select("afa_method", "dataset", "rank"),
            on=["afa_method", "dataset"],
            how="left",
        )
        .with_columns(
            pl.col("rank").cast(pl.String).fill_null("—").alias("label")
        )
    )
    heatmap = (
        p9.ggplot(grid, p9.aes("dataset", "afa_method", fill="rank"))
        + p9.geom_tile(color="white", size=0.5)
        + p9.geom_text(p9.aes(label="label"), size=9)
        + p9.scale_fill_gradient(
            low="#70c7c5",
            high="#f3f8f7",
            limits=(1, len(methods)),
            breaks=list(range(1, len(methods) + 1)),
            na_value="#d4d4d4",
        )
        + p9.scale_x_discrete(limits=datasets)
        + p9.scale_y_discrete(
            limits=list(reversed(methods)),
            labels={
                method: METHOD_DISPLAY_NAMES.get(method, method)
                for method in methods
            },
        )
        + p9.labs(
            title="Method ranks by dataset",
            subtitle=f"{classifier_label} · hard-budget AUC · 1 = best",
            x="",
            y="",
            fill="Rank",
            caption=(
                "Gray / —: missing result. Ranks are among available methods "
                "within each dataset.\n"
                "Rows ordered by average rank on shared datasets; "
                "ties receive average ranks."
            ),
        )
        + p9.theme_minimal(base_size=11)
        + p9.theme(
            figure_size=(13, 7),
            axis_text_x=p9.element_text(rotation=45, ha="right"),
            panel_grid=p9.element_blank(),
        )
    )
    return average_plot, heatmap


METHOD_CATEGORIES = {
    "ma2018_external": "Myopic",
    "covert2023": "Myopic",
    "gadgil2023": "Myopic",
    "jafa": "Non-myopic",
    "ol_with_mask": "Non-myopic",
    "odin_model_based": "Non-myopic",
    "aaco": "Non-myopic",
    "cae": "Static",
}


def category_rankings(
    common: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    unknown = set(common["afa_method"].to_list()) - METHOD_CATEGORIES.keys()
    if unknown:
        msg = f"No method category assigned for {sorted(unknown)}"
        raise ValueError(msg)
    per_dataset = (
        common.with_columns(
            pl.col("afa_method")
            .replace_strict(METHOD_CATEGORIES)
            .alias("category")
        )
        .group_by("classifier", "dataset", "category")
        .agg(
            pl.col("rank").mean().alias("mean_method_rank"),
            pl.len().alias("n_methods"),
        )
        .sort("dataset", "category")
    )
    summary = (
        per_dataset.group_by("classifier", "category")
        .agg(
            pl.col("mean_method_rank").mean().alias("average_rank"),
            pl.col("n_methods").first(),
            pl.len().alias("n_datasets"),
        )
        .sort("average_rank", "category")
    )
    return per_dataset, summary


def create_category_plot(
    common: pl.DataFrame, summary: pl.DataFrame
) -> p9.ggplot:
    classifier = (
        "Built-in predictors"
        if common["classifier"][0] == "builtin"
        else "External classifier"
    )
    categories: list[str] = summary["category"].to_list()
    n_datasets = common["dataset"].n_unique()
    n_methods = common["afa_method"].n_unique()
    labels = summary.with_columns(
        pl.col("average_rank")
        .map_elements(lambda value: f"{value:.2f}", return_dtype=pl.String)
        .alias("label")
    )
    return (
        p9.ggplot(labels, p9.aes("average_rank", "category"))
        + p9.geom_point(color="#007f86", size=6)
        + p9.geom_text(
            p9.aes(label="label"), nudge_y=-0.23, size=11, color="#00676c"
        )
        + p9.scale_y_discrete(limits=list(reversed(categories)))
        + p9.scale_x_continuous(
            breaks=list(range(1, n_methods + 1)),
            limits=(0.7, n_methods + 0.3),
        )
        + p9.labs(
            title="How do acquisition strategies compare?",
            subtitle=f"{classifier} · {n_datasets} shared datasets",
            x="Mean method rank (lower is better)",
            y="",
            caption=(
                "Methods ranked by hard-budget AUC within each dataset.\n"
                "Equal method weights within categories; equal dataset weights.\n"
                "Static represents one evaluated method."
            ),
        )
        + p9.theme_minimal(base_size=14)
        + p9.theme(figure_size=(9, 4.8), panel_grid_minor=p9.element_blank())
    )


def main(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Path,
) -> None:
    ranked, common, summary = prepare_rankings(pl.read_csv(input_path))
    average_plot, heatmap = create_plots(ranked, common, summary)
    category_datasets, category_summary = category_rankings(common)
    category_plot = create_category_plot(common, category_summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_csv(output_dir / "average_ranks.csv")
    category_datasets.write_csv(output_dir / "category_dataset_ranks.csv")
    category_summary.write_csv(output_dir / "category_average_ranks.csv")
    common.sort("dataset", "rank").write_csv(
        output_dir / "shared_dataset_rankings.csv"
    )
    for name, plot in [
        ("average_rank", average_plot),
        ("rank_heatmap", heatmap),
        ("category_average_rank", category_plot),
    ]:
        for extension in ["png", "svg"]:
            plot.save(
                output_dir / f"{name}.{extension}", dpi=180, verbose=False
            )
    typer.echo(f"Saved ranking plots and summary tables to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
