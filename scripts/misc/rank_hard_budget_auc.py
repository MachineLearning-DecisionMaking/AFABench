"""
Rank merged prediction tables by classifier-specific hard-budget AUC.

Uses final predictions, macro-F1 on the specified datasets and accuracy on
others. Defaults match the recovered February 2026 snapshot. Integrates only
between observed budget endpoints (no extrapolation), per seed pair, then
averages AUCs. Missing methods stay absent; inconsistent coverage raises.

Example:
    uv run python scripts/misc/rank_hard_budget_auc.py INPUT.parquet OUTPUT_DIR
"""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
from sklearn.metrics import accuracy_score, f1_score

CURVE_KEYS = ["dataset", "afa_method", "train_seed", "eval_seed"]
POINT_KEYS = [*CURVE_KEYS, "eval_hard_budget"]
DEFAULT_F1_DATASETS = ["physionet", "bank_marketing"]


class Classifier(StrEnum):
    EXTERNAL = "external"
    BUILTIN = "builtin"


def aggregate_predictions(
    input_path: Path, classifier: Classifier = Classifier.EXTERNAL
) -> pl.DataFrame:
    return (
        pl.scan_parquet(input_path)
        .filter(
            (pl.col("classifier") == classifier.value)
            & (pl.col("action_performed") == 0)
            & pl.col("eval_hard_budget").is_not_null()
            & pl.col("predicted_class").is_not_null()
            & pl.col("train_soft_budget_param").is_null()
            & pl.col("eval_soft_budget_param").is_null()
        )
        .group_by(
            *POINT_KEYS,
            "classifier",
            "train_hard_budget",
            "true_class",
            "predicted_class",
        )
        .len()
        .collect(engine="streaming")
    )


def calculate_metrics(
    counts: pl.DataFrame, f1_datasets: list[str]
) -> pl.DataFrame:
    if counts.is_empty():
        msg = "No final predictions found for the selected classifier and hard budgets"
        raise ValueError(msg)
    variants = counts.group_by(POINT_KEYS).agg(
        pl.col("train_hard_budget").n_unique().alias("variants")
    )
    if variants["variants"].max() != 1:
        msg = "Multiple training budgets at the same curve point"
        raise ValueError(msg)
    rows: list[dict[str, object]] = []
    for group in counts.partition_by(POINT_KEYS):
        row = {key: group[key][0] for key in POINT_KEYS}
        metric = "macro_f1" if row["dataset"] in f1_datasets else "accuracy"
        true = group["true_class"].to_numpy()
        predicted = group["predicted_class"].to_numpy()
        weights = group["len"].to_numpy()
        score = (
            f1_score(
                true,
                predicted,
                sample_weight=weights,
                average="macro",
            )
            if metric == "macro_f1"
            else accuracy_score(true, predicted, sample_weight=weights)
        )
        result: dict[str, object] = dict(row)
        result.update(
            metric=metric,
            classifier=group["classifier"][0],
            score=float(score),
            n_predictions=int(weights.sum()),
            train_hard_budget=group["train_hard_budget"][0],
        )
        rows.append(result)
    return pl.DataFrame(rows).sort(POINT_KEYS)


def calculate_auc(metrics: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in metrics.partition_by("dataset"):
        name = dataset["dataset"][0]
        expected_grid = sorted(dataset["eval_hard_budget"].unique().to_list())
        expected_seeds = set(
            dataset.select("train_seed", "eval_seed").iter_rows()
        )
        for method in dataset.partition_by("afa_method"):
            label = f"{name}/{method['afa_method'][0]}"
            seeds = set(method.select("train_seed", "eval_seed").iter_rows())
            if seeds != expected_seeds:
                msg = f"Inconsistent seed coverage for {label}"
                raise ValueError(msg)
            for unsorted in method.partition_by(["train_seed", "eval_seed"]):
                curve = unsorted.sort("eval_hard_budget")
                grid = curve["eval_hard_budget"].to_list()
                if grid != expected_grid or len(grid) < 2:
                    msg = (
                        f"Inconsistent or insufficient budget grid for {label}: "
                        f"{grid}; expected {expected_grid}"
                    )
                    raise ValueError(msg)
                x = curve["eval_hard_budget"].to_numpy()
                if not np.isfinite(x).all() or not np.all(np.diff(x) > 0):
                    msg = f"Invalid budget grid for {label}: {grid}"
                    raise ValueError(msg)
                area = float(np.trapezoid(curve["score"].to_numpy(), x))
                result: dict[str, object] = {
                    key: curve[key][0] for key in CURVE_KEYS
                }
                result.update(
                    metric=curve["metric"][0],
                    classifier=curve["classifier"][0],
                    auc=area,
                    normalized_auc=area / float(x[-1] - x[0]),
                    budget_min=float(x[0]),
                    budget_max=float(x[-1]),
                    n_budgets=len(x),
                )
                rows.append(result)
    return pl.DataFrame(rows).sort(CURVE_KEYS)


def rank_methods(aucs: pl.DataFrame) -> pl.DataFrame:
    return (
        aucs.group_by("dataset", "afa_method", "metric", "classifier")
        .agg(
            pl.col("auc").mean().alias("mean_auc"),
            pl.col("auc").std().alias("std_auc"),
            pl.col("normalized_auc").mean().alias("mean_normalized_auc"),
            pl.col("normalized_auc").std().alias("std_normalized_auc"),
            pl.len().alias("n_seed_pairs"),
            pl.col("budget_min").first(),
            pl.col("budget_max").first(),
            pl.col("n_budgets").first(),
        )
        .with_columns(
            pl.col("mean_auc")
            .rank(method="average", descending=True)
            .over("dataset")
            .alias("rank")
        )
        .sort("dataset", "rank", "afa_method")
    )


def main(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Path,
    f1_dataset: Annotated[
        list[str],
        typer.Option(help="Macro-F1 dataset; repeat to replace defaults."),
    ] = DEFAULT_F1_DATASETS,
    classifier: Classifier = Classifier.EXTERNAL,
) -> None:
    counts = aggregate_predictions(input_path, classifier)
    metrics = calculate_metrics(counts, f1_dataset)
    aucs = calculate_auc(metrics)
    rankings = rank_methods(aucs)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts.write_parquet(output_dir / "confusion_counts.parquet")
    metrics.write_parquet(output_dir / "budget_metrics.parquet")
    metrics.write_csv(output_dir / "budget_metrics.csv")
    aucs.write_csv(output_dir / "seed_auc.csv")
    rankings.write_csv(output_dir / "dataset_rankings.csv")
    rankings.select("dataset", "afa_method", "rank").pivot(
        on="afa_method", index="dataset", values="rank"
    ).write_csv(output_dir / "rank_matrix.csv")
    typer.echo(
        f"Ranked {rankings.height} dataset/method pairs in {output_dir}"
    )


if __name__ == "__main__":
    typer.run(main)
