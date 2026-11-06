"""Summarize a dataset-aware missing-training-data evaluation matrix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sklearn.metrics import f1_score

_EXPERIMENT_PATTERN = re.compile(
    r"^method-(?P<method>.+?)\+mechanism-(?P<mechanism>.+?)"
    r"\+p-(?P<p>.+?)\+strategy-(?P<strategy>.+?)"
    r"\+instance-(?P<instance>\d+)"
    r"\+train_hard_budget-(?P<train_budget>[^+]+)"
    r"\+eval_hard_budget-(?P<eval_budget>[^+]+)$"
)
_RESTORATION_PATTERN = re.compile(
    r"dataset-(?P<dataset>[^/]+)/"
    r"mechanism-(?P<mechanism>[^/]+)\+p-(?P<p>[^/]+)/"
    r"instance-(?P<instance>\d+)/(?P<strategy>[^/]+)/"
    r"(?P<split>train|val)\.bundle/manifest\.json$"
)
_STRATEGY_DISPLAY = {
    "complete": "Complete data",
    "restricted": "Restricted-action training",
    "mean_fill": "Mean completion",
    "zero_fill": "Zero fill",
    "mice_fill": "MICE",
    "knn_fill": "kNN",
    "pvae_label_conditioned": "Generative restoration",
    "pvae_label_free": "PVAE (label-free)",
    "pvae_stepwise": "Stepwise (PVAE)",
    "pvae_oracle": "PVAE (oracle)",
    "pvae_multi_draw": "PVAE (5 draws)",
    "true_completion": "True completion",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--restored-root", type=Path, required=True)
    parser.add_argument("--instance-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--action-output", type=Path, required=True)
    parser.add_argument("--restoration-output", type=Path, required=True)
    parser.add_argument(
        "--base-method",
        action="append",
        default=[],
        metavar="VARIANT=BASE",
    )
    return parser.parse_args()


def _base_method_mapping(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        try:
            variant, base = value.split("=", maxsplit=1)
        except ValueError as error:
            msg = f"Expected VARIANT=BASE, got {value!r}."
            raise ValueError(msg) from error
        mapping[variant] = base
    return mapping


def _metadata(path: Path) -> dict[str, object]:
    match = _EXPERIMENT_PATTERN.fullmatch(path.parent.name)
    if match is None:
        msg = f"Could not parse experiment metadata from {path}."
        raise ValueError(msg)
    dataset_dir = path.parent.parent.name
    if not dataset_dir.startswith("dataset-"):
        msg = f"Could not parse dataset metadata from {path}."
        raise ValueError(msg)
    values = match.groupdict()
    strategy = str(values["strategy"])
    return {
        "dataset": dataset_dir.removeprefix("dataset-"),
        "method": str(values["method"]),
        "mechanism": str(values["mechanism"]),
        "p": float(str(values["p"])),
        "strategy": strategy,
        "instance": int(str(values["instance"])),
        "train_hard_budget": float(str(values["train_budget"])),
        "eval_hard_budget": float(str(values["eval_budget"])),
        "strategy_display_name": _STRATEGY_DISPLAY[strategy],
    }


def _final_rows(frame: pd.DataFrame) -> pd.DataFrame:
    final = frame.loc[frame["action_performed"] == 0].copy()
    if final.empty:
        msg = "No completed evaluation episodes found."
        raise ValueError(msg)
    return final


def _instance_metrics(
    path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    frame = pd.read_parquet(path)
    final = _final_rows(frame)
    metadata = _metadata(path)
    external_prediction = final["external_predicted_class"]
    has_missing_prediction = external_prediction.isna().to_numpy().any()
    if bool(has_missing_prediction):
        msg = f"External classifier predictions are missing from {path}."
        raise ValueError(msg)

    selections = final["prev_selections_performed"].map(len)
    result = metadata | {
        "n_samples": len(final),
        "accuracy": float((external_prediction == final["true_class"]).mean()),
        "f_score": float(
            f1_score(
                final["true_class"],
                external_prediction,
                average="macro",
                zero_division=cast("Any", 0),
            )
        ),
        "mean_selections": float(selections.mean()),
        "mean_cost": float(final["accumulated_cost"].mean()),
        "forced_stop_rate": float(final["forced_stop"].mean()),
    }

    action_counts: dict[int, int] = {}
    for episode in final["prev_selections_performed"]:
        for selection in episode:
            selection_idx = int(selection)
            action_counts[selection_idx] = (
                action_counts.get(selection_idx, 0) + 1
            )
    action_rows = [
        metadata
        | {
            "selection": selection,
            "acquisitions_per_sample": count / len(final),
            "share_of_acquisitions": count
            / max(sum(action_counts.values()), 1),
        }
        for selection, count in sorted(action_counts.items())
    ]
    return result, action_rows


def _add_complete_data_gaps(
    frame: pd.DataFrame, base_methods: dict[str, str]
) -> pd.DataFrame:
    references = frame.loc[
        frame["strategy"] == "complete",
        ["dataset", "method", "instance", "accuracy", "f_score"],
    ].rename(
        columns={
            "method": "reference_method",
            "accuracy": "complete_accuracy",
            "f_score": "complete_f_score",
        }
    )
    output = frame.copy()
    output["reference_method"] = output["method"].replace(base_methods)
    output = output.merge(
        references,
        on=["dataset", "reference_method", "instance"],
        how="left",
        validate="many_to_one",
    )
    output["accuracy_gap_to_complete"] = (
        output["accuracy"] - output["complete_accuracy"]
    )
    output["f_score_gap_to_complete"] = (
        output["f_score"] - output["complete_f_score"]
    )
    return output.drop(columns="reference_method")


def _aggregate(instance_frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "dataset",
        "method",
        "mechanism",
        "p",
        "strategy",
        "strategy_display_name",
        "train_hard_budget",
        "eval_hard_budget",
    ]
    metric_columns = [
        "accuracy",
        "f_score",
        "accuracy_gap_to_complete",
        "f_score_gap_to_complete",
        "mean_selections",
        "mean_cost",
        "forced_stop_rate",
    ]
    grouped = instance_frame.groupby(group_columns, dropna=False)
    mean = grouped[metric_columns].mean().add_suffix("_mean")
    sem = grouped[metric_columns].sem().add_suffix("_sem")
    count = cast("pd.Series", grouped["instance"].nunique()).to_frame(
        name="n_instances"
    )
    return count.join(mean).join(sem).reset_index()


def _restoration_metrics(restored_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(restored_root.glob("**/manifest.json")):
        match = _RESTORATION_PATTERN.search(path.as_posix())
        if match is None:
            continue
        manifest = json.loads(path.read_text())
        metadata = manifest["metadata"]
        values = match.groupdict()
        rows.append(
            {
                "dataset": values["dataset"],
                "mechanism": values["mechanism"],
                "p": float(values["p"]),
                "instance": int(values["instance"]),
                "strategy": values["strategy"],
                "strategy_display_name": _STRATEGY_DISPLAY[values["strategy"]],
                "split": values["split"],
                "imputation_rmse": metadata.get("imputation_rmse"),
            }
        )
    return pd.DataFrame.from_records(rows)


def _evaluation_paths(input_root: Path) -> list[Path]:
    """Discover only dataset-scoped artifacts from the current layout."""
    return sorted(input_root.glob("dataset-*/**/eval_data.parquet"))


def main() -> None:
    args = _parse_args()
    instance_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    for path in _evaluation_paths(args.input_root):
        instance_result, experiment_actions = _instance_metrics(path)
        instance_rows.append(instance_result)
        action_rows.extend(experiment_actions)
    if not instance_rows:
        msg = f"No evaluation parquet files found under {args.input_root}."
        raise FileNotFoundError(msg)

    instance_frame = _add_complete_data_gaps(
        pd.DataFrame.from_records(instance_rows),
        _base_method_mapping(args.base_method),
    )
    summary_frame = _aggregate(instance_frame)
    action_frame = pd.DataFrame.from_records(action_rows)
    restoration_frame = _restoration_metrics(args.restored_root)

    for output in [
        args.instance_output,
        args.summary_output,
        args.action_output,
        args.restoration_output,
    ]:
        output.parent.mkdir(parents=True, exist_ok=True)
    instance_frame.to_csv(args.instance_output, index=False)
    summary_frame.to_csv(args.summary_output, index=False)
    action_frame.to_csv(args.action_output, index=False)
    restoration_frame.to_csv(args.restoration_output, index=False)


if __name__ == "__main__":
    main()
