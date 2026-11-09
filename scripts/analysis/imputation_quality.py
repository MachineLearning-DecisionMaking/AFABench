"""Write imputation RMSE and restored-data scores for restoration views."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from sklearn.metrics import f1_score

from afabench.core.bundle_system.bundle import load_bundle

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path("extra/output/missing_data")
OUTPUT = ROOT / "summary/val/imputation_quality.csv"
FIELDS = [
    "dataset",
    "instance",
    "strategy",
    "split",
    "rmse",
    "restored_accuracy",
    "complete_accuracy",
    "restored_f_score",
    "complete_f_score",
]


def masked_rmse(
    restored: torch.Tensor, truth: torch.Tensor, missing: torch.Tensor
) -> float:
    repeats = len(restored) // len(truth)
    truth = truth.repeat(repeats, *([1] * (truth.ndim - 1)))
    if not missing.any():
        return 0.0
    return float((restored[missing] - truth[missing]).square().mean().sqrt())


def scores(
    classifier: Any,  # noqa: ANN401
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    flat = features.flatten(start_dim=1)
    with torch.no_grad():
        logits = classifier(
            flat,
            torch.ones_like(flat, dtype=torch.bool),
            feature_shape=features.shape[1:],
        )
    predicted = logits.argmax(-1).cpu().numpy()
    true = labels.argmax(-1).cpu().numpy()
    f_score = f1_score(
        true, predicted, average="macro", zero_division=cast("Any", 0)
    )
    return float((predicted == true).mean()), float(f_score)


def _views(namespace: str, rate: str) -> Iterator[tuple[Path, str, str]]:
    for base in ("views/base", "views/restored"):
        pattern = f"dataset-*/mechanism-{rate}/instance-*/*/*.bundle"
        for path in sorted((ROOT / base / namespace).glob(pattern)):
            if path.parent.name == "restricted":
                continue
            dataset = path.parts[-5].removeprefix("dataset-")
            instance = path.parts[-3].removeprefix("instance-")
            yield path, dataset, instance


def rows(namespace: str, rate: str) -> Iterator[dict[str, object]]:
    for path, dataset, instance in _views(namespace, rate):
        split = path.stem
        reference, _ = load_bundle(
            ROOT / "datasets" / namespace / dataset / instance / path.name
        )
        truth, labels = cast("Any", reference).get_all_data()
        view = cast("Any", load_bundle(path)[0])
        row: dict[str, object] = {
            "dataset": dataset,
            "instance": instance,
            "strategy": path.parent.name,
            "split": split,
            "rmse": masked_rmse(
                view.features, truth, ~view.source_availability
            ),
        }
        if split == "val":
            classifier, _ = load_bundle(
                ROOT
                / "classifier"
                / namespace
                / f"dataset-{dataset}"
                / f"instance-{instance}.bundle",
                device=torch.device("cpu"),
            )
            (
                row["restored_accuracy"],
                row["restored_f_score"],
            ) = scores(classifier, view.features, view.labels)
            (
                row["complete_accuracy"],
                row["complete_f_score"],
            ) = scores(classifier, truth, labels)
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("namespaces", nargs="+")
    parser.add_argument("--rate", default="mcar+p-0.7")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for namespace in arguments.namespaces:
            for row in rows(namespace, arguments.rate):
                writer.writerow(row)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
