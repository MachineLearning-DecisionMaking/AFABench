"""Convert saved Mother-Tree TabICL v2 batches into acquisition tasks."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from afabench.components.methods.discriminative.l2m_tabicl.task import Task


@dataclass(frozen=True)
class MotherTreeTask:
    task: Task
    context_set_size: int
    source_path: Path
    dataset_index: int
    class_values: tuple[int, ...]


def load_mother_tree_batch(
    path: Path,
    *,
    expected_features: int | None = None,
    max_classes: int = 10,
) -> list[MotherTreeTask]:
    """
    Read dense or nested v2 batches without acquiring padded columns.

    Keep only the first seq_lens[i] rows and d[i] active columns. Preserve
    train_sizes[i] as the context/query split. Class IDs are mapped to a
    contiguous range using the context labels, without using query labels
    to define that mapping. No generator dependency is needed to read files.
    """
    if max_classes < 2:
        message = f"max_classes={max_classes} must be at least 2"
        raise ValueError(message)
    if expected_features is not None and expected_features < 1:
        message = f"expected_features={expected_features} must be positive"
        raise ValueError(message)
    batch = torch.load(path, map_location="cpu", weights_only=True)
    features = batch["X"]
    labels = batch["y"]
    if features.ndim != 3 or labels.ndim != 2:
        message = "Expected X with three dimensions and y with two"
        raise ValueError(message)
    n_tasks = int(batch["batch_size"])
    if n_tasks < 1 or features.size(0) != n_tasks or labels.size(0) != n_tasks:
        message = f"batch_size={n_tasks} does not match X and y"
        raise ValueError(message)
    dimensions = _metadata(batch["d"], n_tasks, "d")
    lengths = _metadata(batch["seq_lens"], n_tasks, "seq_lens")
    splits = _metadata(batch["train_sizes"], n_tasks, "train_sizes")
    tasks = []
    for index in range(n_tasks):
        n_features = dimensions[index]
        n_rows = lengths[index]
        context_size = splits[index]
        feature_rows = features[index]
        label_rows = labels[index]
        if not (
            1 <= n_features <= feature_rows.shape[1]
            and 2 <= context_size < n_rows <= feature_rows.shape[0]
            and n_rows <= label_rows.shape[0]
        ):
            message = (
                f"Invalid metadata for dataset {index}: "
                f"d={n_features}, seq_len={n_rows}, train_size={context_size}"
            )
            raise ValueError(message)
        if expected_features is not None and n_features != expected_features:
            message = (
                f"Dataset {index} has {n_features} active features; "
                f"the policy requires {expected_features}"
            )
            raise ValueError(message)
        active_features = feature_rows[:n_rows, :n_features].float().clone()
        active_labels = label_rows[:n_rows]
        if not torch.isfinite(active_features).all():
            message = f"Dataset {index} has nonfinite active feature values"
            raise ValueError(message)
        encoded_labels, class_values = _encode_labels(
            active_labels, context_size, max_classes, index
        )
        tasks.append(
            MotherTreeTask(
                task=Task(
                    features=active_features,
                    labels=F.one_hot(
                        encoded_labels, len(class_values)
                    ).float(),
                    feature_mask=torch.ones_like(
                        active_features, dtype=torch.bool
                    ),
                ),
                context_set_size=context_size,
                source_path=path,
                dataset_index=index,
                class_values=class_values,
            )
        )
    return tasks


def _metadata(value: torch.Tensor, n_tasks: int, name: str) -> list[int]:
    if value.shape != (n_tasks,) or not (
        torch.isfinite(value).all() and (value == value.long()).all()
    ):
        message = f"{name} must contain {n_tasks} integer metadata values"
        raise ValueError(message)
    return value.long().tolist()


def _encode_labels(
    active_labels: torch.Tensor,
    context_size: int,
    max_classes: int,
    index: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if not (
        torch.isfinite(active_labels).all()
        and (active_labels >= 0).all()
        and (active_labels == active_labels.long()).all()
    ):
        message = f"Dataset {index} labels must be nonnegative integers"
        raise ValueError(message)
    active_labels = active_labels.long()
    classes = active_labels[:context_size].unique(sorted=True)
    if not 2 <= len(classes) <= max_classes:
        message = (
            f"Dataset {index} has {len(classes)} context classes; "
            f"expected between 2 and {max_classes}"
        )
        raise ValueError(message)
    if not torch.isin(active_labels[context_size:], classes).all():
        message = f"Dataset {index} has query classes absent from context"
        raise ValueError(message)
    encoded_labels = torch.searchsorted(classes, active_labels)
    return encoded_labels, tuple(classes.tolist())
