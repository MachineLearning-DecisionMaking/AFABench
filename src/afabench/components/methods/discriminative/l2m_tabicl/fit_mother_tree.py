"""Fit TFM-Synth on separate Mother-Tree training and validation tasks."""

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

from torch import nn

from afabench.components.methods.discriminative.l2m_tabicl.fit import (
    FitResult,
    fit_policy,
)
from afabench.components.methods.discriminative.l2m_tabicl.mother_tree import (
    MotherTreeTask,
)
from afabench.components.methods.discriminative.l2m_tabicl.policy import (
    AcquisitionPolicy,
)
from afabench.components.methods.discriminative.l2m_tabicl.predictor import (
    FrozenTabICLPredictor,
)


def fit_on_mother_tree(
    predictor: FrozenTabICLPredictor,
    policy: AcquisitionPolicy,
    training_tasks: Sequence[MotherTreeTask],
    validation_tasks: Sequence[MotherTreeTask],
    *,
    n_steps: int,
    checkpoint_interval: int,
    temperature: float = 0.1,
    learning_rate: float = 1e-4,
    seed: int = 0,
    checkpoint_path: Path | None = None,
    resume_path: Path | None = None,
    checkpoint_identity: dict[str, object] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    n_training_evaluation_tasks: int = 0,
    time_budget_seconds: float | None = None,
    checkpoint_callback: Callable[[int], None] | None = None,
) -> FitResult:
    """
    Fit only the policy, respecting each task's saved context/query split.

    Training cycles through its supplied offline task pool. Validation
    always uses its separate pool and fixed random trajectories. Repeated
    files or identical datasets across the pools are rejected.
    """
    _check_task_pools(training_tasks, validation_tasks, policy)
    if n_training_evaluation_tasks > len(training_tasks):
        message = "Training evaluation task count exceeds training pool size"
        raise ValueError(message)
    n_validation = len(validation_tasks)

    def task_factory(task_seed: int) -> MotherTreeTask:
        index = task_seed - seed
        if index < n_validation:
            return validation_tasks[index]
        return training_tasks[(index - n_validation) % len(training_tasks)]

    return fit_policy(
        predictor,
        policy,
        task_factory,
        n_steps=n_steps,
        n_validation_tasks=n_validation,
        checkpoint_interval=checkpoint_interval,
        context_set_size=None,
        temperature=temperature,
        learning_rate=learning_rate,
        seed=seed,
        checkpoint_path=checkpoint_path,
        resume_path=resume_path,
        checkpoint_identity=checkpoint_identity,
        stop_requested=stop_requested,
        n_training_evaluation_tasks=n_training_evaluation_tasks,
        time_budget_seconds=time_budget_seconds,
        checkpoint_callback=checkpoint_callback,
    )


def _check_task_pools(
    training_tasks: Sequence[MotherTreeTask],
    validation_tasks: Sequence[MotherTreeTask],
    policy: AcquisitionPolicy,
) -> None:
    if not training_tasks or not validation_tasks:
        message = "Training and validation pools must both contain tasks"
        raise ValueError(message)
    last_layer = policy.network[-1]
    if not isinstance(last_layer, nn.Linear):
        message = "The L2M policy must end with a linear layer"
        raise TypeError(message)
    for record in (*training_tasks, *validation_tasks):
        if record.task.features.shape[1] != last_layer.out_features:
            message = (
                f"Task has {record.task.features.shape[1]} active features; "
                f"policy requires {last_layer.out_features}"
            )
            raise ValueError(message)
    training_ids = {
        (record.source_path.resolve(), record.dataset_index)
        for record in training_tasks
    }
    validation_ids = {
        (record.source_path.resolve(), record.dataset_index)
        for record in validation_tasks
    }
    if training_ids & validation_ids:
        message = "Training and validation pools share a source dataset"
        raise ValueError(message)
    if {_fingerprint(record) for record in training_tasks} & {
        _fingerprint(record) for record in validation_tasks
    }:
        message = (
            "Training and validation pools contain identical dataset contents"
        )
        raise ValueError(message)


def _fingerprint(record: MotherTreeTask) -> str:
    digest = hashlib.sha256()
    for value in (record.task.features, record.task.labels):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
