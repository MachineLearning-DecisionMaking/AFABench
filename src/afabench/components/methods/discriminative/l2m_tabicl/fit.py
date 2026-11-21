"""Fit a policy on generated tasks and restore its best validation state."""

import logging
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from afabench.components.methods.discriminative.l2m_tabicl.task import Task
from afabench.components.methods.discriminative.l2m_tabicl.mother_tree import (
    MotherTreeTask,
)
from afabench.components.methods.discriminative.l2m_tabicl.policy import (
    AcquisitionPolicy,
)
from afabench.components.methods.discriminative.l2m_tabicl.predictor import (
    FrozenTabICLPredictor,
)
from afabench.components.methods.discriminative.l2m_tabicl.train import (
    train_on_task,
    trajectory_loss,
)
from afabench.components.methods.discriminative.l2m_tabicl.training_state import (
    load_training_state,
    save_training_state,
)


@dataclass(frozen=True)
class FitRecord:
    step: int
    training_loss: float | None
    validation_loss: float
    mean_training_loss: float | None = None
    training_evaluation_loss: float | None = None


@dataclass(frozen=True)
class FitResult:
    best_step: int
    best_validation_loss: float
    history: list[FitRecord]


def fit_policy(  # noqa: C901, PLR0915
    predictor: FrozenTabICLPredictor,
    policy: AcquisitionPolicy,
    task_factory: Callable[[int], Task | MotherTreeTask],
    *,
    n_steps: int,
    n_validation_tasks: int,
    checkpoint_interval: int,
    context_set_size: int | None,
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
    Train on supplied tasks, validate on fixed tasks, restore the best policy.

    The factory must reproducibly generate a task from its supplied seed.
    Training and validation receive disjoint task seeds. Validation also
    repeats fixed random trajectories and uses deterministic selections.
    The initial policy is a checkpoint candidate, allowing zero updates
    to be preferred when the tiny run does not improve validation loss.
    MotherTreeTask records preserve their own context/query split; pass
    context_set_size=None to use it. Plain Tasks require an explicit size.
    Optional atomic disk checkpoints retain the latest training state and
    best policy. Resume requires the same task factory and fit settings.
    """
    _check_counts(n_steps, n_validation_tasks, checkpoint_interval)
    started = time.monotonic()
    if n_training_evaluation_tasks < 0:
        message = "n_training_evaluation_tasks must be nonnegative"
        raise ValueError(message)
    if time_budget_seconds is not None and (
        not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0
    ):
        message = "time_budget_seconds must be finite and positive"
        raise ValueError(message)

    _check_parameters(predictor, seed, learning_rate, temperature)

    # Generate validation tasks once and reuse them at every checkpoint.
    validation_tasks = [
        _task_and_context(task_factory(seed + index), context_set_size)
        for index in range(n_validation_tasks)
    ]

    # Reserve separate seeds for training tasks and random trajectories.
    first_training_seed = seed + n_validation_tasks
    first_trajectory_seed = first_training_seed + n_steps
    training_evaluation_tasks = [
        _task_and_context(
            task_factory(first_training_seed + index), context_set_size
        )
        for index in range(n_training_evaluation_tasks)
    ]

    training_generator = torch.Generator().manual_seed(
        first_trajectory_seed + n_validation_tasks
    )

    # The authors' selection helper samples through PyTorch's global RNG.
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def evaluate(tasks: list[tuple[Task, int]]) -> float:
        policy.eval()

        with torch.no_grad():
            losses = [
                float(
                    trajectory_loss(
                        predictor,
                        policy,
                        task,
                        context_set_size=task_context_size,
                        temperature=temperature,
                        generator=torch.Generator().manual_seed(
                            first_trajectory_seed + index
                        ),
                        deterministic=True,
                    )
                )
                for index, (task, task_context_size) in enumerate(tasks)
            ]

        if not all(math.isfinite(loss) for loss in losses):
            message = f"Validation losses must be finite; got {losses}"
            raise FloatingPointError(message)

        return sum(losses) / len(losses)

    def validate() -> float:
        return evaluate(validation_tasks)

    # Include the initial policy so training cannot force us to keep
    # a policy with worse validation loss.
    identity: dict[str, object] = {
        "n_steps": n_steps,
        "n_validation_tasks": n_validation_tasks,
        "checkpoint_interval": checkpoint_interval,
        "context_set_size": context_set_size,
        "temperature": temperature,
        "learning_rate": learning_rate,
        "seed": seed,
        "n_training_evaluation_tasks": n_training_evaluation_tasks,
        "inputs": checkpoint_identity,
    }
    state = (
        load_training_state(
            resume_path, identity, policy, optimizer, training_generator
        )
        if resume_path is not None
        else None
    )
    best_loss, best_step, best_state, history, start_step = _initial_progress(
        state, policy, validate
    )
    if state is None and training_evaluation_tasks:
        history[0] = replace(
            history[0],
            training_evaluation_loss=evaluate(training_evaluation_tasks),
        )

    def persist(step: int) -> None:
        if checkpoint_path is not None:
            save_training_state(
                checkpoint_path,
                identity,
                policy,
                optimizer,
                training_generator,
                step=step,
                best_step=best_step,
                best_loss=best_loss,
                best_policy=best_state,
                history=[asdict(record) for record in history],
            )

    persist(start_step)
    if checkpoint_path is not None and state is None:
        shutil.copy2(checkpoint_path, checkpoint_path.with_name("initial.pt"))
    if checkpoint_callback is not None and state is None:
        checkpoint_callback(0)
    window_losses = []
    for step in range(start_step + 1, n_steps + 1):
        task, task_context_size = _task_and_context(
            task_factory(first_training_seed + step - 1), context_set_size
        )
        training_loss = train_on_task(
            predictor,
            policy,
            optimizer,
            task,
            context_set_size=task_context_size,
            temperature=temperature,
            generator=training_generator,
        )
        window_losses.append(training_loss)

        # Always validate the final step as well.
        stopping = (stop_requested is not None and stop_requested()) or (
            time_budget_seconds is not None
            and time.monotonic() - started >= time_budget_seconds
        )
        if step % checkpoint_interval and step != n_steps and not stopping:
            continue

        validation_loss = validate()

        history.append(
            FitRecord(
                step,
                training_loss,
                validation_loss,
                sum(window_losses) / len(window_losses),
                evaluate(training_evaluation_tasks)
                if training_evaluation_tasks
                else None,
            )
        )
        window_losses.clear()
        logging.getLogger(__name__).info(
            "Step %d/%d: training_loss=%.6f validation_loss=%.6f",
            step,
            n_steps,
            training_loss,
            validation_loss,
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_step = step

            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in policy.state_dict().items()
            }

        persist(step)
        if checkpoint_callback is not None:
            checkpoint_callback(step)
        if stopping:
            break

    policy.load_state_dict(best_state)
    policy.eval()

    return FitResult(best_step, best_loss, history)


def _check_counts(
    n_steps: int,
    n_validation_tasks: int,
    checkpoint_interval: int,
) -> None:
    counts = {
        "n_steps": n_steps,
        "n_validation_tasks": n_validation_tasks,
        "checkpoint_interval": checkpoint_interval,
    }

    for name, value in counts.items():
        if value < 1:
            message = f"{name}={value} must be at least 1"
            raise ValueError(message)


def _task_and_context(
    item: Task | MotherTreeTask, context_set_size: int | None
) -> tuple[Task, int]:
    if isinstance(item, MotherTreeTask):
        if (
            context_set_size is not None
            and context_set_size != item.context_set_size
        ):
            message = (
                "Configured context size conflicts with Mother-Tree metadata"
            )
            raise ValueError(message)
        return item.task, item.context_set_size
    if context_set_size is None:
        message = "Tasks without split metadata require context_set_size"
        raise ValueError(message)
    return item, context_set_size


def _check_parameters(
    predictor: FrozenTabICLPredictor,
    seed: int,
    learning_rate: float,
    temperature: float,
) -> None:
    if seed < 0:
        message = f"seed={seed} must be nonnegative"
        raise ValueError(message)

    for name, value in {
        "learning_rate": learning_rate,
        "temperature": temperature,
    }.items():
        if not math.isfinite(value) or value <= 0:
            message = f"{name}={value} must be finite and positive"
            raise ValueError(message)

    if any(parameter.requires_grad for parameter in predictor.parameters()):
        message = "All TabICL parameters must be frozen before policy fitting"
        raise ValueError(message)


def _initial_progress(
    state: dict[str, Any] | None,
    policy: AcquisitionPolicy,
    validate: Callable[[], float],
) -> tuple[float, int, dict[str, torch.Tensor], list[FitRecord], int]:
    if state is not None:
        return (
            state["best_loss"],
            state["best_step"],
            state["best_policy"],
            [FitRecord(**record) for record in state["history"]],
            state["step"],
        )
    best_loss = validate()
    best_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.state_dict().items()
    }
    return best_loss, 0, best_state, [FitRecord(0, None, best_loss)], 0
