"""
Policy-only updates over L2M-style random acquisition trajectories.

Task generation stays outside this module so the synthetic prior can be
replaced without changing the acquisition objective.
"""

import torch
from torch.nn import functional as F

from afabench.components.methods.discriminative.l2m_tabicl.task import Task
from afabench.components.methods.discriminative.l2m_tabicl.acquisition import (
    acquisition_loss,
)
from afabench.components.methods.discriminative.l2m_tabicl.policy import (
    AcquisitionPolicy,
)
from afabench.components.methods.discriminative.l2m_tabicl.predictor import (
    FrozenTabICLPredictor,
)
from afabench.components.methods.discriminative.l2m_tabicl.preprocessing import (
    ContextPreprocessor,
)


def trajectory_loss(
    predictor: FrozenTabICLPredictor,
    policy: AcquisitionPolicy,
    task: Task,
    *,
    context_set_size: int,
    temperature: float,
    generator: torch.Generator,
    deterministic: bool = False,
) -> torch.Tensor:
    """
    Average one-step acquisition losses along a random trajectory.

    At each state the policy's selection is scored, but a separate random
    selection advances the trajectory, matching the authors' L2M scheme.
    This initial version requires finite, fully available numerical data.
    With deterministic selections, repeat the generator seed to repeat
    validation trajectories as well.
    """
    if not 2 <= context_set_size < len(task.features):
        message = (
            f"context_set_size={context_set_size} must be between 2 "
            f"and {len(task.features) - 1}"
        )
        raise ValueError(message)

    if not torch.isfinite(task.features).all():
        message = "task.features must contain only finite numerical values"
        raise ValueError(message)

    if not task.feature_mask.bool().all():
        message = "Retrospective missingness is not supported yet"
        raise ValueError(message)

    if task.features.ndim != 2 or task.features.shape[1] < 1:
        message = f"Invalid task.features shape={task.features.shape}"
        raise ValueError(message)

    if task.labels.ndim != 2 or len(task.labels) != len(task.features):
        message = f"Invalid task.labels shape={task.labels.shape}"
        raise ValueError(message)

    n_classes = task.labels.shape[1]
    if n_classes < 2:
        message = f"n_classes={n_classes} must be at least 2"
        raise ValueError(message)

    device = next(predictor.parameters()).device

    # Fit preprocessing only on the context features.
    preprocessor = ContextPreprocessor(task.features[:context_set_size])
    features = preprocessor.transform(task.features).to(device)
    labels = task.labels.argmax(dim=-1).to(device)

    context = features[:context_set_size]
    query = features[context_set_size:]

    context_labels = labels[:context_set_size].unsqueeze(0)
    query_labels = labels[context_set_size:]
    baseline = preprocessor.missing_baseline.to(device)

    # Start with every query feature hidden.
    mask = torch.zeros_like(query)
    total_loss = query.new_zeros(())
    n_features = query.shape[1]

    for state in range(n_features):
        # Score the policy's selection from the current state.
        total_loss = total_loss + acquisition_loss(
            predictor,
            policy,
            context_features=context,
            context_labels=context_labels,
            query_features=query,
            query_labels=query_labels,
            feature_mask=mask,
            missing_baseline=baseline,
            n_classes=n_classes,
            temperature=temperature,
            deterministic=deterministic,
        )

        if state == n_features - 1:
            break

        # Advance using a separate random selection, as in L2M.
        # This transition does not depend on policy parameters.
        random_scores = torch.randn(
            mask.shape,
            generator=generator,
            device="cpu",
        ).to(device)

        selected = random_scores.masked_fill(mask.bool(), -torch.inf).argmax(
            dim=-1
        )

        random_selection = F.one_hot(selected, n_features).to(mask.dtype)

        mask = torch.maximum(mask, random_selection)

    # Every state contains the same number of eligible queries.
    return total_loss / n_features


def train_on_task(
    predictor: FrozenTabICLPredictor,
    policy: AcquisitionPolicy,
    optimizer: torch.optim.Optimizer,
    task: Task,
    *,
    context_set_size: int,
    temperature: float,
    generator: torch.Generator,
) -> float:
    """Apply one policy update using all states of one synthetic task."""
    policy.train()
    optimizer.zero_grad()

    loss = trajectory_loss(
        predictor,
        policy,
        task,
        context_set_size=context_set_size,
        temperature=temperature,
        generator=generator,
    )

    if not torch.isfinite(loss):
        message = f"Acquisition loss must be finite; got {loss.item()}"
        raise FloatingPointError(message)

    loss.backward()
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    ):
        optimizer.zero_grad()
        message = "Policy gradients must be finite; update was not applied"
        raise FloatingPointError(message)
    optimizer.step()

    return float(loss.detach())
