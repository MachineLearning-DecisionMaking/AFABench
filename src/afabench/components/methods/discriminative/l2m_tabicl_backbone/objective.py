"""
Myopic L2M acquisitions evaluated on numerical data by full TabICL.

The reference path evaluates each query with its own permitted labeled
prefix. Policy trials never advance the random training trajectory.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch.nn import functional as F

from afabench.components.methods.discriminative.l2m.models import (
    L2MModel,
    UpstreamBatch,
)
from afabench.components.methods.discriminative.l2m_tabicl.predictor import (
    FrozenTabICLPredictor,
)
from afabench.components.methods.discriminative.l2m_tabicl_backbone.numerical import (
    NumericalTransform,
)

type States = Float[torch.Tensor, "batch queries features"]


def safe_batch(batch: UpstreamBatch) -> UpstreamBatch:
    """Zero retrospective missing values before upstream multiplies masks."""
    result = dict(batch)
    for values, available in (("xc", "rc"), ("xt", "rt")):
        x, r = batch[values], batch[available]
        if not ((r == 0) | (r == 1)).all():
            msg = f"{available} must be binary"
            raise ValueError(msg)
        if not torch.isfinite(x[r.bool()]).all():
            msg = f"{values} has nonfinite available measurements"
            raise ValueError(msg)
        result[values] = torch.where(r.bool(), x, 0.0)
    return result


def select(logits: States, available: States, temperature: float) -> States:
    """Straight-through Gumbel selection; exhausted rows return zeros."""
    if not temperature > 0 or not torch.isfinite(logits).all():
        msg = "temperature must be positive and logits finite"
        raise ValueError(msg)
    flat = logits.reshape(-1, logits.shape[-1])
    valid = available.bool().reshape_as(flat)
    indices = valid.any(-1).nonzero().flatten()
    selected = torch.zeros_like(flat)
    if len(indices):
        active = flat[indices].masked_fill(~valid[indices], -torch.inf)
        chosen = F.gumbel_softmax(active, tau=temperature, hard=True, dim=-1)
        selected = selected.index_copy(0, indices, chosen)
    return selected.reshape_as(logits)


def prefix_logits(
    predictor: FrozenTabICLPredictor,
    batch: UpstreamBatch,
    updated: States,
    *,
    task: int,
    query: int,
    n_classes: int,
) -> torch.Tensor:
    """Context plus full retrospectively available *preceding* queries."""
    context = torch.cat((batch["xc"][task], batch["xt"][task, :query]))
    available = torch.cat((batch["rc"][task], batch["rt"][task, :query]))
    context = torch.where(available.bool(), context, torch.nan)
    labels = torch.cat((batch["yc"][task], batch["yt"][task, :query]))
    labels = (
        labels[..., 0].long() if labels.shape[-1] == 1 else labels.argmax(-1)
    )
    transform = NumericalTransform(context)
    x = batch["xt"][task, query : query + 1]
    mask = updated[task, query : query + 1]
    table = torch.cat((transform(context), transform.reveal(x, mask)))
    return predictor(
        table.unsqueeze(0), labels.unsqueeze(0), n_classes=n_classes
    )[0, 0]


@dataclass
class Objective:
    loss: torch.Tensor
    valid_states: int
    final_mask: States


def trajectory_objective(
    model: L2MModel,
    predictor: FrozenTabICLPredictor,
    batch: UpstreamBatch,
    *,
    temperature: float,
    validation_seed: int | None = None,
) -> Objective:
    """Normalize one-acquisition losses along independent random paths."""
    batch = safe_batch(batch)
    mask = torch.zeros_like(batch["xt"])
    losses: list[torch.Tensor] = []
    # Validation uses fixed Gumbel draws and random trajectories, without
    # consuming training RNG. Training uses the caller's seeded generator.
    devices = [batch["xt"].device.index or 0] if batch["xt"].is_cuda else []
    with torch.random.fork_rng(
        devices=devices, enabled=validation_seed is not None
    ):
        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        for _ in range(model.n_features):
            available = (1 - mask) * batch["rt"]
            valid = available.bool().any(-1)
            if not valid.any():
                break
            scores = model.ltm.select_action(
                {**batch, "maskt": mask}, autoreg=True
            )
            selected = select(scores, available, temperature)
            updated = mask + (1 - mask) * selected
            for task, query in valid.nonzero().tolist():
                logits = prefix_logits(
                    predictor,
                    batch,
                    updated,
                    task=task,
                    query=query,
                    n_classes=model.n_classes,
                )
                target = batch["yt"][task, query]
                target = (
                    target[0].long()
                    if model.label_dim == 1
                    else target.argmax(-1)
                )
                losses.append(
                    F.cross_entropy(logits.unsqueeze(0), target.reshape(1))
                )
            # Independent random acquisition advances the actual path.
            random = torch.rand_like(scores).masked_fill(~available.bool(), -1)
            action = F.one_hot(random.argmax(-1), model.n_features)
            mask = torch.maximum(mask, action * valid.unsqueeze(-1)).detach()
    if not losses:
        return Objective(mask.sum() * 0, 0, mask)
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        msg = "hybrid acquisition loss is nonfinite"
        raise FloatingPointError(msg)
    return Objective(loss, len(losses), mask)


def optimizer_for(
    model: L2MModel, *, policy_lr: float, backbone_lr: float
) -> torch.optim.Adam:
    model.requires_grad_(True)
    model.ltm.predictor.requires_grad_(False)
    return torch.optim.Adam(
        [
            {"params": model.ltm.selector.parameters(), "lr": policy_lr},
            {
                "params": [
                    *model.ltm.embedder.parameters(),
                    *model.ltm.encoder.parameters(),
                ],
                "lr": backbone_lr,
            },
        ]
    )
