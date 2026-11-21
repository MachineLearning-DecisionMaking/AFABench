"""
L2M-style myopic policy updates with random-prefix training states.

Local adaptation of the supplied afa_training_step: fixed labeled context,
classification only, individual features, unit acquisition costs. No L2M
sampler, autoregressive query-label context, or pipeline runner is required.
Training-only hidden query values enter only through a permitted reveal.
"""

import math
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch.nn import functional as F

from afabench.components.methods.discriminative.icl_policy.model import (
    ICLPolicy,
)

type Features = Float[torch.Tensor, "batch instances features"]
type Labels = Int[torch.Tensor, "batch instances"]
type Mask = Bool[torch.Tensor, "batch instances features"]


@dataclass
class AcquisitionBatch:
    """Numeric task values; masks hide values before model-owned preprocessing."""

    context_features: Features
    context_labels: Labels
    query_features: Features
    query_labels: Labels
    context_mask: Mask
    query_available: Mask

    def validate(self, model: ICLPolicy, n_classes: int) -> None:
        c, q = self.context_features, self.query_features
        if (
            c.ndim != 3
            or q.ndim != 3
            or (c.shape[0] != q.shape[0] or c.shape[-1] != q.shape[-1])
            or min(c.shape[:2]) < 1
            or q.shape[1] < 1
        ):
            message = "Need nonempty [batch, context/query, features] tensors"
            raise ValueError(message)
        if q.shape[-1] != model.policy.n_features:
            message = (
                f"query feature width must equal {model.policy.n_features}"
            )
            raise ValueError(message)
        if not 2 <= n_classes <= model.tabicl.max_classes:
            message = f"n_classes={n_classes} exceeds the classifier capacity"
            raise ValueError(message)
        for values, mask, labels in (
            (c, self.context_mask, self.context_labels),
            (q, self.query_available, self.query_labels),
        ):
            if mask.shape != values.shape or mask.dtype != torch.bool:
                message = (
                    "Feature masks must be Boolean and match feature shapes"
                )
                raise ValueError(message)
            if labels.shape != values.shape[:2] or labels.dtype != torch.long:
                message = "Labels must be int64 [batch, instances] class IDs"
                raise ValueError(message)
            if (
                mask.device != c.device
                or labels.device != c.device
                or values.device != c.device
            ):
                message = "All batch tensors must share a device"
                raise ValueError(message)
            if not torch.isfinite(values[mask]).all():
                message = "Available/observed feature values must be finite"
                raise ValueError(message)
            if not ((labels >= 0) & (labels < n_classes)).all():
                message = f"Labels must be in [0, {n_classes})"
                raise ValueError(message)


@dataclass
class TrainingResult:
    loss: float
    scored_states: int
    optimizer_steps: int


def select_feature(
    logits: Features, valid: Mask, temperature: float
) -> Features:
    """Hard one-hot forward, soft gradient backward; exhausted rows are zero."""
    if not math.isfinite(temperature) or temperature <= 0:
        message = f"temperature={temperature} must be finite and positive"
        raise ValueError(message)
    if logits.shape != valid.shape or valid.dtype != torch.bool:
        message = "valid must be Boolean with the same shape as logits"
        raise ValueError(message)
    active = valid.any(-1)
    if not active.any():
        return torch.zeros_like(logits)
    # P05 VALID SELECTIONS: do not softmax an exhausted all--inf vector.
    allowed = logits[active].masked_fill(~valid[active], -torch.inf)
    choices = F.gumbel_softmax(allowed, tau=temperature, hard=True, dim=-1)
    output = torch.zeros_like(logits)
    output[active] = choices
    return output


def build_masked_table(
    batch: AcquisitionBatch, query_mask: Features
) -> Features:
    """Hide values with NaN; imputation belongs to the model interface."""
    if (
        query_mask.requires_grad
        or not ((query_mask == 0) | (query_mask == 1)).all()
    ):
        message = "Use a hard observed mask here; relaxed reveals use classify_reveal"
        raise ValueError(message)
    context = torch.where(
        batch.context_mask, batch.context_features, torch.nan
    )
    visible = query_mask.bool() & batch.query_available
    query = torch.where(visible, batch.query_features, torch.nan)
    return torch.cat((context, query), dim=1)


def make_optimizer(
    model: ICLPolicy,
    *,
    policy_lr: float = 1e-4,
    backbone_lr: float = 1e-5,
) -> torch.optim.Adam:
    """Policy plus every pretrained component, at separate learning rates."""
    if any(not math.isfinite(v) or v <= 0 for v in (policy_lr, backbone_lr)):
        message = "policy_lr and backbone_lr must be finite and positive"
        raise ValueError(message)
    groups = [
        {
            "params": list(model.policy.parameters()),
            "lr": policy_lr,
            "name": "policy",
        }
    ]
    pretrained = [p for p in model.tabicl.parameters() if p.requires_grad]
    if pretrained:
        groups.append(
            {"params": pretrained, "lr": backbone_lr, "name": "tabicl"}
        )
    return torch.optim.Adam(groups)


def _validate_optimizer(
    model: ICLPolicy, optimizer: torch.optim.Optimizer
) -> None:
    # P07 UPDATES: every trainable parameter must be optimized exactly once.
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    optimized = [
        id(p) for group in optimizer.param_groups for p in group["params"]
    ]
    if set(optimized) != expected or len(optimized) != len(expected):
        message = (
            "Optimizer must include every trainable parameter exactly once"
        )
        raise ValueError(message)


def afa_training_step(
    model: ICLPolicy,
    batch: AcquisitionBatch,
    optimizer: torch.optim.Optimizer,
    *,
    n_classes: int,
    max_steps: int,
    temperature: float = 0.1,
    inner_loop_steps: int = 1,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> TrainingResult:
    """
    Average post-reveal CE over valid query/state pairs per inner update.

    Cold-start random-prefix states, one feature per step. max_steps is a
    hard count of unit-cost selections per query, not a soft budget. Loss
    and counts returned describe the last inner iteration, as in L2M.
    Scheduler advances once per call only if an optimizer update occurred.
    """
    batch.validate(model, n_classes)
    if max_steps < 1 or inner_loop_steps < 1:
        message = "max_steps and inner_loop_steps must be positive"
        raise ValueError(message)
    if not math.isfinite(temperature) or temperature <= 0:
        message = f"temperature={temperature} must be finite and positive"
        raise ValueError(message)
    _validate_optimizer(model, optimizer)
    model.train()
    updates = 0
    last_loss = 0.0
    total_states = 0
    # Each random-prefix step consumes exactly one available feature per
    # active query. Its total number of scored states is known in advance.
    # Accumulate gradients per state rather than retaining every state's
    # TabICL graph until the end; keep one Adam update for the whole batch.
    expected_states = int(
        batch.query_available.sum(-1).clamp(max=max_steps).sum().item()
    )
    for _ in range(inner_loop_steps):
        mask = torch.zeros_like(batch.query_available)
        total_loss = 0.0
        total_states = 0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(min(max_steps, model.policy.n_features)):
            valid = batch.query_available & ~mask
            active = valid.any(-1)
            if not active.any():
                break
            # No query labels enter the model; they are used only for CE.
            current = model(
                build_masked_table(batch, mask.to(batch.query_features.dtype)),
                batch.context_labels,
                n_classes=n_classes,
            )
            selected = select_feature(
                current.policy_scores, valid, temperature
            )
            # Preserve hard acquisition forward + soft reveal gradient.
            logits = model.classify_reveal(
                build_masked_table(batch, mask.to(batch.query_features.dtype)),
                batch.context_labels,
                torch.where(
                    batch.query_available, batch.query_features, torch.nan
                ),
                selected,
                n_classes=n_classes,
            )
            loss = F.cross_entropy(
                logits[active], batch.query_labels[active], reduction="sum"
            )
            if not torch.isfinite(loss):
                message = "Nonfinite acquisition loss"
                raise FloatingPointError(message)
            total_loss += float(loss.detach())
            (loss / expected_states).backward()
            total_states += int(active.sum().item())
            # Random available selections advance states, NOT policy choices.
            with torch.no_grad():
                random = select_feature(
                    torch.zeros_like(selected), valid, temperature
                )
                mask = mask | random.bool()
            del current, selected, logits, loss
        if total_states:
            assert total_states == expected_states, (
                "Scored state count differs from loss normalization"
            )
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                message = "Nonfinite acquisition gradient"
                raise FloatingPointError(message)
            optimizer.step()
            updates += 1
            last_loss = total_loss / total_states
    if scheduler is not None and updates:
        scheduler.step()
    return TrainingResult(last_loss, total_states, updates)
