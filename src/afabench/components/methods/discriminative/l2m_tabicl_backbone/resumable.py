"""Bounded, exact-restart fitting for the backbone experiment's two stages."""

import time
from collections.abc import Callable
from math import inf
from pathlib import Path
from typing import Any

import torch
from torch import nn

from afabench.components.methods.discriminative.l2m_tabicl.training_state import (
    load_training_state,
    save_training_state,
)


def _never_stop() -> bool:
    return False


def fit_resumable(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    checkpoint: Path,
    identity: dict[str, object],
    target_steps: int,
    interval: int,
    training_step: Callable[[], dict[str, float]],
    validation_loss: Callable[[], float],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    time_budget_seconds: float = inf,
    stop_requested: Callable[[], bool] = _never_stop,
    logger: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, object]:
    if min(target_steps, interval) < 1 or time_budget_seconds <= 0:
        msg = "target_steps, interval and time budget must be positive"
        raise ValueError(msg)
    started = time.monotonic()
    identity = {**identity, "n_steps": target_steps}
    generator = torch.Generator().manual_seed(0)
    state = _restore(
        model, optimizer, checkpoint, identity, generator, scheduler
    )
    step, best_step = state["step"], state["best_step"]
    best_loss, best_state, history = (
        state["best_loss"],
        state["best_state"],
        state["history"],
    )

    def validate() -> None:
        nonlocal best_loss, best_state, best_step
        model.eval()
        with torch.no_grad():
            loss = validation_loss()
        _check_loss(loss)
        if loss < best_loss:
            best_loss, best_step = loss, step
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        history[-1]["validation_loss"] = loss

    def persist() -> None:
        save_training_state(
            checkpoint,
            identity,
            model,
            optimizer,
            generator,
            step=step,
            best_step=best_step,
            best_loss=best_loss,
            best_state=best_state,
            history=history,
            scheduler={} if scheduler is None else scheduler.state_dict(),
        )

    if not history:
        history.append({"step": 0.0})
        validate()
        persist()
        _emit(logger, history[-1])
    while step < target_steps:
        if (
            stop_requested()
            or time.monotonic() - started >= time_budget_seconds
        ):
            break
        model.train()
        metrics = training_step()
        step += 1
        history.append({**metrics, "step": float(step)})
        if step % interval == 0 or step == target_steps:
            validate()
            persist()
        _emit(logger, history[-1])
    # Persist the latest optimizer/model even when the selected best policy
    # comes from an earlier update. Resume must continue from latest.
    if "validation_loss" not in history[-1]:
        validate()
    persist()
    model.load_state_dict(best_state)
    return {
        "step": step,
        "target_steps": target_steps,
        "complete": step == target_steps,
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "history": history,
    }


def _restore(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint: Path,
    identity: dict[str, object],
    generator: torch.Generator,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> dict[str, Any]:
    if checkpoint.exists():
        state = load_training_state(
            checkpoint, identity, model, optimizer, generator
        )
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        return state
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    initial = checkpoint.with_name("initial.pt")
    if initial.exists():
        msg = f"initial checkpoint exists without restart state: {initial}"
        raise FileExistsError(msg)
    torch.save(
        {k: v.detach().cpu() for k, v in model.state_dict().items()}, initial
    )
    return {
        "step": 0,
        "best_step": 0,
        "best_loss": float("inf"),
        "best_state": {},
        "history": [],
    }


def _emit(
    logger: Callable[[dict[str, float]], None] | None,
    metrics: dict[str, float],
) -> None:
    if logger is not None:
        logger(metrics)


def _check_loss(loss: float) -> None:
    if not float("-inf") < loss < float("inf"):
        msg = f"validation_loss={loss} must be finite"
        raise FloatingPointError(msg)
