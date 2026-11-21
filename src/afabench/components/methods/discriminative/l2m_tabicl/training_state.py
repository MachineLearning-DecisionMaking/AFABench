"""Atomic restart state for policy fitting; frozen weights stay in the input."""

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_training_state(
    path: Path,
    identity: dict[str, object],
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    **progress: object,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    state = {
        "format_version": 1,
        "identity": identity,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "generator": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()
        if torch.cuda.is_initialized()
        else [],
        **progress,
    }
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    # A readable progress sidecar distinguishes an early stop from completion.
    summary = {
        "step": progress["step"],
        "target_steps": identity["n_steps"],
        "best_step": progress["best_step"],
        "best_validation_loss": progress["best_loss"],
        "history": progress["history"],
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if torch.cuda.is_initialized()
        else 0,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()
        if torch.cuda.is_initialized()
        else 0,
    }
    summary_path = path.with_suffix(".json")
    summary_temporary = summary_path.with_suffix(".json.tmp")
    summary_temporary.write_text(json.dumps(summary, indent=2) + "\n")
    summary_temporary.replace(summary_path)


def load_training_state(
    path: Path,
    identity: dict[str, object],
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["format_version"] != 1 or state["identity"] != identity:
        message = f"Resume settings or input hashes differ from {path}"
        raise ValueError(message)
    policy.load_state_dict(state["policy"])
    optimizer.load_state_dict(state["optimizer"])
    generator.set_state(state["generator"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state
