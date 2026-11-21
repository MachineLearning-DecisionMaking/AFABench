"""Resumable policy training and later evaluation on the predictor manifest."""

import fcntl
import hashlib
import json
import math
import os
import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import typer
from torch import Tensor
from torch.nn import functional as F

from afabench.components.methods.discriminative.icl_policy.model import (
    ICLPolicy,
)
from afabench.components.methods.discriminative.icl_policy.tasks import (
    make_sampler,
    sample_task,
)
from afabench.components.methods.discriminative.icl_policy.train import (
    AcquisitionBatch,
    afa_training_step,
    build_masked_table,
    make_optimizer,
)
from afabench.core.bundle_system.bundle import load_bundle
from afabench.core.types import AFADataset

app = typer.Typer()


def file_hash(path: Path) -> str:
    if path.is_dir():
        digest = hashlib.sha256()
        for item in sorted(path.rglob("*")):
            if item.is_file():
                digest.update(str(item.relative_to(path)).encode())
                digest.update(bytes.fromhex(file_hash(item)))
        return digest.hexdigest()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_data(path: Path) -> tuple[Tensor, Tensor, Tensor]:
    bundle = load_bundle(path, expect=AFADataset)
    dataset = bundle.obj
    x, y = dataset.get_all_data()
    if x.ndim != 2 or not torch.isfinite(x).all():
        message = f"Need finite tabular features at {path}"
        raise ValueError(message)
    return x.float(), y.argmax(-1).long(), dataset.get_generation_indices()


def load_inputs(manifest: Path, entry_index: int) -> dict[str, Any]:
    entries = json.loads(manifest.read_text())
    if not 0 <= entry_index < len(entries):
        message = f"entry_index={entry_index} outside manifest"
        raise ValueError(message)
    entry = entries[entry_index]
    train_path = Path(entry["train_bundle"])
    val_path = train_path.parent / "val.bundle"
    train_x, train_y, train_ids = read_data(train_path)
    val_x, val_y, val_ids = read_data(val_path)
    if set(train_ids.tolist()) & set(val_ids.tolist()):
        message = "Training and validation generation indices overlap"
        raise ValueError(message)
    if train_x.shape[-1] != val_x.shape[-1]:
        message = "Training/validation feature shapes differ"
        raise ValueError(message)
    return {
        "entry": entry,
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "n_classes": int(max(train_y.max(), val_y.max())) + 1,
        "train_hash": file_hash(train_path),
        "val_hash": file_hash(val_path),
    }


def real_batch(
    data: dict[str, Any],
    config: dict[str, Any],
    split: str,
    device: torch.device,
) -> AcquisitionBatch:
    generator = torch.Generator().manual_seed(int(config["seed"]) + 2026)
    x = data["train_x"]
    n = min(int(config["validation_context"]), len(x))
    # Stratified first pass ensures every training class occurs in context.
    per_class = [
        torch.where(data["train_y"] == c)[0] for c in range(data["n_classes"])
    ]
    if (
        any(len(indices) == 0 for indices in per_class)
        or n < data["n_classes"]
    ):
        message = "Context must contain every task class"
        raise ValueError(message)
    first = torch.tensor(
        [
            int(v[torch.randint(len(v), (1,), generator=generator)].item())
            for v in per_class
        ]
    )
    others = torch.randperm(len(x), generator=generator)
    others = others[~torch.isin(others, first)]
    indices = torch.cat((first, others))[:n]
    cx = x[indices]
    qx, qy = data[f"{split}_x"], data[f"{split}_y"]
    limit = int(config["validation_queries"])
    ids = (
        torch.randperm(len(qx), generator=generator)[:limit]
        if limit
        else torch.arange(len(qx))
    )
    context = cx.unsqueeze(0).to(device)
    query = qx[ids].unsqueeze(0).to(device)
    return AcquisitionBatch(
        context,
        data["train_y"][indices].unsqueeze(0).to(device),
        query,
        qy[ids].unsqueeze(0).to(device),
        torch.ones_like(context, dtype=torch.bool),
        torch.ones_like(query, dtype=torch.bool),
    )


@torch.no_grad()
def evaluate_curve(
    model: ICLPolicy,
    batch: AcquisitionBatch,
    n_classes: int,
    budget: int,
    chunk_size: int,
    seed: int,
) -> list[dict[str, float | str]]:
    """Paired learned/random acquisition with the same selected raw classifier."""
    curves = []
    budget = min(budget, model.policy.n_features)
    for strategy in ("policy", "random"):
        losses = [0.0] * (budget + 1)
        correct = [0] * (budget + 1)
        total = batch.query_features.shape[1]
        generator = torch.Generator(
            device=batch.query_features.device
        ).manual_seed(seed)
        for start in range(0, total, chunk_size):
            end = start + chunk_size
            part = AcquisitionBatch(
                batch.context_features,
                batch.context_labels,
                batch.query_features[:, start:end],
                batch.query_labels[:, start:end],
                batch.context_mask,
                batch.query_available[:, start:end],
            )
            mask = torch.zeros_like(part.query_available)
            for step in range(budget + 1):
                output = model(
                    build_masked_table(part, mask.float()),
                    part.context_labels,
                    n_classes=n_classes,
                )
                losses[step] += float(
                    F.cross_entropy(
                        output.class_logits.flatten(0, 1),
                        part.query_labels.flatten(),
                        reduction="sum",
                    )
                )
                correct[step] += int(
                    (output.class_logits.argmax(-1) == part.query_labels).sum()
                )
                if step < budget:
                    valid = part.query_available & ~mask
                    scores = (
                        output.policy_scores
                        if strategy == "policy"
                        else torch.rand(
                            mask.shape, generator=generator, device=mask.device
                        )
                    )
                    selected = scores.masked_fill(~valid, -torch.inf).argmax(
                        -1
                    )
                    mask.scatter_(-1, selected.unsqueeze(-1), True)
        curves.extend(
            {
                "strategy": strategy,
                "budget": float(step),
                "cross_entropy": losses[step] / total,
                "accuracy": correct[step] / total,
            }
            for step in range(budget + 1)
        )
    return curves


def atomic_save(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def identity_for(
    data: dict[str, Any], checkpoint: Path, config: dict[str, Any]
) -> dict[str, Any]:
    root = Path("src/afabench/components/methods/discriminative/icl_policy")
    return {
        "entry": data["entry"],
        "config": config,
        "checkpoint_hash": file_hash(checkpoint),
        "train_hash": data["train_hash"],
        "val_hash": data["val_hash"],
        "sources": {str(p): file_hash(p) for p in sorted(root.rglob("*.py"))},
        "runner_hash": file_hash(Path(__file__)),
        "torch_version": str(torch.__version__),
    }


@contextmanager
def locked_output(output: Path) -> Iterator[None]:
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def weights_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@app.command()
def train(  # noqa: C901, PLR0912, PLR0915
    output: Path,
    checkpoint: Path,
    manifest: Path = Path("experiments/predictor_eval/datasets.json"),
    config_path: Path = Path("experiments/icl_policy/config.json"),
    entry_index: int = 0,
    device: str = "cuda",
    max_updates: int = 0,
    time_budget_seconds: float = 34200,
    threads: int = 2,
) -> None:
    """Train/resume one manifest entry; max-updates bounds a pilot invocation."""
    if max_updates < 0 or time_budget_seconds <= 0 or threads < 1:
        message = "max-updates >= 0, time budget > 0, threads >= 1 required"
        raise typer.BadParameter(message)
    config = json.loads(config_path.read_text())
    positive = (
        "steps",
        "batch_size",
        "context_size",
        "query_size",
        "checkpoint_interval",
        "validation_context",
        "validation_queries",
        "validation_budget",
        "query_batch_size",
        "pool_size",
    )
    if (
        any(int(config[k]) < 1 for k in positive)
        or min(float(config["policy_lr"]), float(config["backbone_lr"])) <= 0
    ):
        message = (
            "Training dimensions/counts and learning rate must be positive"
        )
        raise typer.BadParameter(message)
    if (
        config["task_prior"] != "l2m_real_pool_bnn_mar"
        or config["precision"] != "float32"
    ):
        message = (
            "Only the documented real-pool prior and float32 are supported"
        )
        raise typer.BadParameter(message)
    if (
        config.get("preprocessing_profile")
        != "official_numeric_default_ensemble"
    ):
        message = "Use the official numeric ensemble preprocessing profile"
        raise typer.BadParameter(message)
    torch.set_num_threads(threads)
    torch.manual_seed(int(config["seed"]))
    target_device = torch.device(device)
    data = load_inputs(manifest, entry_index)
    identity = identity_for(data, checkpoint, config)
    with locked_output(output):
        state_path = output / "latest.pt"
        if not state_path.exists() and any(
            p.name != ".lock" for p in output.iterdir()
        ):
            raise FileExistsError(output)
        model = ICLPolicy.from_checkpoint(
            checkpoint,
            n_features=data["train_x"].shape[-1],
            training_mode=config["training_mode"],
            n_estimators=int(config["n_estimators"]),
            softmax_temperature=float(config["softmax_temperature"]),
            ensemble_seed=int(config["ensemble_seed"]),
        ).to(target_device)
        optimizer = make_optimizer(
            model,
            policy_lr=float(config["policy_lr"]),
            backbone_lr=float(config["backbone_lr"]),
        )
        batch = real_batch(data, config, "val", target_device)
        generator = torch.Generator().manual_seed(int(config["seed"]))
        pool = data["train_x"][
            torch.randperm(len(data["train_x"]), generator=generator)[
                : int(config["pool_size"])
            ]
        ]
        if len(pool) < int(config["context_size"]) + int(config["query_size"]):
            message = "Training pool is smaller than a task"
            raise ValueError(message)
        sampler = make_sampler(pool, data["n_classes"])
        step, best_step, best_loss = 0, 0, math.inf
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        if state_path.exists():
            state = torch.load(
                state_path, map_location="cpu", weights_only=True
            )
            if (
                state.get("format_version") != 2
                or state["identity"] != identity
            ):
                message = "Resume rejected: inputs, source, environment or configuration changed"
                raise ValueError(message)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"])
            if state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            step, best_step, best_loss = (
                state["step"],
                state["best_step"],
                state["best_loss"],
            )
            best_state = state["best_model_state"]
        stop = False

        def request_stop(_signal: int, _frame: object) -> None:
            nonlocal stop
            stop = True

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(sig, request_stop)
        start = time.monotonic()
        starting_step = step
        max_acquisitions = (
            int(config["max_acquisitions"]) or model.policy.n_features
        )
        frozen_hash = weights_hash(model.tabicl)

        def persist() -> None:
            nonlocal best_loss, best_step, best_state
            with torch.random.fork_rng(
                devices=list(range(torch.cuda.device_count()))
                if target_device.type == "cuda"
                else []
            ):
                curve = evaluate_curve(
                    model,
                    batch,
                    data["n_classes"],
                    int(config["validation_budget"]),
                    int(config["query_batch_size"]),
                    int(config["seed"]),
                )
            loss = next(
                float(v["cross_entropy"])
                for v in reversed(curve)
                if v["strategy"] == "policy"
            )
            if not math.isfinite(loss):
                message = "Nonfinite validation loss"
                raise FloatingPointError(message)
            if loss < best_loss:
                best_loss, best_step = loss, step
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            atomic_save(
                state_path,
                {
                    "identity": identity,
                    "step": step,
                    "best_step": best_step,
                    "best_loss": best_loss,
                    "format_version": 2,
                    "model_state": model.state_dict(),
                    "best_model_state": best_state,
                    "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if target_device.type == "cuda"
                    else [],
                },
            )
            summary = {
                "step": step,
                "target_steps": config["steps"],
                "complete": step == config["steps"],
                "best_step": best_step,
                "best_validation_loss": best_loss,
                "validation_curve": curve,
                "elapsed_seconds_this_invocation": time.monotonic() - start,
                "initial_tabicl_hash": frozen_hash,
                "current_tabicl_hash": weights_hash(model.tabicl),
                "training_mode": model.training_mode,
                "policy_hash": weights_hash(model.policy),
                "dataset": data["entry"],
                "device": device,
            }
            temporary = output / "status.json.tmp"
            temporary.write_text(json.dumps(summary, indent=2) + "\n")
            temporary.replace(output / "status.json")
            with (output / "validation.jsonl").open("a") as log:
                log.write(json.dumps(summary) + "\n")
            print(json.dumps(summary), flush=True)

        if step == 0:
            persist()
        while step < int(config["steps"]):
            if (
                stop
                or time.monotonic() - start >= time_budget_seconds
                or (max_updates and step - starting_step >= max_updates)
            ):
                break
            task = sample_task(
                sampler,
                batch_size=int(config["batch_size"]),
                context_size=int(config["context_size"]),
                query_size=int(config["query_size"]),
                device=target_device,
            )
            result = afa_training_step(
                model,
                task,
                optimizer,
                n_classes=data["n_classes"],
                max_steps=max_acquisitions,
                temperature=float(config["temperature"]),
            )
            if not math.isfinite(result.loss) or result.optimizer_steps != 1:
                message = "Invalid loss or task with no valid acquisitions"
                raise FloatingPointError(message)
            grad_norm = (
                sum(
                    float(p.grad.detach().square().sum())
                    for p in model.policy.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            if not math.isfinite(grad_norm):
                message = "Nonfinite policy gradient"
                raise FloatingPointError(message)
            step += 1
            with (output / "training.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        {
                            "step": step,
                            "policy_grad_norm": grad_norm,
                            **asdict(result),
                        }
                    )
                    + "\n"
                )
            if step % int(config["checkpoint_interval"]) == 0:
                persist()
        persist()
        actual_hash = weights_hash(model.tabicl)
        if model.training_mode == "frozen" and actual_hash != frozen_hash:
            message = "Frozen TabICL weights changed"
            raise AssertionError(message)


@app.command()
def evaluate(
    run: Path,
    output: Path,
    checkpoint: Path,
    manifest: Path = Path("experiments/predictor_eval/datasets.json"),
    entry_index: int = 0,
    device: str = "cuda",
    max_test: int = 0,
) -> None:
    """Evaluate the jointly selected model after all requested steps finish."""
    torch.set_num_threads(2)
    if output.exists():
        raise FileExistsError(output)
    state = torch.load(
        run / "latest.pt", map_location="cpu", weights_only=True
    )
    config = state["identity"]["config"]
    if state["step"] != config["steps"]:
        message = "Training incomplete; final test evaluation is blocked"
        raise ValueError(message)
    data = load_inputs(manifest, entry_index)
    if identity_for(data, checkpoint, config) != state["identity"]:
        message = (
            "Evaluation input/source identity differs from the training run"
        )
        raise ValueError(message)
    test_x, test_y, test_ids = read_data(Path(data["entry"]["test_bundle"]))
    train_ids = read_data(Path(data["entry"]["train_bundle"]))[2]
    val_ids = read_data(
        Path(data["entry"]["train_bundle"]).parent / "val.bundle"
    )[2]
    if set(test_ids.tolist()) & (
        set(train_ids.tolist()) | set(val_ids.tolist())
    ):
        message = "Test instances overlap training or validation"
        raise ValueError(message)
    data.update(test_x=test_x, test_y=test_y)
    model = ICLPolicy.from_checkpoint(
        checkpoint,
        n_features=test_x.shape[-1],
        training_mode=config["training_mode"],
        n_estimators=int(config["n_estimators"]),
        softmax_temperature=float(config["softmax_temperature"]),
        ensemble_seed=int(config["ensemble_seed"]),
    ).to(device)
    model.load_state_dict(state["best_model_state"])
    test_config = {**config, "validation_queries": max_test}
    batch = real_batch(data, test_config, "test", torch.device(device))
    curve = evaluate_curve(
        model,
        batch,
        data["n_classes"],
        model.policy.n_features,
        int(config["query_batch_size"]),
        int(config["seed"]),
    )
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "entry": data["entry"],
                "curve": curve,
                "test_hash": file_hash(Path(data["entry"]["test_bundle"])),
                "best_step": state["best_step"],
                "n_test": batch.query_features.shape[1],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    app()
