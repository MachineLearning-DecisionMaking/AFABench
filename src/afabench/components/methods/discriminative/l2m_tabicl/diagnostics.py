"""Fixed synthetic acquisition episodes for monitoring, not model selection."""

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.nn import functional as F

from afabench.components.methods.discriminative.l2m_tabicl.afa_methods import (
    L2MTabICLAFAMethod,
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
from afabench.components.unmaskers.config import UnmaskerConfig


@torch.inference_mode()
def record_acquisition(
    predictor: FrozenTabICLPredictor,
    policy: AcquisitionPolicy,
    tasks: Sequence[MotherTreeTask],
    unmasker: UnmaskerConfig,
    path: Path,
    step: int,
) -> None:
    records = []
    for strategy in ("learned", "random"):
        losses, correct, counts = [], 0, 0
        for index, record in enumerate(tasks):
            task, size = record.task, record.context_set_size
            method = L2MTabICLAFAMethod(
                predictor,
                policy,
                task.features[:size],
                task.labels[:size],
                unmasker=unmasker,
            )
            query = task.features[size:].to(method.device)
            truth = task.labels[size:].argmax(-1).to(method.device)
            mask = torch.zeros_like(query, dtype=torch.bool)
            order = (
                torch.rand(
                    query.shape,
                    generator=torch.Generator().manual_seed(700000 + index),
                )
                .argsort(-1)
                .to(query.device)
            )
            for position in range(min(3, query.shape[1])):
                observed = torch.where(mask, query, float("nan"))
                choice = (
                    method.act(observed, mask).flatten() - 1
                    if strategy == "learned"
                    else order[:, position]
                )
                mask.scatter_(1, choice[:, None], True)
            logits = method.predict(
                torch.where(mask, query, float("nan")), mask
            )
            losses.append(
                float(F.cross_entropy(logits, truth, reduction="sum"))
            )
            correct += int((logits.argmax(-1) == truth).sum())
            counts += len(query)
        records.append(
            {
                "step": step,
                "strategy": strategy,
                "budget": 3,
                "queries": counts,
                "accuracy": correct / counts,
                "nll": sum(losses) / counts,
            }
        )
    path.mkdir(parents=True, exist_ok=True)
    destination = path / f"step-{step:06d}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n")
    temporary.replace(destination)
