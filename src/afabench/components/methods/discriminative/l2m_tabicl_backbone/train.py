"""Fit the experimental hybrid using upstream real-feature/BNN tasks."""

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

import torch

from afabench.components.methods.discriminative.l2m.afa_methods import (
    require_direct_unmasker,
)
from afabench.components.methods.discriminative.l2m.models import (
    L2MModel,
)
from afabench.components.methods.discriminative.l2m.train import (
    _draw_context_set,
)
from afabench.components.methods.discriminative.l2m.upstream.utils import (
    MLP,
)
from afabench.components.methods.discriminative.l2m.upstream_fit import (
    draw_pretraining_pool,
    make_sampler,
)
from afabench.components.methods.discriminative.l2m_tabicl.predictor import (
    FrozenTabICLPredictor,
)
from afabench.components.methods.discriminative.l2m_tabicl_backbone.afa_methods import (
    L2MTabICLBackboneAFAMethod,
)
from afabench.components.methods.discriminative.l2m_tabicl_backbone.config import (
    BackboneTrainingConfig,
)
from afabench.components.methods.discriminative.l2m_tabicl_backbone.objective import (
    optimizer_for,
    trajectory_objective,
)
from afabench.fit.inputs import FitInputs

log = logging.getLogger(__name__)


def gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    norms = [
        p.grad.detach().square().sum()
        for p in parameters
        if p.grad is not None
    ]
    return float(torch.stack(norms).sum().sqrt()) if norms else 0.0


def train_backbone(
    cfg: BackboneTrainingConfig,
    *,
    inputs: FitInputs,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
) -> L2MTabICLBackboneAFAMethod:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = _initialize_model(cfg, inputs, device)
    dataset = inputs.train_dataset()
    predictor_path = Path(cfg.tabicl_checkpoint_path)
    with predictor_path.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    predictor = FrozenTabICLPredictor(predictor_path).to(device)
    if model.n_classes > predictor.model.max_classes:
        msg = "dataset class count exceeds TabICL checkpoint capacity"
        raise ValueError(msg)
    pool = draw_pretraining_pool(
        dataset, size=cfg.pretraining_pool_size, seed=cfg.seed
    )
    if not torch.isfinite(pool).all():
        msg = "official real-feature sampler requires a finite training pool"
        raise ValueError(msg)
    sampler = make_sampler(pool, label_dim=model.label_dim)
    validation = [
        sampler.sample(
            batch_size=1,
            max_num_points=cfg.sequence_length,
            device=str(device),
            num_observed=model.n_features,
            seed=cfg.seed + 100000 + i,
        )
        for i in range(cfg.n_validation_tasks)
    ]
    optimizer = optimizer_for(
        model, policy_lr=cfg.policy_lr, backbone_lr=cfg.backbone_lr
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for step in range(cfg.n_steps + 1):
        metrics: dict[str, float] = {"step": float(step)}
        if step:
            model.train()
            batch = sampler.sample(
                batch_size=cfg.batch_size,
                max_num_points=cfg.sequence_length,
                device=str(device),
                num_observed=model.n_features,
            )
            metrics.update(
                _training_step(
                    model, predictor, batch, optimizer, cfg.temperature
                )
            )
        if step % cfg.checkpoint_interval == 0 or step == cfg.n_steps:
            model.eval()
            with torch.no_grad():
                objectives = [
                    trajectory_objective(
                        model,
                        predictor,
                        batch,
                        temperature=cfg.temperature,
                        validation_seed=cfg.seed + i,
                    )
                    for i, batch in enumerate(validation)
                ]
                count = sum(o.valid_states for o in objectives)
                if not count:
                    msg = "validation has no valid acquisition states"
                    raise ValueError(msg)
                loss = (
                    sum(float(o.loss) * o.valid_states for o in objectives)
                    / count
                )
            metrics["validation_loss"] = loss
            if loss < best_loss:
                best_loss = loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            log.info("Hybrid step %d: %s", step, metrics)
        history.append(metrics)
        if metric_logger is not None:
            metric_logger(metrics)
    if best_state is None:
        msg = "hybrid never reached a finite validation loss"
        raise RuntimeError(msg)
    model.load_state_dict(best_state)
    context_features, context_labels = _draw_context_set(
        inputs.val_dataset().get_all_data(),
        n_features=model.n_features,
        context_set_size=cfg.context_set_size,
        seed=cfg.seed,
    )
    result = L2MTabICLBackboneAFAMethod(
        model.cpu().eval(),
        predictor.cpu(),
        context_features,
        context_labels,
        unmasker=cfg.unmasker,
    )
    result.training_record = {
        "tabicl_sha256": checkpoint_hash,
        "history": history,
        "best_validation_loss": best_loss,
        "preprocessing": "context_mean_imputation_none_single_estimator",
    }
    return result


def _initialize_model(
    cfg: BackboneTrainingConfig, inputs: FitInputs, device: torch.device
) -> L2MModel:
    require_direct_unmasker(cfg.unmasker)
    if cfg.soft_budget_param is not None or cfg.hard_budget is None:
        msg = "hybrid requires a hard budget and no soft-budget parameter"
        raise ValueError(msg)
    model = inputs.pretrained_model(L2MModel).to(device)
    dataset = inputs.train_dataset()
    if (model.n_features, model.n_classes) != (
        dataset.feature_shape.numel(),
        dataset.label_shape.numel(),
    ):
        msg = "pretrained L2M dimensions do not match the dataset"
        raise ValueError(msg)
    if not 0 < cfg.hard_budget <= model.n_features:
        msg = "hard_budget must be between one and the feature count"
        raise ValueError(msg)
    if model.model_config["dropout"] != 0:
        msg = "L2M checkpoint dropout must be zero"
        raise ValueError(msg)
    # A predictor-pretraining checkpoint's selector is not treated as a
    # pretrained policy. Reinitialize with the authors' normal head init.
    model.ltm.selector = MLP(
        int(model.model_config["d_model"]),
        model.n_features,
        int(model.model_config["dim_feedforward"]),
        int(model.model_config["pred_depth"]),
        policy_head=True,
    ).to(device)
    if cfg.policy_checkpoint_path is not None:
        model.ltm.selector.load_state_dict(
            torch.load(
                cfg.policy_checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        )
    return model


def _training_step(
    model: L2MModel,
    predictor: FrozenTabICLPredictor,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Adam,
    temperature: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    objective = trajectory_objective(
        model, predictor, batch, temperature=temperature
    )
    if objective.valid_states:
        objective.loss.backward()
        for parameter in model.parameters():
            if (
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ):
                msg = "L2M gradient is nonfinite"
                raise FloatingPointError(msg)
        metrics.update(
            {
                "training_loss": float(objective.loss.detach()),
                "valid_states": float(objective.valid_states),
                "policy_gradient_norm": gradient_norm(
                    list(model.ltm.selector.parameters())
                ),
                "backbone_gradient_norm": gradient_norm(
                    [
                        *model.ltm.embedder.parameters(),
                        *model.ltm.encoder.parameters(),
                    ]
                ),
            }
        )
        optimizer.step()
    return metrics
