"""
L2M training stage: upstream's `real_train_afa` on the fit contract.

The pretrained `L2MModel` is loaded and trained with the vendored
`afa_training_step` (upstream's `sequential` strategy): every step rolls out
a trajectory of random acquisitions, and at each state the policy's
straight-through Gumbel-softmax action is scored by the classifier's loss
after acquiring it. Every parameter is trained, the policy head at
`policy_lr` and the rest at `backbone_lr`, both constant, at a fixed
temperature. Afterwards the context set is drawn from the validation split
with the contract seed and packaged with the model as an
`L2MAFAMethod`.
"""

from collections.abc import Callable

import torch

from afabench.components.methods.discriminative.l2m.afa_methods import (
    L2MAFAMethod,
    require_direct_unmasker,
)
from afabench.components.methods.discriminative.l2m.config import (
    L2MTrainingConfig,
)
from afabench.components.methods.discriminative.l2m.models import (
    L2MModel,
)
from afabench.components.methods.discriminative.l2m.upstream.experiment_utils import (
    eval_afa,
)
from afabench.components.methods.discriminative.l2m.upstream.train_utils import (
    afa_training_step,
)
from afabench.components.methods.discriminative.l2m.upstream.utils import (
    ConcreteSelector,
)
from afabench.components.methods.discriminative.l2m.upstream_fit import (
    UpstreamAccelerator,
    draw_feature_pool,
    fit_upstream,
    generate_validation_tasks,
    make_sampler,
    upstream_config,
)
from afabench.core.types import Features, Label
from afabench.fit.inputs import FitInputs


def train_l2m(
    cfg: L2MTrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
    *,
    inputs: FitInputs,
) -> L2MAFAMethod:
    _check_config(cfg, inputs)
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    train_dataset = inputs.train_dataset()
    model = inputs.pretrained_model(L2MModel).to(device)
    shapes = (
        train_dataset.feature_shape.numel(),
        train_dataset.label_shape.numel(),
    )
    if (model.n_features, model.n_classes) != shapes:
        msg = (
            f"The pretrained model has {model.n_features} features and "
            f"{model.n_classes} classes; the train split has {shapes[0]} "
            f"and {shapes[1]}"
        )
        raise ValueError(msg)
    sampler = make_sampler(
        draw_feature_pool(
            train_dataset,
            feature_source=cfg.feature_source,
            size=cfg.pretraining_pool_size,
            sequence_length=cfg.sequence_length,
            seed=cfg.seed,
        ),
        label_dim=model.label_dim,
    )
    config = upstream_config(
        model,
        dataset_key=cfg.dataset_key,
        sequence_length=cfg.sequence_length,
        n_validation_tasks=cfg.n_validation_tasks,
        seed=cfg.seed,
        device=device,
    )
    generate_validation_tasks(config, sampler, task="afa")
    accelerator = UpstreamAccelerator(
        device, mixed_precision=cfg.mixed_precision
    )
    for parameter in model.parameters():
        parameter.requires_grad = True
    policy_parameters = list(model.ltm.selector.parameters())
    policy_parameter_ids = {id(parameter) for parameter in policy_parameters}
    optimizer = torch.optim.Adam(
        [
            {"params": policy_parameters, "lr": cfg.policy_lr},
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if id(parameter) not in policy_parameter_ids
                ],
                "lr": cfg.backbone_lr,
            },
        ]
    )
    selector_fn = ConcreteSelector()

    def training_step() -> float:
        batch = sampler.sample(
            batch_size=cfg.batch_size,
            max_num_points=cfg.sequence_length,
            # Upstream annotates the device as a string.
            device=str(device),
            num_observed=model.n_features,
        )
        return float(
            afa_training_step(
                config,
                model.ltm,
                sampler,
                batch,
                optimizer,
                None,
                accelerator,
                selector_fn,
                cfg.temperature,
            )
        )

    def validation_loss() -> float:
        loss, _ = eval_afa(
            config, sampler, model.ltm, accelerator, cfg.temperature
        )
        return float(loss)

    fit_upstream(
        model,
        accelerator,
        stage="train",
        n_steps=cfg.n_steps,
        checkpoint_interval=cfg.checkpoint_interval,
        training_step=training_step,
        validation_loss=validation_loss,
        metric_logger=metric_logger,
    )
    context_features, context_labels = _draw_context_set(
        inputs.val_dataset().get_all_data(),
        n_features=model.n_features,
        context_set_size=cfg.context_set_size,
        seed=cfg.seed,
    )
    return L2MAFAMethod(
        model.cpu().eval(),
        context_features,
        context_labels,
        unmasker=cfg.unmasker,
    )


def _check_config(cfg: L2MTrainingConfig, inputs: FitInputs) -> None:
    require_direct_unmasker(cfg.unmasker)
    n_validation = len(inputs.val_dataset())
    if n_validation < 1:
        msg = f"The validation split has {n_validation} instances; need at least 1"
        raise ValueError(msg)
    n_selections = inputs.unmasker().get_n_selections(
        inputs.train_dataset().feature_shape
    )
    # L2M has no stop action, so the budget must leave a selection for
    # every act call of the hard-budget setting.
    if cfg.hard_budget is None or not 0 < cfg.hard_budget < n_selections:
        msg = (
            f"hard_budget={cfg.hard_budget} must be between 1 and "
            f"the number of selections minus one ({n_selections - 1})"
        )
        raise ValueError(msg)


def _draw_context_set(
    validation_data: tuple[Features, Label],
    *,
    n_features: int,
    context_set_size: int,
    seed: int,
) -> tuple[Features, Label]:
    """Draw up to context_set_size validation instances with the contract seed."""
    features, labels = validation_data
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(features), generator=generator)[
        : min(len(features), context_set_size)
    ]
    return (
        features.reshape(len(features), n_features)[indices].clone(),
        labels.reshape(len(labels), -1)[indices].clone(),
    )
