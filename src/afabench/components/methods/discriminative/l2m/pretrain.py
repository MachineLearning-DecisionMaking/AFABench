"""
L2M pretraining stage: upstream's `real_train` on the fit contract.

The classifier learns to predict query instances' labels from a context set
on tasks from upstream's task prior, with the vendored
`predictor_training_step`: every step acquires random features one at a
time, from none to all, and sums the classifier loss over the trajectory.
The learning rate warms up linearly and decays linearly to zero, as
`transformers.get_linear_schedule_with_warmup`.
"""

from collections.abc import Callable
from dataclasses import asdict
from functools import partial

import torch

from afabench.components.methods.discriminative.l2m.config import (
    L2MPretrainingConfig,
)
from afabench.components.methods.discriminative.l2m.models import (
    L2MModel,
)
from afabench.components.methods.discriminative.l2m.upstream.experiment_utils import (
    eval as upstream_eval,
)
from afabench.components.methods.discriminative.l2m.upstream.train_utils import (
    predictor_training_step,
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
from afabench.fit.inputs import FitInputs

#TODO: try to use existing benchmarks if possible
def pretrain_l2m(
    cfg: L2MPretrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
    *,
    inputs: FitInputs,
) -> L2MModel:
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    train_dataset = inputs.train_dataset()
    model = L2MModel.build(
        train_dataset.feature_shape.numel(),
        train_dataset.label_shape.numel(),
        **asdict(cfg.architecture),
    ).to(device)
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
    generate_validation_tasks(config, sampler, task="predictor")
    accelerator = UpstreamAccelerator(
        device, mixed_precision=cfg.mixed_precision
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        partial(
            linear_schedule_with_warmup,
            warmup_steps=cfg.warmup_steps,
            n_steps=cfg.n_steps,
        ),
    )
    selector_fn = ConcreteSelector()

    def training_step() -> float:
        batch = sampler.sample(
            batch_size=cfg.batch_size,
            max_num_points=cfg.sequence_length,
            num_observed=model.n_features,
            # Upstream annotates the device as a string.
            device=str(device),
        )
        return float(
            predictor_training_step(
                config,
                model.ltm,
                sampler,
                batch,
                optimizer,
                scheduler,
                accelerator,
                selector_fn,
            )
        )

    def validation_loss() -> float:
        loss, _ = upstream_eval(config, sampler, model.ltm, accelerator)
        return float(loss)

    fit_upstream(
        model,
        accelerator,
        stage="pretrain",
        n_steps=cfg.n_steps,
        checkpoint_interval=cfg.checkpoint_interval,
        training_step=training_step,
        validation_loss=validation_loss,
        metric_logger=metric_logger,
    )
    return model.cpu().eval()


def linear_schedule_with_warmup(
    step: int, *, warmup_steps: int, n_steps: int
) -> float:
    """Scale the learning rate as `transformers.get_linear_schedule_with_warmup`."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (n_steps - step) / max(1, n_steps - warmup_steps))
