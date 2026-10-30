"""
What both L2M fit stages share around the vendored code.

The stages follow upstream's `real_train` and `real_train_afa`
(`src/experiments/real.py` at the vendored commit): every step draws a
batch of tasks with `RealDataSampler.sample` and takes one vendored training
step; at step 0 and every `checkpoint_interval` steps after it, the vendored
validation function scores a fixed batch of validation tasks, and the state
with the lowest validation loss is kept. As upstream, the last step is not
validated unless it falls on the interval.

The task prior is upstream's: tasks draw instances from the pretraining
pool, normalize each feature within the task, and label them with a random
BNN. Only the pool's features are used; the train split's labels are
discarded. The pool is the train split's features (`FeatureSource.real`) or
a synthetic draw of the dataset's feature count (`FeatureSource.synthetic`).
"""

import logging
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from types import MethodType
from typing import Any

import torch

from afabench.components.methods.discriminative.l2m.config import (
    FeatureSource,
)
from afabench.components.methods.discriminative.l2m.models import (
    L2MModel,
)
from afabench.components.methods.discriminative.l2m.upstream.data_utils import (
    RealDataSampler,
    TabularDataset,
)
from afabench.components.methods.discriminative.l2m.upstream.experiment_utils import (
    gen_evalset,
)
from afabench.core.types import AFADataset, Features

log = logging.getLogger(__name__)

# `gen_evalset` treats this one experiment name as upstream's regression
# simulation rather than a real dataset.
_UPSTREAM_SIMULATION = "sim"
# Upstream's `inner_loop_steps` for MiniBooNE: one update per batch.
_INNER_LOOP_STEPS = 1
# The synthetic feature source's box, this benchmark's own choice as in the
# earlier port; the paper does not describe it.
_SYNTHETIC_FEATURE_LOW = -2.0
_SYNTHETIC_FEATURE_HIGH = 2.0


class UpstreamAccelerator:
    """
    The part of `accelerate.Accelerator` the vendored code calls.

    With mixed precision, as `Accelerator(mixed_precision="bf16")`,
    `autocast` runs its block in bf16 autocast, and inside `prepared` the
    model's forward pass does too, with floating outputs cast back to
    float32. Without it both are plain float32.
    """

    def __init__(self, device: torch.device, *, mixed_precision: bool):
        self.device: torch.device = device
        self.mixed_precision: bool = mixed_precision

    def autocast(self) -> torch.autocast:
        return torch.autocast(
            self.device.type,
            dtype=torch.bfloat16,
            enabled=self.mixed_precision,
        )

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    @contextmanager
    def prepared(self, model: L2MModel) -> Generator[None]:
        """Wrap the forward pass of upstream's model as `prepare` does."""
        ltm = model.ltm
        forward = ltm.forward

        # Upstream's `LTM.forward`, whose flags its callers pass by keyword.
        def autocast_forward(
            _: object,
            batch: dict[str, torch.Tensor],
            *,
            reduce_ll: bool = True,
            autoreg: bool = True,
        ) -> object:
            with self.autocast():
                outputs = forward(batch, reduce_ll=reduce_ll, autoreg=autoreg)
            return _to_float32(outputs)

        ltm.forward = MethodType(autocast_forward, ltm)
        try:
            yield
        finally:
            del ltm.forward


def _to_float32(outputs: object) -> object:
    if isinstance(outputs, Mapping):
        return {key: _to_float32(value) for key, value in outputs.items()}
    if isinstance(outputs, torch.Tensor) and outputs.dtype in {
        torch.float16,
        torch.bfloat16,
    }:
        return outputs.float()
    return outputs


def draw_pretraining_pool(
    train_dataset: AFADataset,
    *,
    size: int,
    sequence_length: int,
    seed: int,
) -> Features:
    """Draw up to `size` train instances' features as the pretraining pool."""
    features, _ = train_dataset.get_all_data()
    # A task draws its instances from the pool without replacement.
    if len(features) < sequence_length:
        msg = (
            f"The train split has {len(features)} instances, fewer than "
            f"sequence_length={sequence_length}"
        )
        raise ValueError(msg)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(features), generator=generator)[
        : min(len(features), size)
    ]
    return features.reshape(len(features), -1)[indices].clone()


def draw_synthetic_pool(*, size: int, n_features: int, seed: int) -> Features:
    """
    Draw `size` instances uniformly from the box [-2, 2] in every feature.

    Stub of the synthetic feature prior of the earlier independent L2M port
    (`l2m_synthetic_feature_prior`, `task_sampler.py` at `f814beec^`). That
    port drew fresh features for every task; here the tasks sample from one
    fixed pool, as they do from the real pretraining pool, and label and
    missingness come from upstream's task prior. Features are normalized
    within each task, so the box only sets the shape of the marginals.
    """
    generator = torch.Generator().manual_seed(seed)
    return _SYNTHETIC_FEATURE_LOW + (
        _SYNTHETIC_FEATURE_HIGH - _SYNTHETIC_FEATURE_LOW
    ) * torch.rand(size, n_features, generator=generator)


def draw_feature_pool(
    train_dataset: AFADataset,
    *,
    feature_source: FeatureSource,
    size: int,
    sequence_length: int,
    seed: int,
) -> Features:
    """
    Draw the pool the tasks' instances are sampled from.

    The real source samples the train split's features, which it caps at
    the split's size; the synthetic source ignores the data and takes only
    its feature count. A task needs `sequence_length` instances, so
    `size` must be at least that, as the configs check.
    """
    match feature_source:
        case FeatureSource.real:
            return draw_pretraining_pool(
                train_dataset,
                size=size,
                sequence_length=sequence_length,
                seed=seed,
            )
        case FeatureSource.synthetic:
            return draw_synthetic_pool(
                size=size,
                n_features=train_dataset.feature_shape.numel(),
                seed=seed,
            )


def make_sampler(pool: Features, *, label_dim: int) -> RealDataSampler:
    """
    Upstream's task sampler over the pretraining pool.

    `RealDataSampler` also takes a test dataset, read only by its evaluation
    sampler, which the fit stages do not call.
    """
    dataset = TabularDataset(
        data=pool,
        dim_y=label_dim,
        feature_names=[f"feature_{i}" for i in range(pool.shape[1])],
        dataframe=None,
        free_indices=[],
    )
    return RealDataSampler(dataset, dataset)


def upstream_config(
    model: L2MModel,
    *,
    dataset_key: str,
    sequence_length: int,
    n_validation_tasks: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Build the config dictionary the vendored functions read."""
    if dataset_key == _UPSTREAM_SIMULATION:
        msg = f"dataset_key={dataset_key!r} would select upstream's simulation"
        raise ValueError(msg)
    return {
        **model.model_config,
        "experiment": dataset_key,
        "num_points": sequence_length,
        "num_available_features": model.n_features,
        "inner_loop_steps": _INNER_LOOP_STEPS,
        "eval_batch_size": n_validation_tasks,
        "train_seed": seed,
        "device": device,
    }


def generate_validation_tasks(
    config: dict[str, Any], sampler: RealDataSampler, *, task: str
) -> None:
    """Store `gen_evalset`'s validation tasks where the vendored code reads them."""
    config.setdefault("eval_sets", {})[task] = gen_evalset(
        config, sampler, task=task
    )


def fit_upstream(
    model: L2MModel,
    accelerator: UpstreamAccelerator,
    *,
    stage: str,
    n_steps: int,
    checkpoint_interval: int,
    training_step: Callable[[], float],
    validation_loss: Callable[[], float],
    metric_logger: Callable[[dict[str, float]], None] | None,
) -> None:
    """
    Fit `model` in place with upstream's loop, then load its best state.

    `stage` names the log lines and the `l2m_<stage>/` metrics.
    """
    if n_steps < 1:
        msg = f"n_steps={n_steps} must be at least 1"
        raise ValueError(msg)
    best_loss = torch.inf
    best_state: dict[str, torch.Tensor] | None = None
    with accelerator.prepared(model):
        model.train()
        for step in range(n_steps):
            loss = training_step()
            if step % checkpoint_interval:
                continue
            model.eval()
            with torch.no_grad():
                loss_on_validation = validation_loss()
            log.info(
                "L2M %s step %d: train loss %.4f, validation loss %.4f",
                stage,
                step,
                loss,
                loss_on_validation,
            )
            if metric_logger is not None:
                metric_logger(
                    {
                        f"l2m_{stage}/step": float(step),
                        f"l2m_{stage}/train_loss": loss,
                        f"l2m_{stage}/val_loss": loss_on_validation,
                    }
                )
            if loss_on_validation < best_loss:
                best_loss = loss_on_validation
                best_state = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }
            model.train()
    # Step 0 always validates; a non-finite loss there keeps no state.
    if best_state is None:
        msg = f"L2M {stage} never reached a finite validation loss"
        raise RuntimeError(msg)
    model.load_state_dict(best_state)
