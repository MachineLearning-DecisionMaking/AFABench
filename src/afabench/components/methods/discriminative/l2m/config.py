from dataclasses import dataclass
from enum import StrEnum

from afabench.fit.contract import (
    PretrainingContract,
    TrainingContract,
    store_contract_config,
)


class TaskPrior(StrEnum):
    """
    The task prior an L2M fit stage draws its tasks from; values are names.

    The pool priors draw a task's instances from a pretraining pool and
    label them with upstream's random BNN. `tabicl_v1` generates features and
    labels together, with no pool.
    """

    # The train split's features.
    real_pool = "real_pool"
    # Uniform draws on a box, with only the dataset's feature count.
    synthetic_pool = "synthetic_pool"
    # The TabICL v1 prior, with the dataset's feature and class counts.
    tabicl_v1 = "tabicl_v1"

    @property
    def uses_pool(self) -> bool:
        return self is not TaskPrior.tabicl_v1


# `RealDataSampler.sample` always gives a task this many context instances
# (its `num_ctx` default), so a task needs more instances than this.
UPSTREAM_CONTEXT_SET_SIZE = 50


def _check_task_shape(
    *,
    task_prior: TaskPrior,
    sequence_length: int,
    pretraining_pool_size: int,
    n_validation_tasks: int,
    task_workers: int,
) -> None:
    if sequence_length <= UPSTREAM_CONTEXT_SET_SIZE:
        msg = (
            f"sequence_length={sequence_length} must exceed the "
            f"{UPSTREAM_CONTEXT_SET_SIZE} context instances of every task"
        )
        raise ValueError(msg)
    # A task draws its instances from the pretraining pool without
    # replacement.
    if task_prior.uses_pool and pretraining_pool_size < sequence_length:
        msg = (
            f"pretraining_pool_size={pretraining_pool_size} must be at "
            f"least sequence_length={sequence_length}"
        )
        raise ValueError(msg)
    if n_validation_tasks < 1:
        msg = f"n_validation_tasks={n_validation_tasks} must be at least 1"
        raise ValueError(msg)
    if task_workers < 0:
        msg = f"task_workers={task_workers} must not be negative"
        raise ValueError(msg)
    # Only the TabICL v1 prior is slow enough to generate ahead of use.
    if task_workers and task_prior.uses_pool:
        msg = (
            f"task_workers={task_workers} has no effect with "
            f"task_prior={task_prior.value}; only tabicl_v1 generates "
            f"tasks in worker processes"
        )
        raise ValueError(msg)


# Not frozen: OmegaConf would make the nested node read-only.
@dataclass(kw_only=True)
class L2MArchitectureConfig:
    """The upstream `LTM` hyperparameters other than the dataset's shapes."""

    model_dim: int
    embedding_depth: int
    prediction_depth: int
    n_layers: int
    n_heads: int
    feedforward_dim: int
    dropout: float


@dataclass(frozen=True, kw_only=True)
class L2MPretrainingConfig(PretrainingContract):
    task_prior: TaskPrior
    # Processes generating tasks ahead of use; 0 generates them in the fit.
    task_workers: int
    architecture: L2MArchitectureConfig
    # Tasks per step, and instances per task.
    batch_size: int
    sequence_length: int
    n_steps: int
    lr: float
    warmup_steps: int
    checkpoint_interval: int
    n_validation_tasks: int
    # Train-split instances whose features a pool task prior draws from.
    pretraining_pool_size: int
    # bf16 autocast, as upstream's `Accelerator(mixed_precision="bf16")`.
    mixed_precision: bool

    def __post_init__(self) -> None:
        _check_task_shape(
            task_prior=self.task_prior,
            sequence_length=self.sequence_length,
            pretraining_pool_size=self.pretraining_pool_size,
            n_validation_tasks=self.n_validation_tasks,
            task_workers=self.task_workers,
        )


store_contract_config(name="pretrain_l2m", config_class=L2MPretrainingConfig)


@dataclass(frozen=True, kw_only=True)
class L2MTrainingConfig(TrainingContract):
    # The pretrained model's task prior, and its worker processes.
    task_prior: TaskPrior
    task_workers: int
    # Tasks per step, and instances per task.
    batch_size: int
    sequence_length: int
    n_steps: int
    # Policy head learning rate, and the lower one of everything else.
    policy_lr: float
    backbone_lr: float
    # Fixed straight-through Gumbel-softmax temperature.
    temperature: float
    checkpoint_interval: int
    n_validation_tasks: int
    pretraining_pool_size: int
    mixed_precision: bool
    # Validation-split instances stored in the method bundle.
    context_set_size: int

    def __post_init__(self) -> None:
        _check_task_shape(
            task_prior=self.task_prior,
            sequence_length=self.sequence_length,
            pretraining_pool_size=self.pretraining_pool_size,
            n_validation_tasks=self.n_validation_tasks,
            task_workers=self.task_workers,
        )
        if self.context_set_size < 1:
            msg = (
                f"context_set_size={self.context_set_size} must be at least 1"
            )
            raise ValueError(msg)


store_contract_config(name="train_l2m", config_class=L2MTrainingConfig)
