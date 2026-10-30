from dataclasses import dataclass
from enum import StrEnum

from afabench.fit.contract import (
    PretrainingContract,
    TrainingContract,
    store_contract_config,
)

#TODO: add datasets per better synth generation
class FeatureSource(StrEnum):
    """Where the features of L2M's tasks come from; config values are the names."""

    real = "real"
    synthetic = "synthetic"


# `RealDataSampler.sample` always gives a task this many context instances
# (its `num_ctx` default), so a task needs more instances than this.
UPSTREAM_CONTEXT_SET_SIZE = 50


def _check_task_shape(
    *,
    sequence_length: int,
    pretraining_pool_size: int,
    n_validation_tasks: int,
) -> None:
    if sequence_length <= UPSTREAM_CONTEXT_SET_SIZE:
        msg = (
            f"sequence_length={sequence_length} must exceed the "
            f"{UPSTREAM_CONTEXT_SET_SIZE} context instances of every task"
        )
        raise ValueError(msg)
    # A task draws its instances from the pretraining pool without
    # replacement.
    if pretraining_pool_size < sequence_length:
        msg = (
            f"pretraining_pool_size={pretraining_pool_size} must be at "
            f"least sequence_length={sequence_length}"
        )
        raise ValueError(msg)
    if n_validation_tasks < 1:
        msg = f"n_validation_tasks={n_validation_tasks} must be at least 1"
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
    # See `draw_feature_pool`.
    feature_source: FeatureSource
    architecture: L2MArchitectureConfig
    # Tasks per step, and instances per task.
    batch_size: int
    sequence_length: int
    n_steps: int
    lr: float
    warmup_steps: int
    checkpoint_interval: int
    n_validation_tasks: int
    # Train-split instances whose features the task prior draws from.
    pretraining_pool_size: int
    # bf16 autocast, as upstream's `Accelerator(mixed_precision="bf16")`.
    mixed_precision: bool

    def __post_init__(self) -> None:
        _check_task_shape(
            sequence_length=self.sequence_length,
            pretraining_pool_size=self.pretraining_pool_size,
            n_validation_tasks=self.n_validation_tasks,
        )


store_contract_config(name="pretrain_l2m", config_class=L2MPretrainingConfig)


@dataclass(frozen=True, kw_only=True)
class L2MTrainingConfig(TrainingContract):
    # The pretrained model's feature source, see `draw_feature_pool`.
    feature_source: FeatureSource
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
            sequence_length=self.sequence_length,
            pretraining_pool_size=self.pretraining_pool_size,
            n_validation_tasks=self.n_validation_tasks,
        )
        if self.context_set_size < 1:
            msg = (
                f"context_set_size={self.context_set_size} must be at least 1"
            )
            raise ValueError(msg)


store_contract_config(name="train_l2m", config_class=L2MTrainingConfig)
