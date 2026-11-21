"""Configuration for the experimental numerical-data hybrid."""

from dataclasses import dataclass

from afabench.fit.contract import TrainingContract, store_contract_config


@dataclass(frozen=True, kw_only=True)
class BackboneTrainingConfig(TrainingContract):
    tabicl_checkpoint_path: str
    batch_size: int = 1
    sequence_length: int = 64
    n_steps: int = 100
    policy_lr: float = 1e-4
    backbone_lr: float = 1e-5
    temperature: float = 0.1
    checkpoint_interval: int = 10
    n_validation_tasks: int = 2
    pretraining_pool_size: int = 5000
    context_set_size: int = 96
    policy_checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        if not 50 < self.sequence_length <= self.pretraining_pool_size:
            msg = "require 50 < sequence_length <= pretraining_pool_size"
            raise ValueError(msg)
        for name in (
            "batch_size",
            "n_steps",
            "checkpoint_interval",
            "n_validation_tasks",
        ):
            if getattr(self, name) < 1:
                msg = f"{name} must be positive"
                raise ValueError(msg)
        for name in ("policy_lr", "backbone_lr", "temperature"):
            if not 0 < getattr(self, name) < float("inf"):
                msg = f"{name} must be positive and finite"
                raise ValueError(msg)
        if self.context_set_size < 2:
            msg_0 = "context_set_size must be at least two"
            raise ValueError(msg_0)


store_contract_config(
    name="train_l2m_tabicl_backbone", config_class=BackboneTrainingConfig
)
