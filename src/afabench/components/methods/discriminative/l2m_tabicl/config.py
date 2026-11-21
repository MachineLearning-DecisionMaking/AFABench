"""AFABench training settings for TFM-Synth with Mother-Tree v2 data."""

import math
from dataclasses import dataclass, field

from afabench.fit.contract import TrainingContract, store_contract_config


@dataclass(frozen=True, kw_only=True)
class L2MTabICLTrainingConfig(TrainingContract):
    tabicl_checkpoint_path: str
    synthetic_prior: str
    training_prior_dir: str
    validation_prior_dir: str
    training_batch_limit: int
    validation_batch_limit: int
    context_set_size: int
    n_steps: int
    checkpoint_interval: int
    temperature: float
    policy_lr: float
    n_training_evaluation_tasks: int = 0
    acquisition_diagnostic_interval: int = 0
    time_budget_seconds: float | None = None
    training_task_limit: int | None = None
    validation_task_limit: int | None = None
    checkpoint_path: str | None = None
    resume_path: str | None = None
    # Filled from the selected files before training; saved in provenance.
    training_prior_hashes: dict[str, str] = field(default_factory=dict)
    validation_prior_hashes: dict[str, str] = field(default_factory=dict)
    tabicl_checkpoint_sha256: str = ""

    def __post_init__(self) -> None:  # noqa: C901
        for name in (
            "n_training_evaluation_tasks",
            "acquisition_diagnostic_interval",
        ):
            if getattr(self, name) < 0:
                message = f"{name} must be nonnegative"
                raise ValueError(message)
        if self.time_budget_seconds is not None and (
            not math.isfinite(self.time_budget_seconds)
            or self.time_budget_seconds <= 0
        ):
            message = "time_budget_seconds must be finite and positive"
            raise ValueError(message)
        for name in (
            "training_batch_limit",
            "validation_batch_limit",
            "n_steps",
            "checkpoint_interval",
        ):
            value = getattr(self, name)
            if value < 1:
                message = f"{name}={value} must be at least 1"
                raise ValueError(message)
        if self.context_set_size < 2:
            message = "context_set_size must be at least 2"
            raise ValueError(message)
        for name in ("temperature", "policy_lr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                message = f"{name}={value} must be finite and positive"
                raise ValueError(message)
        for name in ("training_task_limit", "validation_task_limit"):
            value = getattr(self, name)
            if value is not None and value < 1:
                message = f"{name}={value} must be at least 1 or null"
                raise ValueError(message)


store_contract_config(
    name="train_l2m_tabicl", config_class=L2MTabICLTrainingConfig
)
