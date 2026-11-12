"""The per-dataset L2M settings the pipeline resolves for every run."""

from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.l2m.config import (
    UPSTREAM_CONTEXT_SET_SIZE,
    L2MPretrainingConfig,
    L2MTrainingConfig,
    TabICLPriorType,
    TaskPrior,
)

# The settings of conf/scripts/{pretrain_model,train_method}/l2m/
# dataset_override/, for every task prior: sequence_length,
# pretraining_pool_size and n_steps of the two stages. The authors' numbers
# for MiniBooNE (sequence length, pool and training steps) and METABRIC; the
# pretraining steps of the cube and MiniBooNE are the thesis choice.
EXPECTED = {
    "cube": (600, 600, 100_000, 50_000),
    "miniboone_subset": (1000, 5000, 100_000, 50_000),
    "metabric": (500, 1000, 100_000, 50_000),
}
# Validation instances the stored context set can hold, at most.
VALIDATION_SIZES = {"cube": 200, "miniboone_subset": 25_918, "metabric": 379}
# The pipeline's method parameters of each run (workflow/conf/
# pretrain_mappings/all.yaml).
RUNS = {
    "l2m": ["task_prior=real_pool"],
    "l2m_synth": ["task_prior=synthetic"],
    "l2m_better_synth": [
        "task_prior=tabicl_v1",
        "prior_type=prior_bag_plus_tree",
    ],
}
CONTRACT = [
    "train_dataset_bundle_path=train.bundle",
    "val_dataset_bundle_path=val.bundle",
    "classifier_bundle_path=classifier.bundle",
    "save_path=out.bundle",
    "initializer=cold",
    "unmasker=direct",
    "device=cpu",
    "seed=0",
]


def _compose(stage: str, dataset: str, overrides: list[str]) -> object:
    config_dir = Path(f"conf/scripts/{stage}/l2m").resolve()
    stage_overrides = (
        []
        if stage == "pretrain_model"
        else [
            "pretrained_model_bundle_path=model.bundle",
            "method_name=l2m",
            "hard_budget=5",
            "soft_budget_param=null",
        ]
    )
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                *CONTRACT,
                *stage_overrides,
                f"dataset_key={dataset}",
                *overrides,
            ],
        )
    return OmegaConf.to_object(cfg)


@pytest.mark.parametrize("dataset", EXPECTED)
@pytest.mark.parametrize("method", RUNS)
def test_every_run_resolves_the_dataset_settings_for_both_stages(
    method: str, dataset: str
) -> None:
    sequence_length, pool_size, pretrain_steps, train_steps = EXPECTED[dataset]

    pretraining = cast(
        "L2MPretrainingConfig",
        _compose("pretrain_model", dataset, RUNS[method]),
    )
    training = cast(
        "L2MTrainingConfig", _compose("train_method", dataset, RUNS[method])
    )

    for cfg in (pretraining, training):
        assert cfg.sequence_length == sequence_length
        assert cfg.pretraining_pool_size == pool_size
        assert cfg.batch_size == 8
        assert cfg.dataset_key == dataset
    assert (pretraining.n_steps, training.n_steps) == (
        pretrain_steps,
        train_steps,
    )
    # Both stages use the same task prior.
    assert pretraining.task_prior is training.task_prior
    assert pretraining.prior_type is training.prior_type
    assert pretraining.task_workers == training.task_workers
    # The stored context set is covered by what training queries see, up to
    # the authors' cap on it.
    context_size = min(training.context_set_size, VALIDATION_SIZES[dataset])
    assert sequence_length - UPSTREAM_CONTEXT_SET_SIZE >= context_size


@pytest.mark.parametrize(
    ("method", "task_prior", "prior_type", "task_workers"),
    [
        ("l2m", TaskPrior.real_pool, TabICLPriorType.prior_bag, 0),
        ("l2m_synth", TaskPrior.synthetic, TabICLPriorType.prior_bag, 0),
        (
            "l2m_better_synth",
            TaskPrior.tabicl_v1,
            TabICLPriorType.prior_bag_plus_tree,
            4,
        ),
    ],
)
def test_the_runs_differ_in_their_task_prior_only(
    method: str,
    task_prior: TaskPrior,
    prior_type: TabICLPriorType,
    task_workers: int,
) -> None:
    cfg = cast(
        "L2MPretrainingConfig",
        _compose("pretrain_model", "cube", RUNS[method]),
    )

    assert cfg.task_prior is task_prior
    assert cfg.prior_type is prior_type
    assert cfg.task_workers == task_workers


@pytest.mark.parametrize(
    ("method", "max_grad_norm"),
    [("l2m", None), ("l2m_synth", None), ("l2m_better_synth", 1.0)],
)
def test_only_the_tabicl_prior_clips_the_gradient_in_both_stages(
    method: str, max_grad_norm: float | None
) -> None:
    pretraining = cast(
        "L2MPretrainingConfig",
        _compose("pretrain_model", "metabric", RUNS[method]),
    )
    training = cast(
        "L2MTrainingConfig",
        _compose("train_method", "metabric", RUNS[method]),
    )

    assert pretraining.max_grad_norm == max_grad_norm
    assert training.max_grad_norm == max_grad_norm


@pytest.mark.parametrize("stage", ["pretrain_model", "train_method"])
def test_a_command_line_override_beats_the_gradient_bound(stage: str) -> None:
    cfg = cast(
        "L2MPretrainingConfig | L2MTrainingConfig",
        _compose(
            stage, "cube", ["task_prior=tabicl_v1", "max_grad_norm=null"]
        ),
    )

    assert cfg.max_grad_norm is None


def test_a_command_line_override_beats_the_dataset_settings() -> None:
    cfg = cast(
        "L2MPretrainingConfig",
        _compose("pretrain_model", "metabric", ["sequence_length=300"]),
    )

    assert cfg.sequence_length == 300


def test_a_dataset_without_a_pool_that_fills_a_task_fails_at_composition() -> (
    None
):
    with pytest.raises(ValueError, match="pretraining_pool_size=600"):
        _compose(
            "pretrain_model",
            "cube",
            ["task_prior=real_pool", "sequence_length=1000"],
        )


def test_sub_priors_are_rejected_for_the_pool_task_priors() -> None:
    with pytest.raises(
        ValueError, match="only applies to task_prior=tabicl_v1"
    ):
        _compose(
            "pretrain_model",
            "cube",
            ["task_prior=synthetic", "prior_type=tree_only"],
        )
