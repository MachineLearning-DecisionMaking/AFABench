"""Fit TFM-Synth from Mother-Tree v2 batches and save an AFABench bundle."""

import signal
from threading import Event
from typing import cast

import hydra
import torch
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.l2m_tabicl.benchmark_train import (
    record_prior_inputs,
    train_l2m_tabicl,
)
from afabench.components.methods.discriminative.l2m_tabicl.config import (
    L2MTabICLTrainingConfig,
)
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result
from afabench.fit.smoke_test import limit


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/l2m_tabicl",
    config_name="config",
)
def main(cfg: L2MTabICLTrainingConfig) -> None:
    cfg = cast("L2MTabICLTrainingConfig", OmegaConf.to_object(cfg))
    cfg = limit(
        cfg,
        context_set_size=8,
        n_steps=1,
        training_batch_limit=1,
        validation_batch_limit=1,
        training_task_limit=1,
        validation_task_limit=1,
        checkpoint_interval=1,
        n_training_evaluation_tasks=min(cfg.n_training_evaluation_tasks, 1),
    )
    if cfg.smoke_test:
        torch.set_num_threads(1)
    cfg = record_prior_inputs(cfg)
    inputs = load_inputs(cfg)
    stopping = Event()
    previous_handler = signal.signal(
        signal.SIGUSR1, lambda _signum, _frame: stopping.set()
    )
    with fit_run(
        cfg, tags=["tfm_synth", "mother_tree_v2"], inputs=inputs
    ) as logger:
        try:
            method = train_l2m_tabicl(
                cfg,
                inputs=inputs,
                metric_logger=logger.log,
                stop_requested=stopping.is_set,
            )
            save_result(method, cfg, inputs=inputs)
        finally:
            signal.signal(signal.SIGUSR1, previous_handler)


if __name__ == "__main__":
    main()
