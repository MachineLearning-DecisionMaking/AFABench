from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.l2m.config import (
    L2MTrainingConfig,
    TaskPrior,
)
from afabench.components.methods.discriminative.l2m.train import (
    train_l2m,
)
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result
from afabench.fit.smoke_test import limit


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/l2m",
    config_name="config",
)
def main(cfg: L2MTrainingConfig) -> None:
    cfg = cast("L2MTrainingConfig", OmegaConf.to_object(cfg))
    # Model width comes from the pretrained model, so the smoke test runs
    # the real heads on short sequences and a small context set.
    cfg = limit(
        cfg,
        # Every task has 50 context instances, the sampler's own default.
        # The TabICL v1 prior rejects nearly every draw with 8 classes in
        # fewer than about 100 instances.
        sequence_length=100 if cfg.task_prior is TaskPrior.tabicl_v1 else 64,
        pretraining_pool_size=64,
        task_workers=0,
        n_steps=2,
        checkpoint_interval=1,
        n_validation_tasks=2,
        context_set_size=8,
    )
    inputs = load_inputs(cfg)
    with fit_run(cfg, tags=["l2m"], inputs=inputs) as metric_logger:
        result = train_l2m(cfg, metric_logger=metric_logger.log, inputs=inputs)
        save_result(result, cfg, inputs=inputs)


if __name__ == "__main__":
    main()
