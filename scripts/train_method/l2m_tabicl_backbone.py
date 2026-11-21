"""Train the experimental L2M-backbone/full-TabICL hybrid."""

from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.l2m_tabicl_backbone.config import (
    BackboneTrainingConfig,
)
from afabench.components.methods.discriminative.l2m_tabicl_backbone.train import (
    train_backbone,
)
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result
from afabench.fit.smoke_test import limit


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/l2m_tabicl_backbone",
    config_name="config",
)
def main(cfg: BackboneTrainingConfig) -> None:
    cfg = cast("BackboneTrainingConfig", OmegaConf.to_object(cfg))
    cfg = limit(
        cfg,
        sequence_length=51,
        batch_size=1,
        pretraining_pool_size=64,
        n_steps=1,
        checkpoint_interval=1,
        n_validation_tasks=1,
        context_set_size=8,
    )
    inputs = load_inputs(cfg)
    with fit_run(cfg, tags=["l2m_tabicl_backbone"], inputs=inputs) as logger:
        result = train_backbone(cfg, inputs=inputs, metric_logger=logger.log)
        save_result(result, cfg, inputs=inputs)


if __name__ == "__main__":
    main()
