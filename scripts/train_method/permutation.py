import logging
from pathlib import Path
from typing import cast

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import AUROC
from torchrl.modules import MLP
from tqdm import tqdm

from afabench.components.methods.discriminative.common.datasets import (
    prepare_datasets,
)
from afabench.components.methods.discriminative.common.utils import (
    afa_discriminative_training_prep,
)
from afabench.components.methods.static.pt.config import (
    PermutationTrainingConfig,
)
from afabench.components.methods.static.pt.models import BaseModel
from afabench.components.methods.static.pt.static_methods import (
    StaticBaseMethod,
)
from afabench.components.methods.static.pt.utils import transform_dataset
from afabench.fit.run import fit_run, save_result
from afabench.fit.smoke_test import limit
from afabench.fit.tensor_batches import (
    TensorBatchDataset,
    passthrough_batch,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/permutation",
    config_name="config",
)
def main(cfg: PermutationTrainingConfig) -> None:
    cfg = cast("PermutationTrainingConfig", OmegaConf.to_object(cfg))
    log.debug(cfg)

    assert cfg.hard_budget is not None, "hard_budget must be configured"
    hard_budget = cfg.hard_budget
    device = torch.device(cfg.device)
    torch.set_float32_matmul_precision("medium")
    cfg = limit(
        cfg,
        selector={"nepochs": 1, "patience": 1},
        classifier={"nepochs": 1},
    )

    with fit_run(cfg, tags=["permutation"]) as metric_logger:
        train_dataset, val_dataset, _, _, class_weights = (
            afa_discriminative_training_prep(
                train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
                val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
                initializer_cfg=cfg.initializer,
                unmasker_cfg=cfg.unmasker,
            )
        )
        assert class_weights is not None
        class_weights = class_weights.to(device)
        train_loader, val_loader, d_in, d_out = prepare_datasets(
            train_dataset, val_dataset, cfg.batch_size
        )

        def auroc_metric(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return AUROC(task="multiclass", num_classes=d_out)(
                pred.softmax(dim=1), y
            )

        predictors: dict[int, nn.Module] = {}
        selected_history: dict[int, list[int]] = {}
        num_features = list(range(1, hard_budget + 1))

        model = MLP(
            in_features=d_in,
            out_features=d_out,
            num_cells=cfg.selector.num_cells,
            activation_class=nn.ReLU,
        )
        basemodel = BaseModel(model).to(device)
        basemodel.fit(
            train_loader,
            val_loader,
            lr=cfg.selector.lr,
            nepochs=cfg.selector.nepochs,
            loss_fn=nn.CrossEntropyLoss(weight=class_weights),
            verbose=False,
            metric_logger=metric_logger.log,
            metric_prefix="permutation_selector",
        )

        permutation_importance = np.zeros(d_in)
        rng = np.random.default_rng(cfg.seed)
        x_train, _ = train_dataset.get_all_data()
        for i in tqdm(range(d_in)):
            x_val, y_val = val_dataset.get_all_data()
            x_val = x_val.clone()
            y_val = y_val.argmax(dim=1).long()
            shuffled_indices = rng.choice(len(x_train), size=len(x_val))
            x_val[:, i] = x_train[shuffled_indices, i]
            with torch.no_grad():
                pred = model(x_val.to(device)).cpu()
                permutation_importance[i] = -auroc_metric(pred, y_val)

        ranked_features = np.argsort(permutation_importance)[::-1]

        for num in num_features:
            selected_features = ranked_features[:num]
            selected_history[num] = selected_features.tolist()
            train_subset = transform_dataset(
                train_dataset, selected_features.copy()
            )
            val_subset = transform_dataset(
                val_dataset, selected_features.copy()
            )
            train_subset_loader = DataLoader(
                TensorBatchDataset(*train_subset.tensors),
                collate_fn=passthrough_batch,
                batch_size=128,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
            )
            val_subset_loader = DataLoader(
                TensorBatchDataset(*val_subset.tensors),
                collate_fn=passthrough_batch,
                batch_size=1024,
                pin_memory=True,
            )

            model = MLP(
                in_features=num,
                out_features=d_out,
                num_cells=cfg.classifier.num_cells,
                activation_class=nn.ReLU,
            )
            predictor = BaseModel(model).to(device)
            predictor.fit(
                train_subset_loader,
                val_subset_loader,
                lr=cfg.classifier.lr,
                nepochs=cfg.classifier.nepochs,
                loss_fn=nn.CrossEntropyLoss(weight=class_weights),
                verbose=False,
                metric_logger=metric_logger.log,
                metric_prefix=f"permutation_classifier/{num}_features",
            )

            predictors[num] = model

        static_method = StaticBaseMethod(selected_history, predictors, device)

    save_result(static_method, cfg)


if __name__ == "__main__":
    main()
