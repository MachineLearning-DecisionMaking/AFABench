import logging
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchrl.modules import MLP

from afabench.components.methods.discriminative.common.datasets import (
    prepare_datasets,
)
from afabench.components.methods.static.cae.config import CAETrainingConfig
from afabench.components.methods.static.common.models import BaseModel
from afabench.components.methods.static.common.static_methods import (
    ConcreteMask,
    DifferentiableSelector,
    StaticBaseMethod,
)
from afabench.components.methods.static.common.utils import transform_dataset
from afabench.core.utils import get_class_frequencies
from afabench.fit.inputs import FitInputs
from afabench.fit.tensor_batches import (
    TensorBatchDataset,
    passthrough_batch,
)

log = logging.getLogger(__name__)


def train_tabular(
    cfg: CAETrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
    *,
    inputs: FitInputs,
) -> StaticBaseMethod:
    log.debug(cfg)
    assert cfg.hard_budget is not None, "hard_budget must be configured"
    print(str(cfg))
    device = torch.device(cfg.device)
    torch.set_float32_matmul_precision("medium")
    train_dataset = inputs.train_dataset()
    val_dataset = inputs.val_dataset()
    _, train_labels = train_dataset.get_all_data()
    class_weights = 1 / get_class_frequencies(train_labels)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset,
        val_dataset,
        cfg.batch_size,
        smoke_test=cfg.smoke_test,
    )

    model = MLP(
        in_features=d_in,
        out_features=d_out,
        num_cells=cfg.selector.num_cells,
        activation_class=nn.ReLU,
    )
    selector_layer = ConcreteMask(d_in, cfg.hard_budget)
    diff_selector = DifferentiableSelector(
        model=model,
        selector_layer=selector_layer,
    ).to(device)
    diff_selector.fit(
        train_loader,
        val_loader,
        lr=cfg.selector.lr,
        nepochs=cfg.selector.nepochs,
        loss_fn=nn.CrossEntropyLoss(weight=class_weights),
        patience=cfg.selector.patience,
        verbose=False,
        metric_logger=metric_logger,
        metric_prefix="cae_selector",
    )

    logits = selector_layer.logits.cpu().data.numpy()
    ranked_features = np.sort(logits.argmax(axis=1))

    if len(np.unique(ranked_features)) != cfg.hard_budget:
        print(
            f"{len(np.unique(ranked_features))} selected instead of {
                cfg.hard_budget
            }, appending extras"
        )
    num_extras = cfg.hard_budget - len(np.unique(ranked_features))
    remaining_features = np.setdiff1d(np.arange(d_in), ranked_features)
    ranked_features = np.sort(
        np.concatenate(
            [np.unique(ranked_features), remaining_features[:num_extras]]
        )
    )

    predictors: dict[int, nn.Module] = {}
    selected_history: dict[int, list[int]] = {}

    num_features = list(range(1, cfg.hard_budget + 1))
    for num in num_features:
        selected_features = ranked_features[:num]
        selected_history[num] = selected_features.tolist()

        train_subset = transform_dataset(train_dataset, selected_features)
        val_subset = transform_dataset(val_dataset, selected_features)

        train_subset_loader = DataLoader(
            TensorBatchDataset(*train_subset.tensors),
            collate_fn=passthrough_batch,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        val_subset_loader = DataLoader(
            TensorBatchDataset(*val_subset.tensors),
            collate_fn=passthrough_batch,
            batch_size=cfg.batch_size,
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
            metric_logger=metric_logger,
            metric_prefix=f"cae_classifier/{num}_features",
        )

        predictors[num] = model

    static_method = StaticBaseMethod(selected_history, predictors, device)

    return static_method
