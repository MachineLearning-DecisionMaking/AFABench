import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf

from afabench.components.classifiers import WrappedMaskedMLPClassifier
from afabench.components.classifiers.config import (
    TrainMaskedMLPClassifierConfig,
)
from afabench.components.classifiers.lit_models import LitMaskedMLPClassifier
from afabench.components.methods.rl.common.dataset_utils import (
    DataModuleFromDatasets,
)
from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.core.naming import infer_dataset_key_from_class_name
from afabench.core.utils import (
    get_class_frequencies,
    initialize_wandb_run,
    set_seed,
)
from afabench.datasets.training_views import TrainingDatasetView
from afabench.training.supervised_learning import lightning_root
from afabench.training.tensor_batches import (
    TensorBatchDataset,
    passthrough_batch,
)

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from afabench.core.types import AFADataset, Features, Label

log = logging.getLogger(__name__)


def _batch_tensors(
    dataset: "AFADataset",
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    flat = features.flatten(start_dim=1)
    if not isinstance(dataset, TrainingDatasetView):
        return flat, labels
    availability = dataset.selection_availability.flatten(start_dim=1)
    log.info(
        f"Training on a view: {availability.float().mean().item():.3f} of "
        "feature values are selectable."
    )
    return flat, labels, availability


def _lightning_log_dir(save_path: str) -> Path:
    save_path_parts = Path(save_path).with_suffix("").parts
    return lightning_root() / "train_classifier" / Path(*save_path_parts)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_classifier/masked_mlp_classifier",
    config_name="config",
)
def main(cfg: TrainMaskedMLPClassifierConfig) -> None:
    cfg = cast("TrainMaskedMLPClassifierConfig", OmegaConf.to_object(cfg))
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    if cfg.use_wandb:
        _run = initialize_wandb_run(
            cfg=asdict(cfg),
            job_type="classifier_training",
            tags=["masked_mlp_classifier"],
        )

    # Handle smoke test
    if cfg.smoke_test:
        log.info("Smoke test mode: reducing epochs and batch size")
        cfg.epochs = 2
        cfg.batch_size = min(cfg.batch_size, 32)

    # Load datasets via bundle system
    train_dataset, train_manifest = load_bundle(Path(cfg.train_dataset_path))
    dataset_name = infer_dataset_key_from_class_name(
        train_manifest["class_name"]
    )
    train_dataset = cast("AFADataset", cast("object", train_dataset))
    val_dataset, _ = load_bundle(Path(cfg.val_dataset_path))
    val_dataset = cast("AFADataset", cast("object", val_dataset))

    # Get dimensions (flatten for MLP)
    feature_shape = train_dataset.feature_shape
    n_features = feature_shape.numel()
    n_classes = train_dataset.label_shape[0]

    train_features, train_labels = train_dataset.get_all_data()
    val_features, val_labels = val_dataset.get_all_data()
    datamodule = DataModuleFromDatasets(
        train_dataset=cast(
            "Dataset[tuple[Features, Label]]",
            cast(
                "object",
                TensorBatchDataset(
                    *_batch_tensors(
                        train_dataset, train_features, train_labels
                    )
                ),
            ),
        ),
        val_dataset=cast(
            "Dataset[tuple[Features, Label]]",
            cast(
                "object",
                TensorBatchDataset(
                    *_batch_tensors(val_dataset, val_features, val_labels)
                ),
            ),
        ),
        batch_size=cfg.batch_size,
        collate_fn=passthrough_batch,
    )

    # Class weights
    train_class_probabilities = get_class_frequencies(train_labels)

    # Model
    lit_model = LitMaskedMLPClassifier(
        n_features=n_features,
        n_classes=n_classes,
        num_cells=tuple(cfg.num_cells),
        dropout=cfg.dropout,
        class_probabilities=train_class_probabilities,
        min_masking_probability=cfg.min_masking_probability,
        max_masking_probability=cfg.max_masking_probability,
        lr=cfg.lr,
    )

    # Training
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_many_observations",
        save_top_k=1,
        mode="min",
    )

    log_dir = _lightning_log_dir(cfg.save_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    trainer_logger = (
        WandbLogger(save_dir="extra/logs/wandb")
        if cfg.use_wandb
        else CSVLogger(save_dir=str(log_dir), name="masked_mlp_classifier")
    )

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=trainer_logger,
        accelerator=cfg.device,
        devices=1,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        enable_progress_bar=sys.stderr.isatty(),
    )

    trainer.fit(lit_model, datamodule)

    # Load best checkpoint
    best_checkpoint = checkpoint_callback.best_model_path
    if not best_checkpoint:
        msg = "No checkpoint saved during training"
        raise RuntimeError(msg)

    best_lit_model = LitMaskedMLPClassifier.load_from_checkpoint(
        best_checkpoint,
        n_features=n_features,
        n_classes=n_classes,
        num_cells=tuple(cfg.num_cells),
        dropout=cfg.dropout,
        class_probabilities=train_class_probabilities,
        min_masking_probability=cfg.min_masking_probability,
        max_masking_probability=cfg.max_masking_probability,
        lr=cfg.lr,
    )

    # Wrap and save
    wrapped_classifier = WrappedMaskedMLPClassifier(
        module=best_lit_model.classifier, device=device
    )

    save_bundle(
        obj=wrapped_classifier,
        path=Path(cfg.save_path),
        metadata={
            "train_dataset_path": cfg.train_dataset_path,
            "val_dataset_path": cfg.val_dataset_path,
            "dataset_name": dataset_name,
            "seed": cfg.seed,
            "num_cells": list(cfg.num_cells),
            "dropout": cfg.dropout,
            "n_features": n_features,
            "n_classes": n_classes,
            "epochs": cfg.epochs,
            "min_masking_probability": cfg.min_masking_probability,
            "max_masking_probability": cfg.max_masking_probability,
        },
    )
    log.info(f"Saved classifier to: {cfg.save_path}")


if __name__ == "__main__":
    main()
