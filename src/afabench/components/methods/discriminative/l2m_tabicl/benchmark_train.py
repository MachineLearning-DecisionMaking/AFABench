"""Train TFM-Synth from saved Mother-Tree batches on the AFABench contract."""

import hashlib
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch

from afabench.components.methods.discriminative.l2m.afa_methods import (
    require_direct_unmasker,
)
from afabench.components.methods.discriminative.l2m.train import (
    _draw_context_set,
)
from afabench.components.methods.discriminative.l2m_tabicl.afa_methods import (
    L2MTabICLAFAMethod,
)
from afabench.components.methods.discriminative.l2m_tabicl.config import (
    L2MTabICLTrainingConfig,
)
from afabench.components.methods.discriminative.l2m_tabicl.diagnostics import (
    record_acquisition,
)
from afabench.components.methods.discriminative.l2m_tabicl.fit_mother_tree import (
    fit_on_mother_tree,
)
from afabench.components.methods.discriminative.l2m_tabicl.mother_tree import (
    MotherTreeTask,
    load_mother_tree_batch,
)
from afabench.components.methods.discriminative.l2m_tabicl.policy import (
    AcquisitionPolicy,
)
from afabench.components.methods.discriminative.l2m_tabicl.predictor import (
    FrozenTabICLPredictor,
)
from afabench.fit.inputs import FitInputs

log = logging.getLogger(__name__)


def record_prior_inputs(
    cfg: L2MTabICLTrainingConfig,
) -> L2MTabICLTrainingConfig:
    """Resolve file selection and record content hashes before fitting."""
    _check_config(cfg)
    return replace(
        cfg,
        training_prior_hashes=_select_batches(
            cfg.training_prior_dir, cfg.training_batch_limit
        ),
        validation_prior_hashes=_select_batches(
            cfg.validation_prior_dir, cfg.validation_batch_limit
        ),
        tabicl_checkpoint_sha256=_file_hash(Path(cfg.tabicl_checkpoint_path)),
    )


def train_l2m_tabicl(  # noqa: C901
    cfg: L2MTabICLTrainingConfig,
    *,
    inputs: FitInputs,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> L2MTabICLAFAMethod:
    require_direct_unmasker(cfg.unmasker)
    _check_config(cfg)
    train_dataset = inputs.train_dataset()
    n_features = train_dataset.feature_shape.numel()
    n_classes = train_dataset.label_shape.numel()
    if len(train_dataset.feature_shape) != 1:
        message = "This initial variant requires flat numerical features"
        raise ValueError(message)
    if cfg.hard_budget is None or not 0 < cfg.hard_budget < n_features:
        message = (
            f"hard_budget={cfg.hard_budget} must be between 1 and "
            f"{n_features - 1}"
        )
        raise ValueError(message)
    validation_dataset = inputs.val_dataset()
    if (
        validation_dataset.feature_shape != train_dataset.feature_shape
        or validation_dataset.label_shape != train_dataset.label_shape
        or len(validation_dataset) < 2
    ):
        message = "Validation data must match train shapes and have at least two rows"
        raise ValueError(message)
    if (
        _file_hash(Path(cfg.tabicl_checkpoint_path))
        != cfg.tabicl_checkpoint_sha256
    ):
        message = (
            "TabICL checkpoint hash is missing or changed before training"
        )
        raise ValueError(message)
    device = torch.device(cfg.device)
    predictor = FrozenTabICLPredictor(Path(cfg.tabicl_checkpoint_path)).to(
        device
    )
    if not 2 <= n_classes <= predictor.model.max_classes:
        message = (
            f"n_classes={n_classes} exceeds the supported TabICL class range"
        )
        raise ValueError(message)
    policy = AcquisitionPolicy(
        n_features,
        representation_dim=predictor.model.embed_dim
        * predictor.model.row_num_cls,
    ).to(device)
    training_tasks = _load_pool(
        cfg.training_prior_hashes,
        n_features,
        predictor.model.max_classes,
        cfg.training_task_limit,
    )
    validation_tasks = _load_pool(
        cfg.validation_prior_hashes,
        n_features,
        predictor.model.max_classes,
        cfg.validation_task_limit,
    )
    log.info(
        "Fitting on %d Mother-Tree tasks; validating on %d separate tasks",
        len(training_tasks),
        len(validation_tasks),
    )

    def diagnostic(step: int) -> None:
        if (
            cfg.acquisition_diagnostic_interval
            and cfg.checkpoint_path
            and (
                step % cfg.acquisition_diagnostic_interval == 0
                or step == cfg.n_steps
                or (stop_requested is not None and stop_requested())
            )
        ):
            record_acquisition(
                predictor,
                policy,
                validation_tasks[:8],
                cfg.unmasker,
                Path(cfg.checkpoint_path).parent / "acquisition",
                step,
            )

    result = fit_on_mother_tree(
        predictor,
        policy,
        training_tasks,
        validation_tasks,
        n_steps=cfg.n_steps,
        checkpoint_interval=cfg.checkpoint_interval,
        temperature=cfg.temperature,
        learning_rate=cfg.policy_lr,
        seed=cfg.seed,
        checkpoint_path=Path(cfg.checkpoint_path)
        if cfg.checkpoint_path
        else None,
        resume_path=Path(cfg.resume_path) if cfg.resume_path else None,
        checkpoint_identity={
            "tabicl": cfg.tabicl_checkpoint_sha256,
            "training": list(cfg.training_prior_hashes.values()),
            "validation": list(cfg.validation_prior_hashes.values()),
            "training_task_limit": cfg.training_task_limit,
            "validation_task_limit": cfg.validation_task_limit,
            "diagnostic_interval": cfg.acquisition_diagnostic_interval,
            "source_hashes": {
                name: _file_hash(Path(__file__).with_name(name))
                for name in (
                    "fit.py",
                    "train.py",
                    "acquisition.py",
                    "policy.py",
                    "predictor.py",
                    "preprocessing.py",
                    "fit_mother_tree.py",
                    "diagnostics.py",
                    "benchmark_train.py",
                    "training_state.py",
                    "config.py",
                    "afa_methods.py",
                    "mother_tree.py",
                )
            },
        },
        stop_requested=stop_requested,
        n_training_evaluation_tasks=cfg.n_training_evaluation_tasks,
        time_budget_seconds=cfg.time_budget_seconds,
        checkpoint_callback=diagnostic
        if cfg.acquisition_diagnostic_interval
        else None,
    )
    for record in result.history:
        metrics = {
            "l2m_tabicl/step": float(record.step),
            "l2m_tabicl/validation_loss": record.validation_loss,
        }
        if record.training_loss is not None:
            metrics["l2m_tabicl/training_loss"] = record.training_loss
        if record.mean_training_loss is not None:
            metrics["l2m_tabicl/mean_training_loss"] = (
                record.mean_training_loss
            )
        if record.training_evaluation_loss is not None:
            metrics["l2m_tabicl/training_evaluation_loss"] = (
                record.training_evaluation_loss
            )
        log.info("Mother-Tree policy fit: %s", metrics)
        if metric_logger is not None:
            metric_logger(metrics)
    log.info("Restored policy from step %d", result.best_step)
    # The real validation split supplies labeled context, not policy updates.
    context_features, context_labels = _draw_context_set(
        validation_dataset.get_all_data(),
        n_features=n_features,
        context_set_size=cfg.context_set_size,
        seed=cfg.seed,
    )
    return L2MTabICLAFAMethod(
        predictor,
        policy,
        context_features,
        context_labels,
        unmasker=cfg.unmasker,
    )


def _check_config(cfg: L2MTabICLTrainingConfig) -> None:
    if cfg.synthetic_prior != "mother_tree_tabiclv2":
        message = f"Unsupported synthetic_prior={cfg.synthetic_prior!r}"
        raise ValueError(message)
    if cfg.soft_budget_param is not None:
        message = "This initial variant supports only hard-budget evaluation"
        raise ValueError(message)
    if cfg.pretrained_model_bundle_path is not None:
        message = (
            "Use tabicl_checkpoint_path; this script fits the policy itself"
        )
        raise ValueError(message)
    if cfg.context_set_size < 2:
        message = f"context_set_size={cfg.context_set_size} must be at least 2"
        raise ValueError(message)


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _select_batches(directory: str, limit: int) -> dict[str, str]:
    if limit < 1:
        message = f"Batch limit={limit} must be at least 1"
        raise ValueError(message)
    paths = sorted(Path(directory).glob("batch_*.pt"))[:limit]
    if not paths:
        message = f"No batch_*.pt files found in {directory}"
        raise FileNotFoundError(message)
    return {str(path.resolve()): _file_hash(path) for path in paths}


def _load_pool(
    files: dict[str, str],
    n_features: int,
    max_classes: int,
    task_limit: int | None,
) -> list[MotherTreeTask]:
    if not files:
        message = (
            "Prior file hashes are missing; call record_prior_inputs first"
        )
        raise ValueError(message)
    records = []
    for filename, expected_hash in files.items():
        path = Path(filename)
        if _file_hash(path) != expected_hash:
            message = f"Prior batch changed before loading: {filename}"
            raise ValueError(message)
        records.extend(
            load_mother_tree_batch(
                path, expected_features=n_features, max_classes=max_classes
            )
        )
        if task_limit is not None and len(records) >= task_limit:
            return records[:task_limit]
    return records
