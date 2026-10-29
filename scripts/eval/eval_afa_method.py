import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast, final

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from afabench.components.initializers.utils import (
    get_afa_initializer_from_config,
)
from afabench.components.unmaskers.utils import (
    get_afa_unmasker_from_config,
)
from afabench.core.bundle_system.bundle import LoadedBundle, load_bundle
from afabench.core.class_probabilities import as_class_probability_fn
from afabench.core.provenance import (
    ProvenanceRecord,
    capture_provenance,
    shared_dataset_identity,
)
from afabench.core.types import (
    AFAClassifier,
    AFADataset,
    AFAMethod,
    SupportsForcedAcquisition,
)
from afabench.core.utils import (
    set_seed,
)
from afabench.evaluation.config import EvalConfig
from afabench.evaluation.eval import eval_afa_method
from afabench.evaluation.provenance import save_evaluation_table
from afabench.evaluation.schemas import (
    SavedEvaluationSchema,
    identity_column,
)
from afabench.fit.smoke_test import eval_settings

if TYPE_CHECKING:
    import polars as pl
    from wandb.sdk.wandb_run import Run

    from afabench.core.types import AFAInitializer, AFAUnmasker

log = logging.getLogger(__name__)


class MethodRecordStageError(ValueError):
    """Raised when a method bundle's provenance record is not from training."""


@final
class AFAEvaluator:
    """
    Evaluate a method bundle and save an identified evaluation table.

    `initializer_name` is the name of the initializer config the evaluation
    selected, written as the table's `initializer` column; the config itself
    holds only the class and its arguments.
    """

    def __init__(self, cfg: EvalConfig, *, initializer_name: str):
        self._cfg = cfg
        self._initializer_name = initializer_name
        self._seed: int | None = None
        self._provenance: ProvenanceRecord | None = None
        self._wandb_run: Run | None = None
        self._method_bundle: LoadedBundle[AFAMethod] | None = None
        self._method: AFAMethod | None = None
        self._unmasker: AFAUnmasker | None = None
        self._initializer: AFAInitializer | None = None
        self._dataset_bundle: LoadedBundle[AFADataset] | None = None
        self._dataset: AFADataset | None = None
        self._classifier_bundle: LoadedBundle[AFAClassifier] | None = None
        self._external_classifier: AFAClassifier | None = None
        self._n_selection_choices: int | None = None
        self._selection_costs: torch.Tensor | None = None
        self._df_eval: pl.DataFrame | None = None

    def run(self) -> None:
        # The evaluator owns the seed (ADR 0002): a null seed is drawn once
        # and the drawn integer is what every component receives
        self._seed = set_seed(self._cfg.seed)
        self._init_wandb()
        self._smoke_test_override()
        self._load()
        self._set_seeds()
        self._set_soft_budget()
        self._set_hard_budget()
        self._set_selection_info()
        self._exec()
        self._save()

    def _load(
        self,
    ) -> None:
        # Load method
        device = torch.device(self._cfg.device)
        # Paths are passed as given, so the provenance record keeps them
        # as passed (ADR 0002)
        self._method_bundle = load_bundle(
            self._cfg.method_bundle_path, expect=AFAMethod, device=device
        )
        self._method = self._method_bundle.obj
        log.info(f"Loaded AFA method from {self._cfg.method_bundle_path}")

        # Load unmasker
        self._unmasker = get_afa_unmasker_from_config(self._cfg.unmasker)
        log.info(f"Loaded {self._cfg.unmasker.class_name}")

        # Load initializer
        self._initializer = get_afa_initializer_from_config(
            self._cfg.initializer
        )
        log.info(f"Loaded {self._cfg.initializer.class_name} initializer")

        # Load dataset
        self._dataset_bundle = load_bundle(
            self._cfg.dataset_bundle_path, expect=AFADataset
        )
        self._dataset = self._dataset_bundle.obj
        log.info(f"Loaded dataset from {self._cfg.dataset_bundle_path}")

        # Load external classifier if specified
        if self._cfg.classifier_bundle_path is not None:
            self._classifier_bundle = load_bundle(
                self._cfg.classifier_bundle_path,
                expect=AFAClassifier,
                device=device,
            )
            self._external_classifier = self._classifier_bundle.obj
            log.info(
                f"Loaded external classifier from {self._cfg.classifier_bundle_path}."
            )
        else:
            self._classifier_bundle = None
            self._external_classifier = None
            log.info(
                "No external classifier provided; using builtin classifier."
            )

    def _init_wandb(self) -> None:
        if self._cfg.use_wandb:
            self._wandb_run = wandb.init(
                job_type="evaluation",
                config=asdict(self._cfg),
                dir="logs/wandb",
            )
            log.info(
                f"W&B run initialized: {self._wandb_run.name} ({self._wandb_run.id})"
            )
            log.info(f"W&B run URL: {self._wandb_run.url}")
        else:
            self._wandb_run = None

    def _smoke_test_override(self) -> None:
        if self._cfg.smoke_test:
            default_n_samples = (
                self._cfg.eval_only_n_samples
                if self._cfg.eval_only_n_samples is not None
                else self._cfg.batch_size * 2
            )
            self._cfg.eval_only_n_samples, self._cfg.batch_size = (
                eval_settings(
                    smoke_test=True,
                    default_n_samples=default_n_samples,
                    default_batch_size=self._cfg.batch_size,
                )
            )

    def _set_seeds(self) -> None:
        assert self._seed is not None
        assert self._method is not None
        self._method.set_seed(self._seed)
        assert self._unmasker is not None
        self._unmasker.set_seed(self._seed)
        assert self._initializer is not None
        self._initializer.set_seed(self._seed)

    def _set_soft_budget(self) -> None:
        assert self._method is not None

        # Some methods require a soft budget parameter set during evaluation instead of training
        if self._cfg.soft_budget_param is not None:
            self._method.set_cost_param(cost_param=self._cfg.soft_budget_param)

    def _set_hard_budget(self) -> None:
        if self._cfg.hard_budget is not None:
            # Optional optimisation: preserve the method's preferred selection
            # rather than falling back to the first available selection.
            # Evaluation enforces acquisition even if the method still stops.
            if isinstance(self._method, SupportsForcedAcquisition):
                self._method.force_acquisition = True
            log.info(
                "Enabled evaluator-enforced acquisition for hard-budget evaluation."
            )

    def _set_selection_info(self) -> None:
        assert self._unmasker is not None
        assert self._dataset is not None

        self._selection_costs = self._unmasker.get_selection_costs(
            feature_costs=self._dataset.get_feature_acquisition_costs()
        )
        log.info(
            "Selection costs summary: n=%d, min=%.4f, max=%.4f, mean=%.4f.",
            self._selection_costs.numel(),
            self._selection_costs.min().item(),
            self._selection_costs.max().item(),
            self._selection_costs.mean().item(),
        )
        self._n_selection_choices = self._unmasker.get_n_selections(
            feature_shape=self._dataset.feature_shape
        )

    def _exec(self) -> None:
        assert self._method is not None
        assert self._unmasker is not None
        assert self._n_selection_choices is not None
        assert self._initializer is not None
        assert self._dataset is not None
        assert self._selection_costs is not None

        hard_budget_str = (
            f"hard budget {self._cfg.hard_budget}"
            if self._cfg.hard_budget is not None
            else "no hard budget"
        )
        log.info(
            "Starting evaluation with batch size %s and hard budget %s.",
            self._cfg.batch_size,
            hard_budget_str,
        )

        df_eval = eval_afa_method(
            afa_action_fn=self._method.act,
            afa_unmask_fn=self._unmasker.unmask,
            n_selection_choices=self._n_selection_choices,
            afa_initialize_fn=self._initializer.initialize,
            dataset=self._dataset,
            external_afa_predict_fn=as_class_probability_fn(
                self._external_classifier,
                output_kind=self._external_classifier.output_kind,
            )
            if self._external_classifier is not None
            else None,
            builtin_afa_predict_fn=as_class_probability_fn(
                self._method.predict, output_kind=self._method.output_kind
            )
            if self._method.has_builtin_classifier
            else None,
            only_n_samples=self._cfg.eval_only_n_samples,
            device=torch.device(self._cfg.device),
            selection_budget=self._cfg.hard_budget,
            batch_size=self._cfg.batch_size,
            selection_costs=self._selection_costs.tolist(),
            seed=self._seed,
            force_acquisition=self._cfg.hard_budget is not None,
        )

        assert self._method_bundle is not None
        method_record = self._method_bundle.manifest.provenance
        self._provenance = self._capture_provenance(method_record)
        identity = _identity_columns(
            self._provenance,
            method_record,
            initializer_name=self._initializer_name,
            cfg=self._cfg,
        )
        self._df_eval = SavedEvaluationSchema.validate(
            df_eval.with_columns(
                identity_column(name, value, len(df_eval))
                for name, value in identity.items()
            )
        )

    def _capture_provenance(
        self, method_record: ProvenanceRecord | None
    ) -> ProvenanceRecord:
        """Capture the record after smoke-test overrides, with the seed used."""
        assert self._seed is not None
        assert self._method_bundle is not None
        assert self._dataset_bundle is not None
        inputs = [
            self._method_bundle.as_input("method"),
            self._dataset_bundle.as_input("eval_dataset"),
        ]
        if self._classifier_bundle is not None:
            inputs.append(self._classifier_bundle.as_input("classifier"))
        dataset_record = self._dataset_bundle.manifest.provenance
        return capture_provenance(
            stage="evaluation",
            resolved_config=asdict(self._cfg),
            seed=self._seed,
            smoke_test=self._cfg.smoke_test,
            device=self._cfg.device,
            inputs=inputs,
            dataset_identity=shared_dataset_identity(dataset_record),
            method_name=(
                None if method_record is None else method_record.method_name
            ),
            split=None if dataset_record is None else dataset_record.split,
        )

    def _save(self) -> None:
        assert self._df_eval is not None
        save_path = Path(self._cfg.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        assert self._provenance is not None
        save_evaluation_table(
            SavedEvaluationSchema.validate(self._df_eval),
            save_path,
            provenance=self._provenance,
        )
        log.info(f"Saved evaluation data to Parquet at: {save_path}")

        log.info(f"Evaluation results saved to: {self._cfg.save_path}")

        if self._wandb_run:
            self._wandb_run.finish()


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/eval",
    config_name="config",
)
def main(cfg: EvalConfig) -> None:
    cfg = cast("EvalConfig", OmegaConf.to_object(cfg))
    log.debug(cfg)
    torch.set_float32_matmul_precision("medium")

    evaluator = AFAEvaluator(
        cfg,
        initializer_name=HydraConfig.get().runtime.choices["initializer"],
    )
    evaluator.run()


def _identity_columns(
    record: ProvenanceRecord,
    method_record: ProvenanceRecord | None,
    *,
    initializer_name: str,
    cfg: EvalConfig,
) -> dict[str, object]:
    """
    Return the constant-per-table identity columns of ADR 0002.

    Dataset identity and the method name come from the evaluation's own
    record; training identity from the method bundle's record, null for a
    method bundle written without one.
    """
    return {
        "afa_method": record.method_name,
        "dataset": record.dataset_key,
        "dataset_realization_index": record.dataset_realization_index,
        "eval_split": record.split,
        "initializer": initializer_name,
        "train_seed": None if method_record is None else method_record.seed,
        "train_hard_budget": _training_setting(method_record, "hard_budget"),
        "train_soft_budget_param": _training_setting(
            method_record, "soft_budget_param"
        ),
        "eval_seed": record.seed,
        "eval_hard_budget": cfg.hard_budget,
        "eval_soft_budget_param": cfg.soft_budget_param,
    }


def _training_setting(
    method_record: ProvenanceRecord | None, name: str
) -> object | None:
    """
    Read a training contract setting from the method bundle's record.

    A record built without the setting leaves it unknown, so null.
    """
    if method_record is None:
        return None
    if method_record.stage != "training":
        msg = (
            f"Method bundle has a {method_record.stage!r} provenance record; "
            "expected a 'training' record."
        )
        raise MethodRecordStageError(msg)
    return method_record.resolved_config.get(name)


if __name__ == "__main__":
    main()
