"""
Write output trees whose artifacts carry provenance records, for release tests.

Bundles and evaluation tables are laid out where the workflow writes them,
with the records and identity columns the pipeline stages would write, but
their payloads are a few bytes. A bundle's data holds its path, so no two
bundles share a content hash.
"""

import json
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from afabench.core.bundle_system.bundle import compute_content_hash
from afabench.core.job_record import ExitStatus, JobIdentity
from afabench.core.job_record import Stage as JobStage
from afabench.core.provenance import (
    PROVENANCE_VERSION,
    Compute,
    Environment,
    InputRole,
    ProvenanceInput,
    ProvenanceRecord,
    Split,
    Stage,
)
from afabench.evaluation.provenance import save_evaluation_table
from afabench.evaluation.schemas import IDENTITY_DTYPES
from afabench.release.manifest import CodeIdentity
from afabench.release.workflow_config import (
    record_workflow_config,
    resolve_workflow_config,
)
from test import job_record_examples

TAG = "initializer-cold"
INITIALIZER = "cold"
EVAL_SPLIT = "test"
CLEAN = CodeIdentity(commit="c" * 40, dirty=False)


def provenance(
    stage: Stage,
    *,
    code: CodeIdentity = CLEAN,
    smoke_test: bool = False,
    seed: int = 0,
    method_name: str | None = None,
    dataset_key: str | None = None,
    dataset_realization_index: int | None = None,
    split: Split | None = None,
    inputs: Iterable[ProvenanceInput] = (),
) -> ProvenanceRecord:
    return ProvenanceRecord(
        provenance_version=PROVENANCE_VERSION,
        stage=stage,
        created_at="2026-10-01T00:00:00+00:00",
        code_commit=code.commit,
        code_dirty=code.dirty,
        resolved_config={"stage": stage},
        seed=seed,
        smoke_test=smoke_test,
        method_name=method_name,
        dataset_key=dataset_key,
        dataset_realization_index=dataset_realization_index,
        split=split,
        inputs=list(inputs),
        environment=Environment(
            python_version="3.12.10",
            afabench_version=None,
            torch_version="2",
            numpy_version="2",
            polars_version="2",
            lockfile_sha256=None,
            platform="test",
        ),
        compute=Compute(
            device="cpu",
            accelerator_name=None,
            cuda_version=None,
            cudnn_version=None,
            float32_matmul_precision="medium",
            cudnn_deterministic=True,
            cudnn_benchmark=False,
            deterministic_algorithms=False,
        ),
    )


def write_bundle(
    root: Path,
    path: str,
    record: ProvenanceRecord | None,
    *,
    class_name: str = "Fake",
    content: str = "",
) -> None:
    """
    Lay out a bundle as `save_bundle` does, without a record if `None`.

    `data/weights.bin` holds `content`; `data/path.txt` holds the path.
    """
    bundle = root / path
    (bundle / "data").mkdir(parents=True)
    (bundle / "data/weights.bin").write_text(content)
    (bundle / "data/path.txt").write_text(path)
    manifest: dict[str, object] = {
        "bundle_version": "1.1.0" if record else "1.0.0",
        "class_name": class_name,
        "class_version": None,
        "metadata": {},
    }
    if record is not None:
        manifest["provenance"] = record.to_json_dict()
        manifest["content_hash"] = compute_content_hash(bundle / "data")
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))


def write_job_record(
    root: Path,
    path: str,
    *,
    stage: JobStage = "training",
    exit_status: ExitStatus = "completed",
    job_duration_seconds: float = 60.0,
    smoke_test: bool = False,
) -> None:
    """Write a job record at `path` under `root` as the job wrapper would."""
    record = job_record_examples.job_record(
        JobIdentity(
            stage=stage,
            name="alpha",
            dataset_key="cube",
            dataset_realization_index=0,
            train_seed=0,
            train_hard_budget=3,
        ),
        job_duration_seconds=job_duration_seconds,
        exit_status=exit_status,
        exit_code=0 if exit_status == "completed" else 1,
        cpus=4,
        time_limit_minutes=120,
        cpu_model="Test CPU",
        host="node1",
        slurm_job_id="42",
        code_commit=CLEAN.commit,
        smoke_test=smoke_test,
    )
    job_record_examples.write_job_record(root / path, record)


def bundle_input(root: Path, role: InputRole, path: str) -> ProvenanceInput:
    """Return the input entry a job reading the bundle at `path` records."""
    manifest = json.loads((root / path / "manifest.json").read_text())
    return ProvenanceInput(
        role=role,
        path=f"output/{path}",
        class_name=manifest["class_name"],
        content_hash=manifest.get("content_hash"),
    )


@dataclass(frozen=True, kw_only=True)
class Evaluation:
    """The identity columns of one evaluation."""

    method: str
    dataset: str
    realization: int
    train_hard_budget: float | None
    train_soft_budget_param: float | None
    eval_hard_budget: float | None
    eval_soft_budget_param: float | None
    initializer: str = INITIALIZER
    eval_split: str = EVAL_SPLIT

    def identity(self) -> dict[str, object]:
        return {
            "afa_method": self.method,
            "dataset": self.dataset,
            "dataset_realization_index": self.realization,
            "eval_split": self.eval_split,
            "initializer": self.initializer,
            "train_seed": self.realization,
            "train_hard_budget": self.train_hard_budget,
            "train_soft_budget_param": self.train_soft_budget_param,
            "eval_seed": self.realization,
            "eval_hard_budget": self.eval_hard_budget,
            "eval_soft_budget_param": self.eval_soft_budget_param,
        }


def raw_table(evaluation: Evaluation) -> pl.DataFrame:
    """One forced-stop episode with external predictions only."""
    frame = pl.DataFrame(
        {
            "episode_id": [0, 0],
            "generation_index": [5, 5],
            "split_index": [0, 0],
            "step": [0, 1],
            "action_performed": [3, 0],
            "builtin_predicted_probabilities": pl.Series(
                [None, None], dtype=pl.List(pl.Float64)
            ),
            "external_predicted_probabilities": [[0.25, 0.75], [0.75, 0.25]],
            "true_class": [0, 0],
            "accumulated_cost": [1.0, 1.0],
            "forced_stop": [False, True],
        }
    )
    return _with_identity(frame, evaluation)


def transformed_table(evaluation: Evaluation, content: str) -> pl.DataFrame:
    """One external prediction; column `release` holds `content`."""
    frame = pl.DataFrame(
        {
            "classifier": ["external"],
            "predicted_probabilities": [[0.25, 0.75]],
            "true_class": pl.Series([0], dtype=pl.UInt64),
            "n_selections_performed": pl.Series([1], dtype=pl.UInt64),
            "release": [content],
        }
    )
    return _with_identity(frame, evaluation)


def _with_identity(
    frame: pl.DataFrame, evaluation: Evaluation
) -> pl.DataFrame:
    return frame.with_columns(
        pl.Series(name, [value] * len(frame), dtype=IDENTITY_DTYPES[name])
        for name, value in evaluation.identity().items()
    )


def write_evaluation(
    root: Path,
    path: str,
    evaluation: Evaluation,
    record: ProvenanceRecord | None,
    *,
    content: str = "",
    raw: bool = True,
    transformed: bool = True,
) -> None:
    """Write the raw and transformed tables of `path` under their folders."""
    for folder, wanted, frame in [
        ("eval_results", raw, raw_table(evaluation)),
        (
            "eval_results_transformed",
            transformed,
            transformed_table(evaluation, content),
        ),
    ]:
        if not wanted:
            continue
        table = root / folder / path
        table.parent.mkdir(parents=True, exist_ok=True)
        save_evaluation_table(frame, table, provenance=record)


def dataset_bundle(dataset: str, realization: int, split: str) -> str:
    return f"datasets/{dataset}/{realization}/{split}.bundle"


def classifier_bundle(
    dataset: str, realization: int, method: str | None = None
) -> str:
    owner = "" if method is None else f"method-{method}+"
    return (
        f"trained_classifiers/{TAG}/{owner}dataset-{dataset}+"
        f"realization_index-{realization}.bundle"
    )


def pretrained_model_bundle(name: str, dataset: str, realization: int) -> str:
    return (
        f"pretrained_models/{TAG}/{name}/"
        f"dataset-{dataset}+realization_index-{realization}/"
        f"pretrain_seed-{realization}/model.bundle"
    )


@dataclass(frozen=True, kw_only=True)
class Budgets:
    """Budget values as the workflow writes them into paths."""

    train_hard: str
    eval_hard: str
    train_soft: str = "null"
    eval_soft: str = "null"


HARD_3 = Budgets(train_hard="3", eval_hard="3")
SOFT_HALF = Budgets(
    train_hard="null", eval_hard="null", train_soft="0.5", eval_soft="0.5"
)


@dataclass(frozen=True, kw_only=True)
class Method:
    name: str
    budgets: list[Budgets]
    pretrained_model: str | None = None
    own_classifier: bool = False


def method_folder(method: Method, dataset: str, realization: int) -> str:
    pretrain = (
        "NO_PRETRAIN"
        if method.pretrained_model is None
        else f"pretrain_seed-{realization}"
    )
    return (
        f"{method.name}/dataset-{dataset}+realization_index-{realization}/"
        f"{pretrain}/"
    )


def method_bundle(
    method: Method, dataset: str, realization: int, budgets: Budgets
) -> str:
    return (
        f"trained_methods/{TAG}/{method_folder(method, dataset, realization)}"
        f"train_seed-{realization}+train_hard_budget-{budgets.train_hard}+"
        f"train_soft_budget_param-{budgets.train_soft}/method.bundle"
    )


def evaluation_table(
    method: Method, dataset: str, realization: int, budgets: Budgets
) -> str:
    """Return the path of an evaluation's tables below their folders."""
    return (
        f"eval_split-{EVAL_SPLIT}/{TAG}/"
        f"{method_folder(method, dataset, realization)}"
        f"train_seed-{realization}+train_hard_budget-{budgets.train_hard}+"
        f"train_soft_budget_param-{budgets.train_soft}/"
        f"eval_seed-{realization}+eval_hard_budget-{budgets.eval_hard}+"
        f"eval_soft_budget_param-{budgets.eval_soft}/eval_data.parquet"
    )


ALPHA = Method(name="alpha", budgets=[HARD_3, SOFT_HALF])
BETA = Method(
    name="beta",
    budgets=[HARD_3],
    pretrained_model="shared",
    own_classifier=True,
)


@dataclass(frozen=True, kw_only=True)
class Catalog:
    """What `write_catalog` writes, before anything is omitted."""

    methods: list[Method] = field(default_factory=lambda: [ALPHA, BETA])
    datasets: list[str] = field(default_factory=lambda: ["cube"])
    realizations: list[int] = field(default_factory=lambda: [0])
    smoke_test: bool = False
    # Code per stage; an absent stage is clean code
    code: Mapping[Stage, CodeIdentity] = field(default_factory=dict)


def record_run_config(
    root: Path,
    configfile: Path,
    *,
    profile: Path | None = None,
    overrides: tuple[str, ...] = (),
) -> None:
    """Stand in for the configuration recorded by an executing workflow."""
    record = resolve_workflow_config(
        profile=profile, configfiles=[configfile], overrides=overrides
    )
    record_workflow_config(
        root,
        profile=record.profile,
        configfiles=[Path(item.path) for item in record.configfiles],
        overrides=record.overrides,
        merged=record.merged,
    )


def write_catalog(
    root: Path,
    catalog: Catalog,
    *,
    content: str = "",
    omit: Iterable[str] = (),
) -> None:
    """
    Write every artifact of `catalog` as the pipeline would, then drop `omit`.

    Each bundle's `weights.bin` and each transformed table's `release`
    column hold `content`. `omit` names bundles and tables (under their
    folder) removed after everything is written, so the artifacts that read
    them still record their content hash.
    """
    writer = _CatalogWriter(root=root, catalog=catalog, content=content)
    for dataset in catalog.datasets:
        writer.write_dataset(dataset)
    for path in omit:
        if (root / path).is_dir():
            shutil.rmtree(root / path)
        else:
            (root / path).unlink()


@dataclass(frozen=True, kw_only=True)
class _CatalogWriter:
    root: Path
    catalog: Catalog
    content: str

    def record(self, stage: Stage, **fields: object) -> ProvenanceRecord:
        return provenance(
            stage,
            code=self.catalog.code.get(stage, CLEAN),
            smoke_test=self.catalog.smoke_test,
            **fields,  # pyright: ignore[reportArgumentType]
        )

    def bundle(self, path: str, stage: Stage, **fields: object) -> None:
        write_bundle(
            self.root,
            path,
            self.record(stage, **fields),
            content=self.content,
        )

    def input(self, role: InputRole, path: str) -> ProvenanceInput:
        return bundle_input(self.root, role, path)

    def training_inputs(
        self, dataset: str, index: int
    ) -> list[ProvenanceInput]:
        return [
            self.input(
                "train_dataset", dataset_bundle(dataset, index, "train")
            ),
            self.input("val_dataset", dataset_bundle(dataset, index, "val")),
        ]

    def write_dataset(self, dataset: str) -> None:
        catalog = self.catalog
        for index in catalog.realizations:
            for split in ["train", "val", "test"]:
                self.bundle(
                    dataset_bundle(dataset, index, split),
                    "dataset_generation",
                    seed=index,
                    dataset_key=dataset,
                    dataset_realization_index=index,
                    split=split,
                )
        # Each dataset realization has its own classifiers.
        owners = [None] + [m.name for m in catalog.methods if m.own_classifier]
        pretrained_models = sorted(
            {m.pretrained_model for m in catalog.methods} - {None}
        )
        for index in catalog.realizations:
            for owner in owners:
                self.bundle(
                    classifier_bundle(dataset, index, owner),
                    "classifier_training",
                    seed=index,
                    method_name=owner,
                    dataset_key=dataset,
                    dataset_realization_index=index,
                    inputs=self.training_inputs(dataset, index),
                )
            for name in pretrained_models:
                assert name is not None
                self.bundle(
                    pretrained_model_bundle(name, dataset, index),
                    "pretraining",
                    seed=index,
                    dataset_key=dataset,
                    dataset_realization_index=index,
                    inputs=[
                        *self.training_inputs(dataset, index),
                        self.input(
                            "classifier", classifier_bundle(dataset, index)
                        ),
                    ],
                )
            for method in catalog.methods:
                for budgets in method.budgets:
                    self.write_method(method, dataset, index, budgets)

    def write_method(
        self, method: Method, dataset: str, index: int, budgets: Budgets
    ) -> None:
        """Write a method's training run and its evaluation."""
        classifier = classifier_bundle(
            dataset, index, method.name if method.own_classifier else None
        )
        inputs = [
            *self.training_inputs(dataset, index),
            self.input("classifier", classifier),
        ]
        if method.pretrained_model is not None:
            inputs.append(
                self.input(
                    "pretrained_model",
                    pretrained_model_bundle(
                        method.pretrained_model, dataset, index
                    ),
                )
            )
        trained = method_bundle(method, dataset, index, budgets)
        self.bundle(
            trained,
            "training",
            seed=index,
            method_name=method.name,
            dataset_key=dataset,
            dataset_realization_index=index,
            inputs=inputs,
        )
        write_evaluation(
            self.root,
            evaluation_table(method, dataset, index, budgets),
            Evaluation(
                method=method.name,
                dataset=dataset,
                realization=index,
                train_hard_budget=_budget(budgets.train_hard),
                train_soft_budget_param=_budget(budgets.train_soft),
                eval_hard_budget=_budget(budgets.eval_hard),
                eval_soft_budget_param=_budget(budgets.eval_soft),
            ),
            self.record(
                "evaluation",
                seed=index,
                method_name=method.name,
                dataset_key=dataset,
                dataset_realization_index=index,
                split=EVAL_SPLIT,
                inputs=[
                    self.input(
                        "eval_dataset",
                        dataset_bundle(dataset, index, EVAL_SPLIT),
                    ),
                    self.input("method", trained),
                    self.input("classifier", classifier),
                ],
            ),
            content=self.content,
        )


def _budget(value: str) -> float | None:
    return None if value == "null" else float(value)
