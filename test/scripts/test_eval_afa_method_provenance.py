"""Evaluation tables identify themselves at write time (ADR 0002)."""

import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
import pytest
import torch
from polars.testing import assert_frame_equal

from afabench.components.classifiers import (
    UniformDummyAFAClassifier,
    WrappedMaskedMLPClassifier,
)
from afabench.components.classifiers.models import MaskedMLPClassifier
from afabench.components.initializers.config import InitializerConfig
from afabench.components.methods.dummy.config import RandomDummyTrainConfig
from afabench.components.methods.dummy.with_classifier import (
    SequentialWithClassifierAFAMethod,
)
from afabench.components.methods.dummy.without_classifier import (
    RandomWithoutClassifierAFAMethod,
)
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import (
    bundle_provenance,
    read_manifest,
    save_bundle,
)
from afabench.core.provenance import (
    DatasetIdentity,
    ProvenanceInput,
    Split,
    capture_provenance,
)
from afabench.datasets import CubeDataset
from afabench.evaluation.config import EvalConfig
from afabench.evaluation.provenance import (
    evaluation_table_provenance,
    save_evaluation_table,
)
from afabench.evaluation.schemas import SavedEvaluationSchema
from afabench.fit.run import save_result
from afabench.testing.provenance import placeholder_provenance
from scripts.eval.eval_afa_method import AFAEvaluator, MethodRecordStageError

REPO_ROOT = Path(__file__).parents[2]
TRAIN_SEED = 11
IDENTITY_DTYPES = {
    "afa_method": "String",
    "dataset": "String",
    "dataset_realization_index": "UInt64",
    "eval_split": "String",
    "initializer": "String",
    "train_seed": "UInt64",
    "train_hard_budget": "Float64",
    "train_soft_budget_param": "Float64",
    "eval_seed": "UInt64",
    "eval_hard_budget": "Float64",
    "eval_soft_budget_param": "Float64",
}


@dataclass(frozen=True)
class EvalInputs:
    method: Path
    dataset: Path
    classifier: Path


@pytest.fixture
def eval_inputs(tmp_path: Path) -> EvalInputs:
    """Bundles of dataset realization 3, and a method trained on it."""
    splits: list[tuple[Split, int]] = [("train", 0), ("val", 1), ("test", 2)]
    for split, seed in splits:
        save_bundle(
            CubeDataset(n_samples=12, seed=seed),
            tmp_path / f"{split}.bundle",
            metadata={},
            provenance=placeholder_provenance(
                dataset_key="cube", dataset_realization_index=3, split=split
            ),
        )
    save_bundle(
        WrappedMaskedMLPClassifier(
            MaskedMLPClassifier(n_features=20, n_classes=8, num_cells=(4,)),
            device=torch.device("cpu"),
        ),
        tmp_path / "classifier.bundle",
        metadata={},
        provenance=placeholder_provenance(
            "classifier_training", dataset_realization_index=0
        ),
    )
    contract = RandomDummyTrainConfig(
        train_dataset_bundle_path=str(tmp_path / "train.bundle"),
        val_dataset_bundle_path=str(tmp_path / "val.bundle"),
        classifier_bundle_path=str(tmp_path / "classifier.bundle"),
        save_path=str(tmp_path / "method.bundle"),
        method_name="random_dummy",
        initializer=InitializerConfig(
            class_name="RandomInitializer",
            kwargs={"num_initial_features": 0},
        ),
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        dataset_key="cube",
        device="cpu",
        seed=TRAIN_SEED,
        hard_budget=2,
        soft_budget_param=None,
    )
    save_result(RandomWithoutClassifierAFAMethod(n_classes=8), contract)
    return EvalInputs(
        method=tmp_path / "method.bundle",
        dataset=tmp_path / "test.bundle",
        classifier=tmp_path / "classifier.bundle",
    )


def eval_config(
    eval_inputs: EvalInputs, save_path: Path, *, seed: int | None
) -> EvalConfig:
    return EvalConfig(
        method_bundle_path=str(eval_inputs.method),
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        # A random initializer, so a dropped seed would show
        initializer=InitializerConfig(
            class_name="RandomInitializer",
            kwargs={"num_initial_features": 1},
        ),
        dataset_bundle_path=str(eval_inputs.dataset),
        save_path=str(save_path),
        classifier_bundle_path=str(eval_inputs.classifier),
        seed=seed,
        device="cpu",
        eval_only_n_samples=None,
        batch_size=8,
        hard_budget=2,
        soft_budget_param=None,
        smoke_test=True,
    )


def test_evaluation_table_round_trips_its_provenance_record(
    tmp_path: Path,
) -> None:
    record = capture_provenance(
        stage="evaluation",
        resolved_config={"nested": {"values": [1, 2.5, None]}, "seed": None},
        seed=9,
        smoke_test=False,
        device="cpu",
        inputs=[
            ProvenanceInput(
                role="method",
                path="method.bundle",
                class_name="RandomWithoutClassifierAFAMethod",
                content_hash="sha256:abc",
            ),
            ProvenanceInput(
                role="classifier",
                path="classifier.bundle",
                class_name="WrappedMaskedMLPClassifier",
                content_hash=None,
            ),
        ],
        dataset_identity=DatasetIdentity(
            dataset_key="cube", dataset_realization_index=None
        ),
        method_name="random_dummy",
        split="test",
    )
    path = tmp_path / "eval_data.parquet"

    save_evaluation_table(pl.DataFrame({"a": [1]}), path, provenance=record)

    assert evaluation_table_provenance(path) == record


def test_evaluation_table_without_a_record_reports_it_absent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval_data.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(path)

    assert evaluation_table_provenance(path) is None


def test_evaluator_records_its_inputs_and_identity(
    eval_inputs: EvalInputs, tmp_path: Path
) -> None:
    save_path = tmp_path / "eval_data.parquet"
    cfg = eval_config(eval_inputs, save_path, seed=5)

    AFAEvaluator(cfg, initializer_name="warm").run()

    record = evaluation_table_provenance(save_path)
    assert record is not None
    assert record.stage == "evaluation"
    assert record.seed == 5
    assert record.smoke_test is True
    assert record.method_name == "random_dummy"
    assert (
        record.dataset_key,
        record.dataset_realization_index,
        record.split,
    ) == ("cube", 3, "test")
    assert record.inputs == [
        ProvenanceInput(
            role="method",
            path=str(eval_inputs.method),
            class_name="RandomWithoutClassifierAFAMethod",
            content_hash=read_manifest(eval_inputs.method)["content_hash"],
        ),
        ProvenanceInput(
            role="eval_dataset",
            path=str(eval_inputs.dataset),
            class_name="CubeDataset",
            content_hash=read_manifest(eval_inputs.dataset)["content_hash"],
        ),
        ProvenanceInput(
            role="classifier",
            path=str(eval_inputs.classifier),
            class_name="WrappedMaskedMLPClassifier",
            content_hash=read_manifest(eval_inputs.classifier)["content_hash"],
        ),
    ]
    # The config as the evaluator used it, after its smoke-test overrides
    assert record.resolved_config == asdict(cfg)
    assert record.resolved_config["batch_size"] == 2
    assert record.resolved_config["eval_only_n_samples"] == 4


def test_evaluator_reads_each_input_manifest_once(
    eval_inputs: EvalInputs,
    tmp_path: Path,
    manifest_reads: Counter[Path],
) -> None:
    AFAEvaluator(
        eval_config(eval_inputs, tmp_path / "eval_data.parquet", seed=5),
        initializer_name="warm",
    ).run()

    assert manifest_reads == {
        eval_inputs.method: 1,
        eval_inputs.dataset: 1,
        eval_inputs.classifier: 1,
    }


def test_evaluation_table_columns_identify_it_without_afabench(
    eval_inputs: EvalInputs, tmp_path: Path
) -> None:
    save_path = tmp_path / "eval_data.parquet"
    AFAEvaluator(
        eval_config(eval_inputs, save_path, seed=5), initializer_name="warm"
    ).run()

    # A results-only user reads the table with plain Polars
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys\n"
            "import polars as pl\n"
            f"table = pl.read_parquet({str(save_path)!r}).select({list(IDENTITY_DTYPES)!r})\n"
            "assert not any(m.startswith('afabench') for m in sys.modules)\n"
            "print(json.dumps({\n"
            "    'dtypes': {k: str(v) for k, v in table.schema.items()},\n"
            "    'rows': table.unique().to_dicts(),\n"
            "}))\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    table = json.loads(completed.stdout)
    assert table["dtypes"] == {k: str(v) for k, v in IDENTITY_DTYPES.items()}
    assert table["rows"] == [
        {
            "afa_method": "random_dummy",
            "dataset": "cube",
            "dataset_realization_index": 3,
            "eval_split": "test",
            "initializer": "warm",
            "train_seed": TRAIN_SEED,
            "train_hard_budget": 2.0,
            "train_soft_budget_param": None,
            "eval_seed": 5,
            "eval_hard_budget": 2.0,
            "eval_soft_budget_param": None,
        }
    ]
    SavedEvaluationSchema.validate(pl.read_parquet(save_path))


def test_evaluation_table_drops_internal_bookkeeping_columns(
    eval_inputs: EvalInputs, tmp_path: Path
) -> None:
    save_path = tmp_path / "eval_data.parquet"
    AFAEvaluator(
        eval_config(eval_inputs, save_path, seed=5), initializer_name="warm"
    ).run()

    saved = pl.read_parquet(save_path)
    assert "prev_selections_performed" not in saved
    assert "idx" not in saved


def test_training_settings_missing_from_the_method_record_are_null(
    eval_inputs: EvalInputs, tmp_path: Path
) -> None:
    # A record built by hand, as `docs/how-to/repository_adopter/add_method.md` allows, need
    # not carry the training contract's budgets
    save_bundle(
        RandomWithoutClassifierAFAMethod(n_classes=8),
        eval_inputs.method,
        metadata={},
        provenance=placeholder_provenance(
            "training", seed=TRAIN_SEED, method_name="random_dummy"
        ),
    )
    save_path = tmp_path / "eval_data.parquet"

    AFAEvaluator(
        eval_config(eval_inputs, save_path, seed=5), initializer_name="warm"
    ).run()

    table = pl.read_parquet(save_path)
    assert table["train_seed"].unique().to_list() == [TRAIN_SEED]
    assert table["train_hard_budget"].is_null().all()
    assert table["train_soft_budget_param"].is_null().all()


def test_method_bundle_with_a_record_of_another_stage_is_rejected(
    eval_inputs: EvalInputs, tmp_path: Path
) -> None:
    save_bundle(
        RandomWithoutClassifierAFAMethod(n_classes=8),
        eval_inputs.method,
        metadata={},
        provenance=placeholder_provenance("pretraining"),
    )
    cfg = eval_config(eval_inputs, tmp_path / "eval_data.parquet", seed=5)

    with pytest.raises(MethodRecordStageError, match="'pretraining'"):
        AFAEvaluator(cfg, initializer_name="warm").run()


def test_null_seed_is_resolved_once_and_reproduces_the_table(
    eval_inputs: EvalInputs, tmp_path: Path
) -> None:
    first_path = tmp_path / "first.parquet"
    AFAEvaluator(
        eval_config(eval_inputs, first_path, seed=None),
        initializer_name="warm",
    ).run()
    first = pl.read_parquet(first_path)
    record = evaluation_table_provenance(first_path)
    assert record is not None
    assert set(first["eval_seed"]) == {record.seed}

    rerun_path = tmp_path / "rerun.parquet"
    AFAEvaluator(
        eval_config(eval_inputs, rerun_path, seed=record.seed),
        initializer_name="warm",
    ).run()

    assert_frame_equal(pl.read_parquet(rerun_path), first)


# A Hydra subprocess takes about 7 s, too slow for the default suite
@pytest.mark.pipeline
def test_hand_run_evaluation_identifies_its_table(
    eval_inputs: EvalInputs, tmp_path: Path
) -> None:
    save_path = tmp_path / "eval_data.parquet"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eval/eval_afa_method.py",
            f"method_bundle_path={eval_inputs.method}",
            "initializer=warm",
            "unmasker=direct",
            f"dataset_bundle_path={eval_inputs.dataset}",
            f"save_path={save_path}",
            f"classifier_bundle_path={eval_inputs.classifier}",
            "device=cpu",
            "hard_budget=2",
            "batch_size=4",
            "smoke_test=True",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]

    table = pl.read_parquet(save_path)
    method = bundle_provenance(eval_inputs.method)
    dataset = bundle_provenance(eval_inputs.dataset)
    record = evaluation_table_provenance(save_path)
    assert method is not None
    assert dataset is not None
    assert record is not None
    identity = table.select(list(IDENTITY_DTYPES)).unique()
    assert identity.to_dicts() == [
        {
            "afa_method": method.method_name,
            "dataset": dataset.dataset_key,
            "dataset_realization_index": dataset.dataset_realization_index,
            "eval_split": dataset.split,
            "initializer": "warm",
            "train_seed": method.seed,
            "train_hard_budget": method.resolved_config["hard_budget"],
            "train_soft_budget_param": None,
            "eval_seed": record.seed,
            "eval_hard_budget": 2.0,
            "eval_soft_budget_param": None,
        }
    ]


@pytest.mark.parametrize("with_external", [False, True])
def test_evaluator_adapts_declared_logits_and_round_trips_probabilities(
    eval_inputs: EvalInputs, tmp_path: Path, *, with_external: bool
) -> None:

    classifier = UniformDummyAFAClassifier(n_classes=8)
    save_bundle(
        SequentialWithClassifierAFAMethod(classifier),
        eval_inputs.method,
        metadata={},
        provenance=placeholder_provenance("training"),
    )
    path = tmp_path / "probabilities.parquet"
    cfg = eval_config(eval_inputs, path, seed=5)
    if not with_external:
        cfg.classifier_bundle_path = None

    AFAEvaluator(cfg, initializer_name="warm").run()

    table = pl.read_parquet(path)
    SavedEvaluationSchema.validate(table)
    assert all(
        vector == [0.125] * 8
        for vector in table["builtin_predicted_probabilities"].to_list()
    )
    if with_external:
        assert all(
            len(vector) == 8 and sum(vector) == pytest.approx(1.0)
            for vector in table["external_predicted_probabilities"]
        )
    else:
        assert table["external_predicted_probabilities"].is_null().all()
    assert evaluation_table_provenance(path) is not None
    assert (
        table.group_by("episode_id").agg("step")["step"].to_list()
        == [[0, 1, 2]] * 4
    )
