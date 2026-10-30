"""The L2M fit stages and method, around the vendored code."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from afabench.components.initializers.config import InitializerConfig
from afabench.components.methods.discriminative.l2m.afa_methods import (
    L2MAFAMethod,
)
from afabench.components.methods.discriminative.l2m.config import (
    FeatureSource,
    L2MArchitectureConfig,
    L2MPretrainingConfig,
    L2MTrainingConfig,
)
from afabench.components.methods.discriminative.l2m.models import (
    L2MModel,
)
from afabench.components.methods.discriminative.l2m.pretrain import (
    linear_schedule_with_warmup,
    pretrain_l2m,
)
from afabench.components.methods.discriminative.l2m.train import (
    train_l2m,
)
from afabench.components.methods.discriminative.l2m.upstream_fit import (
    draw_synthetic_pool,
)
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.datasets import CubeDataset
from afabench.fit.inputs import load_inputs
from afabench.testing.provenance import placeholder_provenance

#TODO: see if we need to adapt our model to this
N_FEATURES = 20
N_CLASSES = 8
# Every upstream task has 50 context instances, so a task needs more.
SEQUENCE_LENGTH = 56
CPU = torch.device("cpu")

#TODO: try to get already existing model loaded, instead of training new test partners
@pytest.fixture(autouse=True)
def _single_thread() -> Iterator[None]:
    # The fits are many tiny kernels, which stall when every test worker
    # runs its own pool of intra-op threads.
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n_threads)


def _save(obj: object, path: Path) -> str:
    save_bundle(obj, path, metadata={}, provenance=placeholder_provenance())
    return str(path)


def _contract_values(
    tmp_path: Path, train_dataset: CubeDataset | None = None
) -> dict[str, object]:
    return {
        "train_dataset_bundle_path": _save(
            train_dataset or CubeDataset(n_samples=96, seed=0),
            tmp_path / "train.bundle",
        ),
        "val_dataset_bundle_path": _save(
            CubeDataset(n_samples=32, seed=1), tmp_path / "val.bundle"
        ),
        "classifier_bundle_path": "unused.bundle",
        "save_path": str(tmp_path / "script_owned.bundle"),
        "initializer": InitializerConfig(
            "RandomInitializer", {"num_initial_features": 0}
        ),
        "unmasker": UnmaskerConfig("DirectUnmasker", {}),
        "dataset_key": "cube",
        "device": "cpu",
        "seed": 0,
    }


def _pretraining_config(
    tmp_path: Path,
    *,
    train_dataset: CubeDataset | None = None,
    mixed_precision: bool = False,
    feature_source: FeatureSource = FeatureSource.real,
) -> L2MPretrainingConfig:
    return L2MPretrainingConfig(
        **_contract_values(tmp_path, train_dataset),  # pyright: ignore[reportArgumentType]
        feature_source=feature_source,
        architecture=L2MArchitectureConfig(
            model_dim=8,
            embedding_depth=1,
            prediction_depth=1,
            n_layers=1,
            n_heads=2,
            feedforward_dim=16,
            dropout=0.0,
        ),
        batch_size=2,
        sequence_length=SEQUENCE_LENGTH,
        n_steps=3,
        lr=1e-3,
        warmup_steps=1,
        checkpoint_interval=2,
        n_validation_tasks=2,
        pretraining_pool_size=64,
        mixed_precision=mixed_precision,
    )


def _small_model(n_features: int, n_classes: int) -> L2MModel:
    torch.manual_seed(3)
    return L2MModel.build(
        n_features,
        n_classes,
        model_dim=8,
        embedding_depth=1,
        prediction_depth=1,
        n_layers=1,
        n_heads=2,
        feedforward_dim=16,
        dropout=0.0,
    )


def _training_config(
    tmp_path: Path,
    *,
    hard_budget: int | None = 3,
    feature_source: FeatureSource = FeatureSource.real,
) -> L2MTrainingConfig:
    return L2MTrainingConfig(
        **_contract_values(tmp_path),  # pyright: ignore[reportArgumentType]
        feature_source=feature_source,
        method_name="l2m",
        pretrained_model_bundle_path=_save(
            _small_model(N_FEATURES, N_CLASSES), tmp_path / "pretrained.bundle"
        ),
        hard_budget=hard_budget,
        soft_budget_param=None,
        batch_size=2,
        sequence_length=SEQUENCE_LENGTH,
        n_steps=3,
        policy_lr=1e-3,
        backbone_lr=1e-4,
        temperature=0.1,
        checkpoint_interval=2,
        n_validation_tasks=2,
        pretraining_pool_size=64,
        mixed_precision=False,
        context_set_size=8,
    )


def _method(n_classes: int = 2) -> L2MAFAMethod:
    generator = torch.Generator().manual_seed(0)
    return L2MAFAMethod(
        _small_model(3, n_classes),
        torch.randn(6, 3, generator=generator),
        torch.eye(n_classes)[torch.arange(6) % n_classes],
        unmasker=UnmaskerConfig("DirectUnmasker", {}),
    )


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_pretraining_returns_a_model_that_survives_its_bundle(
    tmp_path: Path, mixed_precision: bool
) -> None:
    cfg = _pretraining_config(tmp_path, mixed_precision=mixed_precision)

    model = pretrain_l2m(cfg, inputs=load_inputs(cfg))

    assert not Path(cfg.save_path).exists()
    restored = load_bundle(
        Path(_save(model, tmp_path / "caller.bundle")), device=CPU
    ).obj
    assert isinstance(restored, L2MModel)
    for original, loaded in zip(
        model.parameters(), restored.parameters(), strict=True
    ):
        assert torch.equal(original, loaded)
    # The bf16 forward pass is unwrapped once the fit ends.
    assert "forward" not in vars(model.ltm)


def test_pretraining_is_reproducible_from_its_seed(tmp_path: Path) -> None:
    models = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        cfg = _pretraining_config(tmp_path / name)
        models.append(pretrain_l2m(cfg, inputs=load_inputs(cfg)))

    for first, second in zip(
        models[0].parameters(), models[1].parameters(), strict=True
    ):
        assert torch.equal(first, second)


def test_pretraining_never_reads_train_labels(tmp_path: Path) -> None:
    relabelled = CubeDataset(n_samples=96, seed=0)
    relabelled.labels = relabelled.labels.roll(1, dims=0)
    models = []
    for name, train_dataset in [("a", None), ("b", relabelled)]:
        (tmp_path / name).mkdir()
        cfg = _pretraining_config(tmp_path / name, train_dataset=train_dataset)
        models.append(pretrain_l2m(cfg, inputs=load_inputs(cfg)))

    for first, second in zip(
        models[0].parameters(), models[1].parameters(), strict=True
    ):
        assert torch.equal(first, second)


def test_pretraining_caps_pool_at_train_size(tmp_path: Path) -> None:
    cfg = replace(_pretraining_config(tmp_path), pretraining_pool_size=200)

    model = pretrain_l2m(cfg, inputs=load_inputs(cfg))

    assert isinstance(model, L2MModel)


def test_pretraining_requires_a_train_split_that_fills_a_task(
    tmp_path: Path,
) -> None:
    cfg = _pretraining_config(
        tmp_path,
        train_dataset=CubeDataset(n_samples=SEQUENCE_LENGTH - 1, seed=0),
    )
    with pytest.raises(ValueError, match=f"sequence_length={SEQUENCE_LENGTH}"):
        pretrain_l2m(cfg, inputs=load_inputs(cfg))


def test_synthetic_pool_is_uniform_on_the_box_and_seeded() -> None:
    pool = draw_synthetic_pool(size=500, n_features=N_FEATURES, seed=4)

    assert pool.shape == (500, N_FEATURES)
    assert pool.min() >= -2.0
    assert pool.max() <= 2.0
    assert torch.equal(
        pool, draw_synthetic_pool(size=500, n_features=N_FEATURES, seed=4)
    )
    assert not torch.equal(
        pool, draw_synthetic_pool(size=500, n_features=N_FEATURES, seed=5)
    )


def test_synthetic_pretraining_ignores_the_train_split_data(
    tmp_path: Path,
) -> None:
    other_features = CubeDataset(n_samples=96, seed=7)
    models = []
    for name, train_dataset in [("a", None), ("b", other_features)]:
        (tmp_path / name).mkdir()
        cfg = _pretraining_config(
            tmp_path / name,
            train_dataset=train_dataset,
            feature_source=FeatureSource.synthetic,
        )
        models.append(pretrain_l2m(cfg, inputs=load_inputs(cfg)))

    for first, second in zip(
        models[0].parameters(), models[1].parameters(), strict=True
    ):
        assert torch.equal(first, second)


def test_synthetic_pretraining_needs_no_full_task_in_the_train_split(
    tmp_path: Path,
) -> None:
    cfg = _pretraining_config(
        tmp_path,
        train_dataset=CubeDataset(n_samples=SEQUENCE_LENGTH - 1, seed=0),
        feature_source=FeatureSource.synthetic,
    )

    model = pretrain_l2m(cfg, inputs=load_inputs(cfg))

    assert (model.n_features, model.n_classes) == (N_FEATURES, N_CLASSES)


def test_synthetic_training_returns_a_method(tmp_path: Path) -> None:
    cfg = _training_config(tmp_path, feature_source=FeatureSource.synthetic)

    method = train_l2m(cfg, inputs=load_inputs(cfg))

    assert method.context_features.shape == (8, N_FEATURES)


def test_training_returns_a_method_that_survives_its_bundle(
    tmp_path: Path,
) -> None:
    cfg = _training_config(tmp_path)

    method = train_l2m(cfg, inputs=load_inputs(cfg))

    assert method.context_features.shape == (8, N_FEATURES)
    assert method.context_labels.shape == (8, N_CLASSES)
    restored = load_bundle(
        Path(_save(method, tmp_path / "caller.bundle")), device=CPU
    ).obj
    assert isinstance(restored, L2MAFAMethod)
    features = torch.randn(5, N_FEATURES)
    mask = torch.rand(5, N_FEATURES) < 0.5
    assert torch.equal(
        restored.act(features * mask, mask), method.act(features * mask, mask)
    )
    torch.testing.assert_close(
        restored.predict(features * mask, mask),
        method.predict(features * mask, mask),
    )


@pytest.mark.parametrize("context_set_size", [31, 32, 33])
def test_training_caps_context_set_at_validation_size(
    tmp_path: Path,
    context_set_size: int,
) -> None:
    cfg = replace(
        _training_config(tmp_path), context_set_size=context_set_size
    )

    method = train_l2m(cfg, inputs=load_inputs(cfg))

    assert method.context_features.shape == (
        min(32, context_set_size),
        N_FEATURES,
    )
    assert method.context_labels.shape == (
        min(32, context_set_size),
        N_CLASSES,
    )


@pytest.mark.parametrize("hard_budget", [None, 0, N_FEATURES])
def test_training_requires_a_hard_budget_below_the_selection_count(
    tmp_path: Path, hard_budget: int | None
) -> None:
    cfg = _training_config(tmp_path, hard_budget=hard_budget)
    with pytest.raises(ValueError, match=f"hard_budget={hard_budget}"):
        train_l2m(cfg, inputs=load_inputs(cfg))


def test_config_rejects_tasks_without_query_instances(tmp_path: Path) -> None:
    cfg = _pretraining_config(tmp_path)
    with pytest.raises(ValueError, match="sequence_length=50"):
        L2MPretrainingConfig(
            **{**vars(cfg), "sequence_length": 50, "pretraining_pool_size": 50}
        )


def test_binary_logits_match_upstream_sigmoid_prediction() -> None:
    model = _small_model(3, 2).eval()
    generator = torch.Generator().manual_seed(1)
    mask = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
    batch = {
        "xc": torch.randn(1, 4, 3, generator=generator),
        "yc": torch.tensor([[[0.0], [1.0], [1.0], [0.0]]]),
        "rc": torch.ones(1, 4, 3),
        "xt": torch.randn(1, 2, 3, generator=generator),
        "yt": torch.ones(1, 2, 1),
        "rt": torch.ones(1, 2, 3),
        "maskt": mask,
    }

    with torch.no_grad():
        logits = model(batch)
        distribution, _ = model.ltm.predict(batch)

    assert logits.shape == (1, 2, 2)
    torch.testing.assert_close(
        logits.softmax(dim=-1)[..., 1], distribution.probs[..., 0]
    )
    assert torch.equal(
        model.upstream_labels(torch.eye(2)), torch.tensor([[0.0], [1.0]])
    )


def test_predict_ignores_query_labels_and_batching() -> None:
    method = _method(n_classes=3)
    features = torch.tensor(
        [[1.0, 0.0, 0.0], [-4.0, 20.0, 0.0], [0.0, 0.0, 100.0]]
    )
    mask = features != 0

    torch.testing.assert_close(
        method.predict(features, mask, label=torch.eye(3)),
        method.predict(features, mask),
    )
    torch.testing.assert_close(
        method.predict(features[:1], mask[:1]),
        method.predict(features, mask)[:1],
    )
    assert torch.equal(
        method.act(features[:1], mask[:1]), method.act(features, mask)[:1]
    )


def test_act_selects_an_unperformed_feature_without_stopping() -> None:
    method = _method()
    performed = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
            [True, True, False],
            [True, False, True],
            [False, True, True],
        ]
    )
    features = torch.randn(len(performed), 3)

    actions = method.act(
        features, torch.zeros_like(performed), selection_mask=performed
    ).squeeze(-1)

    # Action k selects feature k - 1; action 0 would stop.
    assert torch.all((actions >= 1) & (actions <= 3))
    assert not performed.gather(1, (actions - 1).unsqueeze(-1)).any()
    assert torch.equal(actions[2:], torch.tensor([3, 2, 1]))


def test_act_rejects_exhausted_selections() -> None:
    method = _method()
    mask = torch.tensor([[True, True, True], [False, False, False]])
    with pytest.raises(ValueError, match=r"selection.*exhausted"):
        method.act(torch.zeros(2, 3), mask)


def test_linear_schedule_matches_transformers() -> None:
    factors = [
        linear_schedule_with_warmup(step, warmup_steps=2, n_steps=6)
        for step in range(7)
    ]
    assert factors == [0.0, 0.5, 1.0, 0.75, 0.5, 0.25, 0.0]
