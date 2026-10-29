"""Evaluation rows identify the instance each episode evaluated (ADR 0004)."""

from pathlib import Path
from typing import cast

import torch

from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.core.types import (
    AFAAction,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)
from afabench.datasets import CubeDataset
from afabench.evaluation.eval import eval_afa_method
from afabench.testing.helpers import get_direct_unmask_fn
from afabench.testing.provenance import placeholder_provenance


def initialize_fully_observed(
    features: Features,
    label: Label | None = None,  # noqa: ARG001
    feature_shape: torch.Size | None = None,  # noqa: ARG001
) -> FeatureMask:
    return torch.ones_like(features, dtype=torch.bool)


def test_sampled_evaluation_records_the_evaluated_instances(
    tmp_path: Path,
) -> None:
    # A shuffled subset, so split and generation indices differ
    source = CubeDataset(n_samples=30, seed=0)
    bundle_path = tmp_path / "test.bundle"
    save_bundle(
        source.create_subset(torch.randperm(30)[:20].tolist()),
        bundle_path,
        metadata={},
        provenance=placeholder_provenance(),
    )
    dataset = cast("CubeDataset", load_bundle(bundle_path).obj)
    seen: list[torch.Tensor] = []

    def record_and_stop(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
        selection_mask: SelectionMask | None = None,  # noqa: ARG001
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> AFAAction:
        seen.append(masked_features.clone())
        return torch.zeros((masked_features.shape[0], 1), dtype=torch.long)

    result = eval_afa_method(
        afa_action_fn=record_and_stop,
        afa_unmask_fn=get_direct_unmask_fn(),
        n_selection_choices=20,
        afa_initialize_fn=initialize_fully_observed,
        dataset=dataset,
        only_n_samples=7,
        batch_size=3,
        seed=4,
    )

    # Every episode stops at once, so episode e saw row e of `seen`
    evaluated = torch.cat(seen)
    episodes = result.sort("episode_id")
    split_index = torch.tensor(episodes["split_index"].to_list())
    generation_index = torch.tensor(episodes["generation_index"].to_list())
    assert len(split_index.unique()) == 7
    torch.testing.assert_close(dataset.features[split_index], evaluated)
    torch.testing.assert_close(source.features[generation_index], evaluated)
    assert torch.equal(
        dataset.get_generation_indices()[split_index], generation_index
    )
