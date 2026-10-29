"""Evaluation consumes and retains class-index-ordered probabilities."""

import re
from typing import cast

import polars as pl
import pytest
import torch
from jaxtyping import Shaped

from afabench.core.class_probabilities import as_class_probability_fn
from afabench.core.types import (
    AFAAction,
    AFAPredictFn,
    ClassifierOutputKind,
    ClassProbabilities,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)
from afabench.evaluation.eval import process_batch
from afabench.testing.helpers import (
    get_direct_unmask_fn,
)

# Deliberately invalid ranks and dtypes are inputs to rejection tests.
type PredictionInput = Shaped[torch.Tensor, "..."]


def observed_probabilities(
    masked_features: MaskedFeatures,  # noqa: ARG001
    feature_mask: FeatureMask,
    label: Label | None = None,  # noqa: ARG001
    feature_shape: torch.Size | None = None,  # noqa: ARG001
) -> ClassProbabilities:
    return torch.where(
        feature_mask,
        torch.tensor([0.125, 0.875]),
        torch.tensor([0.75, 0.25]),
    )


def stop_when_observed(
    masked_features: MaskedFeatures,  # noqa: ARG001
    feature_mask: FeatureMask,
    selection_mask: SelectionMask | None = None,  # noqa: ARG001
    label: Label | None = None,  # noqa: ARG001
    feature_shape: torch.Size | None = None,  # noqa: ARG001
) -> AFAAction:
    return (~feature_mask).long()


def evaluate_probabilities(
    external: AFAPredictFn | None = observed_probabilities,
    builtin: AFAPredictFn | None = observed_probabilities,
) -> pl.DataFrame:
    features = torch.tensor([[1.0], [2.0]])
    mask = torch.tensor([[False], [True]])
    return process_batch(
        afa_action_fn=stop_when_observed,
        afa_unmask_fn=get_direct_unmask_fn(),
        n_selection_choices=1,
        features=features,
        initial_feature_mask=mask,
        initial_masked_features=features.masked_fill(~mask, 0.0),
        true_label=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        external_afa_predict_fn=external,
        builtin_afa_predict_fn=builtin,
    )


def test_process_batch_retains_probabilities_at_every_recorded_time_step() -> (
    None
):
    result = evaluate_probabilities()

    assert result[["episode_id", "step"]].to_numpy().tolist() == [
        [0, 0],
        [1, 0],
        [0, 1],
    ]
    expected = [[0.75, 0.25], [0.125, 0.875], [0.125, 0.875]]
    assert result["external_predicted_probabilities"].to_list() == expected
    assert result["builtin_predicted_probabilities"].to_list() == expected


def test_evaluation_leaves_class_decisions_to_consumers() -> None:
    result = evaluate_probabilities()

    assert "builtin_predicted_class" not in result.columns
    assert "external_predicted_class" not in result.columns


def constant_output(output: PredictionInput) -> AFAPredictFn:
    def classify(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> ClassProbabilities:
        return output[: len(masked_features)]

    return classify


@pytest.mark.parametrize("classifier", ["external", "builtin"])
@pytest.mark.parametrize(
    "output",
    [
        torch.tensor([0.5, 0.5]),
        torch.full((1, 2), 0.5),
        torch.full((2, 3), 1 / 3),
        torch.ones((2, 1)),
        torch.full((2, 1, 2), 0.5),
    ],
)
def test_evaluation_rejects_incompatible_probability_shapes(
    classifier: str, output: PredictionInput
) -> None:
    supplied = {classifier: constant_output(output)}
    with pytest.raises(
        ValueError,
        match=rf"{classifier}.*shape.*{re.escape(str(tuple(output.shape)))}",
    ):
        evaluate_probabilities(**supplied)


@pytest.mark.parametrize("classifier", ["external", "builtin"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_evaluation_rejects_nonfinite_probabilities(
    classifier: str, value: float
) -> None:
    supplied = {classifier: constant_output(torch.tensor([[value, 0.5]] * 2))}
    with pytest.raises(ValueError, match=rf"{classifier}.*finite.*{value}"):
        evaluate_probabilities(**supplied)


@pytest.mark.parametrize("classifier", ["external", "builtin"])
@pytest.mark.parametrize("values", [[-0.25, 1.25], [1.1, 0.0]])
def test_evaluation_rejects_probabilities_outside_unit_interval(
    classifier: str, values: list[float]
) -> None:
    supplied = {classifier: constant_output(torch.tensor([values] * 2))}
    with pytest.raises(
        ValueError, match=rf"{classifier}.*\[0, 1\].*{values[0]}"
    ):
        evaluate_probabilities(**supplied)


@pytest.mark.parametrize("classifier", ["external", "builtin"])
@pytest.mark.parametrize("values", [[0.25, 0.25], [0.75, 0.75]])
def test_evaluation_rejects_unnormalized_probability_rows(
    classifier: str, values: list[float]
) -> None:
    supplied = {classifier: constant_output(torch.tensor([values] * 2))}
    with pytest.raises(ValueError, match=rf"{classifier}.*sum.*{sum(values)}"):
        evaluate_probabilities(**supplied)


@pytest.mark.parametrize(
    "output", [torch.tensor([[1, 0]] * 2), torch.tensor([[True, False]] * 2)]
)
def test_evaluation_requires_floating_point_probabilities(
    output: PredictionInput,
) -> None:
    with pytest.raises(TypeError, match=rf"external.*dtype.*{output.dtype}"):
        evaluate_probabilities(external=constant_output(output))


def test_evaluation_converts_only_explicitly_declared_logits_upstream() -> (
    None
):

    # The same normalized-looking values can be logits or probabilities.
    output = constant_output(torch.tensor([[0.75, 0.25]] * 2))
    probabilities = evaluate_probabilities(
        external=as_class_probability_fn(output, output_kind="probabilities"),
        builtin=None,
    )
    logits = evaluate_probabilities(
        external=as_class_probability_fn(output, output_kind="logits"),
        builtin=None,
    )

    assert probabilities["external_predicted_probabilities"].to_list()[0] == [
        0.75,
        0.25,
    ]
    assert logits["external_predicted_probabilities"].to_list()[
        0
    ] == pytest.approx([0.62245933, 0.37754067])


def test_evaluation_adapter_rejects_unknown_output_kind() -> None:

    with pytest.raises(ValueError, match=r"output_kind.*scores"):
        evaluate_probabilities(
            external=as_class_probability_fn(
                observed_probabilities,
                output_kind=cast("ClassifierOutputKind", "scores"),
            )
        )
