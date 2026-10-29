"""
Golden-record parity for `process_batch`.

Issue #25 moved the evaluation loop's bookkeeping onto the compute device and
required that the records it produces stay identical to the previous,
host-side implementation. The golden records below were captured by running
the `dev` implementation of `src/afabench/evaluation/eval.py` at commit e0e26a83
on exactly the batch built here, so any drift in record content, row order or
column dtype fails this test. Compact output is expanded at the public
history-reconstruction seam before comparison with those legacy records.

Every function the loop calls is a pure function of the instance's observed
state, never of its position in the shrinking active set, so the records are
fully determined by the inputs below.
"""

from collections.abc import Callable
from typing import Any

import polars as pl
import pytest
import torch
from polars.testing import assert_frame_equal

from afabench.core.class_probabilities import as_class_probability_fn
from afabench.core.types import (
    AFAAction,
    AFAClassifierFn,
    AFAPredictFn,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)
from afabench.evaluation.eval import process_batch
from afabench.evaluation.history import reconstruct_selection_history
from afabench.testing.helpers import get_direct_unmask_fn

FEATURES = torch.tensor(
    [
        [0.9, 0.1, 0.4, 0.7, 0.2],
        [0.2, 0.8, 0.5, 0.1, 0.6],
        [0.5, 0.5, 0.9, 0.3, 0.8],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.7, 0.6, 0.2, 0.9, 0.1],
        [0.3, 0.9, 0.8, 0.6, 0.4],
    ]
)
N_INSTANCES, N_FEATURES = FEATURES.shape
# Partial initial observations: instance 1 starts with feature 2, instance 4
# with features 0 and 3, instance 5 with feature 1.
INITIAL_FEATURE_MASK = torch.zeros_like(FEATURES, dtype=torch.bool)
INITIAL_FEATURE_MASK[1, 2] = True
INITIAL_FEATURE_MASK[4, [0, 3]] = True
INITIAL_FEATURE_MASK[5, 1] = True
TRUE_CLASSES = torch.tensor([0, 2, 1, 1, 0, 2])
N_CLASSES = 3
# Non-dyadic costs, so accumulated costs carry floating-point rounding that
# both implementations have to reproduce bit for bit.
SELECTION_COSTS = [0.3, 1.1, 0.7, 1.9, 0.4]
SELECTION_BUDGET = 2.5

# Policy: acquire the unobserved feature with the highest score, and stop once
# the observed values sum past a threshold, so instances stop at different
# time steps depending on what they have seen.
ACQUISITION_PREFERENCE = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1])
STOP_THRESHOLD = 1.2

BUILTIN_WEIGHTS = torch.tensor(
    [
        [1.0, -0.5, 0.2],
        [-0.3, 0.8, 0.1],
        [0.4, 0.1, -0.6],
        [-0.2, 0.3, 0.9],
        [0.5, -0.4, 0.3],
    ]
)
EXTERNAL_WEIGHTS = torch.tensor(
    [
        [-0.4, 0.6, 0.1],
        [0.7, -0.2, 0.3],
        [0.2, 0.5, -0.1],
        [0.6, -0.3, 0.4],
        [-0.1, 0.2, 0.8],
    ]
)


def score_policy_action_fn(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
    selection_mask: SelectionMask | None = None,  # noqa: ARG001
    label: Label | None = None,  # noqa: ARG001
    feature_shape: torch.Size | None = None,  # noqa: ARG001
) -> AFAAction:
    scores = ACQUISITION_PREFERENCE + masked_features.roll(1, dims=-1)
    scores = scores.masked_fill(feature_mask, float("-inf"))
    actions = scores.argmax(-1) + 1
    stop = (masked_features.sum(-1) > STOP_THRESHOLD) | feature_mask.all(-1)
    return torch.where(stop, 0, actions).unsqueeze(-1)


def get_linear_predict_fn(weights: torch.Tensor) -> AFAPredictFn:
    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> Label:
        return masked_features @ weights

    logits_fn: AFAClassifierFn = f
    return as_class_probability_fn(logits_fn, output_kind="logits")


SCENARIOS: dict[str, dict[str, Any]] = {
    # Hard budget with non-unit costs; both classifiers.
    "budget": {
        "builtin": True,
        "selection_budget": SELECTION_BUDGET,
        "selection_costs": SELECTION_COSTS,
        "force_acquisition": False,
    },
    # Unit costs and no budget; external classifier only.
    "unlimited_external_only": {
        "builtin": False,
        "selection_budget": None,
        "selection_costs": None,
        "force_acquisition": False,
    },
    # Stops are overridden, so only the budget ends an episode.
    "forced_acquisition": {
        "builtin": True,
        "selection_budget": SELECTION_BUDGET,
        "selection_costs": SELECTION_COSTS,
        "force_acquisition": True,
    },
}


def run_scenario(
    process_batch_fn: Callable[..., pl.DataFrame], scenario: str
) -> pl.DataFrame:
    """Run `process_batch_fn` on the fixed batch with `scenario`'s settings."""
    settings = SCENARIOS[scenario]
    initial_masked_features = FEATURES.clone()
    initial_masked_features[~INITIAL_FEATURE_MASK] = 0.0
    return process_batch_fn(
        afa_action_fn=score_policy_action_fn,
        afa_unmask_fn=get_direct_unmask_fn(),
        n_selection_choices=N_FEATURES,
        features=FEATURES,
        initial_feature_mask=INITIAL_FEATURE_MASK,
        initial_masked_features=initial_masked_features,
        true_label=torch.nn.functional.one_hot(
            TRUE_CLASSES, num_classes=N_CLASSES
        ).float(),
        feature_shape=torch.Size((N_FEATURES,)),
        external_afa_predict_fn=get_linear_predict_fn(EXTERNAL_WEIGHTS),
        builtin_afa_predict_fn=get_linear_predict_fn(BUILTIN_WEIGHTS)
        if settings["builtin"]
        else None,
        selection_budget=settings["selection_budget"],
        selection_costs=settings["selection_costs"],
        force_acquisition=settings["force_acquisition"],
    )


COLUMNS = [
    "prev_selections_performed",
    "action_performed",
    "builtin_predicted_class",
    "external_predicted_class",
    "true_class",
    "accumulated_cost",
    "idx",
    "forced_stop",
]

# Captured from the `dev` implementation at commit e0e26a83. One tuple per
# record, in the order process_batch emits them (time-step major), with fields
# in COLUMNS order:
# (prev_selections, action, builtin, external, true, cost, idx, forced_stop)
GOLDEN_RECORDS: dict[str, list[tuple[Any, ...]]] = {
    "budget": [
        ([], 1, 0, 0, 0, 0.3, 0, False),
        ([], 4, 0, 1, 2, 1.9, 1, False),
        ([], 1, 0, 0, 1, 0.3, 2, False),
        ([], 1, 0, 0, 1, 0.3, 3, False),
        ([], 0, 2, 2, 0, 0.0, 4, False),
        ([], 3, 1, 0, 2, 0.7, 5, False),
        ([0], 2, 0, 1, 0, 1.4000000000000001, 0, False),
        ([3], 1, 0, 1, 2, 2.1999999999999997, 1, False),
        ([0], 2, 0, 1, 1, 1.4000000000000001, 2, False),
        ([0], 2, 0, 1, 1, 1.4000000000000001, 3, False),
        ([2], 0, 1, 0, 2, 0.7, 5, False),
        ([0, 1], 3, 0, 1, 0, 2.1, 0, False),
        ([3, 0], 0, 0, 1, 2, 2.1999999999999997, 1, True),
        ([0, 1], 3, 0, 1, 1, 2.1, 2, False),
        ([0, 1], 3, 1, 0, 1, 2.1, 3, False),
        ([0, 1, 2], 0, 0, 1, 0, 2.1, 0, False),
        ([0, 1, 2], 0, 0, 1, 1, 2.1, 2, False),
        ([0, 1, 2], 0, 0, 1, 1, 2.1, 3, True),
    ],
    "unlimited_external_only": [
        ([], 1, None, 0, 0, 1.0, 0, False),
        ([], 4, None, 1, 2, 1.0, 1, False),
        ([], 1, None, 0, 1, 1.0, 2, False),
        ([], 1, None, 0, 1, 1.0, 3, False),
        ([], 0, None, 2, 0, 0.0, 4, False),
        ([], 3, None, 0, 2, 1.0, 5, False),
        ([0], 2, None, 1, 0, 2.0, 0, False),
        ([3], 1, None, 1, 2, 2.0, 1, False),
        ([0], 2, None, 1, 1, 2.0, 2, False),
        ([0], 2, None, 1, 1, 2.0, 3, False),
        ([2], 0, None, 0, 2, 1.0, 5, False),
        ([0, 1], 3, None, 1, 0, 3.0, 0, False),
        ([3, 0], 2, None, 1, 2, 3.0, 1, False),
        ([0, 1], 3, None, 1, 1, 3.0, 2, False),
        ([0, 1], 3, None, 0, 1, 3.0, 3, False),
        ([0, 1, 2], 0, None, 1, 0, 3.0, 0, False),
        ([3, 0, 1], 0, None, 0, 2, 3.0, 1, False),
        ([0, 1, 2], 0, None, 1, 1, 3.0, 2, False),
        ([0, 1, 2], 4, None, 1, 1, 4.0, 3, False),
        ([0, 1, 2, 3], 5, None, 0, 1, 5.0, 3, False),
        ([0, 1, 2, 3, 4], 0, None, 2, 1, 5.0, 3, False),
    ],
    "forced_acquisition": [
        ([], 1, 0, 0, 0, 0.3, 0, False),
        ([], 4, 0, 1, 2, 1.9, 1, False),
        ([], 1, 0, 0, 1, 0.3, 2, False),
        ([], 1, 0, 0, 1, 0.3, 3, False),
        ([], 1, 2, 2, 0, 0.3, 4, False),
        ([], 3, 1, 0, 2, 0.7, 5, False),
        ([0], 2, 0, 1, 0, 1.4000000000000001, 0, False),
        ([3], 1, 0, 1, 2, 2.1999999999999997, 1, False),
        ([0], 2, 0, 1, 1, 1.4000000000000001, 2, False),
        ([0], 2, 0, 1, 1, 1.4000000000000001, 3, False),
        ([0], 2, 2, 2, 0, 1.4000000000000001, 4, False),
        ([2], 1, 1, 0, 2, 1.0, 5, False),
        ([0, 1], 3, 0, 1, 0, 2.1, 0, False),
        ([3, 0], 0, 0, 1, 2, 2.1999999999999997, 1, True),
        ([0, 1], 3, 0, 1, 1, 2.1, 2, False),
        ([0, 1], 3, 1, 0, 1, 2.1, 3, False),
        ([0, 1], 3, 2, 0, 0, 2.1, 4, False),
        ([2, 0], 2, 1, 0, 2, 2.1, 5, False),
        ([0, 1, 2], 0, 0, 1, 0, 2.1, 0, True),
        ([0, 1, 2], 0, 0, 1, 1, 2.1, 2, True),
        ([0, 1, 2], 0, 0, 1, 1, 2.1, 3, True),
        ([0, 1, 2], 0, 2, 0, 0, 2.1, 4, True),
        ([2, 0, 1], 0, 1, 0, 2, 2.1, 5, True),
    ],
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_records_match_dev_implementation(scenario: str) -> None:
    expected = pl.DataFrame(
        GOLDEN_RECORDS[scenario], schema=COLUMNS, orient="row"
    ).with_columns(pl.col("builtin_predicted_class").cast(pl.Int64))
    compact = run_scenario(process_batch, scenario)
    assert compact["step"].to_list() == [
        len(record[0]) for record in GOLDEN_RECORDS[scenario]
    ]
    # The legacy records stored argmax class decisions, not probabilities.
    actual = (
        compact.with_columns(
            reconstruct_selection_history(compact),
            *[
                pl.Series(
                    f"{classifier}_predicted_class",
                    [
                        None
                        if vector is None
                        else max(range(len(vector)), key=vector.__getitem__)
                        for vector in compact[
                            f"{classifier}_predicted_probabilities"
                        ].to_list()
                    ],
                    dtype=pl.Int64,
                )
                for classifier in ["builtin", "external"]
            ],
        )
        .rename({"episode_id": "idx"})
        .select(COLUMNS)
    )
    assert_frame_equal(actual, expected, check_exact=True, check_dtypes=True)
