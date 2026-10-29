"""Public-seam assertions and deterministic policies for evaluation tests."""

from collections.abc import Sequence

import polars as pl
import torch

from afabench.core.types import (
    AFAAction,
    AFAActionFn,
    AFAUnmaskFn,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)
from afabench.evaluation.eval import process_batch
from afabench.evaluation.history import reconstruct_selection_history


def _row_at_selection_history(
    df: pl.DataFrame, selections: Sequence[int]
) -> dict[str, object]:
    histories = reconstruct_selection_history(df)
    expected = list(selections)
    rows = df.filter(
        pl.Series([history == expected for history in histories.to_list()])
    )
    assert len(rows) == 1, (
        f"Expected one row with selection history {expected}, got:\n{rows}"
    )
    return rows.row(0, named=True)


def assert_where_selections_there_cost(
    df: pl.DataFrame, selections: Sequence[int], cost: float
) -> None:
    assert (
        _row_at_selection_history(df, selections)["accumulated_cost"] == cost
    )


def assert_where_selections_there_action(
    df: pl.DataFrame, selections: Sequence[int], action: int
) -> None:
    assert (
        _row_at_selection_history(df, selections)["action_performed"] == action
    )


def get_deterministic_afa_action_fn(actions: Sequence[int]) -> AFAActionFn:
    """Return an AFAActionFn that outputs actions in a specified order."""
    action_idx = 0

    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
        selection_mask: SelectionMask | None = None,  # noqa: ARG001
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> AFAAction:
        assert masked_features.ndim == 2, (
            "assuming single batch dimension and single feature dimension"
        )
        batch_size = masked_features.shape[0]
        nonlocal action_idx
        action_tensor = (
            torch.ones((batch_size, 1), dtype=torch.long) * actions[action_idx]
        )
        action_idx += 1
        return action_tensor

    return f


def get_batched_deterministic_afa_action_fn(
    actions: Sequence[Sequence[int]],
) -> AFAActionFn:
    """Output planned actions, removing finished instances from the batch end."""
    action_idx = 0
    active_indices = list(range(len(actions)))

    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
        selection_mask: SelectionMask | None = None,  # noqa: ARG001
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> AFAAction:
        assert masked_features.ndim == 2, (
            "assuming single batch dimension and single feature dimension"
        )
        batch_size = masked_features.shape[0]
        nonlocal action_idx, active_indices
        if batch_size < len(active_indices):
            active_indices = active_indices[:batch_size]
        action_tensor = torch.zeros((batch_size, 1), dtype=torch.long)
        for local_idx in range(batch_size):
            action_tensor[local_idx, 0] = actions[active_indices[local_idx]][
                action_idx
            ]
        action_idx += 1
        return action_tensor

    return f


def get_batch_from_costs_and_budget(
    afa_action_fn: AFAActionFn,
    afa_unmask_fn: AFAUnmaskFn,
    selection_budget: float,
    selection_costs: Sequence[float],
    batch_size: int = 2,
    n_features: int = 5,
) -> pl.DataFrame:
    features = torch.zeros((batch_size, n_features))
    true_label = torch.zeros((batch_size, 2))
    true_label[:, :] = torch.tensor([0, 1])
    return process_batch(
        afa_action_fn=afa_action_fn,
        afa_unmask_fn=afa_unmask_fn,
        n_selection_choices=4,
        features=features,
        initial_masked_features=torch.zeros_like(features),
        initial_feature_mask=torch.zeros_like(features, dtype=torch.bool),
        true_label=true_label,
        feature_shape=torch.Size((n_features,)),
        selection_budget=selection_budget,
        selection_costs=selection_costs,
    )


def assert_terminated_after_n_steps(
    df: pl.DataFrame, idx: int, n_steps: int, forced: bool | None = None
) -> None:
    sample_rows = df.filter(pl.col("episode_id") == idx)
    assert len(sample_rows) == n_steps, (
        f"Expected {n_steps} rows for instance {idx}, got {len(sample_rows)}."
    )
    for row in sample_rows.iter_rows(named=True):
        if row["step"] == n_steps - 1:
            assert row["action_performed"] == 0
            if forced is not None:
                assert row["forced_stop"] == forced
        else:
            assert row["action_performed"] != 0
