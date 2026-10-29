import polars as pl
import torch

from afabench.core.types import (
    Features,
    Label,
)
from afabench.evaluation.eval import process_batch
from afabench.evaluation.history import reconstruct_selection_history
from afabench.testing.eval.helpers import get_deterministic_afa_action_fn
from afabench.testing.helpers import (
    get_deterministic_action_fn,
    get_deterministic_afa_predict_fn,
    get_direct_unmask_fn,
    get_random_afa_predict_fn,
)


def process_batch_wrapper(
    actions: list[list[int]],
    features: Features | None = None,
    external_predictions: list[list[int]] | None = None,
    builtin_predictions: list[list[int]] | None = None,
    true_label: Label | None = None,
    selection_budget: float | None = None,
    selection_costs: list[float] | None = None,
) -> pl.DataFrame:
    if features is None:
        features = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert features.ndim == 2, "Only 1D features with batch dim supported"
    n_samples = features.shape[0]
    n_features = features.shape[-1]

    if true_label is None:
        true_label = torch.zeros((n_samples, 4), dtype=torch.float32)
    n_classes = true_label.shape[-1]

    if external_predictions is None:
        external_afa_predict_fn = get_random_afa_predict_fn(
            n_classes=n_classes
        )
    else:
        external_afa_predict_fn = get_deterministic_afa_predict_fn(
            external_predictions, n_classes=n_classes
        )
    if builtin_predictions is None:
        builtin_afa_predict_fn = get_random_afa_predict_fn(n_classes=n_classes)
    else:
        builtin_afa_predict_fn = get_deterministic_afa_predict_fn(
            builtin_predictions, n_classes=n_classes
        )

    initial_feature_mask = torch.zeros_like(features, dtype=torch.bool)
    initial_masked_features = torch.zeros_like(features)
    n_selection_choices = n_features
    return process_batch(
        # afa_action_fn=get_sequential_action_fn(),
        afa_action_fn=get_deterministic_action_fn(actions),
        afa_unmask_fn=get_direct_unmask_fn(),
        n_selection_choices=n_selection_choices,
        features=features,
        initial_feature_mask=initial_feature_mask,
        initial_masked_features=initial_masked_features,
        true_label=true_label,
        feature_shape=torch.Size((n_features,)),
        external_afa_predict_fn=external_afa_predict_fn,
        builtin_afa_predict_fn=builtin_afa_predict_fn,
        selection_budget=selection_budget,
        selection_costs=selection_costs,
    )


def add_time_column(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("step").alias("time"))


def assert_predictions(
    df: pl.DataFrame,
    idx: int,
    expected_predictions: list[int],
    prediction_type: str,
) -> None:
    if prediction_type not in {"external", "builtin"}:
        raise ValueError(prediction_type)
    probabilities = df.filter(pl.col("episode_id") == idx).sort("time")[
        f"{prediction_type}_predicted_probabilities"
    ]
    predictions = [
        max(range(len(vector)), key=vector.__getitem__)
        for vector in probabilities.to_list()
    ]
    assert predictions == expected_predictions


def test_compact_episode_rows() -> None:
    df = process_batch_wrapper(actions=[[1, 2, 0], [0]])
    assert "prev_selections_performed" not in df.columns
    assert "idx" not in df.columns
    assert df[
        ["episode_id", "step", "action_performed"]
    ].to_numpy().tolist() == [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 2],
        [0, 2, 0],
    ]


def test_steps_count_repeated_selections_not_observed_features() -> None:
    features = torch.tensor([[1.0, 2.0]])
    result = process_batch(
        afa_action_fn=get_deterministic_afa_action_fn([2, 2, 0]),
        afa_unmask_fn=get_direct_unmask_fn(),
        n_selection_choices=2,
        features=features,
        initial_feature_mask=torch.tensor([[True, False]]),
        initial_masked_features=torch.tensor([[1.0, 0.0]]),
        true_label=torch.tensor([[1.0, 0.0]]),
    )
    assert result["step"].to_list() == [0, 1, 2]
    assert reconstruct_selection_history(result).to_list() == [[], [1], [1, 1]]


def test_expected_length() -> None:
    """Test that the returned dataframe has an expected number of rows."""
    features = torch.tensor([[1, 2, 3], [4, 5, 6]])
    actions = [[1, 2, 3, 0], [1, 2, 3, 0]]

    df = process_batch_wrapper(features=features, actions=actions)

    # With 3 features, we should have 4 rows for each sample. We make one prediction at 0 features, 1 feature, 2 features, and 3 features
    assert len(df.filter(pl.col("episode_id") == 0)) == 4
    assert len(df.filter(pl.col("episode_id") == 1)) == 4
    assert len(df) == 8


def test_external_predictions() -> None:
    # Batched
    features = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    actions = [[1, 2, 3, 4, 0], [1, 2, 3, 4, 0]]
    external_predictions = [[0, 1, 3, 2, 1], [3, 1, 0, 2, 0]]

    df = process_batch_wrapper(
        features=features,
        actions=actions,
        external_predictions=external_predictions,
    )
    df = add_time_column(df)

    assert_predictions(
        df,
        idx=0,
        expected_predictions=external_predictions[0],
        prediction_type="external",
    )

    assert_predictions(
        df,
        idx=1,
        expected_predictions=external_predictions[1],
        prediction_type="external",
    )


def test_budget_forces_a_stop_and_records_it() -> None:
    """
    The budget check overrides the action, and the override has to be visible.

    `forced_stop` is the column most sensitive to how the active set is indexed,
    since it is written against global sample indices while the actions it
    reacts to are indexed against the shrinking active set.
    """
    features = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    # Both samples want four selections; a budget of two cuts them short.
    actions = [[1, 2, 3, 4, 0], [1, 2, 3, 4, 0]]

    df = add_time_column(
        process_batch_wrapper(
            features=features, actions=actions, selection_budget=2
        )
    )

    for idx in (0, 1):
        sample = df.filter(pl.col("episode_id") == idx).sort("time")
        assert sample["action_performed"].to_list() == [1, 2, 0]
        assert sample["accumulated_cost"].to_list() == [1.0, 2.0, 2.0]
        assert sample["forced_stop"].to_list() == [False, False, True]


def test_non_unit_costs_accumulate_and_bound_the_episode() -> None:
    """A budget is spent in cost, not in count, and the boundary is strict."""
    features = torch.tensor([[1, 2, 3, 4]])
    actions = [[1, 2, 3, 4, 0]]
    # Selecting features 0 then 1 costs 1 + 3 = 4. Feature 2 costs 2 more,
    # which would reach 6: allowed at budget 6, refused at budget 5.
    costs = [1.0, 3.0, 2.0, 10.0]

    within = add_time_column(
        process_batch_wrapper(
            features=features,
            actions=actions,
            selection_budget=6,
            selection_costs=costs,
        )
    ).sort("time")
    assert within["action_performed"].to_list() == [1, 2, 3, 0]
    assert within["accumulated_cost"].to_list() == [1.0, 4.0, 6.0, 6.0]
    assert within["forced_stop"].to_list() == [False, False, False, True]

    beyond = add_time_column(
        process_batch_wrapper(
            features=features,
            actions=actions,
            selection_budget=5,
            selection_costs=costs,
        )
    ).sort("time")
    assert beyond["action_performed"].to_list() == [1, 2, 0]
    assert beyond["accumulated_cost"].to_list() == [1.0, 4.0, 4.0]


def test_samples_that_stop_at_different_times_keep_their_own_history() -> None:
    """
    Once a sample stops, the active set shifts under the ones still running.

    On-demand reconstruction must attribute each action to the right episode,
    not to whatever position it occupied in the shrinking active set.
    """
    features = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    def action_fn(
        masked_features: torch.Tensor,
        feature_mask: torch.Tensor,  # noqa: ARG001
        selection_mask: torch.Tensor | None = None,
        label: torch.Tensor | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> torch.Tensor:
        # Identify samples by what they have taken, not by their position in
        # the active set, which is exactly the thing under test.
        assert selection_mask is not None
        plans = {0: [1, 2, 0], 1: [3, 4, 2, 0]}
        out = torch.zeros(
            (masked_features.shape[0], 1),
            dtype=torch.int,
            device=features.device,
        )
        for row in range(masked_features.shape[0]):
            taken = selection_mask[row]
            sample = 0 if (not taken.any() and row == 0) or taken[0] else 1
            out[row] = plans[sample][int(taken.sum())]
        return out

    df = add_time_column(
        process_batch(
            afa_action_fn=action_fn,
            afa_unmask_fn=get_direct_unmask_fn(),
            n_selection_choices=4,
            features=features,
            initial_feature_mask=torch.zeros_like(features, dtype=torch.bool),
            initial_masked_features=torch.zeros_like(features),
            true_label=torch.zeros((2, 4), dtype=torch.float32),
            feature_shape=torch.Size((4,)),
            external_afa_predict_fn=get_random_afa_predict_fn(n_classes=4),
            builtin_afa_predict_fn=get_random_afa_predict_fn(n_classes=4),
        )
    )

    df = df.with_columns(reconstruct_selection_history(df))
    first = df.filter(pl.col("episode_id") == 0).sort("time")
    second = df.filter(pl.col("episode_id") == 1).sort("time")
    assert first["action_performed"].to_list() == [1, 2, 0]
    assert second["action_performed"].to_list() == [3, 4, 2, 0]
    assert first["prev_selections_performed"].to_list() == [[], [0], [0, 1]]
    assert second["prev_selections_performed"].to_list() == [
        [],
        [2],
        [2, 3],
        [2, 3, 1],
    ]


def test_builtin_predictions() -> None:
    # Batched
    features = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    actions = [[1, 2, 3, 4, 0], [1, 2, 3, 4, 0]]
    builtin_predictions = [[0, 1, 3, 2, 1], [3, 1, 0, 2, 0]]

    df = process_batch_wrapper(
        features=features,
        actions=actions,
        builtin_predictions=builtin_predictions,
    )
    df = add_time_column(df)

    assert_predictions(
        df,
        idx=0,
        expected_predictions=builtin_predictions[0],
        prediction_type="builtin",
    )
    assert_predictions(
        df,
        idx=1,
        expected_predictions=builtin_predictions[1],
        prediction_type="builtin",
    )
