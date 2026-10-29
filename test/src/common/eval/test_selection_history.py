"""Selection history is derived only from complete episode logs."""

import polars as pl
import pytest

from afabench.evaluation.history import reconstruct_selection_history


def test_reconstructs_ordered_histories_after_reordering() -> None:
    frame = pl.DataFrame(
        {
            "episode_id": [7, 2, 7, 7, 2, 7],
            "step": [3, 1, 1, 0, 0, 2],
            "action_performed": [0, 0, 3, 3, 1, 1],
        },
    )
    histories = reconstruct_selection_history(frame)
    assert histories.to_list() == [[2, 2, 0], [0], [2], [], [], [2, 2]]
    assert list(frame.columns) == ["episode_id", "step", "action_performed"]


@pytest.mark.parametrize(
    ("steps", "actions"),
    [
        ([1, 2], [1, 0]),
        ([0, 2], [1, 0]),
        ([0, 0], [1, 0]),
        ([0, 1], [1, 2]),
        ([0, 1], [0, 1]),
        ([0, 1], [0, 0]),
        ([0, -1], [1, 0]),
        ([0, 1], [-1, 0]),
        ([0, 1.5], [1, 0]),
        ([0, 1], [True, False]),
    ],
)
def test_reconstruction_rejects_malformed_episodes(
    steps: list[int | float], actions: list[int | bool]
) -> None:
    frame = pl.DataFrame(
        {"episode_id": [0, 0], "step": steps, "action_performed": actions},
        strict=False,
    )
    with pytest.raises(ValueError, match=r"Episode|nonnegative integers"):
        reconstruct_selection_history(frame)
