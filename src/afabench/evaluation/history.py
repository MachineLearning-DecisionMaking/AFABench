"""On-demand selection histories for a single evaluation artifact."""

import polars as pl

from afabench.evaluation.schemas import validate_episode_log


def reconstruct_selection_history(frame: pl.DataFrame) -> pl.Series:
    """
    Return pre-action histories aligned to input row positions.

    Episode IDs are artifact-local: do not combine independently produced logs
    before calling this function. Stop and initial observations are excluded;
    repeated selections are retained. Materializing all prefixes can require
    quadratic memory, so callers needing only lengths should use ``step``.
    """
    validate_episode_log(frame)
    events = frame.select(
        "episode_id", "step", "action_performed"
    ).with_row_index("position")
    histories: list[list[int]] = [[] for _ in range(len(events))]
    for episode in events.sort("step").partition_by("episode_id"):
        history: list[int] = []
        for position, action in episode.select(
            "position", "action_performed"
        ).iter_rows():
            histories[position] = history.copy()
            if action > 0:
                history.append(int(action) - 1)
    return pl.Series(
        "prev_selections_performed", histories, dtype=pl.List(pl.Int64)
    )
