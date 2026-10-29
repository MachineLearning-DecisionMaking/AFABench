from pathlib import Path

import numpy as np
import polars as pl

from afabench.plotting.config import PlottingDisplayConfig
from afabench.plotting.eval_actions import (
    normalize_heatmap_by_timestep,
    save_action_heatmaps,
)


def test_polars_heatmap_counts_stops_in_the_time_step_denominator() -> None:
    table = pl.DataFrame(
        {
            "action_performed": [1, 2, 0, 2],
            "n_selections_performed": [0, 0, 0, 1],
        },
        schema={
            "action_performed": pl.UInt64,
            "n_selections_performed": pl.UInt64,
        },
    )

    heatmap = normalize_heatmap_by_timestep(table, max_action=2, max_time=2)

    np.testing.assert_allclose(heatmap, [[1 / 3, 0.0, 0.0], [1 / 3, 1.0, 0.0]])
    np.testing.assert_array_equal(
        normalize_heatmap_by_timestep(table.clear(), max_action=2, max_time=2),
        np.zeros((2, 3)),
    )


def test_polars_action_heatmaps_keep_null_budget_groups_and_skip_stops(
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    table = pl.DataFrame(
        {
            "dataset": ["cube", "cube", "mnist"],
            "afa_method": ["jafa", "gdfs", "jafa"],
            "action_performed": [1, 2, 0],
            "n_selections_performed": [0, 0, 0],
            "eval_hard_budget": [None, None, 2.0],
        },
    )

    save_action_heatmaps(
        table, tmp_path, "hard_budget", plotting_config, formats=("png",)
    )

    assert [
        p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.png")
    ] == ["cube/cube_action_heatmap_None.png"]
    assert not (tmp_path / "mnist").exists()
    save_action_heatmaps(
        table.clear(),
        tmp_path / "empty",
        "hard_budget",
        plotting_config,
        formats=("png",),
    )
    assert not (tmp_path / "empty").exists()
