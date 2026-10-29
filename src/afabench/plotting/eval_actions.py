"""
Action heatmaps of a plotting-ready table.

A heatmap shows, for each time step, the share of episodes at that step
that performed each selection; stop actions are left out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from jaxtyping import Float
from tqdm import tqdm

from afabench.evaluation.tables import combine_soft_budget_params

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from afabench.plotting.config import PlottingDisplayConfig

type Heatmap = Float[npt.NDArray[np.float64], "actions time_steps"]

PLOT_FONT_SIZE = 16
PLOT_TITLE_FONT_SIZE = 18
HEATMAP_COLUMNS = 5


def normalize_heatmap_by_timestep(
    df_method: pl.DataFrame,
    max_action: int,
    max_time: int,
) -> Heatmap:
    """
    Create and normalize heatmap for a single method.

    Heatmap shape: [num_actions, num_timesteps]
    Values normalized by number of samples at each timestep.
    Excludes action 0.
    """
    heatmap = np.zeros((int(max_action), int(max_time) + 1))

    actions = df_method["action_performed"].cast(pl.Int64).to_numpy()
    time_steps = df_method["n_selections_performed"].cast(pl.Int64).to_numpy()
    # Skip action 0
    taken = actions > 0
    np.add.at(heatmap, (actions[taken] - 1, time_steps[taken]), 1)

    time_counts = np.bincount(
        time_steps,
        minlength=int(max_time) + 1,
    )
    time_counts = np.maximum(time_counts, 1)
    return heatmap / time_counts


def format_heatmap_axes(
    ax: Axes,
    max_action: float,
    max_time: float,
    method: str,
    plotting_config: PlottingDisplayConfig,
) -> None:
    """Format and label axes for a heatmap subplot."""
    method_name = plotting_config.method_name_mapping.get(method, method)
    ax.set_title(
        method_name,
        fontsize=PLOT_FONT_SIZE,
        fontweight="bold",
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Action index")
    ax.set_xticks(np.arange(0, int(max_time) + 1, max(1, int(max_time) // 5)))
    max_action_int = int(max_action)
    y_ticks = np.arange(
        0, max_action_int + 1, max(1, (max_action_int + 1) // 10)
    )
    ax.set_yticks(y_ticks)
    # Increment y tick labels by 1 to show action numbers (1, 2, 3...) instead of (0, 1, 2...)
    ax.set_yticklabels(y_ticks + 1)


def action_heatmap_figure(
    group_df: pl.DataFrame,
    methods: Sequence[str],
    title: str,
    plotting_config: PlottingDisplayConfig,
) -> Figure:
    """
    Draw one heatmap per method of `group_df`, on shared axis limits.

    `group_df` holds the selections of one dataset and one budget.
    """
    num_methods = len(methods)
    num_rows = (num_methods + HEATMAP_COLUMNS - 1) // HEATMAP_COLUMNS

    # Calculate global max_action and max_time across all methods in group
    global_max_action = cast("int", group_df["action_performed"].max())
    global_max_time = cast("int", group_df["n_selections_performed"].max())

    fig, axes = plt.subplots(
        num_rows,
        HEATMAP_COLUMNS,
        figsize=(3 * HEATMAP_COLUMNS, 4 * num_rows),
        squeeze=False,
    )

    for idx, method in enumerate(methods):
        ax = axes[idx // HEATMAP_COLUMNS, idx % HEATMAP_COLUMNS]
        df_method = group_df.filter(pl.col("afa_method") == method)

        heatmap = normalize_heatmap_by_timestep(
            df_method, global_max_action, global_max_time
        )

        ax.imshow(
            heatmap,
            cmap="Blues",
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )

        format_heatmap_axes(
            ax, global_max_action, global_max_time, method, plotting_config
        )

    # Hide any unused subplots
    for idx in range(num_methods, num_rows * HEATMAP_COLUMNS):
        axes[idx // HEATMAP_COLUMNS, idx % HEATMAP_COLUMNS].set_visible(False)

    fig.suptitle(title, fontsize=PLOT_TITLE_FONT_SIZE, y=0.98)
    plt.subplots_adjust(
        left=0.08, right=0.92, top=0.84, bottom=0.1, wspace=0.3
    )
    return fig


def save_action_heatmaps(
    df: pl.DataFrame,
    output_folder: Path,
    budget_type: str,
    plotting_config: PlottingDisplayConfig,
    formats: Sequence[str] = ("pdf",),
) -> None:
    """
    Save one heatmap figure per dataset and budget, in a folder per dataset.

    In the hard-budget setting a figure holds every method; in the
    soft-budget setting each method gets its own figure per parameter.

    Args:
        df: Rows of one budget setting, with `soft_budget_param` combined
        output_folder: Output folder for plots
        budget_type: Type of budget ("hard_budget" or "soft_budget")
        formats: Output formats (e.g. ("pdf", "svg")). Default: ("pdf",)
    """
    # Filter out action 0
    filtered_df = df.filter(pl.col("action_performed") != 0)
    # Group by dataset and budget (and method for soft budget)
    if budget_type == "hard_budget":
        group_cols = ["dataset", "eval_hard_budget"]
    else:
        group_cols = ["dataset", "afa_method", "soft_budget_param"]

    for group_keys, group_df in tqdm(
        filtered_df.group_by(group_cols, maintain_order=True),
        desc=f"Creating separate {budget_type} plots",
    ):
        if budget_type == "hard_budget":
            dataset_name, hard_budget = group_keys
            extra_title = f"Hard Budget: {hard_budget}"
            filename_suffix = f"_{hard_budget}"
            methods = sorted(group_df["afa_method"].unique().to_list())
            method_filename = ""
        else:
            dataset_name, method_name, soft_budget_param = group_keys
            extra_title = f"Soft Budget Param: {soft_budget_param}"
            filename_suffix = f"_{soft_budget_param}"
            methods = [method_name]
            method_filename = f"_{method_name.replace('/', '_')}"

        # Create subdirectory for this dataset
        dataset_output_folder = output_folder / dataset_name
        dataset_output_folder.mkdir(parents=True, exist_ok=True)

        dataset_display_name = plotting_config.dataset_name_mapping.get(
            dataset_name, dataset_name
        )
        fig = action_heatmap_figure(
            group_df,
            methods,
            f"Action Heatmaps - {dataset_display_name} - {extra_title}",
            plotting_config,
        )

        for fmt in formats:
            output_path = (
                dataset_output_folder
                / f"{dataset_name}{method_filename}_action_heatmap{filename_suffix}.{fmt}"
            )
            fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def save_all_action_heatmaps(
    table: pl.DataFrame,
    output_folder: Path,
    plotting_config: PlottingDisplayConfig,
    formats: Sequence[str] = ("pdf",),
) -> None:
    """
    Save the action heatmaps of a plotting-ready table.

    Hard-budget heatmaps go under `hard_budget/` and soft-budget ones under
    `soft_budget/`, one figure per dataset and budget.
    """
    table = combine_soft_budget_params(table)
    hard_budget_folder = output_folder / "hard_budget"
    soft_budget_folder = output_folder / "soft_budget"
    hard_budget_folder.mkdir(parents=True, exist_ok=True)
    soft_budget_folder.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(
        {
            "font.family": plotting_config.plot_font_family,
            "font.size": PLOT_FONT_SIZE,
        }
    ):
        save_action_heatmaps(
            df=table.filter(pl.col("eval_hard_budget").is_not_null()),
            output_folder=hard_budget_folder,
            budget_type="hard_budget",
            plotting_config=plotting_config,
            formats=formats,
        )
        save_action_heatmaps(
            df=table.filter(pl.col("soft_budget_param").is_not_null()),
            output_folder=soft_budget_folder,
            budget_type="soft_budget",
            plotting_config=plotting_config,
            formats=formats,
        )
