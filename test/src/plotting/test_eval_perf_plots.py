from dataclasses import replace
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from afabench.evaluation.tables import load_plotting_ready_table
from afabench.plotting.config import PlotEvalPerfConfig, PlottingDisplayConfig
from afabench.plotting.eval_perf import (
    EvaluationPlotter,
    add_metric_column,
    get_normal_hard_budget_plot,
    get_per_dataset_plot,
    process_df_every_action,
    process_df_only_stop_action_and_valid_prediction,
)
from scripts.plotting.plot_eval_perf import main

REPO_ROOT = Path(__file__).parents[3]

# The (ddof=1) standard deviation of two values `d` apart is d * SQRT_HALF
SQRT_HALF = 0.5**0.5


def eval_row(
    method: str,
    dataset: str,
    train_seed: int,
    *,
    action: int,
    n_selections: int,
    true_class: int,
    predicted_class: int | None,
    cost: float,
    hard_budget: float | None = None,
    eval_soft_budget_param: float | None = None,
) -> dict[str, object]:
    return {
        "action_performed": action,
        "true_class": true_class,
        "accumulated_cost": cost,
        "forced_stop": False,
        "eval_seed": 0,
        "eval_hard_budget": hard_budget,
        "n_selections_performed": n_selections,
        "predicted_probabilities": (
            None
            if predicted_class is None
            else [0.8, 0.2]
            if predicted_class == 0
            else [0.3, 0.7]
        ),
        "afa_method": method,
        "dataset": dataset,
        "initializer": "cold",
        "train_seed": train_seed,
        "train_hard_budget": None,
        "train_soft_budget_param": None,
        "eval_soft_budget_param": eval_soft_budget_param,
    }


EVAL_SCHEMA = pa.schema(
    {
        "action_performed": pa.uint64(),
        "true_class": pa.uint64(),
        "accumulated_cost": pa.float64(),
        "forced_stop": pa.bool_(),
        "eval_seed": pa.uint64(),
        "eval_hard_budget": pa.float64(),
        "n_selections_performed": pa.uint64(),
        "predicted_probabilities": pa.list_(pa.float64()),
        "afa_method": pa.string(),
        "dataset": pa.string(),
        "initializer": pa.string(),
        "train_seed": pa.uint64(),
        "train_hard_budget": pa.float64(),
        "train_soft_budget_param": pa.float64(),
        "eval_soft_budget_param": pa.float64(),
    }
)


@pytest.fixture
def eval_perf_path(tmp_path: Path) -> Path:
    rows = [
        # JAFA on cube at hard budget 2, two training seeds. Seed 0 makes an
        # intermediate wrong guess, then stops with one of two instances right.
        eval_row(
            "jafa", "cube", 0, action=3, n_selections=0, true_class=1,
            predicted_class=0, cost=0.0, hard_budget=2.0,
        ),
        eval_row(
            "jafa", "cube", 0, action=0, n_selections=2, true_class=1,
            predicted_class=1, cost=2.0, hard_budget=2.0,
        ),
        eval_row(
            "jafa", "cube", 0, action=0, n_selections=1, true_class=0,
            predicted_class=1, cost=1.0, hard_budget=2.0,
        ),
        # Rows without a prediction are ignored
        eval_row(
            "jafa", "cube", 0, action=0, n_selections=1, true_class=0,
            predicted_class=None, cost=1.0, hard_budget=2.0,
        ),
        # Seed 1 gets both instances right
        eval_row(
            "jafa", "cube", 1, action=0, n_selections=2, true_class=1,
            predicted_class=1, cost=2.0, hard_budget=2.0,
        ),
        eval_row(
            "jafa", "cube", 1, action=0, n_selections=2, true_class=0,
            predicted_class=0, cost=2.0, hard_budget=2.0,
        ),
        # A smaller hard budget, evaluated with a single seed
        eval_row(
            "jafa", "cube", 0, action=0, n_selections=1, true_class=1,
            predicted_class=0, cost=1.0, hard_budget=1.0,
        ),
        # GDFS on mnist (an F1 dataset) under a soft budget, single seed
        eval_row(
            "gdfs", "mnist", 0, action=0, n_selections=3, true_class=1,
            predicted_class=1, cost=3.0, eval_soft_budget_param=0.1,
        ),
        eval_row(
            "gdfs", "mnist", 0, action=0, n_selections=3, true_class=0,
            predicted_class=1, cost=3.0, eval_soft_budget_param=0.1,
        ),
    ]  # fmt: skip
    path = tmp_path / "eval_perf.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=EVAL_SCHEMA), path)
    return path


def sorted_rows(df: pl.DataFrame, *keys: str) -> list[dict[str, object]]:
    assert isinstance(df, pl.DataFrame)
    rows = df.to_dicts()
    return sorted(rows, key=lambda row: tuple(str(row[key]) for key in keys))


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.UndefinedMetricWarning"
)
def test_stop_action_metrics_are_aggregated_over_seeds(
    eval_perf_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    plotter = EvaluationPlotter(
        load_plotting_ready_table(eval_perf_path), tmp_path, plotting_config
    )

    assert plotter.df_stop_action is not None
    rows = sorted_rows(
        plotter.df_stop_action, "afa_method", "eval_hard_budget"
    )
    assert rows == [
        {
            "afa_method": "gdfs",
            "dataset": "mnist",
            "eval_hard_budget": None,
            "soft_budget_param": 0.1,
            "mean_auroc": 0.5,
            "std_auroc": None,
            "mean_accuracy": 0.5,
            "std_accuracy": None,
            "mean_f_score": pytest.approx(1 / 3),
            "std_f_score": None,
            "mean_avg_accumulated_cost": 3.0,
            "std_avg_accumulated_cost": None,
            # mnist is scored with F1
            "mean_metric": pytest.approx(1 / 3),
            "std_metric": None,
            "low_metric": None,
            "high_metric": None,
            "low_avg_accumulated_cost": None,
            "high_avg_accumulated_cost": None,
        },
        {
            "afa_method": "jafa",
            "dataset": "cube",
            "eval_hard_budget": 1.0,
            "soft_budget_param": None,
            "mean_auroc": pytest.approx(float("nan"), nan_ok=True),
            "std_auroc": None,
            "mean_accuracy": 0.0,
            "std_accuracy": None,
            "mean_f_score": 0.0,
            "std_f_score": None,
            "mean_avg_accumulated_cost": 1.0,
            "std_avg_accumulated_cost": None,
            "mean_metric": 0.0,
            "std_metric": None,
            "low_metric": None,
            "high_metric": None,
            "low_avg_accumulated_cost": None,
            "high_avg_accumulated_cost": None,
        },
        {
            "afa_method": "jafa",
            "dataset": "cube",
            "eval_hard_budget": 2.0,
            "soft_budget_param": None,
            # Seeds score 0.5 and 1.0
            "mean_auroc": 0.75,
            "std_auroc": pytest.approx(0.5 * SQRT_HALF),
            "mean_accuracy": 0.75,
            "std_accuracy": pytest.approx(0.5 * SQRT_HALF),
            # Seeds score 1/3 and 1.0
            "mean_f_score": pytest.approx(2 / 3),
            "std_f_score": pytest.approx(2 / 3 * SQRT_HALF),
            # Seeds average 1.5 and 2.0
            "mean_avg_accumulated_cost": 1.75,
            "std_avg_accumulated_cost": pytest.approx(0.5 * SQRT_HALF),
            # cube is scored with accuracy
            "mean_metric": 0.75,
            "std_metric": pytest.approx(0.5 * SQRT_HALF),
            "low_metric": pytest.approx(0.75 - 0.5 * SQRT_HALF),
            "high_metric": pytest.approx(0.75 + 0.5 * SQRT_HALF),
            "low_avg_accumulated_cost": pytest.approx(1.75 - 0.5 * SQRT_HALF),
            "high_avg_accumulated_cost": pytest.approx(1.75 + 0.5 * SQRT_HALF),
        },
    ]


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.UndefinedMetricWarning"
)
def test_per_time_step_metrics_use_largest_hard_budget_per_dataset(
    eval_perf_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    plotter = EvaluationPlotter(
        load_plotting_ready_table(eval_perf_path), tmp_path, plotting_config
    )

    assert plotter.df_traj is not None
    # Only JAFA on cube at hard budget 2 remains: budget 1 is not the
    # largest, and the soft budget run has no hard budget at all
    shared = {
        "afa_method": "jafa",
        "dataset": "cube",
        "eval_hard_budget": 2.0,
        "soft_budget_param": None,
    }
    wrong_once = {
        "mean_auroc": pytest.approx(float("nan"), nan_ok=True),
        "std_auroc": None,
        "mean_accuracy": 0.0,
        "std_accuracy": None,
        "mean_f_score": 0.0,
        "std_f_score": None,
        "mean_metric": 0.0,
        "std_metric": None,
        "low_metric": None,
        "high_metric": None,
    }
    assert sorted_rows(plotter.df_traj, "n_selections_performed") == [
        shared | {"n_selections_performed": 0} | wrong_once,
        shared | {"n_selections_performed": 1} | wrong_once,
        shared
        | {
            "n_selections_performed": 2,
            "mean_auroc": pytest.approx(float("nan"), nan_ok=True),
            "std_auroc": pytest.approx(float("nan"), nan_ok=True),
            "mean_accuracy": 1.0,
            "std_accuracy": 0.0,
            "mean_f_score": 1.0,
            "std_f_score": 0.0,
            "mean_metric": 1.0,
            "std_metric": 0.0,
            "low_metric": 1.0,
            "high_metric": 1.0,
        },
    ]


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.UndefinedMetricWarning"
)
def test_all_plots_are_written_for_each_dataset_set(
    eval_perf_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    output_folder = tmp_path / "plots"
    plotter = EvaluationPlotter(
        load_plotting_ready_table(eval_perf_path),
        output_folder,
        plotting_config,
        formats=("png",),
    )
    plotter.generate_all_plots({"all": ["cube", "mnist"]})

    assert sorted(
        path.name for path in (output_folder / "dataset_set-all").iterdir()
    ) == [
        "hard_budget_normal.png",
        "hard_budget_traj.png",
        "soft_budget_2d_errors.png",
        "soft_budget_lines.png",
    ]


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.UndefinedMetricWarning"
)
def test_dataset_set_without_results_gets_no_folder(
    eval_perf_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    output_folder = tmp_path / "plots"
    plotter = EvaluationPlotter(
        load_plotting_ready_table(eval_perf_path),
        output_folder,
        plotting_config,
        formats=("png",),
    )
    plotter.generate_all_plots({"cube_only": ["cube"], "unrun": ["ckd"]})

    assert [path.name for path in output_folder.iterdir()] == [
        "dataset_set-cube_only"
    ]


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.UndefinedMetricWarning"
)
def test_caption_labels_every_plot(
    eval_perf_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    output_folder = tmp_path / "plots"
    caption = "Workflow demonstration, not scientific results"
    plotter = EvaluationPlotter(
        load_plotting_ready_table(eval_perf_path),
        output_folder,
        replace(plotting_config, caption=caption),
        formats=("svg",),
    )
    plotter.generate_all_plots({"all": ["cube", "mnist"]})

    plots = sorted((output_folder / "dataset_set-all").iterdir())
    assert len(plots) == 4
    # Matplotlib keeps each text of an SVG as a comment beside its glyphs.
    for plot in plots:
        assert f"<!-- {caption} -->" in plot.read_text(), plot.name


def test_dataset_metric_selection_supports_auroc_and_f1(
    eval_perf_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    config = replace(
        plotting_config,
        dataset_metrics={"cube": "auroc"},
        default_metric="f1",
    )
    plotter = EvaluationPlotter(
        load_plotting_ready_table(eval_perf_path), tmp_path, config
    )
    assert plotter.df_stop_action is not None
    cube = plotter.df_stop_action.filter(pl.col("dataset") == "cube").sort(
        "eval_hard_budget"
    )
    assert cube["mean_metric"].is_nan().to_list() == [True, False]
    assert cube["mean_metric"][1] == pytest.approx(0.75)
    assert cube["std_metric"][1] == pytest.approx(0.5 * SQRT_HALF)
    mnist = plotter.df_stop_action.filter(pl.col("dataset") == "mnist")
    assert mnist["mean_metric"][0] == pytest.approx(1 / 3)
    plot = get_normal_hard_budget_plot(plotter.df_stop_action, config)
    assert set(plot.data["dataset"]) == {"Cube (AUROC)", "MNIST (F1)"}


@pytest.mark.parametrize(
    "changes", [{"default_metric": "ap"}, {"dataset_metrics": {"cube": "F1"}}]
)
def test_invalid_metric_selection_names_the_bad_value(
    plotting_config: PlottingDisplayConfig, changes: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match=r"ap|F1"):
        replace(plotting_config, **changes)


def test_per_dataset_figure_facets_classifiers_and_all_metrics(
    plotting_config: PlottingDisplayConfig,
) -> None:
    summary = pl.DataFrame(
        [
            {
                "dataset": "cube",
                "afa_method": method,
                "classifier": classifier,
                "eval_hard_budget": 2.0,
                "mean_accuracy": accuracy,
                "std_accuracy": 0.1,
                "mean_f_score": 0.4,
                "std_f_score": 0.2,
                "mean_auroc": auroc,
                "std_auroc": 0.05,
            }
            for method, classifier, accuracy, auroc in [
                ("jafa", "external", 0.75, 0.9),
                ("gdfs", "external", 0.6, 0.8),
                ("jafa", "builtin", 0.5, float("nan")),
            ]
        ]
    )
    plot = get_per_dataset_plot(
        summary, "eval_hard_budget", "Hard budget", plotting_config
    )
    assert set(plot.data["metric"]) == {"Accuracy", "Macro F1", "AUROC"}
    builtin = plot.data.filter(pl.col("classifier") == "Built-in classifier")
    assert set(builtin["afa_method"]) == {"JAFA"}
    assert (
        builtin.filter(pl.col("metric") == "Accuracy")["mean_metric"][0] == 0.5
    )
    assert (
        builtin.filter(pl.col("metric") == "AUROC")["mean_metric"]
        .is_nan()
        .all()
    )
    external = plot.data.filter(
        (pl.col("classifier") == "External classifier")
        & (pl.col("metric") == "AUROC")
    )
    assert external["mean_metric"].to_list() == [0.9, 0.8]
    # Drawing observes the actual facet layout and axis sharing.
    figure = plot.draw()
    assert len(figure.axes) == 6
    assert len({ax.get_xlim() for ax in figure.axes}) == 1
    assert len({ax.get_ylim() for ax in figure.axes}) > 1
    assert figure.axes[0].get_position().y0 > figure.axes[3].get_position().y0


def test_per_dataset_figure_omits_unavailable_builtin_row(
    plotting_config: PlottingDisplayConfig,
) -> None:
    summary = pl.DataFrame(
        [
            {
                "dataset": "cube",
                "afa_method": "gdfs",
                "classifier": "external",
                "eval_hard_budget": 2.0,
                "mean_accuracy": 0.75,
                "std_accuracy": 0.1,
                "mean_f_score": 0.7,
                "std_f_score": 0.1,
                "mean_auroc": 0.9,
                "std_auroc": 0.1,
            }
        ]
    )
    plot = get_per_dataset_plot(
        summary, "eval_hard_budget", "Hard budget", plotting_config
    )
    assert len(plot.draw().axes) == 3
    assert set(plot.data["classifier"]) == {"External classifier"}


def test_per_dataset_families_are_additional_and_keep_budget_variants(
    eval_perf_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    table = load_plotting_ready_table(eval_perf_path).with_columns(
        pl.lit("external").alias("classifier")
    )
    # Only JAFA has built-in results; these must not affect headline summaries.
    builtin = table.filter(pl.col("afa_method") == "jafa").with_columns(
        pl.lit("builtin").alias("classifier"),
        pl.lit([0.9, 0.1]).alias("predicted_probabilities"),
    )
    unavailable = table.filter(pl.col("afa_method") == "gdfs").with_columns(
        pl.lit("builtin").alias("classifier"),
        pl.lit(None, dtype=pl.List(pl.Float64)).alias(
            "predicted_probabilities"
        ),
    )
    both = pl.concat([table, builtin, unavailable])
    output = tmp_path / "usual"
    additional = tmp_path / "per_dataset"
    plotter = EvaluationPlotter(
        table,
        output,
        replace(plotting_config, caption="Additional family"),
        formats=("svg",),
    )
    plotter.generate_all_plots({"all": ["cube", "mnist"]})
    normal_before = {p: p.read_bytes() for p in output.rglob("*.svg")}
    plotter.generate_per_dataset_plots(both, additional)
    assert {p: p.read_bytes() for p in output.rglob("*.svg")} == normal_before
    assert sorted(
        str(p.relative_to(additional)) for p in additional.rglob("*.svg")
    ) == [
        "dataset-cube/hard_budget_normal.svg",
        "dataset-cube/hard_budget_traj.svg",
        "dataset-mnist/soft_budget_2d_errors.svg",
        "dataset-mnist/soft_budget_lines.svg",
    ]
    cube = (additional / "dataset-cube/hard_budget_normal.svg").read_text()
    mnist = (additional / "dataset-mnist/soft_budget_lines.svg").read_text()
    assert "Built-in classifier" in cube
    assert "External classifier" in cube
    assert "Built-in classifier" not in mnist
    for metric in ["Accuracy", "Macro F1", "AUROC"]:
        assert metric in cube
        assert metric in mnist


def test_plotting_script_opt_in_reads_unsplit_classifier_source(
    eval_perf_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    table = load_plotting_ready_table(eval_perf_path).with_columns(
        pl.lit("external").alias("classifier")
    )
    builtin = table.filter(pl.col("afa_method") == "jafa").with_columns(
        pl.lit("builtin").alias("classifier")
    )
    both_path = tmp_path / "both.parquet"
    pl.concat([table, builtin]).write_parquet(both_path)
    config = PlotEvalPerfConfig(
        input=str(eval_perf_path),
        output_folder=str(tmp_path / "usual"),
        formats=["svg"],
        dataset_sets={},
        plotting=plotting_config,
        per_dataset_plots=True,
        per_dataset_input=str(both_path),
        per_dataset_output_folder=str(tmp_path / "additional"),
    )
    main(OmegaConf.structured(config))
    plot = (
        tmp_path / "additional/dataset-cube/hard_budget_normal.svg"
    ).read_text()
    assert "External classifier" in plot
    assert "Built-in classifier" in plot
    assert not list((tmp_path / "usual").rglob("*.svg"))


@pytest.mark.parametrize("relative", ["usual", "usual/per_dataset", "."])
def test_per_dataset_output_cannot_overlap_usual_directory(
    eval_perf_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
    relative: str,
) -> None:
    table = load_plotting_ready_table(eval_perf_path).with_columns(
        pl.lit("external").alias("classifier")
    )
    plotter = EvaluationPlotter(table, tmp_path / "usual", plotting_config)
    with pytest.raises(ValueError, match=r"overlap.*usual"):
        plotter.generate_per_dataset_plots(table, tmp_path / relative)
    assert not (tmp_path / "usual").exists()


def test_default_display_metrics_preserve_headline_choices() -> None:
    config = OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(PlottingDisplayConfig),
            OmegaConf.load(
                REPO_ROOT / "conf/scripts/plotting/common/default.yaml"
            ),
        )
    )
    assert isinstance(config, PlottingDisplayConfig)
    assert config.dataset_metrics == {
        "physionet_challenge_2012": "f1",
        "bank_marketing": "f1",
    }
    assert config.default_metric == "accuracy"
    script = PlotEvalPerfConfig(
        input="input.parquet",
        output_folder="usual",
        formats=[],
        dataset_sets={},
        plotting=config,
    )
    assert script.per_dataset_plots is False
    assert script.per_dataset_input is None
    assert script.per_dataset_output_folder is None


def test_metric_columns_are_selected_without_changing_the_input(
    plotting_config: PlottingDisplayConfig,
) -> None:
    summary = pl.DataFrame(
        {
            "dataset": ["cube", "mnist", "other"],
            "mean_accuracy": [0.75, 0.6, 0.8],
            "std_accuracy": [0.1, 0.2, None],
            "mean_f_score": [0.5, 0.4, 0.7],
            "std_f_score": [0.2, 0.3, None],
            "mean_auroc": [None, 0.9, 0.8],
            "std_auroc": [None, 0.1, None],
        }
    )
    config = replace(
        plotting_config,
        dataset_metrics={"cube": "auroc", "mnist": "f1"},
    )

    selected = add_metric_column(summary, config)

    assert isinstance(selected, pl.DataFrame)
    assert selected["mean_metric"].to_list() == [None, 0.4, 0.8]
    assert selected["std_metric"].to_list() == [None, 0.3, None]
    assert "mean_metric" not in summary.columns
    empty = add_metric_column(summary.clear(), config)
    assert empty.is_empty()
    assert empty.schema == selected.schema


@pytest.mark.parametrize("empty", [False, True])
def test_unavailable_predictions_have_no_plot_summaries(
    plotting_config: PlottingDisplayConfig,
    empty: bool,
) -> None:
    table = pl.DataFrame(
        {
            "action_performed": [0, None, 1],
            "predicted_probabilities": [None, None, None],
        },
        schema={
            "action_performed": pl.UInt64,
            "predicted_probabilities": pl.List(pl.Float64),
        },
    )
    if empty:
        table = table.clear()

    assert (
        process_df_only_stop_action_and_valid_prediction(
            table, plotting_config
        )
        is None
    )
    assert process_df_every_action(table, plotting_config) is None


@pytest.mark.parametrize("empty", [False, True])
def test_plotter_without_predictions_writes_no_figures(
    eval_perf_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
    empty: bool,
) -> None:
    table = load_plotting_ready_table(eval_perf_path).with_columns(
        pl.lit(None, dtype=pl.List(pl.Float64)).alias(
            "predicted_probabilities"
        ),
        pl.lit("external").alias("classifier"),
    )
    if empty:
        table = table.clear()
    output = tmp_path / "usual"
    additional = tmp_path / "additional"

    plotter = EvaluationPlotter(table, output, plotting_config)
    plotter.generate_all_plots({"all": ["cube", "mnist"]})
    plotter.generate_per_dataset_plots(table, additional)

    assert plotter.df_stop_action is None
    assert plotter.df_traj is None
    assert not output.exists()
    assert not additional.exists()


def test_performance_plot_preserves_method_order_and_null_numeric_values(
    plotting_config: PlottingDisplayConfig,
) -> None:
    summary = pl.DataFrame(
        {
            "dataset": ["cube", "cube"],
            "afa_method": ["jafa", "gdfs"],
            "eval_hard_budget": [1, 2],
            "mean_metric": [0.75, 0.6],
            "low_metric": [None, None],
            "high_metric": [None, None],
        },
        schema_overrides={"eval_hard_budget": pl.UInt64},
    )

    plot = get_normal_hard_budget_plot(
        summary, plotting_config, method_order=["gdfs", "jafa"]
    )

    assert isinstance(plot.data, pl.DataFrame)
    assert plot.data["afa_method"].to_list() == ["JAFA", "GDFS"]
    assert plot.data["policy_type"].to_list() == ["Non-myopic", "Myopic"]
    assert plot.scales.get_scales("color").breaks == ["GDFS", "JAFA"]
    assert plot.scales.get_scales("fill").breaks == ["GDFS", "JAFA"]
    assert len(plot.draw().axes) == 1
