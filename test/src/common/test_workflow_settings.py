"""
The workflow config loader: typed settings, rejected at load time when wrong.

Each config mistake below once surfaced late, as a `KeyError` while
Snakemake built the DAG, or never, as a silently different benchmark.
"""

import warnings
from pathlib import Path

import pytest
import yaml
from snakemake.cli import parse_config

from afabench.core.output_layout import OutputLayout
from afabench.core.workflow_settings import ScriptInvocation, load_config
from afabench.release.workflow_config import resolve_workflow_config

REPO_ROOT = Path(__file__).parents[3]
LAYOUT = OutputLayout(root="output", initializer="cold", eval_split="test")


@pytest.mark.parametrize("profile", ["all", "kdd26"])
def test_shipped_configs_load(profile: str) -> None:
    profile_file = REPO_ROOT / "workflow/profiles/pipeline" / profile
    configfiles = yaml.safe_load((profile_file / "config.yaml").read_text())[
        "configfile"
    ]
    record = resolve_workflow_config(
        profile=None,
        configfiles=[REPO_ROOT / path for path in configfiles],
        overrides=[],
    )

    settings = load_config(record.merged)

    assert settings.methods


def _shipped_all_profile(
    soft_budget_profile: str, eval_hard_budget_profile: str = "all"
) -> dict[str, object]:
    """Return the `all` profile's merged config, with other budget profiles."""
    configfiles = [
        path.replace(
            "soft_budget_params/all.yaml",
            f"soft_budget_params/{soft_budget_profile}.yaml",
        ).replace(
            "eval_hard_budgets/all.yaml",
            f"eval_hard_budgets/{eval_hard_budget_profile}.yaml",
        )
        for path in yaml.safe_load(
            (
                REPO_ROOT / "workflow/profiles/pipeline/all/config.yaml"
            ).read_text()
        )["configfile"]
    ]
    merged = dict(
        resolve_workflow_config(
            profile=None,
            configfiles=[REPO_ROOT / path for path in configfiles],
            overrides=[],
        ).merged
    )
    # The smaller soft-budget profiles configure only some methods
    methods = [
        method
        for method in merged["methods"]
        if method in merged["soft_budget_params"]
    ]
    return merged | {"methods": methods, "method_sets": {}}


@pytest.mark.parametrize("soft_budget_profile", ["all", "fast", "single"])
def test_no_method_runs_metabric_in_the_soft_budget_setting(
    soft_budget_profile: str,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        settings = load_config(_shipped_all_profile(soft_budget_profile))

    on_metabric = {
        method: per_dataset["metabric"]
        for method, per_dataset in settings.budget_params.items()
    }
    for method, combinations in on_metabric.items():
        assert combinations, method
        assert all(eval_hard != "null" for _, eval_hard, _, _ in combinations)


@pytest.mark.parametrize("eval_hard_budget_profile", ["all", "fast", "single"])
def test_l2m_trains_one_policy_for_every_metabric_hard_budget(
    eval_hard_budget_profile: str,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        settings = load_config(
            _shipped_all_profile("all", eval_hard_budget_profile)
        )

    combinations = settings.budget_params["l2m"]["metabric"]
    assert {train for train, _, _, _ in combinations} == {7}


L2M_RUNS = {
    "l2m": "",
    "l2m_synth": "task_prior=synthetic",
    "l2m_better_synth": "task_prior=tabicl_v1 prior_type=prior_bag_plus_tree",
}


@pytest.mark.parametrize(("method", "params"), L2M_RUNS.items())
def test_each_l2m_run_trains_one_policy_per_dataset_for_every_budget(
    method: str, params: str
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        settings = load_config(_shipped_all_profile("all"))

    runs = {
        dataset: combinations
        for dataset, combinations in settings.budget_params[method].items()
        if combinations
    }
    # The model's width is fixed by the dataset: one model per dataset.
    assert set(runs) == {"cube", "miniboone_subset", "metabric"}
    trained_at = {
        dataset: {train for train, _, _, _ in combinations}
        for dataset, combinations in runs.items()
    }
    assert trained_at == {
        "cube": {10},
        "miniboone_subset": {7},
        "metabric": {7},
    }
    # The task prior is the only difference between the runs, and both
    # stages use the same one.
    assert settings.pretraining_script(method).script_params == params
    assert settings.training_script(method).script_params == params
    assert settings.pretraining_script(method).script_name == "l2m"
    assert settings.training_script(method).script_name == "l2m"


def test_the_l2m_runs_have_their_own_pretrained_models() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        settings = load_config(_shipped_all_profile("all"))

    runs = list(L2M_RUNS)
    # A model trained on one task prior is not another run's model.
    assert {
        settings.method_to_pretrained_model[method] for method in runs
    } == set(runs)


def test_load_config_uses_pipeline_defaults() -> None:
    settings = load_config(_config())

    assert settings.eval_dataset_split == "test"
    assert settings.dataset_realization_indices == [0, 1, 2, 3, 4]
    assert settings.smoke_test is False
    assert settings.use_wandb is False
    assert settings.list_jobs is False
    assert settings.soft_budget_setting is True
    assert settings.per_dataset_plots is False
    assert settings.reuse_release is False
    assert settings.plotting.dataset_metrics == {}
    assert settings.plotting.default_metric is None
    assert settings.initializer == "cold"
    assert settings.output_root == "output/production"


def test_realization_override_is_normalized_for_release_identity_checks() -> (
    None
):
    override = parse_config(["dataset_realization_indices=[0,1,7]"])
    settings = load_config(_config() | override)
    assert settings.dataset_realization_indices == [0, 1, 7]


@pytest.mark.parametrize("indices", [[True], [-1], [0.5], ["invalid"], "0,1"])
def test_realizations_require_nonnegative_integer_indices(
    indices: object,
) -> None:
    with pytest.raises(
        (TypeError, ValueError), match="dataset_realization_indices"
    ):
        load_config(_config() | {"dataset_realization_indices": indices})


def test_every_pipeline_stage_runs_by_default() -> None:
    stages = load_config(_config()).stages

    assert stages.dataset_generation is True
    assert stages.training is True
    assert stages.evaluation is True


def test_stages_turn_off_only_the_named_stages() -> None:
    stages = load_config(
        _config()
        | {"stages": {"dataset_generation": False, "training": False}}
    ).stages

    assert stages.dataset_generation is False
    assert stages.training is False
    assert stages.evaluation is True


def test_stages_read_the_strings_a_config_override_passes() -> None:
    # Snakemake passes the values of a mapping override as strings.
    override = parse_config(
        [
            "stages={dataset_generation: False, training: false, evaluation: TRUE}"
        ]
    )

    stages = load_config(_config() | override).stages

    assert stages.dataset_generation is False
    assert stages.training is False
    assert stages.evaluation is True


@pytest.mark.parametrize(
    ("stages", "error", "message"),
    [
        ({"trainig": False}, ValueError, r"stages.*'trainig'"),
        ({"pretraining": False}, ValueError, r"stages.*'pretraining'"),
        ({"training": "no"}, ValueError, r"stages\.training.*'no'"),
        ({"evaluation": 0}, ValueError, r"stages\.evaluation.*0"),
        ({"evaluation": None}, ValueError, r"stages\.evaluation.*None"),
        (["training"], TypeError, r"stages.*\['training'\]"),
        ("training=false", TypeError, r"stages.*'training=false'"),
    ],
)
def test_invalid_stages_are_rejected(
    stages: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        load_config(_config() | {"stages": stages})


@pytest.mark.parametrize(
    ("value", "expected", "output_root"),
    [
        ("false", False, "output/production"),
        ("true", True, "output/smoke"),
        ("False", False, "output/production"),
        ("True", True, "output/smoke"),
        ("fAlSe", False, "output/production"),
        ("TrUe", True, "output/smoke"),
    ],
)
def test_smoke_test_override_selects_the_correct_output_root(
    *, value: str, expected: bool, output_root: str
) -> None:
    override = parse_config([f"smoke_test={value}"])

    settings = load_config(_config() | override)

    assert settings.output_root == output_root
    assert settings.smoke_test is expected


@pytest.mark.parametrize("key", ["use_wandb", "list_jobs"])
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("false", False),
        ("true", True),
        ("False", False),
        ("True", True),
        ("fAlSe", False),
        ("TrUe", True),
    ],
)
def test_logging_overrides_resolve_to_booleans(
    *, key: str, value: str, expected: bool
) -> None:
    settings = load_config(
        _config() | {"wandb_group": "g"} | parse_config([f"{key}={value}"])
    )

    assert getattr(settings, key) is expected


def test_wandb_logging_needs_a_wandb_group() -> None:
    with pytest.raises(ValueError, match="needs wandb_group"):
        load_config(_config() | {"use_wandb": True})


def test_a_numeric_wandb_group_override_is_a_name() -> None:
    override = parse_config(["use_wandb=true", "wandb_group=2026"])

    assert load_config(_config() | override).wandb_group == "2026"


@pytest.mark.parametrize(
    "key",
    [
        "smoke_test",
        "use_wandb",
        "list_jobs",
        "per_dataset_plots",
        "reuse_release",
    ],
)
@pytest.mark.parametrize("value", ["no", "0", "1", "null", "[]"])
def test_invalid_boolean_overrides_are_rejected(key: str, value: str) -> None:
    override = parse_config([f"{key}={value}"])

    with pytest.raises(ValueError, match=rf"Expected {key}") as error:
        load_config(_config() | override)

    assert str(error.value) == (
        f"Expected {key} to be true or false, got {override[key]!r}."
    )


@pytest.mark.parametrize(
    "key",
    [
        "smoke_test",
        "use_wandb",
        "list_jobs",
        "per_dataset_plots",
        "reuse_release",
    ],
)
def test_null_boolean_settings_are_rejected(key: str) -> None:
    with pytest.raises(ValueError, match=rf"{key}.*None"):
        load_config(_config() | {key: None})


@pytest.mark.parametrize(
    ("value", "output_root"),
    [("false", "output/smoke"), ("true", "output/production")],
)
def test_smoke_test_override_cannot_write_into_the_other_output_root(
    value: str, output_root: str
) -> None:
    override = parse_config(
        [f"smoke_test={value}", f"output_root={output_root}"]
    )

    with pytest.raises(ValueError, match=rf"smoke_test={value}.*output_root"):
        load_config(_config() | override)


def test_smoke_test_writes_under_its_own_output_root() -> None:
    settings = load_config(_config() | {"smoke_test": True})

    assert settings.output_root == "output/smoke"


@pytest.mark.parametrize("smoke_test", [False, True])
def test_an_explicit_output_root_wins(*, smoke_test: bool) -> None:
    settings = load_config(
        _config() | {"smoke_test": smoke_test, "output_root": "elsewhere"}
    )

    assert settings.output_root == "elsewhere"


@pytest.mark.parametrize(
    ("smoke_test", "output_root"),
    [
        (True, "output/production"),
        (True, "output/production/"),
        (True, str(Path("output/production").resolve())),
        (True, "output/production/nested"),
        (True, "output"),
        (False, "output/smoke"),
        (False, "output/smoke/nested"),
        (False, "output"),
    ],
)
def test_an_output_root_cannot_overlap_the_other_kind_of_run(
    *, smoke_test: bool, output_root: str
) -> None:
    config = _config() | {"smoke_test": smoke_test, "output_root": output_root}

    with pytest.raises(ValueError, match=r"output_root"):
        load_config(config)


@pytest.mark.parametrize(
    ("smoke_test", "output_root"),
    [
        (True, "output/smoke/nested"),
        (False, "output/production/nested"),
        (True, "/scratch/afabench/output"),
        (False, "output_elsewhere"),
    ],
)
def test_an_output_root_apart_from_the_other_kind_of_run_is_allowed(
    *, smoke_test: bool, output_root: str
) -> None:
    config = _config() | {"smoke_test": smoke_test, "output_root": output_root}

    assert load_config(config).output_root == output_root


def test_aaco_eval_batch_size_is_pinned_per_dataset() -> None:
    """AACO batches its acquisition, so all.yaml pins its eval batch size."""
    method_options = yaml.safe_load(
        (REPO_ROOT / "workflow/conf/method_options/all.yaml").read_text()
    )["method_options"]
    config = _config(
        method_options=method_options,
        methods=["aaco", "aaco_nn"],
        datasets=["cube", "mnist", "fashion_mnist", "synthetic_mnist"],
        pretrain_mapping={"aaco": {"pretrain_script_name": "aaco"}},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        settings = load_config(config)

    # Image datasets get the smaller pinned size, every other dataset the
    # default. The loader keeps the `default` key alongside the datasets.
    expected = {
        "default": 128,
        "cube": 128,
        "mnist": 32,
        "fashion_mnist": 32,
        "synthetic_mnist": 32,
        "synthetic_mnist_without_noise": 32,
    }
    assert settings.eval_batch_sizes["aaco"] == expected
    assert settings.eval_batch_sizes["aaco_nn"] == expected


def test_missing_eval_batch_size_warns_and_falls_back_to_one() -> None:
    config = _config(
        method_options={"unbatched": {"train_script_name": "unbatched"}},
        methods=["unbatched"],
        datasets=["cube", "mnist"],
    )

    with pytest.warns(UserWarning, match=r"unbatched.*eval_batch_size"):
        settings = load_config(config)

    assert settings.eval_batch_sizes["unbatched"] == {"cube": 1, "mnist": 1}


def test_eval_batch_size_mapping_without_default_is_rejected() -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": {"cube": 8},
            }
        }
    )

    with pytest.raises(ValueError, match=r"'alpha'.*eval_batch_size.*default"):
        load_config(config)


@pytest.mark.parametrize(
    "typo",
    [
        "pretrained_model_nme",
        "hard_budget_ignored_dataset",
        "soft_budget_dataset",
        "eval_batch_sise",
    ],
)
def test_misspelt_method_option_is_rejected(typo: str) -> None:
    config = _config(
        method_options={
            "alpha": {"train_script_name": "alpha", typo: "anything"}
        }
    )

    with pytest.raises(ValueError, match=rf"'alpha'.*{typo}"):
        load_config(config)


# A config passed through JSON, as a release config is, has string keys.
@pytest.mark.parametrize("eval_budget_key", [1, "1"])
def test_eval_to_train_hard_budget_mapping_sets_the_train_hard_budget(
    eval_budget_key: int | str,
) -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": 1,
                "eval_to_train_hard_budget_mapping": {
                    "cube": {eval_budget_key: 3}
                },
            }
        }
    )

    settings = load_config(config)

    assert (3, 1, "null", "null") in settings.budget_params["alpha"]["cube"]


def test_partial_eval_to_train_hard_budget_mapping_is_rejected() -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": 1,
                "eval_to_train_hard_budget_mapping": {"cube": {1: 3}},
            }
        }
    )
    config["eval_hard_budgets"] = {"default": [1, 2]}

    with pytest.raises(
        ValueError, match=r"'alpha'.*'cube'.*eval hard budgets \[2\]"
    ):
        load_config(config)


def test_partial_mapping_is_allowed_on_a_hard_budget_ignored_dataset() -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": 1,
                "eval_to_train_hard_budget_mapping": {"cube": {2: 3}},
                "hard_budget_ignored_datasets": ["cube"],
                "soft_budget_datasets": ["cube"],
            }
        }
    )

    settings = load_config(config)

    assert settings.budget_params["alpha"]["cube"] == [
        ("null", "null", 0.1, 0.1)
    ]


def test_train_hard_budget_trains_every_eval_hard_budget_at_it() -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": 1,
                "use_max_hard_budget_when_training_soft_budget": True,
                "train_hard_budget": {"cube": 7},
                "soft_budget_datasets": ["cube", "mnist"],
            }
        },
        datasets=["cube", "mnist"],
    )
    config["eval_hard_budgets"] = {"default": [1, 3]}

    settings = load_config(config)

    assert settings.budget_params["alpha"] == {
        "cube": [
            (7, 1, "null", "null"),
            (7, 3, "null", "null"),
            (7, "null", 0.1, 0.1),
        ],
        "mnist": [
            (1, 1, "null", "null"),
            (3, 3, "null", "null"),
            (3, "null", 0.1, 0.1),
        ],
    }


def test_train_hard_budget_and_mapping_for_one_dataset_are_rejected() -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": 1,
                "train_hard_budget": {"cube": 7},
                "eval_to_train_hard_budget_mapping": {"cube": {1: 7}},
            }
        }
    )

    with pytest.raises(ValueError, match=r"'alpha'.*\['cube'\].*give one"):
        load_config(config)


def test_method_without_train_script_name_is_rejected() -> None:
    config = _config(method_options={"alpha": {"eval_batch_size": 1}})

    with pytest.raises(ValueError, match=r"'alpha'.*train_script_name"):
        load_config(config)


def test_pretrained_model_missing_from_pretrain_mapping_is_rejected() -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "pretrained_model_name": "shard",
                "eval_batch_size": 1,
            }
        },
        pretrain_mapping={"shared": {"pretrain_script_name": "shared"}},
    )

    with pytest.raises(
        ValueError, match=r"'alpha'.*'shard'.*pretrain_mapping"
    ):
        load_config(config)


@pytest.mark.parametrize(
    "model_config",
    [
        {"pretrain_params": []},
        {"pretrain_script_name": "shared", "pretrain_parms": []},
    ],
)
def test_malformed_pretrain_mapping_entry_is_rejected(
    model_config: dict[str, object],
) -> None:
    config = _config(pretrain_mapping={"shared": model_config})

    with pytest.raises(ValueError, match=r"pretrain_mapping\['shared'\]"):
        load_config(config)


def test_method_missing_from_method_options_is_rejected() -> None:
    config = _config(methods=["alpha", "beta"])

    with pytest.raises(ValueError, match=r"'beta'.*method_options"):
        load_config(config)


def test_method_missing_from_soft_budget_params_is_rejected() -> None:
    config = _config()
    config["soft_budget_params"] = {}

    with pytest.raises(ValueError, match=r"'alpha'.*soft_budget_params"):
        load_config(config)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("hard_budget_ignored_datasets", ["imagenete"]),
        ("soft_budget_datasets", ["imagenete"]),
        ("train_hard_budget", {"imagenete": 7}),
        ("eval_to_train_hard_budget_mapping", {"imagenete": {1: 7}}),
    ],
)
def test_unknown_dataset_key_in_method_options_is_rejected(
    option: str, value: object
) -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": 1,
                option: value,
            }
        }
    )

    with pytest.raises(ValueError, match=rf"'alpha'.*{option}.*'imagenete'"):
        load_config(config)


def test_ignored_dataset_need_not_be_in_this_run() -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": 1,
                "hard_budget_ignored_datasets": ["imagenette"],
                "soft_budget_datasets": ["cube"],
            }
        },
        datasets=["cube"],
    )

    settings = load_config(config)

    assert settings.budget_params["alpha"]["cube"] == [
        (1, 1, "null", "null"),
        ("null", "null", 0.1, 0.1),
    ]


def test_dataset_not_listed_for_soft_budgets_gets_no_soft_budget_runs() -> (
    None
):
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": 1,
                "soft_budget_datasets": ["cube"],
            }
        },
        datasets=["cube", "mnist"],
    )
    config["soft_budget_params"] = {
        "alpha": {"default": [[0.1, 0.1]], "mnist": [[0.2, 0.2]]}
    }

    settings = load_config(config)

    assert settings.budget_params["alpha"] == {
        "cube": [(1, 1, "null", "null"), ("null", "null", 0.1, 0.1)],
        "mnist": [(1, 1, "null", "null")],
    }


def test_pretraining_skips_datasets_a_method_runs_in_neither_setting() -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": 1,
                "pretrained_model_name": "alpha_model",
                "hard_budget_ignored_datasets": ["cube", "mnist"],
                "soft_budget_datasets": ["mnist"],
            }
        },
        datasets=["cube", "mnist"],
        pretrain_mapping={"alpha_model": {"pretrain_script_name": "alpha"}},
    )

    settings = load_config(config)

    assert settings.datasets_used_per_pretrain_name == {
        "alpha_model": ["mnist"]
    }


@pytest.mark.parametrize("soft_budget_setting", [False, "false"])
def test_soft_budget_setting_off_leaves_only_hard_budget_runs(
    soft_budget_setting: object,
) -> None:
    config = _config(datasets=["cube", "mnist"]) | {
        "soft_budget_setting": soft_budget_setting
    }

    settings = load_config(config)

    assert settings.budget_params["alpha"] == {
        "cube": [(1, 1, "null", "null")],
        "mnist": [(1, 1, "null", "null")],
    }


def test_soft_budget_setting_off_skips_pretraining_for_hard_ignored_datasets() -> (
    None
):
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": 1,
                "pretrained_model_name": "alpha_model",
                "hard_budget_ignored_datasets": ["mnist"],
            }
        },
        datasets=["cube", "mnist"],
        pretrain_mapping={"alpha_model": {"pretrain_script_name": "alpha"}},
    ) | {"soft_budget_setting": False}

    settings = load_config(config)

    assert settings.budget_params["alpha"]["mnist"] == []
    assert settings.datasets_used_per_pretrain_name == {
        "alpha_model": ["cube"]
    }


def test_invalid_soft_budget_setting_is_rejected() -> None:
    config = _config() | {"soft_budget_setting": "no"}

    with pytest.raises(ValueError, match="soft_budget_setting"):
        load_config(config)


def test_misspelt_classifier_key_is_rejected() -> None:
    config = _config()
    config["classifier_names"] = {
        "default": {"script_name": "masked_mlp_classifier", "script_parms": []}
    }

    with pytest.raises(ValueError, match=r"'default'.*script_parms"):
        load_config(config)


def _compared_methods_config() -> dict[str, object]:
    """
    Run two of three configured methods.

    Alpha has no pretraining stage and no built-in classifier, beta has
    both, and gamma is configured but not run.
    """
    config = _config(
        method_options={
            "alpha": {"train_script_name": "alpha", "eval_batch_size": 1},
            "beta": {
                "train_script_name": "beta",
                "pretrained_model_name": "shared",
                "classifier": {"script_name": "beta_classifier"},
                "eval_batch_size": 1,
                "soft_budget_datasets": ["cube"],
            },
            "gamma": {
                "train_script_name": "gamma",
                "pretrained_model_name": "shared",
            },
        },
        methods=["alpha", "beta"],
        pretrain_mapping={"shared": {"pretrain_script_name": "shared"}},
    )
    config["soft_budget_params"] = {
        method: {"default": [[0.1, 0.2]]}
        for method in ["alpha", "beta", "gamma"]
    }
    return config


def test_method_sets_hold_only_the_methods_run() -> None:
    config = _compared_methods_config()
    config["method_sets"] = {"pair": ["alpha", "gamma"], "other": ["gamma"]}

    method_sets = load_config(config).method_sets

    assert method_sets == {"all": ["alpha", "beta"], "pair": ["alpha"]}


def test_configured_all_method_set_is_rejected() -> None:
    config = _config() | {"method_sets": {"all": ["alpha"]}}

    with pytest.raises(ValueError, match="'all'"):
        load_config(config)


def test_dataset_sets_keep_only_the_datasets_of_the_run() -> None:
    config = _config(datasets=["cube", "mnist"]) | {
        "dataset_sets": {
            "pair": ["cube", "ckd"],
            "unrun": ["ckd"],
        }
    }

    dataset_sets = load_config(config).dataset_sets

    assert dataset_sets == {"all": ["cube", "mnist"], "pair": ["cube"]}


def test_configured_all_dataset_set_is_rejected() -> None:
    config = _config() | {"dataset_sets": {"all": ["cube"]}}

    with pytest.raises(ValueError, match="'all'"):
        load_config(config)


@pytest.mark.parametrize(
    ("method", "pretrain_folder"),
    [
        ("alpha", "NO_PRETRAIN"),
        ("beta", "pretrain_seed-2"),
    ],
)
def test_evaluation_run_is_seeded_with_its_dataset_realization_index(
    method: str, pretrain_folder: str
) -> None:
    settings = load_config(_compared_methods_config())

    run = settings.evaluation_run(
        method=method,
        dataset="cube",
        dataset_realization_index=2,
        budget_combination=("null", "null", 0.1, 0.2),
    )

    assert run.training.pretrain_folder == pretrain_folder
    assert run.training.train_seed == 2
    assert run.eval_seed == 2
    assert (run.training.train_hard_budget, run.eval_hard_budget) == (
        "null",
        "null",
    )
    assert (
        run.training.train_soft_budget_param,
        run.eval_soft_budget_param,
    ) == (0.1, 0.2)


@pytest.mark.parametrize(
    ("method", "bundle"),
    [
        (
            "alpha",
            "output/trained_classifiers/initializer-cold/"
            "dataset-cube+realization_index-2.bundle",
        ),
        (
            "beta",
            "output/trained_classifiers/initializer-cold/"
            "method-beta+dataset-cube+realization_index-2.bundle",
        ),
    ],
)
def test_classifier_bundle_is_the_built_in_one_if_the_method_has_one(
    method: str, bundle: str
) -> None:
    settings = load_config(_compared_methods_config())

    assert (
        settings.classifier_bundle(
            LAYOUT, method=method, dataset="cube", dataset_realization_index=2
        )
        == bundle
    )


def test_training_script_passes_the_method_specific_params() -> None:
    config = _compared_methods_config()
    config["method_options"]["beta"]["method_specific_params"] = [  # type: ignore[index]
        "lr=0.1",
        "epochs=2",
    ]

    assert load_config(config).training_script("beta") == ScriptInvocation(
        script_name="beta", script_params="lr=0.1 epochs=2"
    )


def test_pretraining_script_passes_the_pretrain_params() -> None:
    config = _compared_methods_config()
    config["pretrain_mapping"] = {
        "shared": {
            "pretrain_script_name": "shared_vae",
            "pretrain_params": ["latent=4"],
        }
    }

    assert load_config(config).pretraining_script(
        "shared"
    ) == ScriptInvocation(script_name="shared_vae", script_params="latent=4")


@pytest.mark.parametrize(
    ("method", "pretrain_folder", "bundles"),
    [
        (
            "beta",
            "pretrain_seed-2",
            [
                "output/pretrained_models/initializer-cold/shared/"
                "dataset-cube+realization_index-2/pretrain_seed-2/model.bundle"
            ],
        ),
        ("alpha", "NO_PRETRAIN", []),
    ],
)
def test_pretrained_model_bundle_is_the_methods_pretrained_model_if_any(
    method: str, pretrain_folder: str, bundles: list[str]
) -> None:
    settings = load_config(_compared_methods_config())

    assert (
        settings.pretrained_model_bundle(
            LAYOUT,
            method=method,
            dataset="cube",
            dataset_realization_index=2,
            pretrain_folder=pretrain_folder,
        )
        == bundles
    )


@pytest.mark.parametrize(
    ("method", "pretrain_folder", "message"),
    [
        (
            "beta",
            "NO_PRETRAIN",
            r"'beta' has a pretraining stage.*pretrain_seed-<seed>/.*"
            r"NO_PRETRAIN/",
        ),
        (
            "alpha",
            "pretrain_seed-2",
            r"'alpha' has no pretraining stage.*NO_PRETRAIN/.*"
            r"pretrain_seed-2/",
        ),
    ],
)
def test_pretrained_model_bundle_rejects_the_other_pretrain_folder(
    method: str, pretrain_folder: str, message: str
) -> None:
    settings = load_config(_compared_methods_config())

    with pytest.raises(ValueError, match=message):
        settings.pretrained_model_bundle(
            LAYOUT,
            method=method,
            dataset="cube",
            dataset_realization_index=2,
            pretrain_folder=pretrain_folder,
        )


def test_unmasker_falls_back_to_the_default() -> None:
    config = _config(datasets=["cube", "mnist"])
    config["unmaskers"] = {"default": "direct", "mnist": "image_patch"}
    settings = load_config(config)

    assert settings.unmasker("cube") == "direct"
    assert settings.unmasker("mnist") == "image_patch"


def test_eval_batch_size_falls_back_to_the_default() -> None:
    config = _config(
        method_options={
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": {"default": 8, "mnist": 2},
            }
        },
        datasets=["cube", "mnist"],
    )
    settings = load_config(config)

    assert settings.eval_batch_size("alpha", "cube") == 8
    assert settings.eval_batch_size("alpha", "mnist") == 2


def test_summary_lists_what_the_run_covers() -> None:
    config = _compared_methods_config()
    config["method_options"]["beta"] |= {  # type: ignore[index]
        "use_max_hard_budget_when_training_soft_budget": True,
        "eval_to_train_hard_budget_mapping": {"cube": {1: 3}},
    }

    summary = load_config(config).summary()

    assert "  dataset realizations: [0, 1, 2, 3, 4]" in summary.splitlines()
    assert "  stages off: []" in summary.splitlines()
    assert (
        "  datasets:\n"
        "    cube:\n"
        "      unmasker: direct\n"
        "      classifier: masked_mlp_classifier\n"
    ) in summary
    assert (
        "    alpha:\n"
        "      train script: alpha\n"
        "      classifier: external\n"
        "      pretrained model: none\n"
        "    beta:\n"
        "      train script: beta\n"
        "      classifier: built-in beta_classifier\n"
        "      pretrained model: shared\n"
    ) in summary
    assert (
        "    beta on cube:\n"
        "      hard budgets: [1 (trained at 3)]\n"
        "      soft-budget parameters (trained at hard budgets [3]): [0.1 -> 0.2]"
    ) in summary
    assert "gamma" not in summary


def test_summary_lists_the_stages_that_are_off() -> None:
    config = _config() | {"stages": {"training": False, "evaluation": False}}

    summary = load_config(config).summary()

    assert "  stages off: [training, evaluation]" in summary.splitlines()


@pytest.mark.parametrize(
    ("wandb_group_url", "shown"),
    [(None, "g"), ("https://wandb.ai/lab/afabench/groups/g", None)],
)
def test_summary_shows_the_wandb_group(
    wandb_group_url: str | None, shown: str | None
) -> None:
    config = _config() | {"use_wandb": True, "wandb_group": "g"}

    summary = load_config(config).summary(wandb_group_url=wandb_group_url)

    assert f"  W&B group: {shown or wandb_group_url}" in summary.splitlines()


def test_summary_shows_no_wandb_group_without_wandb_logging() -> None:
    assert "W&B group" not in load_config(_config()).summary()


def test_summary_says_whether_the_soft_budget_setting_runs() -> None:
    config = _config() | {"soft_budget_setting": False}

    summary = load_config(config).summary()

    assert "  soft-budget setting: False" in summary.splitlines()


def _config(
    *,
    method_options: dict[str, object] | None = None,
    methods: list[str] | None = None,
    datasets: list[str] | None = None,
    pretrain_mapping: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a minimal valid config, with the given parts replaced."""
    if method_options is None:
        method_options = {
            "alpha": {
                "train_script_name": "alpha",
                "eval_batch_size": 1,
                "soft_budget_datasets": datasets or ["cube"],
            }
        }
    if methods is None:
        methods = list(method_options)
    return {
        "pretrain_mapping": pretrain_mapping or {},
        "method_options": method_options,
        "methods": methods,
        "datasets": datasets or ["cube"],
        "unmaskers": {"default": "direct"},
        "eval_hard_budgets": {"default": [1]},
        "soft_budget_params": {
            method: {"default": [[0.1, 0.1]]} for method in methods
        },
        "classifier_names": {"default": "masked_mlp_classifier"},
    }


@pytest.mark.parametrize(
    ("value", "expected"), [(False, False), (True, True), ("TrUe", True)]
)
def test_per_dataset_plots_is_a_validated_opt_in(
    *, value: bool | str, expected: bool
) -> None:
    settings = load_config(_config() | {"per_dataset_plots": value})

    assert settings.per_dataset_plots is expected
    assert f"  per-dataset plots: {expected}" in settings.summary()


def test_plotting_metrics_follow_pipeline_profile_overrides() -> None:
    override = parse_config(
        ["plotting={dataset_metrics: {cube: auroc}, default_metric: f1}"]
    )
    settings = load_config(_config() | override)

    assert settings.plotting.dataset_metrics == {"cube": "auroc"}
    assert settings.plotting.default_metric == "f1"


@pytest.mark.parametrize(
    "plotting", [{"default_metric": "ap"}, {"dataset_metrics": {"cube": "F1"}}]
)
def test_plotting_rejects_unsupported_metrics(
    plotting: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"plotting.*(ap|F1)"):
        load_config(_config() | {"plotting": plotting})


@pytest.mark.parametrize(
    ("plotting", "error"),
    [
        (None, TypeError),
        ("auroc", TypeError),
        ({"default_metric": None}, ValueError),
        ({"plot_metric": "auroc"}, ValueError),
    ],
)
def test_plotting_rejects_malformed_metric_overrides(
    plotting: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match="plotting"):
        load_config(_config() | {"plotting": plotting})
