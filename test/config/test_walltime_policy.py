"""Site walltime lookup and eager configuration validation."""

from pathlib import Path

import pytest
import yaml

from workflow.src.walltime import WalltimePolicy


def policy(config: object) -> WalltimePolicy:
    return WalltimePolicy(
        config,
        methods={"dime", "l2m"},
        pretrained_models={"dime", "shared"},
        datasets={"miniboone", "cube"},
    )


def test_arrhenius_l2m_training_limit_is_dataset_specific() -> None:
    site_path = (
        Path(__file__).resolve().parents[2]
        / "workflow/profiles/site/arrhenius/site.yaml"
    )
    site = yaml.safe_load(site_path.read_text())
    walltime = WalltimePolicy(
        site["walltime"],
        methods={"l2m", "dime", "l2m_tabicl_prior"},
        pretrained_models={"l2m", "l2m_tabicl_prior"},
        datasets={"miniboone_subset", "metabric", "cube"},
    )

    assert walltime.minutes("train_method", "l2m", "miniboone_subset") == 480
    assert walltime.minutes("train_method", "l2m", "metabric") == 480
    assert walltime.minutes("train_method", "l2m", "cube") == 240
    assert walltime.minutes("train_method", "dime", "miniboone_subset") == 240
    assert walltime.minutes("pretrain_model", "l2m", "miniboone_subset") == 240
    assert walltime.minutes("pretrain_model", "l2m", "metabric") == 480


def test_arrhenius_l2m_tabicl_prior_limits_are_the_cube_upper_bounds() -> None:
    site_path = (
        Path(__file__).resolve().parents[2]
        / "workflow/profiles/site/arrhenius/site.yaml"
    )
    site = yaml.safe_load(site_path.read_text())
    walltime = WalltimePolicy(
        site["walltime"],
        methods={"l2m", "l2m_tabicl_prior"},
        pretrained_models={"l2m", "l2m_tabicl_prior"},
        datasets={"miniboone_subset", "metabric", "cube"},
    )

    assert (
        walltime.minutes("pretrain_model", "l2m_tabicl_prior", "cube") == 360
    )
    assert walltime.minutes("train_method", "l2m_tabicl_prior", "cube") == 720
    assert walltime.minutes("pretrain_model", "l2m", "cube") == 240


def test_walltime_uses_the_most_specific_available_limit() -> None:
    walltime = policy(
        {
            "default_hours": 2,
            "rules": {
                "train_method": {
                    "default_hours": 8,
                    "methods": {
                        "dime": {
                            "default_hours": 4,
                            "datasets": {"miniboone": 1.01},
                        }
                    },
                },
                "pretrain_model": {
                    "pretrained_models": {"shared": {"default_hours": 3}}
                },
            },
        }
    )

    assert walltime.minutes("train_method", "dime", "miniboone") == 61
    assert walltime.minutes("train_method", "dime", "cube") == 240
    assert walltime.minutes("train_method", "l2m", "cube") == 480
    assert walltime.minutes("pretrain_model", "shared", "cube") == 180
    assert walltime.minutes("pretrain_model", "dime", "cube") == 120
    assert walltime.minutes("train_classifier", None, "miniboone") == 120
    assert walltime.minutes("plot_time", None, None) == 120


@pytest.mark.parametrize(
    ("hours", "minutes"),
    [(1, 60), (1.01, 61), (0.05, 3), (0.001, 1), (72, 4320)],
)
def test_numeric_hours_round_up_to_whole_minutes(
    hours: float, minutes: int
) -> None:
    assert (
        policy({"default_hours": hours}).minutes("plot_time", None, None)
        == minutes
    )


@pytest.mark.parametrize(
    "hours", [0, -1, True, False, "4", None, float("inf"), float("nan")]
)
@pytest.mark.parametrize("level", ["site", "rule", "method", "dataset"])
def test_invalid_hours_fail_at_every_lookup_level(
    hours: object, level: str
) -> None:
    configs = {
        "site": {"default_hours": hours},
        "rule": {"rules": {"train_method": {"default_hours": hours}}},
        "method": {
            "rules": {
                "train_method": {"methods": {"dime": {"default_hours": hours}}}
            }
        },
        "dataset": {
            "rules": {
                "train_method": {
                    "methods": {"dime": {"datasets": {"miniboone": hours}}}
                }
            }
        },
    }
    with pytest.raises(ValueError, match="positive finite numeric hours"):
        policy(configs[level])


@pytest.mark.parametrize(
    ("config", "diagnostic"),
    [
        ({"default_hour": 2}, "default_hour"),
        ({"rules": {"train_methd": {}}}, "train_methd"),
        ({"rules": {"train_method": {"hours": 2}}}, "hours"),
        (
            {"rules": {"train_method": {"methods": {"diem": {}}}}},
            "diem",
        ),
        (
            {
                "rules": {
                    "pretrain_model": {"pretrained_models": {"shard": {}}}
                }
            },
            "shard",
        ),
        (
            {"rules": {"train_classifier": {"methods": {"dime": {}}}}},
            "methods",
        ),
        (
            {"rules": {"pretrain_model": {"methods": {"dime": {}}}}},
            "methods",
        ),
        (
            {
                "rules": {
                    "eval_method": {
                        "methods": {"dime": {"datasets": {"minibone": 2}}}
                    }
                }
            },
            "minibone",
        ),
        (
            {"rules": {"eval_method": {"methods": {"dime": {"dataset": {}}}}}},
            "dataset",
        ),
    ],
)
def test_unknown_fields_and_selectors_fail_eagerly(
    config: object, diagnostic: str
) -> None:
    with pytest.raises(ValueError, match=diagnostic):
        policy(config)


@pytest.mark.parametrize(
    "config", [None, [], {"rules": []}, {"rules": {"train_method": 4}}]
)
def test_malformed_tables_fail_eagerly(config: object) -> None:
    with pytest.raises(TypeError, match="mapping"):
        policy(config)


def test_absent_table_does_not_replace_snakemake_resources() -> None:
    walltime = policy({})
    assert walltime.minutes("train_method", "dime", "cube") is None
    assert walltime.resources("train_method") == {}


def test_sparse_table_leaves_unmatched_rules_to_snakemake() -> None:
    walltime = policy(
        {
            "rules": {
                "train_method": {"methods": {"dime": {"default_hours": 4}}}
            }
        }
    )
    assert walltime.minutes("train_method", "dime", "cube") == 240
    assert walltime.minutes("train_method", "l2m", "cube") is None
    assert walltime.resources("train_classifier") == {}
