"""
Every script config the pipeline runs composes without missing values.

A method's root config gives every hyperparameter a default, and its
optional `dataset_override/<key>.yaml` only overrides some on one dataset key
(`docs/how-to/add_method.md`). A hyperparameter without a default is left
`???` on every dataset key without an override, and the job fails on the
cluster with `MissingMandatoryValue`. Composing each (script, dataset) pair
of the shipped `all` profile catches it here: the only values left missing
must be those the pipeline passes on the command line, the training
contract fields and the method's script params.
"""

import importlib
import warnings
from dataclasses import fields
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from afabench.core.workflow_settings import load_config
from afabench.fit.contract import PretrainingContract, TrainingContract
from afabench.release.workflow_config import resolve_workflow_config

REPO_ROOT = Path(__file__).parents[2]

_STAGE_CONTRACT_FIELDS = {
    "pretrain_model": frozenset(f.name for f in fields(PretrainingContract)),
    "train_method": frozenset(f.name for f in fields(TrainingContract)),
}


def _register_structured_configs() -> None:
    """Import every config module, as the scripts do, to fill the store."""
    src = REPO_ROOT / "src"
    for module in sorted((src / "afabench/components").rglob("config.py")):
        importlib.import_module(
            ".".join(module.relative_to(src).with_suffix("").parts)
        )


def _pipeline_cases() -> list[tuple[str, str, str, frozenset[str]]]:
    """Return (stage, script, dataset, params keys) for each job kind run."""
    configfiles = yaml.safe_load(
        (REPO_ROOT / "workflow/profiles/pipeline/all/config.yaml").read_text()
    )["configfile"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        settings = load_config(
            resolve_workflow_config(
                profile=None,
                configfiles=[REPO_ROOT / path for path in configfiles],
                overrides=[],
            ).merged
        )

    cases: set[tuple[str, str, str, frozenset[str]]] = set()
    for method in settings.methods:
        invocations = [("train_method", settings.training_script(method))]
        pretrained_model = settings.method_to_pretrained_model.get(method)
        if pretrained_model is not None:
            invocations.append(
                (
                    "pretrain_model",
                    settings.pretraining_script(pretrained_model),
                )
            )
        for dataset, combinations in settings.budget_params[method].items():
            if not combinations:
                continue
            for stage, invocation in invocations:
                params = frozenset(
                    param.split("=", 1)[0]
                    for param in invocation.script_params.split()
                )
                cases.add((stage, invocation.script_name, dataset, params))
    return sorted(cases)


_CASES = _pipeline_cases()


@pytest.mark.parametrize(
    ("stage", "script", "dataset", "params"),
    [
        pytest.param(*case, id=f"{case[0]}/{case[1]}/{case[2]}")
        for case in _CASES
    ],
)
def test_pipeline_script_config_has_no_missing_values(
    stage: str, script: str, dataset: str, params: frozenset[str]
) -> None:
    _register_structured_configs()
    config_dir = REPO_ROOT / "conf/scripts" / stage / script
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name="config",
            overrides=[
                "initializer=cold",
                "unmasker=direct",
                f"dataset_key={dataset}",
            ],
        )

    missing = (
        set(OmegaConf.missing_keys(config))
        - _STAGE_CONTRACT_FIELDS[stage]
        - params
    )

    assert missing == set(), (
        f"{stage}/{script} on {dataset} leaves {sorted(missing)} missing; "
        f"give them defaults in conf/scripts/{stage}/{script}/config.yaml"
    )
