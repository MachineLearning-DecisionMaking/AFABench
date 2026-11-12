"""
Dataset overrides may not set training-contract fields.

Snakemake always overrides contract fields on the command line, so a
contract field set in a dataset override YAML is dead in the pipeline and only
misleads a developer running the script by hand (see issue #43). This test covers every
method under `conf/scripts/`. `NOT_YET_PORTED` lists the methods whose
dataset overrides are known to still set contract fields, pending their
training-contract port (see the open "Port ... to the training contract
helpers" issues); their cases are `xfail(strict=True)` so a forgotten
removal from the set turns into a failure once a port actually fixes the
files (`docs/adr/0001-training-contract-as-library.md`).
"""

from dataclasses import fields
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from afabench.fit.contract import PretrainingContract, TrainingContract

REPO_ROOT = Path(__file__).parents[2]
SCRIPTS_CONF = REPO_ROOT / "conf/scripts"

_STAGE_CONTRACT_FIELDS = {
    "train_method": frozenset(f.name for f in fields(TrainingContract)),
    "pretrain_model": frozenset(f.name for f in fields(PretrainingContract)),
}

NOT_YET_PORTED: dict[str, frozenset[str]] = {}


def _dataset_override_files() -> list[tuple[Path, frozenset[str], bool]]:
    cases = []
    for stage, contract_fields in _STAGE_CONTRACT_FIELDS.items():
        stage_dir = SCRIPTS_CONF / stage
        if not stage_dir.is_dir():
            msg = f"missing scripts conf stage dir: {stage_dir}"
            raise FileNotFoundError(msg)
        not_yet_ported = NOT_YET_PORTED.get(stage, frozenset())
        for method_dir in sorted(stage_dir.iterdir()):
            override_dir = method_dir / "dataset_override"
            if not override_dir.is_dir():
                continue
            is_known_violator = method_dir.name in not_yet_ported
            cases.extend(
                (override_file, contract_fields, is_known_violator)
                for override_file in sorted(override_dir.glob("*.yaml"))
            )
    return cases


def _case_id(override_file: Path) -> str:
    return "/".join(override_file.parts[-4:])


_CASES = _dataset_override_files()
assert _CASES, (
    f"no dataset overrides found under {SCRIPTS_CONF}; "
    "check that the stage/method layout did not change"
)


@pytest.mark.parametrize(
    ("override_file", "contract_fields"),
    [
        pytest.param(
            override_file,
            contract_fields,
            id=_case_id(override_file),
            marks=(
                [
                    pytest.mark.xfail(
                        reason=(
                            f"{override_file.parts[-3]} not yet ported "
                            "to the training contract"
                        ),
                        strict=True,
                    )
                ]
                if is_known_violator
                else []
            ),
        )
        for override_file, contract_fields, is_known_violator in _CASES
    ],
)
def test_dataset_override_does_not_set_a_contract_field(
    override_file: Path, contract_fields: frozenset[str]
) -> None:
    content = OmegaConf.load(override_file)
    keys = {key for key in content if isinstance(key, str)}

    violations = keys & contract_fields

    assert violations == set(), (
        f"{override_file} sets contract field(s) {violations}"
    )
