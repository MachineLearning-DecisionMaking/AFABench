"""
The provenance record: how one artifact was produced.

Design: `docs/adr/0002-provenance-recorded-in-artifacts.md`. The record is
captured once by the library code that writes the artifact, embedded in the
artifact itself, and propagated from inputs to outputs. `capture_provenance`
collects the code, environment and compute facts; callers pass the stage,
the resolved config, the resolved seed, the inputs and the dataset identity.
`null` always means "unknown", never a default.
"""

import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self, cast

import numpy as np
import polars as pl
import torch
import torch.version

from afabench.core.checkout import AFABENCH_CHECKOUT
from afabench.core.code_identity import code_identity

PROVENANCE_VERSION = 1
AFABENCH_DISTRIBUTION = "afabench"

type Stage = Literal[
    "dataset_generation",
    "classifier_training",
    "pretraining",
    "training",
    "evaluation",
]
type Split = Literal["train", "val", "test"]
type InputRole = Literal[
    "train_dataset",
    "val_dataset",
    "eval_dataset",
    "classifier",
    "pretrained_model",
    "method",
]


class UnknownProvenanceVersionError(ValueError):
    """Raised when a record's `provenance_version` is not known."""


class DatasetIdentityMismatchError(ValueError):
    """Raised when inputs that must share a dataset identity disagree."""


@dataclass(frozen=True, kw_only=True)
class ProvenanceInput:
    """One input bundle, as recorded in a record's `inputs`."""

    role: InputRole
    path: str
    class_name: str
    content_hash: str | None


@dataclass(frozen=True, kw_only=True)
class Environment:
    python_version: str
    afabench_version: str | None
    torch_version: str
    numpy_version: str
    polars_version: str
    lockfile_sha256: str | None
    platform: str


@dataclass(frozen=True, kw_only=True)
class Compute:
    device: str
    accelerator_name: str | None
    cuda_version: str | None
    cudnn_version: int | None
    float32_matmul_precision: str
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    deterministic_algorithms: bool


@dataclass(frozen=True, kw_only=True)
class DatasetIdentity:
    dataset_key: str | None
    dataset_realization_index: int | None


@dataclass(frozen=True, kw_only=True)
class ProvenanceRecord:
    provenance_version: int
    stage: Stage
    created_at: str
    code_commit: str | None
    code_dirty: bool | None
    resolved_config: dict[str, Any]
    seed: int
    smoke_test: bool
    method_name: str | None
    dataset_key: str | None
    dataset_realization_index: int | None
    split: Split | None
    inputs: list[ProvenanceInput]
    environment: Environment
    compute: Compute

    @property
    def dataset_identity(self) -> DatasetIdentity:
        return DatasetIdentity(
            dataset_key=self.dataset_key,
            dataset_realization_index=self.dataset_realization_index,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> Self:
        version = data.get("provenance_version")
        if version != PROVENANCE_VERSION:
            msg = (
                f"Unknown provenance_version {version!r}; "
                f"this code reads version {PROVENANCE_VERSION}."
            )
            raise UnknownProvenanceVersionError(msg)
        fields = dict(data)
        fields["inputs"] = [
            ProvenanceInput(**entry) for entry in fields["inputs"]
        ]
        fields["environment"] = Environment(**fields["environment"])
        fields["compute"] = Compute(**fields["compute"])
        return cls(**fields)


def capture_provenance(
    *,
    stage: Stage,
    resolved_config: Mapping[str, Any],
    seed: int,
    smoke_test: bool,
    device: str,
    dataset_identity: DatasetIdentity,
    inputs: Sequence[ProvenanceInput] = (),
    method_name: str | None = None,
    split: Split | None = None,
    checkout: Path | None = None,
) -> ProvenanceRecord:
    """
    Build the record of an artifact being written now.

    `resolved_config` is the script's full configuration as it actually used
    it; a value that is not JSON-serialisable raises `TypeError` naming it.
    `checkout` is the git work tree whose code runs, by default the one
    holding this package; outside a work tree, or with the package installed
    rather than run from a checkout, the code identity is unknown.
    """
    _require_json(resolved_config, path="resolved_config")
    checkout = AFABENCH_CHECKOUT if checkout is None else checkout
    code_commit, code_dirty = code_identity(checkout)
    return ProvenanceRecord(
        provenance_version=PROVENANCE_VERSION,
        stage=stage,
        created_at=datetime.now(UTC).isoformat(),
        code_commit=code_commit,
        code_dirty=code_dirty,
        resolved_config=json.loads(json.dumps(resolved_config)),
        seed=seed,
        smoke_test=smoke_test,
        method_name=method_name,
        dataset_key=dataset_identity.dataset_key,
        dataset_realization_index=dataset_identity.dataset_realization_index,
        split=split,
        inputs=list(inputs),
        environment=_environment(checkout),
        compute=_compute(device),
    )


def shared_dataset_identity(
    *records: ProvenanceRecord | None,
) -> DatasetIdentity:
    """
    Return the dataset identity that the given input records agree on.

    A null record (an input written before ADR 0002) and null fields are
    unknown and cannot disagree. Known values that differ raise
    `DatasetIdentityMismatchError` naming both.
    """
    dataset_key: str | None = None
    dataset_realization_index: int | None = None
    for record in records:
        if record is None:
            continue
        dataset_key = _agree("dataset_key", dataset_key, record.dataset_key)
        dataset_realization_index = _agree(
            "dataset_realization_index",
            dataset_realization_index,
            record.dataset_realization_index,
        )
    return DatasetIdentity(
        dataset_key=dataset_key,
        dataset_realization_index=dataset_realization_index,
    )


def provenance_from_manifest(
    manifest: Mapping[str, Any],
) -> ProvenanceRecord | None:
    """Read the record in a bundle manifest; null for a manifest without one."""
    if "provenance" not in manifest:
        return None
    return ProvenanceRecord.from_json_dict(manifest["provenance"])


def input_from_manifest(
    role: InputRole, path: str, manifest: Mapping[str, Any]
) -> ProvenanceInput:
    """
    Build the `inputs` entry for an input bundle from its manifest.

    `path` is recorded as given. The content hash is the one the input's
    producer recorded, null for a bundle written without one.
    """
    return ProvenanceInput(
        role=role,
        path=path,
        class_name=manifest["class_name"],
        content_hash=manifest.get("content_hash"),
    )


def _agree[T](name: str, known: T | None, value: T | None) -> T | None:
    if known is None:
        return value
    if value is not None and value != known:
        msg = f"Inputs disagree on {name}: {known!r} and {value!r}."
        raise DatasetIdentityMismatchError(msg)
    return known


def _require_json(value: object, *, path: str) -> None:
    if value is None or isinstance(value, bool | int | float | str):
        return
    if isinstance(value, Mapping):
        for key, item in cast("Mapping[object, object]", value).items():
            if not isinstance(key, str):
                msg = f"Config key {key!r} at {path} is not a string."
                raise TypeError(msg)
            _require_json(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(cast("list[object]", value)):
            _require_json(item, path=f"{path}[{index}]")
        return
    msg = (
        f"Config value at {path} is not JSON-serialisable: {value!r} "
        f"({type(value).__name__})."
    )
    raise TypeError(msg)


def _environment(checkout: Path | None) -> Environment:
    lockfile = None if checkout is None else checkout / "uv.lock"
    try:
        afabench_version = importlib.metadata.version(AFABENCH_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        afabench_version = None
    return Environment(
        python_version=platform.python_version(),
        afabench_version=afabench_version,
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        polars_version=pl.__version__,
        lockfile_sha256=(
            hashlib.sha256(lockfile.read_bytes()).hexdigest()
            if lockfile is not None and lockfile.is_file()
            else None
        ),
        platform=platform.platform(),
    )


def _compute(device: str) -> Compute:
    torch_device = torch.device(device)
    accelerator_name = (
        torch.cuda.get_device_name(torch_device)
        if torch_device.type == "cuda" and torch.cuda.is_available()
        else None
    )
    return Compute(
        device=device,
        accelerator_name=accelerator_name,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )
