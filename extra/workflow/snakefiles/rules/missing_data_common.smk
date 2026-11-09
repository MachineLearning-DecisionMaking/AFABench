"""Config, paths, and the experiment matrix shared by the missing-data orchestrations."""

import os
import re
import sys

snakefile_dir = workflow.basedir
workflow_dir = os.path.dirname(os.path.dirname(snakefile_dir))
src_dir = os.path.join(workflow_dir, "src")
sys.path.insert(0, src_dir)

from config import load_config, resolve_device
from missing_data_config import (
    build_method_specs,
    largest_hard_budget,
    policy_training_coordinates,
    strategy_enabled,
)


required_missing_config = {
    "artifact_namespace",
    "eval_dataset_split",
    "missingness",
    "strategies",
}
missing_config = sorted(required_missing_config - set(config))
if missing_config:
    raise ValueError(
        "Missing required missing-data config keys: "
        + ", ".join(missing_config)
    )

if runtime_device := os.environ.get("AFABENCH_DEVICE"):
    config["device"] = runtime_device
_config = load_config(config)

NAMESPACE = str(config["artifact_namespace"])
DATA_NAMESPACE = str(config.get("upstream_namespace", NAMESPACE))
CLASSIFIER_SOURCE = str(config.get("classifier_source", "complete"))
if CLASSIFIER_SOURCE not in {"complete", "view", "restricted_view"}:
    raise ValueError(
        "classifier_source must be 'complete', 'view', or 'restricted_view'"
    )
VIEW_CLASSIFIER_NAMESPACE = str(
    config.get("view_classifier_namespace", NAMESPACE)
)
INSTANCES = [int(value) for value in _config["DATASET_INSTANCE_INDICES"]]
if not INSTANCES:
    raise ValueError("dataset_instance_indices must not be empty")
CLASSIFIER_INSTANCE = INSTANCES[0]
CLASSIFIER_SCOPE = str(config.get("classifier_scope", "shared"))
if CLASSIFIER_SCOPE not in {"shared", "per_instance"}:
    raise ValueError("classifier_scope must be 'shared' or 'per_instance'")

UNSUPPORTED_DATASETS = set(
    str(value)
    for value in config.get("missing_data_unsupported_datasets", [])
)
SELECTED_DATASETS = [str(value) for value in _config["DATASETS"]]
DATASETS = [
    dataset
    for dataset in SELECTED_DATASETS
    if dataset not in UNSUPPORTED_DATASETS
]
excluded_datasets = sorted(set(SELECTED_DATASETS) - set(DATASETS))
if excluded_datasets:
    print(
        "Skipping datasets without flat-feature missingness support: "
        + ", ".join(excluded_datasets)
    )
if not DATASETS:
    raise ValueError("No missing-data-compatible datasets were selected")

BASE_METHODS = [str(value) for value in _config["METHODS"]]
METHOD_OPTIONS = _config["METHOD_OPTIONS"]
METHOD_SPECS = build_method_specs(
    BASE_METHODS,
    METHOD_OPTIONS,
    config.get("missing_data_method_overrides", {}),
    (
        config.get("missing_data_method_variants", {})
        if config.get("include_method_variants", True)
        else {}
    ),
)
PRETRAINED_METHODS = [
    name
    for name, spec in METHOD_SPECS.items()
    if spec.pretrained_model_name is not None
]
UNPRETRAINED_METHODS = [
    name
    for name, spec in METHOD_SPECS.items()
    if spec.pretrained_model_name is None
]
COMMON_STRATEGIES = [str(value) for value in config["strategies"]]
STRATEGY_FILTERS = config.get("strategy_filters", {})
INCLUDE_COMPLETE_DATA = bool(config.get("include_complete_data", True))

UNMASKERS = _config["UNMASKERS"]
CLASSIFIER_NAMES = _config["CLASSIFIER_NAMES"]
METHOD_CLASSIFIER_SCRIPT_NAMES = _config["METHOD_CLASSIFIER_SCRIPT_NAMES"]
METHOD_CLASSIFIER_SCRIPT_PARAMS = _config["METHOD_CLASSIFIER_SCRIPT_PARAMS"]
PRETRAIN_SCRIPT_NAMES = _config["PRETRAIN_SCRIPT_NAMES"]
PRETRAIN_PARAMS = _config["PRETRAIN_PARAMS"]
EVAL_BATCH_SIZES = _config["EVAL_BATCH_SIZES"]
HARD_BUDGET_IGNORED_DATASETS = _config[
    "HARD_BUDGET_IGNORED_DATASETS"
]

DEFAULT_DEVICE = str(_config["DEVICE"])
DEVICE_OVERRIDES = _config["DEVICE_OVERRIDES"]
USE_WANDB = str(bool(_config["USE_WANDB"])).lower()
SMOKE_TEST = bool(_config["SMOKE_TEST"])
SMOKE_TEST_STR = str(SMOKE_TEST).lower()
INITIALIZER = str(_config["INITIALIZER"])
EVAL_SPLIT = str(_config["EVAL_DATASET_SPLIT"])
if EVAL_SPLIT not in {"val", "test"}:
    raise ValueError("eval_dataset_split must be either 'val' or 'test'")

ROOT = "extra/output/missing_data"
RESULTS = "extra/output/paper/experiments/results"
PAPER_ARTIFACTS = bool(config.get("paper_artifacts", False))
if PAPER_ARTIFACTS and (NAMESPACE != "induced" or EVAL_SPLIT != "val" or SMOKE_TEST):
    raise ValueError("paper_artifacts requires the full induced validation study")
PAPER_OUTPUTS = [
    f"{RESULTS}/{name}"
    for name in (
        "conceptual_constants.tex", "exact_study_raw.pdf",
        "exact_study_combined.pdf", "exact_study_collapse.pdf",
        "budget_study.pdf",
        "main_summary_absolute_mcar.pdf",
        "main_summary_variants_grid.pdf", "law_grid.pdf",
        "compute.pdf", "route_structure.tex",
    )
] if PAPER_ARTIFACTS else []
BENCHMARK_ROOT = f"{ROOT}/benchmark/{NAMESPACE}"
HYDRA_WORKFLOW_OVERRIDES = (
    "hydra/job_logging=workflow_console "
    "hydra.run.dir=${SNIC_TMP:-/tmp}/afabench-hydra/"
    "${SLURM_JOB_ID:-local} hydra.output_subdir=null"
)
MISSINGNESS = config["missingness"]
MECHANISMS = [str(value) for value in MISSINGNESS["mechanisms"]]
PROBABILITIES = [str(value) for value in MISSINGNESS["probabilities"]]
MISSING_COMBINATIONS = [
    (mechanism, probability)
    for mechanism in MECHANISMS
    for probability in PROBABILITIES
]

DATASET_GENERATION_PARAMS = config.get("dataset_generation_params", {})
CLASSIFIER_PARAMS = config.get("classifier_params", {})
RESTORATION_PVAE_PARAMS = config.get("restoration_pvae_params", {})
PRETRAIN_RUNTIME_PARAMS = config.get("pretrain_runtime_params", {})
TRAIN_RUNTIME_PARAMS = config.get("train_runtime_params", {})
EVAL_PARAMS = config.get("eval_params", {})
PRETRAIN_REUSE = config.get("missing_data_pretrain_reuse", {})
RESTORATION_BATCH_SIZE = int(config.get("restoration_batch_size", 1024))
EVAL_BATCH_SIZE_OVERRIDE = config.get("eval_batch_size")
STEPWISE_EVAL_BATCH_SIZE = int(config.get("stepwise_eval_batch_size", 16))



def wildcard_pattern(values):
    unique_values = dict.fromkeys(str(value) for value in values)
    return "(?:" + "|".join(re.escape(value) for value in unique_values) + ")"


def runtime_params(mapping, *keys):
    if isinstance(mapping, list):
        return " ".join(str(value) for value in mapping)
    values = [str(value) for value in mapping.get("default", [])]
    for key in dict.fromkeys(str(value) for value in keys if value is not None):
        values.extend(str(value) for value in mapping.get(key, []))
    return " ".join(values)


def method_device(dataset, method):
    spec = METHOD_SPECS[method]
    method_datasets = DEVICE_OVERRIDES.get("method_datasets", {})
    methods = DEVICE_OVERRIDES.get("methods", {})
    for candidate in dict.fromkeys([method, spec.base_method]):
        if dataset in method_datasets.get(candidate, {}):
            return str(method_datasets[candidate][dataset])
        if candidate in methods:
            return str(methods[candidate])
    return resolve_device(
        DEFAULT_DEVICE,
        DEVICE_OVERRIDES,
        dataset=dataset,
    )


def dataset_device(dataset, pretrained_model=None):
    return resolve_device(
        DEFAULT_DEVICE,
        DEVICE_OVERRIDES,
        dataset=dataset,
        pretrained_model=pretrained_model,
    )


def gpu_for_device(device):
    """Reserve one GPU slot exactly when the resolved device uses CUDA."""
    return int(str(device).split(":", maxsplit=1)[0] == "cuda")


def classifier_script_name(dataset):
    classifier = CLASSIFIER_NAMES[dataset]
    if isinstance(classifier, dict):
        return classifier["script_name"]
    return classifier


def classifier_script_params(dataset):
    classifier = CLASSIFIER_NAMES[dataset]
    if isinstance(classifier, dict):
        configured = classifier.get("script_params", [])
    else:
        configured = []
    runtime = runtime_params(CLASSIFIER_PARAMS, dataset)
    return " ".join([*[str(value) for value in configured], runtime]).strip()


def classifier_instance(wildcards):
    if CLASSIFIER_SCOPE == "per_instance":
        return int(wildcards.instance)
    return CLASSIFIER_INSTANCE


def classifier_path(dataset, instance, method=None):
    instance_suffix = (
        f"/instance-{instance}.bundle"
        if CLASSIFIER_SCOPE == "per_instance"
        else ".bundle"
    )
    if method is not None:
        base_method = METHOD_SPECS[method].base_method
        if base_method in METHOD_CLASSIFIER_SCRIPT_NAMES:
            return (
                f"{ROOT}/classifier/{DATA_NAMESPACE}/"
                f"method-{base_method}+dataset-{dataset}{instance_suffix}"
            )
    return (
        f"{ROOT}/classifier/{DATA_NAMESPACE}/dataset-{dataset}{instance_suffix}"
    )


VIEW_CLASSIFIER_OUTPUT = (
    f"{ROOT}/classifier/{VIEW_CLASSIFIER_NAMESPACE}/view/dataset-{{dataset}}/"
    "mechanism-{mechanism}+p-{p}/instance-{instance}/{strategy}.bundle"
)


def policy_classifier(wildcards, method=None):
    if CLASSIFIER_SOURCE != "complete" and wildcards.strategy != "complete":
        return VIEW_CLASSIFIER_OUTPUT.format(
            dataset=wildcards.dataset,
            mechanism=wildcards.mechanism,
            p=wildcards.p,
            instance=wildcards.instance,
            strategy=(
                "restricted"
                if CLASSIFIER_SOURCE == "restricted_view"
                else wildcards.strategy
            ),
        )
    return classifier_path(wildcards.dataset, wildcards.instance, method)


CLASSIFIER_OUTPUT = (
    f"{ROOT}/classifier/{DATA_NAMESPACE}/dataset-{{dataset}}/"
    "instance-{instance}.bundle"
    if CLASSIFIER_SCOPE == "per_instance"
    else f"{ROOT}/classifier/{DATA_NAMESPACE}/dataset-{{dataset}}.bundle"
)
METHOD_CLASSIFIER_OUTPUT = (
    f"{ROOT}/classifier/{DATA_NAMESPACE}/"
    "method-{base_method}+dataset-{dataset}/instance-{instance}.bundle"
    if CLASSIFIER_SCOPE == "per_instance"
    else f"{ROOT}/classifier/{DATA_NAMESPACE}/"
    "method-{base_method}+dataset-{dataset}.bundle"
)
CLASSIFIER_BENCHMARK = (
    f"{BENCHMARK_ROOT}/train_shared_classifier/dataset-{{dataset}}/"
    "instance-{instance}.tsv"
    if CLASSIFIER_SCOPE == "per_instance"
    else f"{BENCHMARK_ROOT}/train_shared_classifier/dataset-{{dataset}}.tsv"
)
METHOD_CLASSIFIER_BENCHMARK = (
    f"{BENCHMARK_ROOT}/train_method_classifier/"
    "method-{base_method}+dataset-{dataset}/instance-{instance}.tsv"
    if CLASSIFIER_SCOPE == "per_instance"
    else f"{BENCHMARK_ROOT}/train_method_classifier/"
    "method-{base_method}+dataset-{dataset}.tsv"
)


def raw_dataset(dataset, instance, split):
    return (
        f"{ROOT}/datasets/{DATA_NAMESPACE}/{dataset}/{instance}/{split}.bundle"
    )


def base_view(dataset, mechanism, probability, instance, strategy, split):
    return (
        f"{ROOT}/views/base/{DATA_NAMESPACE}/dataset-{dataset}/"
        f"mechanism-{mechanism}+p-{probability}/instance-{instance}/"
        f"{strategy}/{split}.bundle"
    )


def restored_view(
    dataset, mechanism, probability, instance, strategy, split
):
    return (
        f"{ROOT}/views/restored/{DATA_NAMESPACE}/dataset-{dataset}/"
        f"mechanism-{mechanism}+p-{probability}/instance-{instance}/"
        f"{strategy}/{split}.bundle"
    )


def training_view(wildcards, split):
    if wildcards.strategy == "complete":
        return raw_dataset(wildcards.dataset, wildcards.instance, split)
    if wildcards.strategy == "pvae_stepwise":
        return base_view(
            wildcards.dataset,
            wildcards.mechanism,
            wildcards.p,
            wildcards.instance,
            "restricted",
            split,
        )
    if wildcards.strategy.startswith("pvae_"):
        return restored_view(
            wildcards.dataset,
            wildcards.mechanism,
            wildcards.p,
            wildcards.instance,
            wildcards.strategy,
            split,
        )
    return base_view(
        wildcards.dataset,
        wildcards.mechanism,
        wildcards.p,
        wildcards.instance,
        wildcards.strategy,
        split,
    )


def incomplete_pvae(dataset, mechanism, probability, instance):
    return (
        f"{ROOT}/restoration_pvae/{DATA_NAMESPACE}/incomplete/dataset-{dataset}/"
        f"mechanism-{mechanism}+p-{probability}/instance-{instance}/"
        "model.bundle"
    )


def oracle_pvae(dataset, instance):
    return (
        f"{ROOT}/restoration_pvae/{DATA_NAMESPACE}/oracle/dataset-{dataset}/"
        f"instance-{instance}/model.bundle"
    )


def method_pretrain(wildcards):
    key = METHOD_SPECS[wildcards.method].pretrained_model_name
    strategy = (
        "restricted"
        if wildcards.strategy == "pvae_stepwise"
        else wildcards.strategy
    )
    reuse_kind = PRETRAIN_REUSE.get(key, {}).get(strategy)
    if reuse_kind == "incomplete_restoration_pvae":
        return incomplete_pvae(
            wildcards.dataset,
            wildcards.mechanism,
            wildcards.p,
            wildcards.instance,
        )
    if reuse_kind == "oracle_restoration_pvae":
        return oracle_pvae(wildcards.dataset, wildcards.instance)
    if reuse_kind is not None:
        raise ValueError(
            f"Unknown pretraining reuse target for {key}/{strategy}: "
            f"{reuse_kind}"
        )
    return (
        f"{ROOT}/pretrained/{NAMESPACE}/{key}/dataset-{wildcards.dataset}/"
        f"mechanism-{wildcards.mechanism}+p-{wildcards.p}+"
        f"strategy-{strategy}+instance-{wildcards.instance}/"
        "model.bundle"
    )


def trained_method(
    dataset,
    method,
    mechanism,
    probability,
    strategy,
    instance,
    train_budget,
):
    return (
        f"{ROOT}/trained/{NAMESPACE}/{method}/dataset-{dataset}/"
        f"mechanism-{mechanism}+p-{probability}+strategy-{strategy}+"
        f"instance-{instance}+train_hard_budget-{train_budget}/method.bundle"
    )


def trained_method_input(wildcards):
    mechanism, probability, strategy = policy_training_coordinates(
        wildcards.mechanism,
        wildcards.p,
        wildcards.strategy,
    )
    return trained_method(
        wildcards.dataset,
        wildcards.method,
        mechanism,
        probability,
        strategy,
        wildcards.instance,
        wildcards.train_budget,
    )


def policy_reuse_guard(wildcards):
    if wildcards.strategy != "true_completion":
        return []
    return [
        training_view(wildcards, "train"),
        training_view(wildcards, "val"),
    ]


def evaluation_path(
    dataset,
    method,
    mechanism,
    probability,
    strategy,
    instance,
    train_budget,
    eval_budget,
):
    return (
        f"{ROOT}/eval/{EVAL_SPLIT}/{NAMESPACE}/dataset-{dataset}/"
        f"method-{method}+mechanism-{mechanism}+p-{probability}+"
        f"strategy-{strategy}+instance-{instance}+"
        f"train_hard_budget-{train_budget}+eval_hard_budget-{eval_budget}/"
        "eval_data.parquet"
    )


BUDGETS = {}
for method in BASE_METHODS:
    for dataset in DATASETS:
        if dataset in HARD_BUDGET_IGNORED_DATASETS[method]:
            continue
        budget = largest_hard_budget(_config["BUDGET_PARAMS"][method][dataset])
        if budget is not None:
            BUDGETS[(method, dataset)] = budget


def experiment_matrix():
    rows = []
    for dataset in DATASETS:
        for method, spec in METHOD_SPECS.items():
            budget = BUDGETS.get((spec.base_method, dataset))
            if budget is None:
                continue
            train_budget, eval_budget = budget
            if INCLUDE_COMPLETE_DATA and spec.include_complete_data:
                for instance in INSTANCES:
                    rows.append(
                        (
                            dataset,
                            method,
                            "none",
                            "0.0",
                            "complete",
                            instance,
                            train_budget,
                            eval_budget,
                        )
                    )
            strategies = (
                list(spec.allowed_strategies)
                if spec.allowed_strategies is not None
                else COMMON_STRATEGIES
            )
            strategies = list(strategies) + list(spec.extra_strategies)
            for mechanism, probability in MISSING_COMBINATIONS:
                for instance in INSTANCES:
                    for strategy in dict.fromkeys(strategies):
                        if not strategy_enabled(
                            STRATEGY_FILTERS,
                            strategy,
                            dataset=dataset,
                            method=method,
                            base_method=spec.base_method,
                            mechanism=mechanism,
                            probability=probability,
                        ):
                            continue
                        rows.append(
                            (
                                dataset,
                                method,
                                mechanism,
                                probability,
                                strategy,
                                instance,
                                train_budget,
                                eval_budget,
                            )
                        )
    return rows


EXPERIMENTS = experiment_matrix()
if not EXPERIMENTS:
    raise ValueError("The selected methods and datasets produce no experiments")
EVALUATIONS = [evaluation_path(*row) for row in EXPERIMENTS]
NATIVE_EVALUATIONS = [
    evaluation_path(*row) for row in EXPERIMENTS if row[2] == "native"
]
SUMMARY_DIR = f"{ROOT}/summary/{EVAL_SPLIT}/{NAMESPACE}"
FIGURE_DIR = f"{ROOT}/figures/{EVAL_SPLIT}/{NAMESPACE}"
NATIVE_AUDIT_OUTPUTS = (
    [f"{ROOT}/analysis/native_legality_{NAMESPACE}_{EVAL_SPLIT}.csv"]
    if NATIVE_EVALUATIONS
    else []
)

configured_strategies = ["complete", "restricted", *COMMON_STRATEGIES]
if CLASSIFIER_SOURCE != "complete" and "true_completion" in COMMON_STRATEGIES:
    raise ValueError(
        f"classifier_source '{CLASSIFIER_SOURCE}' does not support true_completion"
    )
for spec in METHOD_SPECS.values():
    configured_strategies.extend(spec.allowed_strategies or ())
    configured_strategies.extend(spec.extra_strategies)

wildcard_constraints:
    dataset=wildcard_pattern(DATASETS),
    mechanism=wildcard_pattern(["none", *MECHANISMS]),
    p=wildcard_pattern(["0.0", *PROBABILITIES]),
    strategy=wildcard_pattern(configured_strategies),
    method=wildcard_pattern(METHOD_SPECS),
    pretrain_key=wildcard_pattern(PRETRAIN_SCRIPT_NAMES),
    train_budget=r"[0-9.]+",
    eval_budget=r"[0-9.]+"
# The cluster runner selects its architecture-specific environment.
shell.prefix(os.environ.get("AFABENCH_SHELL_PREFIX", ""))


onstart:
    if not os.environ.get("AFABENCH_RUN_MANIFEST"):
        from argparse import Namespace
        from datetime import datetime, timezone
        from pathlib import Path
        from scripts.workflow.write_run_manifest import write_manifest

        write_manifest(Path.cwd(), dict(config), Namespace(
            profile=workflow.main_snakefile,
            run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}",
            device=DEFAULT_DEVICE, cores=workflow.cores,
            mem_mb=workflow.global_resources.get("mem_mb"),
            job_mem_mb=None, gpu_workers=workflow.global_resources.get("gpu", 0),
            mps=False, archive_dir=None, remove_archived_source=False,
            snakemake_args=sys.argv[1:],
        ))


# The plots orchestration takes results as given, so the figure rules must not
# reach back into the experiment DAG.
BUILD_EXPERIMENTS = globals().get("BUILD_EXPERIMENTS", True)
COMPUTE_TRIGGERS = EVALUATIONS if BUILD_EXPERIMENTS else []
ROUTE_CLASSIFIERS = [
    classifier_path(dataset, instance)
    for dataset in DATASETS
    for instance in INSTANCES
] if BUILD_EXPERIMENTS else []
