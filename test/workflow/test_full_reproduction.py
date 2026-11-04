"""Single-invocation full reproduction through the real orchestration."""

import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from test.workflow.submission_harness import WorkflowHarness
from test.workflow.test_compute_estimate_planning import (
    planned_allocation,
    submitted_allocation,
)
from test.workflow.test_cpu_processing_execution import processing_workflow

# Membership of the retired workflow/conf/methods/gpu.yaml, whose
# methods were trained and evaluated in the separate GPU invocation.
FORMER_GPU_METHODS = {
    "jafa",
    "odin_model_free",
    "odin_model_based",
    "gdfs",
    "eddi_builtin",
    "eddi_external",
    "dime",
}
# Methods added since, which run on the GPU nodes and have jobs on the cube.
# (l2m has none there: it ignores the cube.)
GPU_METHODS_ADDED_SINCE = {"l2m_tabicl_prior"}
# Membership of the retired workflow/conf/methods/cpu.yaml.
FORMER_CPU_METHODS = {
    "ol_without_mask",
    "ol_with_mask",
    "random_dummy",
    "sequential_dummy",
    "cae",
    "permutation",
    "aaco",
    "aaco_nn",
}
KDD26_METHODS = {
    "jafa",
    "odin_model_based",
    "ol_with_mask",
    "gdfs",
    "eddi_external",
    "dime",
    "cae",
    "aaco",
}


def planned_jobs(output: str) -> list[dict[str, str]]:
    jobs = []
    for block in output.split("\nrule ")[1:]:
        rule, _, body = block.partition(":")
        job = {"rule": rule}
        for line in body.splitlines():
            key, _, value = line.strip().partition(": ")
            if key in {"wildcards", "resources"}:
                job[key] = value
        shell = body.partition("Shell command:")[2]
        for argument in shell.split():
            if argument.startswith("device="):
                job["device"] = argument.removeprefix("device=")
        jobs.append(job)
    return jobs


# The two shipped pipeline profiles, run on a cluster as documented: with
# the pipeline launcher's --site and pipeline profile overrides.
CLUSTER_PIPELINE_PROFILES = ["all", "kdd26"]
CLUSTER_SITE = ["--site", "examples/mixed-gres"]
SMALL_SELECTION = ["datasets=[cube]", "dataset_realization_indices=[0]"]


@pytest.mark.parametrize(
    ("pipeline_profile", "methods"),
    [
        (
            "all",
            FORMER_GPU_METHODS | FORMER_CPU_METHODS | GPU_METHODS_ADDED_SINCE,
        ),
        ("kdd26", KDD26_METHODS),
    ],
)
def test_cluster_pipeline_profile_declares_former_six_stage_hardware(
    tmp_path: Path, pipeline_profile: str, methods: set[str]
) -> None:
    workflow = WorkflowHarness(tmp_path)
    shutil.rmtree(tmp_path / "output/smoke")

    result = workflow.run_launcher(
        [
            pipeline_profile,
            *CLUSTER_SITE,
            "-n",
            *SMALL_SELECTION,
            "list_jobs=true",
        ]
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "deprecated" not in output
    devices: dict[tuple[str, str], set[str]] = {}
    for job in planned_jobs(output):
        gpu = "gres=gpu:T4:1" in job.get("resources", "")
        if job["rule"] in {"train_method", "eval_method"}:
            method = job["wildcards"].split(",")[0].removeprefix("method=")
            devices.setdefault((job["rule"], method), set()).add(job["device"])
            assert gpu is (job["device"] == "cuda"), job
        elif job["rule"] in {"train_classifier", "pretrain_model"}:
            assert job["device"] == "cuda", job
            assert gpu, job
        elif job["rule"] != "all":
            assert "gpu=0" in job["resources"], job
            assert "slurm_partition=cpu-queue" in job["resources"], job
    for rule in ["train_method", "eval_method"]:
        assert {
            method for stage, method in devices if stage == rule
        } == methods
        for method in methods:
            assert devices[rule, method] == (
                {"cuda"}
                if method in FORMER_GPU_METHODS | GPU_METHODS_ADDED_SINCE
                else {"cpu"}
            )


def test_cpu_execution_file_runs_every_job_on_cpu(tmp_path: Path) -> None:
    workflow = WorkflowHarness(tmp_path)
    shutil.rmtree(tmp_path / "output/smoke")

    result = workflow.run_invocation(
        ["--profile", "workflow/profiles/pipeline/all"],
        "--dry-run",
        "--config",
        "datasets=[cube]",
        "dataset_realization_indices=[0]",
        "smoke_test=True",
        "execution_file=workflow/profiles/execution/cpu.yaml",
        "list_jobs=true",
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "deprecated" not in output
    jobs = [job for job in planned_jobs(output) if job["rule"] != "all"]
    assert {job["rule"] for job in jobs} >= {
        "dataset_generation",
        "train_classifier",
        "pretrain_model",
        "train_method",
        "eval_method",
        "plot_eval_perf",
    }
    for job in jobs:
        assert job.get("device", "cpu") == "cpu", job
        assert "gpu=0" in job["resources"], job
        assert "gres=," in job["resources"], job


def test_alvis_profile_plans_cuda_methods_and_cpu_processing(
    tmp_path: Path,
) -> None:
    workflow = processing_workflow(tmp_path)

    result = workflow.run(
        "--dry-run",
        "--workflow-profile",
        str(tmp_path / "workflow/profiles/site/alvis"),
        target="all",
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    jobs = [job for job in planned_jobs(output) if job["rule"] != "all"]
    assert "dataset_generation" in {job["rule"] for job in jobs}
    for job in jobs:
        cuda = job["rule"] in {"train_method", "eval_method"}
        assert job.get("device", "cpu") == ("cuda" if cuda else "cpu"), job
        resources = dict(
            item.split("=", 1) for item in job["resources"].split(", ")
        )
        assert resources["slurm_partition"] == "alvis", job
        assert resources["slurm_account"] == "NAISS2026-4-39", job
        assert resources["gres"] == ("gpu:T4:1" if cuda else ""), job
        assert resources["gpu"] == "0", job
        assert resources["slurm_extra"] == "", job


METHODS = ["alpha", "beta", "gamma", "delta"]
# Two dataset realizations, so per-realization wiring cannot pass by
# accident.
REALIZATIONS = ["0", "1"]


def full_benchmark_workflow(root: Path) -> WorkflowHarness:
    workflow = processing_workflow(root)
    shutil.rmtree(root / "output/smoke")
    workflow.config.update(
        {
            "methods": METHODS,
            "dataset_realization_indices": [int(k) for k in REALIZATIONS],
            "method_options": {
                "alpha": {
                    "train_script_name": "alpha",
                    "pretrained_model_name": "shared",
                    "eval_batch_size": 1,
                    "classifier": {"script_name": "special"},
                },
                "beta": {
                    "train_script_name": "beta",
                    "pretrained_model_name": "shared",
                    "eval_batch_size": 1,
                },
                "gamma": {
                    "train_script_name": "gamma",
                    "pretrained_model_name": "other",
                    "eval_batch_size": 1,
                },
                "delta": {"train_script_name": "delta", "eval_batch_size": 1},
            },
            "soft_budget_params": {
                method: {"default": []} for method in METHODS
            },
            "pretrain_mapping": {
                "shared": {"pretrain_script_name": "shared"},
                "other": {"pretrain_script_name": "other"},
            },
            "method_sets": {"heatmap_comparison": METHODS},
            "execution": {
                "defaults": {
                    "classifier_training": "cuda",
                    "pretraining": "cpu",
                    "training": "cpu",
                    "evaluation": "cpu",
                },
                "methods": {
                    "alpha": {
                        "classifier_training": "cpu",
                        "training": "cuda",
                        "evaluation": "cpu",
                    },
                    "beta": {"evaluation": "cuda"},
                },
                "pretrained_models": {"shared": "cuda"},
            },
        }
    )
    for script in [
        "train_method/gamma.py",
        "train_method/delta.py",
        "train_classifier/special.py",
        "pretrain_model/other.py",
    ]:
        shutil.copyfile(
            root / "scripts/train_method/alpha.py", root / "scripts" / script
        )
    # Computational stubs fail unless their declared bundle inputs exist, so
    # a job submitted before its prerequisites completed cannot pass.
    for directory in [
        "train_method",
        "train_classifier",
        "pretrain_model",
        "eval",
    ]:
        for path in (root / "scripts" / directory).glob("*.py"):
            path.write_text(
                "import sys\n"
                "from pathlib import Path\n"
                "inputs = dict(arg.split('=', 1) for arg in sys.argv[1:])\n"
                "for key, value in inputs.items():\n"
                "    if key.endswith(('_bundle_path', '_dataset_path')) and value != 'null':\n"
                "        assert Path(value).is_dir(), (key, value)\n"
                + path.read_text()
            )
    return workflow


# Jobs whose resolved execution choice is cuda in full_benchmark_workflow.
GPU_JOBS = {
    ("train_classifier", "cube"),
    ("pretrain_model", "shared"),
    ("train_method", "alpha"),
    ("eval_method", "beta"),
}
# Every job of the tiny full graph, each submitted exactly once.
FULL_GRAPH_JOBS = {
    "dataset_generation": 2,
    "train_classifier": 2,
    "train_classifier_for_method": 2,
    "pretrain_model": 4,
    "train_method": 8,
    "eval_method": 8,
    "transform_eval_data": 8,
    # Both the reserved all set and heatmap_comparison need their own tables
    # and one performance plot job for each classifier.
    "merge_eval_perf": 2,
    "split_by_classifier_type": 2,
    "collect_job_records": 1,
    "plot_eval_perf": 4,
    "plot_eval_actions": 1,
    "plot_time": 1,
}


def test_full_graph_plans_every_job_once(tmp_path: Path) -> None:
    workflow = full_benchmark_workflow(tmp_path)
    result = workflow.run("--dry-run", "--printshellcmds", target="all")
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    jobs = Counter(
        job["rule"] for job in planned_jobs(output) if job["rule"] != "all"
    )
    assert jobs == FULL_GRAPH_JOBS


def submitted_job(args: list[str]) -> tuple[str, str]:
    """Return the rule and first wildcard value of a captured submission."""
    comment = args[args.index("--comment") + 1].removeprefix("rule_")
    rule, _, wildcards = comment.partition("_wildcards_")
    return rule, wildcards.split("_")[0]


@dataclass
class FullGraphRun:
    root: Path
    jobs: list[tuple[str, str]]
    submissions: list[list[str]]
    calls: list[tuple[str, dict[str, str]]]

    def position(self, rule: str, identity: str = "") -> int:
        return next(
            index
            for index, (job_rule, job_identity) in enumerate(self.jobs)
            if job_rule == rule and job_identity.startswith(identity)
        )


@pytest.fixture(scope="module")
def full_graph_run(tmp_path_factory: pytest.TempPathFactory) -> FullGraphRun:
    root = tmp_path_factory.mktemp("full-graph")
    workflow = full_benchmark_workflow(root)
    result = workflow.run(
        "--workflow-profile",
        str(root / "workflow/profiles/site/examples/mixed-gres"),
        "--slurm-init-seconds-before-status-checks",
        "0",
        "--seconds-between-status-checks",
        "1",
        target="all",
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    submissions = workflow.submissions()
    return FullGraphRun(
        root,
        [submitted_job(args) for args in submissions],
        submissions,
        workflow.script_arguments(),
    )


@pytest.mark.pipeline
def test_full_graph_submits_every_job_once(
    full_graph_run: FullGraphRun,
) -> None:
    comments = [
        args[args.index("--comment") + 1]
        for args in full_graph_run.submissions
    ]
    assert len(set(comments)) == len(comments)
    assert Counter(rule for rule, _ in full_graph_run.jobs) == FULL_GRAPH_JOBS
    # Every job but collect_job_records, which runs for real, runs a stub.
    assert len(full_graph_run.calls) == len(comments) - 1
    plots = (
        full_graph_run.root
        / "output/smoke/plots/eval_split-test/initializer-cold"
    )
    assert len(list(plots.rglob("fixture.svg"))) == 6


@pytest.mark.pipeline
def test_full_graph_plan_matches_its_submissions(
    tmp_path: Path, full_graph_run: FullGraphRun
) -> None:
    workflow = full_benchmark_workflow(tmp_path)

    plan = workflow.plan(
        "--workflow-profile",
        str(tmp_path / "workflow/profiles/site/examples/mixed-gres"),
        target="all",
    )

    assert plan.returncode == 0, plan.stdout + plan.stderr
    assert sorted(map(planned_allocation, workflow.planned_jobs())) == sorted(
        map(submitted_allocation, full_graph_run.submissions)
    )


@pytest.mark.pipeline
def test_full_graph_submissions_mix_cpu_and_gpu_allocations(
    full_graph_run: FullGraphRun,
) -> None:
    for args, job in zip(
        full_graph_run.submissions, full_graph_run.jobs, strict=True
    ):
        gpu = job in GPU_JOBS
        assert args[args.index("-p") + 1] == (
            "gpu-queue" if gpu else "cpu-queue"
        )
        assert args[args.index("-A") + 1] == (
            "gpu-account" if gpu else "cpu-account"
        )
        gpu_requests = [
            arg for arg in args if arg.startswith(("--gres", "--gpus"))
        ]
        assert gpu_requests == (["--gres=gpu:T4:1"] if gpu else [])
        method_job = job[0] in {"train_method", "eval_method"}
        assert args[args.index("-t") + 1] == ("600" if method_job else "120")
        assert f"--cpus-per-task={8 if method_job else 1}" in args


@pytest.mark.pipeline
def test_full_graph_job_records_carry_submitted_allocations(
    full_graph_run: FullGraphRun,
) -> None:
    records = [
        json.loads(path.read_text())
        for path in (full_graph_run.root / "output/smoke").rglob(
            "*.job_record.json"
        )
    ]
    # Every job but aggregation and visualization leaves one.
    assert len(records) == 34
    assert sum(record["device"] == "cuda" for record in records) == 8
    for record in records:
        method_job = record["stage"] in {"training", "evaluation"}
        assert record["cpus"] == (8 if method_job else 1), record
        assert record["gpus"] == (1 if record["device"] == "cuda" else 0)


@pytest.mark.pipeline
def test_full_graph_submits_jobs_after_their_dependencies(
    full_graph_run: FullGraphRun,
) -> None:
    position = full_graph_run.position
    assert full_graph_run.jobs[0] == ("dataset_generation", "cube")
    for method, pretrained, classifier in [
        ("alpha", "shared", ("train_classifier_for_method", "alpha")),
        ("beta", "shared", ("train_classifier", "cube")),
        ("gamma", "other", ("train_classifier", "cube")),
        ("delta", None, ("train_classifier", "cube")),
    ]:
        training = position("train_method", method)
        evaluation = position("eval_method", method)
        assert position("train_classifier", "cube") < training
        if pretrained is not None:
            assert position("pretrain_model", pretrained) < training
        assert training < evaluation
        assert position(*classifier) < evaluation
        assert evaluation < position("transform_eval_data", method)
        assert position("transform_eval_data", method) < position(
            "merge_eval_perf"
        )
        assert position("transform_eval_data", method) < position(
            "collect_job_records"
        )
    assert position("merge_eval_perf") < position("split_by_classifier_type")
    assert position("split_by_classifier_type") < position("plot_eval_perf")
    assert position("collect_job_records") < position("plot_time")


@pytest.mark.pipeline
def test_full_graph_scripts_receive_resolved_devices(
    full_graph_run: FullGraphRun,
) -> None:
    devices = {
        (script.removeprefix("scripts/"), args["save_path"]): args["device"]
        for script, args in full_graph_run.calls
        if "device" in args
    }
    assert {(script, device) for (script, _), device in devices.items()} == {
        ("train_classifier/masked_mlp_classifier.py", "cuda"),
        ("train_classifier/special.py", "cpu"),
        ("pretrain_model/shared.py", "cuda"),
        ("pretrain_model/other.py", "cpu"),
        ("train_method/alpha.py", "cuda"),
        ("train_method/beta.py", "cpu"),
        ("train_method/gamma.py", "cpu"),
        ("train_method/delta.py", "cpu"),
        ("eval/eval_afa_method.py", "cuda"),
        ("eval/eval_afa_method.py", "cpu"),
    }
    for (script, save_path), device in devices.items():
        if script == "eval/eval_afa_method.py":
            assert device == ("cuda" if "/beta/" in save_path else "cpu")


def realization_of(args: dict[str, str]) -> str:
    """Return the dataset realization a script call's save path belongs to."""
    match = re.search(r"realization_index-(\d+)", args["save_path"])
    assert match is not None, args["save_path"]
    return match.group(1)


@pytest.mark.pipeline
def test_full_graph_preserves_contract_and_shared_prerequisites(
    full_graph_run: FullGraphRun,
) -> None:
    shared = {
        realization_of(args): args["save_path"]
        for script, args in full_graph_run.calls
        if script.endswith("pretrain_model/shared.py")
    }
    assert shared == {
        k: "output/smoke/pretrained_models/initializer-cold/shared/"
        f"dataset-cube+realization_index-{k}/pretrain_seed-{k}/model.bundle"
        for k in REALIZATIONS
    }
    for script, args in full_graph_run.calls:
        if "train_method/" in script:
            assert args["seed"] == realization_of(args)
            assert args["hard_budget"] == "1"
            assert args["soft_budget_param"] == "null"
            assert args["smoke_test"] == "True"
            assert args["initializer"] == "cold"
            assert args["unmasker"] == "direct"
            assert args["dataset_key"] == "cube"
        if script.endswith(("train_method/alpha.py", "train_method/beta.py")):
            assert (
                args["pretrained_model_bundle_path"]
                == (shared[realization_of(args)])
            )


@pytest.mark.pipeline
def test_full_graph_trains_classifiers_per_dataset_realization(
    full_graph_run: FullGraphRun,
) -> None:
    classifiers = {
        (script.removeprefix("scripts/"), args["save_path"]): args
        for script, args in full_graph_run.calls
        if "train_classifier/" in script
    }
    assert set(classifiers) == {
        (
            f"train_classifier/{script}.py",
            "output/smoke/trained_classifiers/initializer-cold/"
            f"{owner}dataset-cube+realization_index-{k}.bundle",
        )
        for script, owner in [
            ("masked_mlp_classifier", ""),
            ("special", "method-alpha+"),
        ]
        for k in REALIZATIONS
    }
    for args in classifiers.values():
        k = realization_of(args)
        assert args["train_dataset_path"] == (
            f"output/smoke/datasets/cube/{k}/train.bundle"
        )
        assert args["val_dataset_path"] == (
            f"output/smoke/datasets/cube/{k}/val.bundle"
        )
        assert args["seed"] == k


@pytest.mark.pipeline
def test_full_graph_hands_each_stage_its_realizations_classifier(
    full_graph_run: FullGraphRun,
) -> None:
    stages = ("pretrain_model/", "train_method/", "eval/eval_afa_method.py")
    calls = [
        (script, args)
        for script, args in full_graph_run.calls
        if any(stage in script for stage in stages)
    ]
    # Two pretrained models, four methods trained and evaluated, per
    # dataset realization.
    assert len(calls) == 10 * len(REALIZATIONS)
    for script, args in calls:
        k = realization_of(args)
        owner = (
            "method-alpha+"
            if "train_method/alpha.py" in script
            or "/alpha/" in args["save_path"]
            else ""
        )
        assert args["classifier_bundle_path"] == (
            "output/smoke/trained_classifiers/initializer-cold/"
            f"{owner}dataset-cube+realization_index-{k}.bundle"
        ), (script, args["save_path"])


@pytest.mark.parametrize(
    "execution", [{"methods": {"alpha": {"training": "cuda"}}}, {}]
)
def test_cli_config_without_site_file_fails_before_submission(
    tmp_path: Path, execution: dict[str, object]
) -> None:
    # Snakemake replaces a workflow profile's whole `config` with the CLI one.
    workflow = WorkflowHarness(tmp_path)
    workflow.config["execution"] = execution

    result = workflow.run(
        "--workflow-profile",
        "workflow/profiles/site/examples/mixed-gres",
        "--config",
        "smoke_test=True",
        target="all_train_methods",
    )

    assert result.returncode != 0
    assert "workflow/run.py --site" in result.stdout + result.stderr
    assert workflow.submissions() == []


# Each job starts a nested Snakemake in the fake sbatch, so this takes ~10 s.
@pytest.mark.optional
def test_cli_config_with_repeated_site_file_keeps_site_allocations(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["execution"] = {"methods": {"alpha": {"training": "cuda"}}}

    result = workflow.submit_first_wave(
        2,
        "--workflow-profile",
        "workflow/profiles/site/examples/mixed-gres",
        "--config",
        "smoke_test=True",
        "execution_site_file=workflow/profiles/site/examples/mixed-gres/site.yaml",
        target="all_train_methods",
    )

    gpu_job = next(
        (
            args
            for args in workflow.submissions()
            if submitted_job(args) == ("train_method", "alpha")
        ),
        None,
    )
    assert gpu_job is not None, result.stdout
    assert gpu_job[gpu_job.index("-p") + 1] == "gpu-queue"
    assert gpu_job[gpu_job.index("-A") + 1] == "gpu-account"
    assert "--gres=gpu:T4:1" in gpu_job


@pytest.mark.pipeline
@pytest.mark.parametrize("pipeline_profile", CLUSTER_PIPELINE_PROFILES)
def test_cluster_pipeline_profile_submits_declared_hardware(
    tmp_path: Path, pipeline_profile: str
) -> None:
    workflow = WorkflowHarness(tmp_path)
    (tmp_path / "scripts/pretrain_model").mkdir()
    for script in [
        "pretrain_model/jafa.py",
        "train_method/jafa.py",
        "train_method/aaco.py",
    ]:
        shutil.copyfile(
            tmp_path / "scripts/train_method/alpha.py",
            tmp_path / "scripts" / script,
        )

    result = workflow.run_launcher(
        [
            pipeline_profile,
            *CLUSTER_SITE,
            "--target",
            "all_eval_methods",
            "methods=[jafa,aaco]",
            "use_wandb=False",
            "smoke_test=True",
            *SMALL_SELECTION,
            "--",
            # Submit each dependency wave at once; the plugin waits per wave.
            "--jobs",
            "100",
            "--slurm-init-seconds-before-status-checks",
            "0",
            "--seconds-between-status-checks",
            "1",
        ],
        timeout=600,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    submissions = workflow.submissions()
    jobs = [submitted_job(args) for args in submissions]
    assert {rule for rule, _ in jobs} == {
        "pretrain_model",
        "train_method",
        "eval_method",
    }
    assert {identity for _, identity in jobs} == {"jafa", "aaco"}
    for args, (rule, identity) in zip(submissions, jobs, strict=True):
        gpu = rule == "pretrain_model" or identity == "jafa"
        assert args[args.index("-p") + 1] == (
            "gpu-queue" if gpu else "cpu-queue"
        )
        assert ("--gres=gpu:T4:1" in args) is gpu
    for script, args in workflow.script_arguments():
        gpu = "jafa" in args["save_path"]
        assert args["device"] == ("cuda" if gpu else "cpu"), script


@pytest.mark.pipeline
def test_local_cpu_smoke_runs_the_same_full_graph(tmp_path: Path) -> None:
    workflow = full_benchmark_workflow(tmp_path)
    del workflow.config["execution"]
    workflow.config["execution_file"] = "workflow/profiles/execution/cpu.yaml"

    result = workflow.run("--executor", "local", target="all")

    assert result.returncode == 0, result.stdout + result.stderr
    assert workflow.submissions() == []
    calls = workflow.script_arguments()
    # Every job but collect_job_records, which runs for real, runs a stub.
    assert len(calls) == sum(FULL_GRAPH_JOBS.values()) - 1
    for script, args in calls:
        if "device" in args:
            assert args["device"] == "cpu", script
            assert args["smoke_test"] == "True", script
    plots = tmp_path / "output/smoke/plots/eval_split-test/initializer-cold"
    assert len(list(plots.rglob("fixture.svg"))) == 6
