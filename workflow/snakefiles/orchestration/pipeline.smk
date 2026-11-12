"""
The orchestration pipeline: datasets, classifiers, pretraining, training,
evaluation and plots, less the pipeline stages `stages` turns off.

Runtime filters (--config, select subsets to run):
    stages (mapping, default=every stage on): Booleans that turn off whole
        pipeline stages: dataset_generation, training (pretraining,
        training and classifier training) and evaluation, each default
        true. A stage that is off defines no rules, so the run reads its
        outputs as they are on disk, never schedules its jobs, and fails the
        plan with MissingInputException where one is missing. An unknown
        stage or a value other than true or false fails load_config. The
        resolved configuration lists the stages that are off.
        Transformation, aggregation and visualization always define rules.
        Evaluate existing trained methods:
            "stages={dataset_generation: false, training: false}"
        Transform, aggregate and plot existing evaluation tables:
            "stages={dataset_generation: false, training: false,
            evaluation: false}"
    reuse_release (bool, default=False): Reuse existing restored release
        artifacts after regenerating omitted dataset bundles. First run
        --target all_generate_datasets without this option, then run with
        reuse_release=true "stages={dataset_generation: false}". The launcher's
        --configfile mode manages these phases automatically. Requires
        <output_root>/release_manifest.json and all dataset inputs of existing
        released artifacts on the selected dataset realizations. Their actual
        bundle content hashes must match the release's recorded input hashes;
        unknown or mismatched hashes fail before planning. Only verified
        dataset edges to existing released outputs ignore mtimes; missing
        outputs and ordinary development retain normal dependencies.
        Internal release_preview=true permits a dry run or estimate to show
        reuse conditional on matching regenerated data; execution refuses it.
    methods (list[str], required): Subset of methods from method_options.yaml
    datasets (list[str], required): Subset of datasets to run. Every dataset
        key needs a file conf/components/dataset_key/<key>.yaml.
    dataset_realization_indices (list[int], default=[0,1,2,3,4]): Subset of random seeds
    soft_budget_setting (bool, default=True): false runs only the
        hard-budget setting, as if no method listed any dataset in
        soft_budget_datasets. The launcher's --no-soft-budget sets
        it:
            uv run workflow/run.py --no-soft-budget "datasets=[cube]" -n
    per_dataset_plots (bool, default=False): Request additional per-dataset
        Accuracy, Macro F1 and AUROC figures, with external and built-in
        classifier rows. The external usual plot job owns a separately
        tracked sibling directory once per method set, using the unsplit
        merged table; built-in usual plots remain separate. The launcher
        sets it with --per-dataset-plots:
            uv run workflow/run.py --per-dataset-plots "datasets=[cube]" -n
        No optional targets are requested with the default false.
    plotting (mapping, default={}): Headline metric overrides, restricted
        to dataset_metrics (dataset key -> accuracy, f1 or auroc) and
        default_metric (accuracy, f1 or auroc). Omitted choices retain the
        Hydra plotting defaults. The mapping is forwarded to both usual
        classifier jobs, not used to replace the optional three metrics:
            uv run workflow/run.py "plotting={dataset_metrics: {cube: auroc}}" -n
        Invalid keys, types and metric identifiers fail load_config.
        See docs/reference/evaluation_plots.md for output paths and config.
    device (str, default='cpu'): Deprecated invocation-wide device for
        computational jobs, with a warning. Cannot be combined with execution.
    execution_file (str, default='workflow/profiles/execution/default.yaml'):
        YAML file holding the execution mapping below. Use
        workflow/profiles/execution/cpu.yaml to run every job on CPU, as
        local runs without GPUs need. Ignored when execution or the
        deprecated device is given.
    execution (mapping, default=the execution_file's mapping): CPU/cuda defaults for the pipeline
        stages classifier_training, pretraining, training and evaluation.
        methods.<name> overrides training, evaluation and method-specific
        classifier_training; pretrained_models overrides pretraining by named
        model. External classifiers use only the classifier_training default. Overrides take precedence over defaults.
        Shipped declarations: workflow/profiles/execution/{default,cpu}.yaml.
    execution_site_file (str, required for SLURM submission): The site
        profile's YAML allocation map (<site>/site.yaml); submitting without
        one fails before any job. The pipeline launcher (workflow/run.py
        --site <site>) sets it, also with overrides, which replace the site
        profile's config. Alternatively provide execution_site in a
        configuration file.
        An allocation may name the venv its jobs run in
        (execution_site.<cpu|gpu>.venv, relative to the repository root),
        on a cluster whose nodes differ in architecture: the pipeline
        launcher then runs Snakemake from the login node's venv and refuses
        to submit while a venv is missing or built from another uv.lock,
        and the site profile's precommand puts workflow/venvs/bin, whose
        python runs the node's venv, on PATH.
        See docs/how-to/run_on_slurm.md and
        docs/adr/0015-one-venv-per-architecture.md.
    use_wandb (bool, default=False): Enable W&B logging; the launcher's
        --wandb sets it. Needs WANDB_ENTITY and WANDB_PROJECT set, and
        wandb_group, or load fails. The classifier training, pretraining,
        training and evaluation rules then pass each job its W&B run ID,
        name and group in WANDB_RUN_ID, WANDB_NAME and WANDB_RUN_GROUP
        (workflow/src/wandb_runs.py), and the terminal links the run when
        the job is submitted or starts and when it finishes.
    wandb_group (str, required with use_wandb): The W&B group every W&B
        run of this invocation joins, linked in the resolved
        configuration. The launcher's --wandb names a new one per
        invocation; --wandb-group <name> sets it, to continue a group:
            uv run workflow/run.py --wandb-group all-20261012T121005Z-k3f9
    list_jobs (bool, default=False): Print every job's messages (its rule
        block, shell command, submission and completion) in the terminal,
        not only the resolved configuration, job counts, progress, each
        submitted or started and each finished job's rule and wildcards,
        and errors. A real run's log file under .snakemake/log has them either
        way (workflow/src/job_output.py).
    smoke_test (bool, default=False): Run smoke tests, under their own
        output root
    output_root (str, default='output/production', or
        'output/smoke' with smoke_test=true): Where the run writes its
        bundles, evaluation tables, job records, merged results and plots.
        It cannot overlap the other kind of run's default root, so smoke
        outputs never satisfy a real run.
    initializer (str, default='cold'): Initialization strategy, a file in
        conf/components/initializer/
    eval_dataset_split (str, default='test'): Dataset split for evaluation

Training contract:
    The pretrain_model and train_method rules (rules/training.smk) pass every
    pretraining and training script the plain `key=value` training contract
    rendered by workflow/src/contract_arguments.py: dataset, classifier
    and pretrained-model bundle paths, save_path, method_name (the rule's
    method wildcard), initializer, unmasker, dataset_key, hard_budget,
    soft_budget_param, device, seed, use_wandb and smoke_test (pretraining
    receives no method name, no pretrained model and no budgets). Methods
    add their own arguments through method_specific_params and
    pretrain_params. See docs/adr/0001-training-contract-as-library.md.
    Every bundle a rule writes carries a provenance record in its manifest
    (docs/adr/0002-provenance-recorded-in-artifacts.md): the training
    scripts record the contract's seed, inputs and method name; the
    classifier and dataset generation scripts record theirs.

Job records:
    Every computational job (dataset generation, classifier training,
    pretraining, training, evaluation and transformation) runs its script
    through `python -m afabench.core.job_record`, rendered by
    workflow/src/job_records.py. It writes a job record, a declared
    output named after the job's artifact with `.job_record.json` in place
    of its suffix (dataset generation: dataset_generation.job_record.json in
    the dataset realization's folder): the job's identity, job duration and resolved
    allocation (device, CPUs, GPUs, time limit). A job whose script fails, or that
    receives SLURM's time-limit SIGTERM, writes its record to the same path
    under the output root's failed_job_records/ instead, one record per attempt;
    Snakemake deletes a failed job's declared outputs. The
    collect_job_records rule, part of `all`, collects every job record
    under the output root, failed attempts included, into the job duration
    table <output_root>/merged_results/job_duration_table.parquet, one row
    per record (afabench.core.job_duration_table). The plot_time rule plots
    from it the job durations of the completed pretraining, method-specific
    classifier training, training, evaluation and transformation jobs of
    `methods` under this initializer and eval split; failed and timed-out
    attempts are left out. See docs/reference/job_records.md and
    docs/adr/0007-job-records-beside-artifacts.md.

Workflow configuration recording:
    Before executing jobs, the main process writes the effective Snakemake
    config, config file paths and hashes, and overrides to
    <output_root>/workflow_config.json. Release creation reads this record
    automatically. The record also freezes pipeline-level benchmark defaults
    for the release's runnable benchmark.yaml; publisher execution/site and
    invocation controls are excluded from that mapping.
    Dry runs, estimates, and invocations with no jobs do not
    replace it; cluster workers never write it.

Execution configuration and required files:
    Methods retain their independent scripts and native bundle/result paths.
    Site profiles own CPU/GPU partition, account and GPU request syntax;
    CPU counts, memory and runtime remain separate resource settings.
    Invalid selected-method and prerequisite execution fails before submission,
    including conflicting device arguments in classifier/pretraining params,
    unknown pretrained_models names and default-resources slurm_extra.
    See docs/how-to/customize_pipeline.md for execution YAML and
    docs/how-to/run_on_slurm.md for site.yaml and resource settings.

Usage:
    Run it with the pipeline launcher, workflow/run.py (uv run workflow/run.py -h).
    Full benchmark, one invocation from an authorized SLURM submit host of a
    single cluster, with the repository, environment and outputs on a shared
    filesystem (execution/default.yaml places the GPU jobs):
        uv run workflow/run.py kdd26 --site <site> -n
    It prints the resolved configuration and the job counts; pass
    list_jobs=true to also list every job with its resources and device
    arguments. Then remove -n.
    Full method set: uv run workflow/run.py --site <site> (pipeline profile all).
    Extend a downloaded benchmark without reconstructing a profile:
        uv run workflow/run.py --configfile output/more_seeds/benchmark.yaml --site <site> -n
    Local CPU smoke test without SLURM or GPUs:
        uv run workflow/run.py --jobs 8 "datasets=[cube]" \
            "dataset_realization_indices=[0]" smoke_test=true \
            execution_file=workflow/profiles/execution/cpu.yaml
    It writes under output/smoke, so a later real run still runs every
    job.
    See docs/how-to/run_pipeline.md and docs/how-to/run_on_slurm.md.
    Compare a method with a benchmark release: download the published
    methods' chains into the output root, then run a pipeline profile
    listing them in `methods` next to the new method. The release is a
    cache: the run reuses what it holds and produces the rest, so the
    outputs are those the profile gives without it (ADR 0012):
        uv run python scripts/release/snapshot.py download <release_id> \
            --repo-id <repo_id> --method random_dummy --method gdfs
        uv run workflow/run.py <pipeline> --jobs 8 -n
    For datasets whose bundles were omitted from the release:
        uv run workflow/run.py <pipeline> --jobs 8 --target all_generate_datasets
        uv run workflow/run.py <pipeline> --jobs 8 reuse_release=true \
            "stages={dataset_generation: false}" -n
    See docs/how-to/manage_releases.md.
    Evaluate trained methods already under the output root, without
    generating datasets or training:
        uv run workflow/run.py <pipeline> --site <site> -n \
            --target all_eval_methods \
            "stages={dataset_generation: false, training: false}"

CPU-only processing:
    Dataset generation (when its stage is on), transformations, aggregation and
    visualization always resolve to CPU, including with legacy device=cuda.
    These fixed stages have no execution defaults/overrides. The profile's
    execution_site.cpu allocation maps their partition/account and clears GPU
    requests; CPU counts, memory and runtime remain independently configurable.
    Conflicting rule allocation overrides fail before any submission. Heavy
    processing is submitted normally, not designated as login-node/local work.
    See docs/how-to/run_on_slurm.md for CPU allocation requirements.

Output namespacing:
    - Every rule addresses dataset, classifier, pretrained-model and method
      bundles, evaluation tables, job records, merged tables, the job
      duration table and plots under the output root (output_root) through
      OUTPUT_LAYOUT (src/afabench/core/output_layout.py), the native layout
      that release manifests and restored benchmark releases also use. No
      rule spells a path under the output root itself.
    - All initializer-dependent artifacts are stored under
      `initializer-<initializer>` to allow side-by-side comparisons
      (for example `cold` vs `missingness`) without overwriting.

Config files (--configfile):
    The merged config is validated by load_config
    (src/afabench/core/workflow_settings.py) through ../settings.smk, which
    defines WORKFLOW_SETTINGS, the typed settings the rules query per method
    and dataset, OUTPUT_LAYOUT and the few scalars the rules read. Unknown
    method_options keys, a missing train_script_name, a
    pretrained_model_name outside pretrain_mapping, a method missing from
    method_options or soft_budget_params, and an ignored dataset that is not
    a dataset key all fail the parse, naming the method, as does a malformed
    pretrain_mapping entry, naming the pretrained model; see
    docs/reference/pipeline_configuration.md.
    Fixed definitions:
        method_options.yaml,
        pretrain_mapping.yaml,
        methods.yaml
        classifier_names.yaml
    Runtime params:
        eval_hard_budgets.yaml,
        soft_budget_params_*.yaml,
        unmaskers.yaml (values are files in conf/components/unmasker/)

    Note: method_options.yaml can include train_hard_budget to train every
    hard-budget run of a method on a dataset with one hard budget, or
    eval_to_train_hard_budget_mapping to give the train hard budget of each
    eval hard budget per method/dataset; a mapping must cover every eval hard
    budget the dataset gets.

    Note: method_options.yaml can include eval_batch_size to specify different
    batch sizes for evaluation per method and dataset. Format:
        eval_batch_size:
          default: <batch_size>
          <dataset_name>: <batch_size>

    Note: method_options.yaml can include hard_budget_ignored_datasets to skip
    hard budget training/evaluation for specific datasets per method. Format:
        hard_budget_ignored_datasets: [dataset1, dataset2, ...]
    When set, hard budget combinations are excluded for those datasets.

    Note: method_options.yaml lists in soft_budget_datasets the datasets a
    method runs soft budget training/evaluation on (opt-in, ADR 0013). Format:
        soft_budget_datasets: [dataset1, dataset2, ...]
    A dataset left out gets no soft budget combinations.

    Note: Pretraining is skipped per pretrained model for datasets that every
    method consuming that model ignores for hard budgets and does not list
    for soft budgets.
"""

include: "../settings.smk"

# A stage that is off defines no rules, so its outputs are read as they are.
STAGES = WORKFLOW_SETTINGS.stages
if STAGES.dataset_generation:
    include: "../rules/dataset_generation.smk"
if STAGES.training:
    include: "../rules/training.smk"
    include: "../rules/classifier_training.smk"
if STAGES.evaluation:
    include: "../rules/evaluation.smk"

include: "../rules/transformations.smk"
include: "../rules/aggregation.smk"
include: "../rules/visualization.smk"
include: "../rules/helpers.smk"
