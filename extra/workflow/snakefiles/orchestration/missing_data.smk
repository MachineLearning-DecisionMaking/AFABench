"""Generic experiments with missing values in training data only.

The workflow consumes the same dataset, method, pretraining, classifier,
unmasker, and hard-budget configuration files as the ordinary evaluation
pipeline. Missingness-specific scientific choices live in
``conf/missing_data/design.yaml``; runtime scale and devices live in the
selected runtime config and execution profile.

Required shared config keys:
    datasets, methods, method_options, pretrain_mapping, eval_hard_budgets,
    soft_budget_params, unmaskers, classifier_names

Missing-data config keys:
    artifact_namespace, missingness, strategies, eval_dataset_split
    Optional ``classifier_scope`` is ``shared`` for legacy experiments or
    ``per_instance`` to fit classifiers on each dataset split independently.
    Optional ``strategy_filters`` restricts one strategy by datasets, methods,
    mechanisms, or probabilities without duplicating the experiment workflow.
    Optional ``include_method_variants`` omits missingness-specific controls
    from a focused experiment while retaining them by default.
    Optional ``classifier_source`` selects the predictor of every policy:
    ``complete`` (default) trains it on complete data, ``view`` on the
    training data of the policy, and ``restricted_view`` on the incomplete
    training data for every policy. ``view_classifier_namespace`` reads these
    predictors from another namespace.
    Optional ``paper_artifacts`` adds the exact study, route analysis, compute
    accounting, and paper figures under ``extra/output/paper/experiments/results``.
    The exact study compares myopic and two-step policies as dimension varies;
    the budget study compares myopic and full-horizon policies on
    parity branches at budgets 2, 3, and 4, with complete-data controls.
    The combined main-paper figure shows both policies across both sweeps.
    This requires the complete induced study on the validation split.
    The ``plots`` target regenerates plots/tables and schedules any missing
    upstream results. The default ``all`` target includes these artifacts.
    A namespace containing the ``native`` mechanism also writes a mandatory
    legality report. Its evaluation traces must retain source indices, prove
    that every action respected factual availability, and omit oracle or
    true-completion strategies.

Every non-local production rule writes a Snakemake benchmark TSV under
``extra/output/missing_data/benchmark/<namespace>``. These records measure the
wall time, CPU time, and peak resident memory of the exact artifact-producing
job without becoming part of the rule's completeness contract.
Hydra's per-invocation configuration and log files are redirected to temporary
storage because the scheduler log and benchmark TSV already retain the command
and resource record.

Device routing:
    ``device`` is the default. Optional ``device_overrides`` may contain
    ``methods``, ``datasets``, ``method_datasets``, and ``pretrained_models``.
    ``AFABENCH_DEVICE`` may replace only that default at execution time without
    replacing a profile's scientific config. Every CUDA-resolved rule consumes
    one ``gpu`` resource; the runner caps those tokens to physical devices.

Example:
    uv run snakemake \
      --profile extra/workflow/profiles/config/missing_data --cores 4
    Add ``--dry-run`` to inspect the DAG, or use ``missing_data_smoke`` for
    a small training/evaluation check. Dataset source requirements are listed
    in docs/tutorials/missing_data_experiments.md.
    To inspect only the synthetic appendix DAG, append ``--dry-run
    extra/output/paper/experiments/results/budget_study.pdf``. This target
    requires no benchmark datasets or trained model bundles.
"""



include: "../rules/missing_data_common.smk"
include: "../rules/missing_data_experiments.smk"
include: "../rules/missing_data_results.smk"


# Run aggregation on the submitting host.
localrules:
    all,
    audit_native_legality,
    summarize_missing_data,
    plot_missing_data,


rule all:
    input:
        EVALUATIONS,
        f"{SUMMARY_DIR}/instance_metrics.csv",
        f"{SUMMARY_DIR}/summary.csv",
        f"{SUMMARY_DIR}/action_rates.csv",
        f"{SUMMARY_DIR}/restoration_rmse.csv",
        FIGURE_DIR,
        NATIVE_AUDIT_OUTPUTS,
        PAPER_OUTPUTS,


rule plots:
    input:
        FIGURE_DIR,
        PAPER_OUTPUTS,
