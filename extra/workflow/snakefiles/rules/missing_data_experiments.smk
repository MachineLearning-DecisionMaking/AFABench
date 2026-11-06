"""Dataset, classifier, view, training, evaluation, and summary rules."""

rule audit_native_legality:
    input:
        NATIVE_EVALUATIONS,
    output:
        NATIVE_AUDIT_OUTPUTS,
    params:
        root=f"{ROOT}/eval/{EVAL_SPLIT}/{NAMESPACE}",
    shell:
        """
        python scripts/analysis/audit_native_legality.py \
            --input-root={params.root} --output={output}
        """


rule generate_missing_data_dataset:
    output:
        [
            directory(
                f"{ROOT}/datasets/{DATA_NAMESPACE}/{{dataset}}/{instance}/{split}.bundle"
            )
            for instance in INSTANCES
            for split in ["train", "val", "test"]
        ]
    benchmark:
        f"{BENCHMARK_ROOT}/generate_dataset/dataset-{{dataset}}.tsv"
    params:
        instances="[" + ",".join(str(value) for value in INSTANCES) + "]",
        seeds="[" + ",".join(str(value) for value in INSTANCES) + "]",
        save_path=lambda wc: f"{ROOT}/datasets/{DATA_NAMESPACE}/{wc.dataset}",
        extra=lambda wc: runtime_params(DATASET_GENERATION_PARAMS, wc.dataset),
    shell:
        """
        python scripts/dataset_generation/generate_dataset.py \
            dataset={wildcards.dataset} instance_indices={params.instances} \
            seeds={params.seeds} save_path={params.save_path} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule train_missing_data_shared_classifier:
    input:
        train=lambda wc: raw_dataset(
            wc.dataset, classifier_instance(wc), "train"
        ),
        val=lambda wc: raw_dataset(
            wc.dataset, classifier_instance(wc), "val"
        ),
    output:
        directory(CLASSIFIER_OUTPUT),
    benchmark:
        CLASSIFIER_BENCHMARK
    resources:
        gpu=lambda wc: gpu_for_device(dataset_device(wc.dataset)),
    params:
        instance=classifier_instance,
        script=lambda wc: classifier_script_name(wc.dataset),
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: dataset_device(wc.dataset),
        extra=lambda wc: classifier_script_params(wc.dataset),
    shell:
        """
        python scripts/train_classifier/{params.script}.py \
            train_dataset_path={input.train} val_dataset_path={input.val} \
            save_path={output} components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} seed={params.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} \
            experiment@_global_={wildcards.dataset} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule train_missing_data_method_classifier:
    input:
        train=lambda wc: raw_dataset(
            wc.dataset, classifier_instance(wc), "train"
        ),
        val=lambda wc: raw_dataset(
            wc.dataset, classifier_instance(wc), "val"
        ),
    output:
        directory(METHOD_CLASSIFIER_OUTPUT),
    benchmark:
        METHOD_CLASSIFIER_BENCHMARK
    resources:
        gpu=lambda wc: gpu_for_device(
            resolve_device(
                DEFAULT_DEVICE,
                DEVICE_OVERRIDES,
                dataset=wc.dataset,
                method=wc.base_method,
            )
        ),
    params:
        instance=classifier_instance,
        script=lambda wc: METHOD_CLASSIFIER_SCRIPT_NAMES[wc.base_method],
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: resolve_device(
            DEFAULT_DEVICE,
            DEVICE_OVERRIDES,
            dataset=wc.dataset,
            method=wc.base_method,
        ),
        extra=lambda wc: METHOD_CLASSIFIER_SCRIPT_PARAMS[wc.base_method],
    shell:
        """
        python scripts/train_classifier/{params.script}.py \
            train_dataset_path={input.train} val_dataset_path={input.val} \
            save_path={output} components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} seed={params.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} \
            experiment@_global_={wildcards.dataset} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule train_missing_data_view_classifier:
    wildcard_constraints:
        strategy=r"(?!complete$)[a-z_]+",
    input:
        train=lambda wc: training_view(wc, "train"),
        val=lambda wc: training_view(wc, "val"),
    output:
        directory(VIEW_CLASSIFIER_OUTPUT),
    benchmark:
        f"{BENCHMARK_ROOT}/train_view_classifier/dataset-{{dataset}}/"
        "mechanism-{mechanism}+p-{p}/instance-{instance}/{strategy}.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(dataset_device(wc.dataset)),
    params:
        script=lambda wc: classifier_script_name(wc.dataset),
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: dataset_device(wc.dataset),
        extra=lambda wc: classifier_script_params(wc.dataset),
    shell:
        """
        python scripts/train_classifier/{params.script}.py \
            train_dataset_path={input.train} val_dataset_path={input.val} \
            save_path={output} components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} seed={wildcards.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} \
            experiment@_global_={wildcards.dataset} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule materialize_missing_training_view:
    input:
        train=lambda wc: raw_dataset(wc.dataset, wc.instance, "train"),
        val=lambda wc: raw_dataset(wc.dataset, wc.instance, "val"),
    output:
        train=directory(
            f"{ROOT}/views/base/{DATA_NAMESPACE}/dataset-{{dataset}}/"
            "mechanism-{mechanism}+p-{p}/instance-{instance}/"
            "{strategy}/train.bundle"
        ),
        val=directory(
            f"{ROOT}/views/base/{DATA_NAMESPACE}/dataset-{{dataset}}/"
            "mechanism-{mechanism}+p-{p}/instance-{instance}/"
            "{strategy}/val.bundle"
        ),
    benchmark:
        f"{BENCHMARK_ROOT}/materialize_view/dataset-{{dataset}}/"
        "mechanism-{mechanism}+p-{p}+instance-{instance}+"
        "strategy-{strategy}.tsv"
    params:
        p_obs=MISSINGNESS["p_obs"],
        p_params=MISSINGNESS["p_params"],
        exclude_inputs=str(bool(MISSINGNESS["exclude_inputs"])).lower(),
    shell:
        """
        python scripts/missing_values/materialize_training_views.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            train_save_path={output.train} val_save_path={output.val} \
            strategy={wildcards.strategy} seed={wildcards.instance} \
            missingness.mechanism={wildcards.mechanism} \
            missingness.p={wildcards.p} missingness.p_obs={params.p_obs} \
            missingness.p_params={params.p_params} \
            missingness.exclude_inputs={params.exclude_inputs} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule pretrain_incomplete_restoration_pvae:
    input:
        train=lambda wc: base_view(
            wc.dataset, wc.mechanism, wc.p, wc.instance, "restricted", "train"
        ),
        val=lambda wc: base_view(
            wc.dataset, wc.mechanism, wc.p, wc.instance, "restricted", "val"
        ),
        classifier=lambda wc: classifier_path(wc.dataset, wc.instance),
    output:
        directory(
            f"{ROOT}/restoration_pvae/{DATA_NAMESPACE}/incomplete/"
            "dataset-{dataset}/mechanism-{mechanism}+p-{p}/"
            "instance-{instance}/model.bundle"
        ),
    benchmark:
        f"{BENCHMARK_ROOT}/pretrain_restoration_pvae/incomplete/"
        "dataset-{dataset}+mechanism-{mechanism}+p-{p}+"
        "instance-{instance}.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(dataset_device(wc.dataset, "pvae")),
    params:
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: dataset_device(wc.dataset, "pvae"),
        extra=lambda wc: runtime_params(
            RESTORATION_PVAE_PARAMS, wc.dataset
        ),
    shell:
        """
        python scripts/pretrain_model/odin.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} seed={wildcards.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} \
            respect_source_availability=true \
            experiment@_global_={wildcards.dataset} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule pretrain_oracle_restoration_pvae:
    input:
        train=lambda wc: raw_dataset(wc.dataset, wc.instance, "train"),
        val=lambda wc: raw_dataset(wc.dataset, wc.instance, "val"),
        classifier=lambda wc: classifier_path(wc.dataset, wc.instance),
    output:
        directory(
            f"{ROOT}/restoration_pvae/{DATA_NAMESPACE}/oracle/"
            "dataset-{dataset}/instance-{instance}/model.bundle"
        ),
    benchmark:
        f"{BENCHMARK_ROOT}/pretrain_restoration_pvae/oracle/"
        "dataset-{dataset}+instance-{instance}.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(dataset_device(wc.dataset, "pvae")),
    params:
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: dataset_device(wc.dataset, "pvae"),
        extra=lambda wc: runtime_params(
            RESTORATION_PVAE_PARAMS, wc.dataset
        ),
    shell:
        """
        python scripts/pretrain_model/odin.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} seed={wildcards.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} \
            respect_source_availability=false \
            experiment@_global_={wildcards.dataset} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


def restoration_pvae_input(wildcards):
    if wildcards.strategy == "pvae_oracle":
        return oracle_pvae(wildcards.dataset, wildcards.instance)
    return incomplete_pvae(
        wildcards.dataset,
        wildcards.mechanism,
        wildcards.p,
        wildcards.instance,
    )


rule restore_missing_training_view:
    input:
        train=lambda wc: base_view(
            wc.dataset, wc.mechanism, wc.p, wc.instance, "restricted", "train"
        ),
        val=lambda wc: base_view(
            wc.dataset, wc.mechanism, wc.p, wc.instance, "restricted", "val"
        ),
        pvae=restoration_pvae_input,
        reference_train=lambda wc: raw_dataset(
            wc.dataset, wc.instance, "train"
        ),
        reference_val=lambda wc: raw_dataset(wc.dataset, wc.instance, "val"),
    output:
        train=directory(
            f"{ROOT}/views/restored/{DATA_NAMESPACE}/dataset-{{dataset}}/"
            "mechanism-{mechanism}+p-{p}/instance-{instance}/"
            "{strategy}/train.bundle"
        ),
        val=directory(
            f"{ROOT}/views/restored/{DATA_NAMESPACE}/dataset-{{dataset}}/"
            "mechanism-{mechanism}+p-{p}/instance-{instance}/"
            "{strategy}/val.bundle"
        ),
    benchmark:
        f"{BENCHMARK_ROOT}/restore_view/dataset-{{dataset}}/"
        "mechanism-{mechanism}+p-{p}+instance-{instance}+"
        "strategy-{strategy}.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(dataset_device(wc.dataset, "pvae")),
    params:
        device=lambda wc: dataset_device(wc.dataset, "pvae"),
    shell:
        """
        python scripts/missing_values/restore_training_views.py \
            train_view_bundle_path={input.train} \
            val_view_bundle_path={input.val} pvae_bundle_path={input.pvae} \
            train_save_path={output.train} val_save_path={output.val} \
            strategy={wildcards.strategy} seed={wildcards.instance} \
            batch_size={RESTORATION_BATCH_SIZE} device={params.device} \
            reference_train_dataset_bundle_path={input.reference_train} \
            reference_val_dataset_bundle_path={input.reference_val} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


def pretrain_extra(wildcards):
    key = wildcards.pretrain_key
    script = PRETRAIN_SCRIPT_NAMES[key]
    params = [PRETRAIN_PARAMS[key]] if PRETRAIN_PARAMS[key] else []
    if script in {"jafa", "odin"}:
        respect = wildcards.strategy == "restricted"
        params.append(
            f"respect_source_availability={str(respect).lower()}"
        )
    params.append(f"experiment@_global_={wildcards.dataset}")
    runtime = runtime_params(
        PRETRAIN_RUNTIME_PARAMS,
        key,
        script,
        wildcards.dataset,
        f"{key}/{wildcards.dataset}",
    )
    if runtime:
        params.append(runtime)
    return " ".join(params)


rule pretrain_missing_data_method:
    input:
        train=lambda wc: training_view(wc, "train"),
        val=lambda wc: training_view(wc, "val"),
        classifier=policy_classifier,
    output:
        directory(
            f"{ROOT}/pretrained/{NAMESPACE}/{{pretrain_key}}/"
            "dataset-{dataset}/mechanism-{mechanism}+p-{p}+"
            "strategy-{strategy}+instance-{instance}/model.bundle"
        ),
    benchmark:
        f"{BENCHMARK_ROOT}/pretrain_method/pretrain-{{pretrain_key}}/"
        "dataset-{dataset}+mechanism-{mechanism}+p-{p}+"
        "strategy-{strategy}+instance-{instance}.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(
            dataset_device(wc.dataset, wc.pretrain_key)
        ),
    params:
        script=lambda wc: PRETRAIN_SCRIPT_NAMES[wc.pretrain_key],
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: dataset_device(wc.dataset, wc.pretrain_key),
        extra=pretrain_extra,
    shell:
        """
        python scripts/pretrain_model/{params.script}.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} seed={wildcards.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


def training_extra(wildcards):
    spec = METHOD_SPECS[wildcards.method]
    params = list(spec.train_params)
    params.append(f"experiment@_global_={wildcards.dataset}")
    if wildcards.strategy == "pvae_stepwise":
        params.append(
            "stepwise_pvae_bundle_path="
            + incomplete_pvae(
                wildcards.dataset,
                wildcards.mechanism,
                wildcards.p,
                wildcards.instance,
            )
        )
    runtime = runtime_params(
        TRAIN_RUNTIME_PARAMS,
        spec.base_method,
        wildcards.method,
        wildcards.dataset,
        f"{wildcards.method}/{wildcards.dataset}",
    )
    if runtime:
        params.append(runtime)
    return " ".join(params)


def stepwise_pvae_input(wildcards):
    if wildcards.strategy != "pvae_stepwise":
        return []
    return incomplete_pvae(
        wildcards.dataset,
        wildcards.mechanism,
        wildcards.p,
        wildcards.instance,
    )


rule train_missing_data_method_with_pretraining:
    wildcard_constraints:
        method=wildcard_pattern(PRETRAINED_METHODS)
    input:
        train=lambda wc: training_view(wc, "train"),
        val=lambda wc: training_view(wc, "val"),
        pretrained=method_pretrain,
        stepwise_pvae=stepwise_pvae_input,
        classifier=policy_classifier,
    output:
        directory(
            f"{ROOT}/trained/{NAMESPACE}/{{method}}/dataset-{{dataset}}/"
            "mechanism-{mechanism}+p-{p}+strategy-{strategy}+"
            "instance-{instance}+train_hard_budget-{train_budget}/"
            "method.bundle"
        ),
    benchmark:
        f"{BENCHMARK_ROOT}/train_method/method-{{method}}/"
        "dataset-{dataset}+mechanism-{mechanism}+p-{p}+"
        "strategy-{strategy}+instance-{instance}+"
        "train_hard_budget-{train_budget}.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(method_device(wc.dataset, wc.method)),
    params:
        script=lambda wc: METHOD_SPECS[wc.method].train_script_name,
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: method_device(wc.dataset, wc.method),
        extra=training_extra,
    shell:
        """
        python scripts/train_method/{params.script}.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            pretrained_model_bundle_path={input.pretrained} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.train_budget} soft_budget_param=null \
            device={params.device} seed={wildcards.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule train_missing_data_method_without_pretraining:
    wildcard_constraints:
        method=wildcard_pattern(UNPRETRAINED_METHODS)
    input:
        train=lambda wc: training_view(wc, "train"),
        val=lambda wc: training_view(wc, "val"),
        classifier=lambda wc: policy_classifier(wc, wc.method),
        stepwise_pvae=stepwise_pvae_input,
    output:
        directory(
            f"{ROOT}/trained/{NAMESPACE}/{{method}}/dataset-{{dataset}}/"
            "mechanism-{mechanism}+p-{p}+strategy-{strategy}+"
            "instance-{instance}+train_hard_budget-{train_budget}/"
            "method.bundle"
        ),
    benchmark:
        f"{BENCHMARK_ROOT}/train_method/method-{{method}}/"
        "dataset-{dataset}+mechanism-{mechanism}+p-{p}+"
        "strategy-{strategy}+instance-{instance}+"
        "train_hard_budget-{train_budget}.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(method_device(wc.dataset, wc.method)),
    params:
        script=lambda wc: METHOD_SPECS[wc.method].train_script_name,
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: method_device(wc.dataset, wc.method),
        extra=training_extra,
    shell:
        """
        python scripts/train_method/{params.script}.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.train_budget} soft_budget_param=null \
            device={params.device} seed={wildcards.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


def eval_batch_size(wildcards):
    if wildcards.strategy == "pvae_stepwise":
        return STEPWISE_EVAL_BATCH_SIZE
    if EVAL_BATCH_SIZE_OVERRIDE is not None:
        if isinstance(EVAL_BATCH_SIZE_OVERRIDE, dict):
            return int(
                EVAL_BATCH_SIZE_OVERRIDE.get(
                    wildcards.dataset,
                    EVAL_BATCH_SIZE_OVERRIDE.get("default"),
                )
            )
        return int(EVAL_BATCH_SIZE_OVERRIDE)
    base_method = METHOD_SPECS[wildcards.method].base_method
    return int(EVAL_BATCH_SIZES[base_method][wildcards.dataset])


rule eval_missing_data_method:
    input:
        dataset=lambda wc: raw_dataset(wc.dataset, wc.instance, EVAL_SPLIT),
        method=trained_method_input,
        reuse_guard=policy_reuse_guard,
        classifier=lambda wc: policy_classifier(wc, wc.method),
    output:
        f"{ROOT}/eval/{EVAL_SPLIT}/{NAMESPACE}/dataset-{{dataset}}/"
        "method-{method}+mechanism-{mechanism}+p-{p}+"
        "strategy-{strategy}+instance-{instance}+"
        "train_hard_budget-{train_budget}+eval_hard_budget-{eval_budget}/"
        "eval_data.parquet",
    # Total subprocess wall time, plus max RSS and CPU time. `benchmark:` rather
    # than a second `output:` so the DAG's completeness check is unaffected.
    # Subtract the eval_data.timing.json phases from this to get fixed startup.
    benchmark:
        f"{ROOT}/eval/{EVAL_SPLIT}/{NAMESPACE}/dataset-{{dataset}}/"
        "method-{method}+mechanism-{mechanism}+p-{p}+"
        "strategy-{strategy}+instance-{instance}+"
        "train_hard_budget-{train_budget}+eval_hard_budget-{eval_budget}/"
        "benchmark.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(method_device(wc.dataset, wc.method)),
    params:
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: method_device(wc.dataset, wc.method),
        batch_size=eval_batch_size,
        respect_native=lambda wc: str(wc.mechanism == "native").lower(),
        extra=lambda wc: runtime_params(
            EVAL_PARAMS,
            METHOD_SPECS[wc.method].base_method,
            wc.method,
            wc.dataset,
            f"{wc.method}/{wc.dataset}",
        ),
    shell:
        """
        python scripts/eval/eval_afa_method.py \
            method_bundle_path={input.method} dataset_bundle_path={input.dataset} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.eval_budget} soft_budget_param=null \
            batch_size={params.batch_size} device={params.device} \
            respect_native_availability={params.respect_native} \
            seed={wildcards.instance} use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST_STR} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule summarize_missing_data:
    input:
        EVALUATIONS,
    output:
        instances=f"{SUMMARY_DIR}/instance_metrics.csv",
        summary=f"{SUMMARY_DIR}/summary.csv",
        actions=f"{SUMMARY_DIR}/action_rates.csv",
        restoration=f"{SUMMARY_DIR}/restoration_rmse.csv",
    params:
        root=f"{ROOT}/eval/{EVAL_SPLIT}/{NAMESPACE}",
        restored=f"{ROOT}/views/restored/{DATA_NAMESPACE}",
        variants=" ".join(
            f"--base-method {name}={spec.base_method}"
            for name, spec in METHOD_SPECS.items()
            if name != spec.base_method
        ),
    shell:
        """
        python scripts/analysis/summarize_missing_data.py \
            --input-root {params.root} --restored-root {params.restored} \
            --instance-output {output.instances} \
            --summary-output {output.summary} \
            --action-output {output.actions} \
            --restoration-output {output.restoration} {params.variants}
        """
