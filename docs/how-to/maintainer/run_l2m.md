# Run L2M

For a [maintainer](../../explanation/user_types.md#maintainer). Run the
three L2M runs (`l2m`, `l2m_synth`, `l2m_better_synth`) next to the other
methods on `cube`, `miniboone_subset` and `metabric`, or change what they
train on. Configuration and files of a run: [L2M](../../reference/l2m.md).
Run commands from the repository root, after
[preparing to run the pipeline](run_the_pipeline.md).

## 1. Smoke-test the three runs on a CPU

```shell
uv run workflow/run.py l2m_comparison --jobs 4 "methods=[l2m,l2m_synth,l2m_better_synth]" smoke_test=true execution_file=workflow/profiles/execution/cpu.yaml
```

Check that every run ends with `train_method` and `eval_method` jobs on the
three datasets. On a CPU the bf16 autocast makes each fit stage take
minutes.

To test one fit stage without the pipeline, point the scripts at the dataset
and classifier bundles of a dataset realization (the first from
`scripts/dataset_generation/generate_dataset.py`; L2M does not use the
classifier but records its provenance):

```shell
uv run python scripts/pretrain_model/l2m.py \
    train_dataset_bundle_path=<train.bundle> val_dataset_bundle_path=<val.bundle> \
    classifier_bundle_path=<classifier.bundle> save_path=<pretrained.bundle> \
    initializer=cold unmasker=direct dataset_key=<dataset> device=cpu seed=0 \
    smoke_test=true mixed_precision=false \
    task_prior=tabicl_v1 prior_type=prior_bag_plus_tree
uv run python scripts/train_method/l2m.py <the same arguments> \
    save_path=<method.bundle> pretrained_model_bundle_path=<pretrained.bundle> \
    method_name=l2m_better_synth hard_budget=7 soft_budget_param=null
```

Take `task_prior` and `prior_type` of the run from the
[table of runs](../../reference/l2m.md#the-runs), and give both scripts the
same ones.

## 2. Run the comparison

```shell
uv run workflow/run.py l2m_comparison --site <your_site> -i
```

Check the L2M rows of every stage for the three datasets, then remove `-i`
to submit. To limit the run, add `"methods=[l2m,l2m_synth,l2m_better_synth]"`
or `"datasets=[cube]"`.

`l2m_better_synth` starts four task worker processes per fit stage: [set `cpus_per_task`](slurm_integration.md#2-set-cpu-memory-and-walltime-requests)
to at least four for its jobs.

## 3. Change a dataset's settings

1. Edit the same keys in both files, so that the stages agree:
   `conf/scripts/pretrain_model/l2m/dataset_override/<dataset>.yaml` and
   `conf/scripts/train_method/l2m/dataset_override/<dataset>.yaml`.
2. Keep `sequence_length - 50` at least the stored context set size, which is
   `context_set_size` or the validation split, whichever is smaller.
3. Keep `pretraining_pool_size` at least `sequence_length` and at most the
   train split's size, so that `l2m` can draw a full task.
4. Update the expected values in `test/config/test_l2m_config.py`, then run
   it with `uv run pytest test/config/test_l2m_config.py`.

## 4. Add a dataset

1. Check the dataset's classes: `l2m_better_synth` needs 2 to 10. If it has
   more, add the dataset to that method's `hard_budget_ignored_datasets` in
   `workflow/conf/method_options/all.yaml`.
2. In the same file, remove the dataset from `hard_budget_ignored_datasets`
   of `l2m`, `l2m_synth` and `l2m_better_synth`, and add it to their
   `train_hard_budget` with a budget below the dataset's number of features.
3. Add its evaluation budgets to `workflow/conf/eval_hard_budgets/all.yaml`.
4. Add both `dataset_override/<dataset>.yaml` files of
   [step 3](#3-change-a-datasets-settings), starting from the closest dataset's.
   Choose `sequence_length` and `batch_size` to fit the GPU's memory.
5. Add the dataset to `workflow/conf/datasets/l2m_comparison.yaml` and to
   the `EXPECTED` and `VALIDATION_SIZES` tables of
   `test/config/test_l2m_config.py`.
6. Preview the plan:

   ```shell
   uv run workflow/run.py l2m_comparison -n "datasets=[<dataset>]" "methods=[l2m,l2m_synth,l2m_better_synth]"
   ```

   Check that the three runs appear, each trained at the budget of step 2.
7. Add requested walltime limits for the dataset to the site profile if the
   default is too short ([requested walltime](../../reference/requested_walltime.md)).
