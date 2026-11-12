# Download, reuse and publish releases

Use [published releases](https://huggingface.co/datasets/afabench/results/tree/main/releases)
to read results without running AFABench, or download their outputs and
configuration to extend a comparison. Run commands below from a synced
checkout unless using the browser-only option.

## Download tables and plots

Choose a release and read its README for coverage and caveats. In
`releases/<release_id>/release_manifest.json`, inspect `evaluations`:
`transformed_path` identifies plotting-ready tables, and `raw_path`
identifies acquisition histories. Browse `output/plots/` for figures.

Download individual files in a browser, with `curl`, or with a Parquet reader:

```text
https://huggingface.co/datasets/afabench/results/resolve/main/releases/<release_id>/output/<path>
```

No AFABench installation or account is needed to read public files.
[Manifest fields](../reference/release_manifest.md) ·
[Table columns](../reference/evaluation_dataframes.md).

## Extend a release

### Download into a fresh output root

Use one release per root:

```shell
uv run python scripts/release/snapshot.py download <release_id> \
    --repo-id afabench/results --all --destination-root output/extended
```

Check the release README for omitted [dataset sources](../reference/datasets/README.md)
and compatible code. Review [comparability](#check-comparability) before
combining new results with released ones.

### Edit the downloaded configuration

Edit `output/extended/benchmark.yaml`, keeping released settings unchanged
except for the additions you intend. Choose the relevant changes:

| Addition | What to edit |
| --- | --- |
| Method | [Implement and register it](add_method.md); append to `methods`, add `method_options` and `soft_budget_params` (even if empty), and any `pretrain_mapping` entry. |
| Dataset | [Implement and register it](add_dataset.md); append to `datasets`, set budgets, classifier and Unmasker, and check each method's exclusions and soft-budget opt-ins. |
| Realizations | Append unused non-negative indices to `dataset_realization_indices`, retaining existing ones. |
| Hard budgets | Extend explicit dataset lists in `eval_hard_budgets`; retain old entries and check training-budget mappings below. |
| Soft budgets | Extend each method/dataset's `soft_budget_params` pairs and ensure the dataset is in `soft_budget_datasets`. |

Optionally extend named `method_sets` or `dataset_sets`; the automatic `all`
sets already include additions. Apply method/dataset setup to the downloaded
config, not a separate pipeline profile.
[Configuration keys](../reference/pipeline_configuration.md).

For new hard budgets, check `method_options`:

- Without a training-budget mapping, allow new training for each new budget.
- With `train_hard_budget`, retain the released training budget to reuse its bundle.
- With `eval_to_train_hard_budget_mapping`, preserve existing entries and map
  every new evaluation budget. Do not change training budgets just to avoid training.
- Check `hard_budget_ignored_datasets`. If
  `use_max_hard_budget_when_training_soft_budget` is enabled, raising the
  largest mapped training budget can also require new soft-budget training.

For soft budgets, keep the method's supported convention: `[value, null]`
for training-time parameters, or `[null, value]` for evaluation-time parameters.
Set at most one value per pair and keep `soft_budget_setting: true`.
New training-time values require training. Retain hard budgets needed by
`use_max_hard_budget_when_training_soft_budget`.

### Preview and investigate unexpected work

```shell
uv run workflow/run.py --configfile output/extended/benchmark.yaml \
    --site <your_site> -i
```

Without GPUs, replace `--site <your_site>` with
`--jobs 8 execution_file=workflow/profiles/execution/cpu.yaml`.

Expect `Run` for additions and missing release coverage, `Reuse` for
published outputs, and `Conditional reuse` for published outputs whose
datasets must be regenerated. Actual reuse requires regenerated dataset
content to match the release's recorded hashes.

Before proceeding, inspect every unexpected `Run` job of a released method
or shared prerequisite in the [preview](../reference/interactive_preview.md):

1. Read its reason; follow `Needs` relations to the upstream cause.
2. For missing outputs, check download coverage and compare budgets and
   realization indices with the release.
3. For changed inputs, parameters or code, restore the release settings or
   use compatible code. Check both training and evaluation budgets.
4. For a new dataset, verify that every intended method has evaluation jobs;
   investigate exclusions if evaluations are missing.
5. Fix the cause and preview again.

If regenerated data differs, fix the source or recompute in a separate
output root; do not bypass the content check.

### Run and check results

Remove `-i` from the preview command; repeat the command to resume.
Inspect `output/extended/merged_results/` and `output/extended/plots/`.
For budget extensions, verify training/evaluation budget columns under
`eval_results_transformed/`. Save the edited `benchmark.yaml` and invocation;
the downloaded manifest is not an inventory of your local extension.

## Check comparability

Check both whether files still load and whether the experiments remain
scientifically comparable. The release's artifact records expose producing
commits, dirty flags and resolved configurations, but do not certify either
property for your checkout.

Review changes to data preparation, splits, feature costs, Unmaskers,
initializers, budget/stop handling, classifiers, method hyperparameters,
batch sizes, seeds and metrics. A file loading successfully does not make
its results comparable. Restored outputs omit Snakemake metadata; explicitly
regenerate outputs affected by code or parameter changes.
[Reuse limits](../explanation/output_snapshots.md#why-a-snapshot-omits-snakemake-metadata).

## Publish a release

### Authenticate

Create a [Hugging Face write token](https://huggingface.co/settings/tokens)
and authenticate without committing credentials:

```shell
uv run hf auth login
uv run hf auth whoami
```

For non-interactive use, set `HF_TOKEN`. Create the public dataset repository
if needed:

```shell
uv run hf repo create afabench/results --repo-type dataset
```

Replace `afabench/results` throughout with your repository id when publishing
elsewhere, and verify write access.

### Save and review the package

1. [Complete the pipeline](run_pipeline.md) and keep only intended outputs
   in the output root. Smoke outputs cannot be published as benchmark results.
2. Review `conf/release/dataset_redistribution.yaml` against the
   [redistribution policy](../reference/release_manifest.md#dataset-redistribution).
   Dataset bundles need explicit permission to be included; omitted datasets'
   models and results remain included. Check applicable terms for those too.
3. Choose an unused id (letters, digits, `.`, `_`, `-`, starting with a
   letter or digit) and a fresh destination outside the repository:

   ```shell
   uv run python scripts/release/snapshot.py save ~/data/afabench-releases/example \
       --release-id example
   ```

   Do not repeat pipeline configuration arguments. Add
   `--release-notes /path/to/notes.md` if needed.
4. Check the generated README's coverage, caveats and output links. Keep the
   package unchanged after saving, including during publication.

### Publish the reviewed package

```shell
uv run python scripts/release/snapshot.py publish ~/data/afabench-releases/example \
    --repo-id afabench/results
```

Open the printed release URL. If validation fails, fix source outputs and
save to a fresh destination; do not edit the manifest to bypass validation.
For dirty or unknown producing code, regenerate from a clean commit or use
`--allow-dirty-code` only if intentional.
[Validation and overrides](../reference/snapshot_command.md#publish).

Never replace a published release: use a new id and release notes for
corrections. Saving or completing a run does not publish anything;
only `publish` and `download` contact Hugging Face.

For selective downloads, snapshot options and overwrite rules, see the
[command reference](../reference/snapshot_command.md).
