# Run the pipeline

Complete the [setup](setup.md), then run
commands from the repository root. For a small worked example, start with
[your first run](../tutorials/first_run.md).

## Choose a profile and preview

The full benchmark creates many jobs; use SLURM for large runs and start
with a small subset when running locally.

Use `all` (the default) for all configured datasets and methods, or `kdd26`
for the KDD '26 configuration. Current code is not guaranteed to reproduce
submitted results.

On a CPU workstation:

```shell
uv run workflow/run.py --jobs 8 \
    execution_file=workflow/profiles/execution/cpu.yaml -i
```

Adjust `--jobs` to available CPUs and memory. For SLURM, first
[configure your site](run_on_slurm.md), then preview from an authorized
submit host:

```shell
uv run workflow/run.py kdd26 --site <your_site> -i
```

Check selected methods, datasets, budgets, job counts and allocations.
Remove `-i` to run; repeat the same command to resume interrupted work.
Use `-n` for a text dry run, or `-e` to [estimate compute](estimate_compute.md).

To change methods, datasets, budgets or CPU/GPU choices, follow
[customize the pipeline](customize_pipeline.md).
[All launcher options](../reference/pipeline_launcher_command.md).

## Inspect planned work and failures

- In the [interactive preview](../reference/interactive_preview.md), open
  jobs to check their reasons, commands and resources. Follow `Needs`
  relations to find the upstream job that triggers a rerun.
- Use `--group-by dataset,method` to compare methods on each dataset.
- Add `list_jobs=true` to a text dry run to print job commands and allocations.
- Read detailed execution logs under `.snakemake/log/` when troubleshooting.

## Log to Weights & Biases

Log in and set the entity and project before adding `--wandb`:

```shell
uv run wandb login
export WANDB_ENTITY=<entity>
export WANDB_PROJECT=afabench
uv run workflow/run.py ... --wandb
```

With direnv, put the entity in gitignored `.envrc.local`; `.envrc` already
sets the project. To continue an existing group, replace `--wandb` with
`--wandb-group <group_name>`. Follow the run and group links printed in the terminal.

## Run a specific stage

Pass a target to build that stage and any missing or outdated prerequisites:

```shell
uv run workflow/run.py ... --target all_eval_methods
```

Other targets are `all_generate_datasets`, `all_train_classifiers`,
`all_pretrain_models` and `all_train_methods`.

To evaluate existing bundles *without* regenerating datasets or training:

```shell
uv run workflow/run.py ... --target all_eval_methods \
    "stages={dataset_generation: false, training: false}"
```

To transform, aggregate and plot existing evaluation tables only:

```shell
uv run workflow/run.py ... \
    "stages={dataset_generation: false, training: false, evaluation: false}"
```

Supply all outputs required from disabled stages. Preview with `-i` before
executing. [Stage controls](../reference/pipeline_configuration.md#stages).

## Save or restore outputs

Choose a fresh snapshot destination outside the checkout:

```shell
uv run python scripts/release/snapshot.py inventory
uv run python scripts/release/snapshot.py save /path/to/snapshot-dir
uv run python scripts/release/snapshot.py restore /path/to/snapshot-dir
```

Check the inventory's sizes and missing provenance before saving.
Restore defaults to the snapshot's production or smoke root; pass
`--destination-root <root>` to change it. Use `--overwrite` only after
reviewing listed conflicts. To save smoke outputs, pass `--source-root output/smoke`.

After restoring, preview the original profile, passing `output_root=<root>`
if needed. Investigate missing outputs and changed inputs. Snapshots omit
Snakemake metadata: explicitly regenerate outputs affected by changed code
or parameters, even if the preview proposes reuse.
[Reuse limits](../explanation/output_snapshots.md#why-a-snapshot-omits-snakemake-metadata).

To share or extend benchmark results, use the [release guide](manage_releases.md).
