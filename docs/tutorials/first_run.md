# Your first pipeline run

This tutorial runs a small CPU smoke test: generate one dataset realization,
train two methods, evaluate them and produce plots. Smoke metrics are for
checking execution, not reporting benchmark results.

## Set up the checkout

Follow the [prerequisites and installation](../how-to/setup.md).
Run the following commands from the repository root after `uv sync`.

## Preview the work

```shell
uv run workflow/run.py --jobs 2 \
    "datasets=[cube]" "dataset_realization_indices=[0]" \
    "methods=[random_dummy,gdfs]" smoke_test=true \
    execution_file=workflow/profiles/execution/cpu.yaml -i
```

The preview shows planned jobs grouped by stage, method and dataset.
On a fresh checkout, expect dataset generation, classifier training,
method training, evaluation and result processing. Confirm that the jobs
use CPU and that the output root is `output/smoke`.
Quit the preview before continuing. Without an interactive terminal, use
`-n` instead of `-i` for a text dry run.

## Run it

Run the same command without `-i`:

```shell
uv run workflow/run.py --jobs 2 \
    "datasets=[cube]" "dataset_realization_indices=[0]" \
    "methods=[random_dummy,gdfs]" smoke_test=true \
    execution_file=workflow/profiles/execution/cpu.yaml
```

Wait for the workflow to finish. The job count advances as prerequisites
complete and dependent jobs become ready.

## Inspect the outputs

Look under `output/smoke/`:

- `eval_results/`: acquisition histories.
- `eval_results_transformed/`: plotting-ready evaluation tables.
- `merged_results/`: combined results.
- `plots/`: comparison figures.

Run the command again. With unchanged inputs, configuration and code,
completed jobs are reused rather than trained again.

Next, use the [running guide](../how-to/run_pipeline.md) for production runs
or read about the [pipeline's moving parts](../explanation/pipeline.md)
before customizing it.
