# Missing training data experiments

```sh
uv sync --locked

# full study
uv run snakemake --profile extra/workflow/profiles/config/missing_data --cores 4

# full study on GPU
uv run snakemake --profile extra/workflow/profiles/config/missing_data --cores 4 \
  --config device=cuda --resources gpu=1

# synthetic controlled study only
uv run snakemake --profile extra/workflow/profiles/config/missing_data --cores 4 \
  extra/output/paper/experiments/results/exact_study_combined.pdf

# figures and tables from existing results
uv run snakemake --profile extra/workflow/profiles/config/missing_data_plots --cores 4

# smoke test (CUBE-NM, one seed, MCAR 0.5)
uv run snakemake --profile extra/workflow/profiles/config/missing_data_smoke --cores 4
```

Figures and tables are written to `extra/output/paper/experiments/results/`, and
runs to `extra/output/missing_data/`.

## Paper and rebuttal runs

```sh
# every profile as one Slurm job each, in dependency order
scripts/workflow/submit_missing_data.sh

# a subset, or everything sequentially on this machine
scripts/workflow/submit_missing_data.sh rebuttal_constant_fill_induced_real
scripts/workflow/submit_missing_data.sh --local cuda

# rebuttal tables and figures from the summaries
uv run python scripts/analysis/imputation_quality.py \
  core_group_missingness_v2 induced_nonuniform_missingness_v2 induced_real_missingness_v2
uv run python scripts/analysis/restoration_tables.py
uv run python -m scripts.plotting.plot_restoration_split
```

`--dry-run` prints the jobs. Each rebuttal profile combines a dataset group
(`rebuttal/core_group.yaml`, `induced_nonuniform.yaml`, `induced_real.yaml`) with
an experiment (`rebuttal/imputers.yaml`, `sklearn_imputers.yaml`, ...); the
experiments of a group read the namespace of `imputers`.

## Configuration

- `extra/workflow/conf/missing_data/study.yaml`: datasets, methods, strategies
- `extra/workflow/conf/missing_data/design.yaml`: missingness mechanisms and rates, training settings
- `extra/workflow/conf/missing_data/smoke.yaml`: smoke test

## Data

Synthetic datasets are generated. Real datasets use the AFABench loaders. NHANES
needs the snapshot in [`extra/data/nhanes_mortality/`](../../extra/data/nhanes_mortality/README.md).
