# Compact evaluation dataframes

## Contract

`afabench.evaluation.schemas` defines Pandera Polars contracts, each
extending `BatchEvaluationSchema` or `EvaluationSchema`:

- `BatchEvaluationSchema`: one row per episode and zero-based `step`,
  returned by `process_batch`.
- `EvaluationSchema`: the same columns plus `generation_index` and
  `split_index`, returned by `eval_afa_method`.
- `SavedEvaluationSchema`: the `EvaluationSchema` columns plus the identity
  columns below, added by `AFAEvaluator` before saving Parquet.

`episode_id` identifies an episode, not a dataset position. It is batch-local
for an individual `process_batch` call, but unique across all batches in one
`eval_afa_method` result. IDs are local to an evaluation artifact: independently
produced results must remain in separate namespaces when reconstructing
histories.

`generation_index` and `split_index` identify the instance an episode
evaluated, and are the same on every row of an episode
(`docs/adr/0004-instances-carry-their-generation-index.md`). `split_index` is
the instance's position in the evaluated dataset, the eval split's dataset
bundle; `generation_index` is its position in the dataset produced by dataset
generation, before splitting. Both are correct whether or not
`eval_only_n_samples` samples a subset. With the dataset key and dataset
realization, `generation_index` identifies the instance across methods and
splits; for a real-world dataset it is also the instance's row in the source.

`step` starts at zero and counts selections performed before the row's action.
An episode with three selections has steps 0, 1, 2, 3; the last row is stop.
Repeated selections count as separate steps, even if no new features appear.
The initializer's observations are not selections and do not increment `step`.

The other evaluation columns are `action_performed`,
`builtin_predicted_probabilities`, `external_predicted_probabilities`,
`true_class`, `accumulated_cost`, and `forced_stop`. Actions are one-based selection IDs;
zero means stop. Recorded actions include budget/forced-acquisition overrides,
not rejected actions or original policy intentions. Predictions use the
pre-action observations, whereas cost includes the current action.
Missing classifiers produce null probability vectors.

Functions return `polars.DataFrame`. Runtime
validation rejects missing, unexpected, or duplicate column names and checks
dtypes, nullability, and values without coercing dtypes. Each episode must
have contiguous steps from zero, no duplicate steps, and exactly one stop at
its final step. Row order is immaterial. Validation also runs after batch
concatenation and before saving.

Probability columns use `List(Float64)`, including all-null columns.
Integer identities, steps and actions use `Int64` in the in-memory evaluation
schemas. Validation does not silently coerce invalid events.

## Class probabilities

Evaluation consumes only class probabilities (`ClassProbabilities`), not
logits. Classifier outputs declare `output_kind` as `logits` or
`probabilities`; upstream adapters (`as_class_probability_fn`) apply softmax
only to declared logits. Semantics are never inferred from tensor values.

A probability tensor has the label's batch/class shape, at least two classes,
finite floating-point values in `[0, 1]`, and row sums of one (absolute
tolerance `1e-6`, relative tolerance `1e-5`). Runtime validation raises
specific exceptions naming invalid shapes or values. Single-logit binary
outputs are not supported: binary classifiers return two-class vectors.

Every recorded time step retains the full vector for each available
classifier, ordered by label class index. Raw Parquet list columns store
`builtin_predicted_probabilities` and `external_predicted_probabilities` as
floating vectors; unavailable classifiers have null vectors. Tables hold no
class decisions: each consumer decides how to turn a vector into a class.
The classifier-long transformed table holds each classifier's vectors in
`predicted_probabilities`; it does not replace missing built-in results with
external results.

## Identity and provenance

A raw evaluation table identifies itself
(`docs/adr/0002-provenance-recorded-in-artifacts.md`), whether the pipeline
or a hand run of `scripts/eval/eval_afa_method.py` wrote it. Its identity
columns are constant per table and use nullable Polars dtypes, which
`polars.read_parquet` restores without `afabench`:

| Column | dtype | Value |
| --- | --- | --- |
| `afa_method` | `String` | Method name, from the method bundle's record. |
| `dataset` | `String` | Dataset key, from the eval dataset bundle's record. |
| `dataset_realization_index` | `UInt64` | Dataset realization, from the eval dataset bundle's record. |
| `eval_split` | `String` | `train`, `val` or `test`: the eval dataset bundle's own split. |
| `initializer` | `String` | Name of the initializer config the evaluation selected, such as `cold`. Never null. |
| `train_seed` | `UInt64` | Seed of the method bundle's record. |
| `train_hard_budget` | `Float64` | `hard_budget` of the method's training contract. |
| `train_soft_budget_param` | `Float64` | `soft_budget_param` of the method's training contract. |
| `eval_seed` | `UInt64` | The seed the evaluation used. Never null: a null configured seed is drawn once and recorded. |
| `eval_hard_budget` | `Float64` | The evaluation's hard budget, null in the soft-budget setting. |
| `eval_soft_budget_param` | `Float64` | The evaluation's soft-budget parameter, null when not given. |

A value whose source bundle was written without a provenance record, or
whose record lacks it, is null, not guessed. The pretraining seed, the Unmasker, the classifier and the rest
of the configuration are in the table's provenance record only. The record
is stored as JSON under the Arrow schema metadata key `afabench.provenance`;
Polars drops it on read, so read it with
`afabench.evaluation.provenance.evaluation_table_provenance(path)`, which
returns null for a table written without one.

The evaluator owns the evaluation seed: it resolves a null seed once, seeds
Python, NumPy and torch with it, and passes the resolved integer to the
method, the Unmasker, the Initializer and the evaluation sampler. The same
seed reproduces the table on CPU.

## On-demand histories

The producer and saved format do not contain `prev_selections_performed`.
Saving every selection-history prefix costs quadratic space per episode;
scalar episode IDs, steps, and actions require linear space instead.

```python
import polars as pl

from afabench.evaluation.history import reconstruct_selection_history
from afabench.evaluation.schemas import SavedEvaluationSchema

results = SavedEvaluationSchema.validate(pl.read_parquet(path))
histories = reconstruct_selection_history(results)
```

`reconstruct_selection_history` returns a Polars `List(Int64)` Series aligned
to the input row positions, even with reordered rows. For
step t it collects the episode's earlier positive actions, subtracting one
from each. It preserves order and duplicates, excluding stop and initial
observations. It validates complete episodes before reconstructing.

This deliberately materializes quadratic-size prefixes only on demand. Use
`step` directly when only the history length is needed. Reconstruct **before**
filtering rows: an individual row or incomplete episode is not self-contained.
Neither histories nor actions alone recover initial masks or the Unmasker's
mapping from selections to features.

## Plotting-ready tables

`scripts/misc/transform_eval_data_pipeline.py` turns one raw table into one
plotting-ready table. It reads identity from the raw table's columns and only
derives plotting columns: `n_selections_performed`, the melt of the
probability columns into `classifier` (`builtin` or `external`) and
`predicted_probabilities`, and nullable dtypes. The output keeps `generation_index`,
`split_index` and every identity column, including
`dataset_realization_index` and `eval_split`, and copies the raw table's
provenance record into its own schema metadata. It discards episode
identity when expanding rows by classifier, so these tables are not episode
logs.

Its identity arguments (`--method`, `--dataset`, `--initializer`,
`--train_seed`, `--train_hard_budget`, `--train_soft_budget_param`,
`--eval_soft_budget_param`) are optional checks. An argument that disagrees
with a non-null column raises `IdentityArgumentMismatchError`; `null` is an
explicit null value and disagrees with a non-null column. A nullable column
whose values are all null is filled from its argument, or left null when the
argument is omitted. The input must validate against `SavedEvaluationSchema`
before transformation: missing columns, incomplete episodes and older table
shapes are rejected. The Snakemake rule passes its wildcards as checks.

### Loading and plotting

`afabench.evaluation.tables.load_plotting_ready_table(path)` reads a
plotting-ready table, or a merge of them, as a Polars DataFrame and casts
each column it knows to its dtype in
`afabench.evaluation.schemas.PLOTTING_READY_DTYPES`: the identity columns
take the dtypes of `IDENTITY_DTYPES`, the rest are `UInt64`, `Float64`,
`Boolean`, `String` or `List(Float64)`. Merging tables whose column types disagree upcasts
those columns to strings, so in a numeric column values that do not parse
as numbers become null. A number the column's dtype cannot hold, such as a
fraction or a negative number in a `UInt64` column, and a value of a
`Boolean` column that is not boolean raise `TypeError`. Absent columns
stay absent. It does not validate the table.

Every evaluation plot reads its table through this function, and the same
module prepares it for plotting:

| Function | Returns |
| --- | --- |
| `combine_soft_budget_params` | The table with `train_soft_budget_param` and `eval_soft_budget_param` replaced by `soft_budget_param`, whichever is set. Raises `SoftBudgetParamConflictError` for a row that sets both. |
| `keep_largest_hard_budget` | The rows of each dataset's largest `eval_hard_budget`. |
| `keep_smallest_soft_budget_param` | The rows of each method and dataset's smallest `soft_budget_param`. |
| `metrics_at_stop` | Accuracy, macro F1, AUROC and mean accumulated cost per method, dataset, seed pair and budget. |
| `metrics_at_every_step` | Accuracy, macro F1 and AUROC per method, dataset, seed pair, budget and `n_selections_performed`. |
| `variance_over_seeds_at_stop`, `variance_over_seeds_at_every_step` | Mean and standard deviation of those metrics over seeds. |
| `apply_exclusions` | The table without each method excluded from a dataset. |

The figures are built in `afabench.plotting`: `eval_perf` (performance
against hard budget, selections and accumulated cost), `eval_actions`
(action heatmaps) and `total_time` (job durations per pipeline stage, from
the job duration table). The scripts under `scripts/plotting/` only load
their input and call these modules.

## Metrics and seed summaries

Pass each classifier's rows separately to the metric interfaces. Accuracy
and macro F1 (`accuracy`, `f_score`) decide each row's most probable class,
the lowest class index on ties. AUROC (`auroc`) uses class
1 probability for binary classification, or macro one-vs-rest over all vector
classes for multiclass classification. It is computed within each run before
across-seed aggregation, at stopping and at each selection-count time step.

A run missing any required class has undefined AUROC (`NaN`), not zero.
If any run in a seed summary has undefined AUROC, both `mean_auroc` and
`std_auroc` remain undefined; undefined runs are not silently dropped.
Standard deviations use the sample convention, so a single seed also has
null standard deviation. The accuracy and F1 summaries are
`mean_accuracy`, `std_accuracy`, `mean_f_score`, `std_f_score`.

[Evaluation performance plots](evaluation_plots.md) documents headline metric
configuration and optional per-dataset three-metric figures. Average precision
(AP) is deferred to [issue #130](https://github.com/MachineLearning-DecisionMaking/AFABench/issues/130).
