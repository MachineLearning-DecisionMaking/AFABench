# Documentation

User documentation follows [Diátaxis](https://diataxis.fr/): every page is
exactly one of four types, and each type has its own folder. Put a new page
in the folder of its type, and when a page serves two types, split it.

Write for a reader using the current system: describe its contract and the
steps to use it. Development chronology, retired interfaces and migration
narratives belong outside these pages. Apply the active-development policy
in `AGENTS.md` when deciding whether compatibility work is needed.

| Folder | Type | A page here | Example |
| --- | --- | --- | --- |
| `tutorials/` | Tutorial | Teaches a newcomer by walking them through a lesson they complete. | None yet. |
| `how-to/` | How-to guide | Tells a user with a goal what to do, as steps. No background beyond what a step needs; link to explanation instead. | `how-to/repository_adopter/add_method.md` |
| `reference/` | Reference | Describes a thing (a file format, a command, a configuration option) completely and neutrally, for lookup. | `reference/release_manifest.md` |
| `explanation/` | Explanation | Says why: design reasons, trade-offs, limits, background. | `explanation/output_snapshots.md` |

Most AFABench users want to run the benchmark, not learn it, so most guides
are how-to guides. A how-to guide names its reader by one of the user types
in [`explanation/user_types.md`](explanation/user_types.md) and links to
that page rather than describing the reader itself. Store each guide under
`how-to/<user_type>/`: `results_only_user`, `repository_adopter`,
`library_user` or `maintainer`. Keep shared guides in one audience's folder
and link to them from the other audiences' indexes rather than duplicating
pages.

When creating or editing a how-to, follow the
[procedure-only writing and review rules](how-to/AGENTS.md).

Name a how-to guide after the task, with a verb (`create_output_snapshots.md`)
and its section headings after steps the reader performs. Name a
reference or explanation page after the thing it describes.

Two folders hold documents that are not user documentation and are outside
the Diátaxis types:

- `adr/`: architecture decision records (`docs/agents/domain.md`).
- `agents/`: how agents use the issue tracker, triage labels and domain
  docs.

Terms are defined in [`CONTEXT.md`](../CONTEXT.md); link to an entry there
instead of defining a term on a page.

## Index

### How-to guides

Guides live in folders by [user type](explanation/user_types.md). Shared
guides appear under each audience they serve, linking to one canonical page.

#### Results-only user

- [Download published results](how-to/results_only_user/download_published_results.md)

#### Repository adopter

- [Add a dataset](how-to/repository_adopter/add_dataset.md)
- [Add a method](how-to/repository_adopter/add_method.md)
- [Add a method to a release](how-to/repository_adopter/add_method_to_a_release.md)
- [Add a dataset to a release](how-to/repository_adopter/add_dataset_to_a_release.md)
- [Add more seeds to a release's datasets and methods](how-to/repository_adopter/add_seeds_to_a_release.md)
- [Add hard budgets or soft-budget parameters to a release](how-to/repository_adopter/add_budget_settings_to_a_release.md)
- [Estimate the compute of a run](how-to/maintainer/estimate_compute.md)
- [Run on SLURM](how-to/maintainer/slurm_integration.md)
- [Choose per-method training and evaluation execution](how-to/maintainer/mixed_execution.md)
- [Choose classifier and pretrained-model execution](how-to/maintainer/prerequisite_execution.md)
- [Use CPU-only dataset and result processing](how-to/maintainer/cpu_processing_execution.md)

#### Library user

- [Install and use the library](how-to/library_user/use_the_library.md)

#### Maintainer

- [Run the pipeline](how-to/maintainer/run_the_pipeline.md)
- [Run on SLURM](how-to/maintainer/slurm_integration.md)
- [Choose per-method training and evaluation execution](how-to/maintainer/mixed_execution.md)
- [Choose classifier and pretrained-model execution](how-to/maintainer/prerequisite_execution.md)
- [Use CPU-only dataset and result processing](how-to/maintainer/cpu_processing_execution.md)
- [Estimate the compute of a run](how-to/maintainer/estimate_compute.md)
- [Add a dataset](how-to/repository_adopter/add_dataset.md)
- [Add a method](how-to/repository_adopter/add_method.md)
- [Run L2M](how-to/maintainer/run_l2m.md)
- [Create output snapshots](how-to/maintainer/create_output_snapshots.md)
- [Restore an output snapshot](how-to/maintainer/restore_an_output_snapshot.md)
- [Publish a benchmark release](how-to/maintainer/publish_a_benchmark_release.md)

### Reference

- [Pipeline configuration](reference/pipeline_configuration.md)
- [Requested walltime](reference/requested_walltime.md)
- [Bundle format](reference/bundle_format.md)
- [L2M](reference/l2m.md): configuration, files and runs
- [Datasets](reference/datasets/README.md): one page per dataset
- [Evaluation dataframes](reference/evaluation_dataframes.md)
- [Evaluation performance plots](reference/evaluation_plots.md)
- [Job records](reference/job_records.md)
- [Release manifest](reference/release_manifest.md)
- [`snapshot.py` command](reference/snapshot_command.md)
- [`workflow/run.py` command](reference/pipeline_launcher_command.md)
- [`estimate-compute` command](reference/estimate_compute_command.md)
- [Interactive preview](reference/interactive_preview.md)
- [Terminology: patch-based unmasking example](reference/terminology.md)

### Explanation

- [User types](explanation/user_types.md)
- [Benchmark releases](explanation/benchmark_releases.md)
- [Output snapshots](explanation/output_snapshots.md)
- [Framework limitations](explanation/limitations.md)
- [L2M trained on the TabICL v1 prior](explanation/l2m_tabicl_prior.md)
- [How the L2M runs relate to the authors' code](explanation/l2m_baseline_audit.md)
- [What GDFS and DIME share, and where the port departs from upstream](explanation/gdfs_dime_shared_code.md)
- Investigations behind ADRs:
  [training-contract duplication](explanation/training_contract_inventory.md),
  [evaluation input cloning](explanation/eval_input_cloning_assessment.md)
