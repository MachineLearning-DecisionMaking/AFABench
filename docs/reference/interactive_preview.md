# Interactive preview

The [interactive preview](../../CONTEXT.md) browses the
[pipeline plan](../../CONTEXT.md) of one pipeline invocation in the terminal.
The [pipeline launcher](pipeline_launcher_command.md#the-interactive-preview)
opens it with `-i` (`--interactive-preview`); steps:
[run the pipeline](../how-to/run_pipeline.md#inspect-planned-work-and-failures).

## Statuses

Each job wears its [job status](../../CONTEXT.md) as a text badge with a
color of its own, and each group counts its jobs by status. All statuses
are shown; each job appears and counts once, a shared prerequisite
included.

| Status | Badge |
| --- | --- |
| Run | `Run`, green |
| Reuse | `Reuse`, blue |
| Conditional reuse | `Conditional reuse`, orange; only in the plan of a config invocation of a downloaded release |

A plan with nothing to run still shows the jobs it reuses. A file an input
needs that no rule of the invocation produces, such as the output of a
stage that is off, is no job of the plan: a job reading it lists it under
`Existing inputs`. Released outputs on datasets that exist are checked
while planning and are `Reuse`; existing dataset content that differs from
the release's record fails planning. Why the preview reuses conditionally:
[previewing a release extension](../how-to/manage_releases.md#preview-and-investigate-unexpected-work).

## Reasons

Each job's details give the reasons for its status, in these words, with
the files or params they name:

| Reason | Status |
| --- | --- |
| `Missing outputs` | Run |
| `Incomplete outputs` | Run |
| `Inputs newer than its outputs` | Run |
| `Inputs another job of this plan rewrites` | Run |
| `The set of inputs changed since the last run` | Run |
| `The code changed since the last run` | Run |
| `Params changed since the last run` | Run |
| `The software environment changed since the last run` | Run |
| `Forced by the invocation` | Run |
| `The rule has no output, so it always runs` | Run |
| `The outputs it is needed for exist and are up to date, and nothing it needs is rerun` | Reuse |
| `Released outputs, reused during execution only if these regenerated dataset bundles match the release's recorded content` | Conditional reuse |
| `Reads outputs that are themselves conditionally reused` | Conditional reuse |

## Planning

The preview opens at once with a planning indicator and plans the invocation
in a child process, `python -m afabench.pipeline_plan`, through the shared
planner that the [compute estimate](estimate_compute_command.md) uses, so both
describe the same [scheduled jobs](../../CONTEXT.md). Quitting while planning
ends the child process.

A reused job whose params or resources cannot be evaluated as for
submission does not fail planning, since a run never submits it: its
identity, allocation and shell command are unavailable, and its details
give the error. The same failure for a scheduled job fails planning, as it
would fail the run.

When planning fails, as it does for an invocation Snakemake rejects, the
preview shows a failure state and the child's output in a scrollable text
area, where text can be selected and `c` copies it all to the clipboard. No
partial plan is shown. Quitting then exits with status 1.

## Grouping, filters and search

The controls above the tree regroup, filter and search the jobs of the plan
already loaded: nothing is planned again, and the invocation, its targets
and configuration stay as they are. Settings last until the preview quits.

| Control | Effect |
| --- | --- |
| Group by | The dimensions of the tree's levels, in order: `stage → method → dataset` (the default, or the launcher's [`--group-by`](pipeline_launcher_command.md#arguments)), any other order of the three, or any one or two of them, such as `dataset → method`. A dimension left out is still in each job's details. |
| Stage, dataset and status filters | Show only the jobs of the chosen pipeline stage, dataset or job status; `All` shows every one. They offer the values and statuses the plan's jobs have. Strict: a job without the chosen value, such as an aggregation job under a dataset filter, is hidden. |
| Method filter | Show the chosen method's jobs and the jobs they need, directly or not, such as their shared prerequisites, and no other shared prerequisite. The other filters still apply to the jobs it keeps. |
| Search (`/`) | Show only the jobs whose rule, stage, method, dataset and tree label contain every whitespace-separated term, regardless of case, with the matching groups expanded. |

Filters and search compose. Groups count the jobs shown. When some jobs are
hidden, the status line counts the shown jobs beside the whole plan's, as
`Shown: 3 of 9 jobs: 1 Run, 2 Reuse | Whole plan: 9 jobs: 5 Run, 4 Reuse`;
when none match, the tree says so. A job hidden by a filter is still listed,
with its status, among the relations of the jobs it needs or is needed by,
and can be inspected from there.

## Panes

| Pane | Contents |
| --- | --- |
| Controls | The grouping, the stage, method, dataset and status filters, and the search field. |
| Status line | The planning state, then the number of jobs and their counts by status, and when some are hidden, the shown jobs' counts beside the whole plan's. `Conditional reuse` is counted only when the plan has it. |
| Release warning | Only when the plan has conditionally reused jobs: regenerated datasets must match the release's recorded content before released outputs are reused during execution, and the preview has not checked them because they do not exist yet. |
| Jobs | The jobs as a tree, grouped as the controls choose, each group labelled with its counts by status. Stages come in pipeline order. Where a method level applies, a [shared prerequisite](../../CONTEXT.md)'s job, such as a dataset generation, classifier training or pretraining job, is under `Shared prerequisites`, a classifier trained for one method under that method, and a job that writes no job record, such as an aggregation or visualization job, under `Across methods`; where a dataset level applies, such a job is under `Across datasets`. A job whose identity is unavailable is under `Identity unavailable` at every level. Each job is in one place and counts once. Jobs below the last level are labelled by their status badge, rule, dataset realization, seeds and training and evaluation budgets. |
| Details | For a group: its dimension value, its jobs by status and its subgroups' counts. For a job: its status and the reasons for it, rule, stage, method, dataset, the classifier or pretrained model a classifier training or pretraining job's record names, dataset realization, seeds, training and evaluation hard budgets and soft-budget parameters, evaluation batch size, wildcards, device, CPUs, GPUs, requested time limit, placement, input and output paths, existing inputs no job of the plan makes, and for a shared prerequisite the methods that consume it. A dimension or identity field the job has none of reads `not applicable`; a job that writes no job record says so instead of listing identity fields. A job whose identity is unavailable gives the error under `Unavailable`, and its dimensions and allocation read `unavailable`. |
| Relations | Below a job's details: the jobs of the plan it needs (`Needs`), then the jobs that need it (`Needed by`), each with its status badge, its tree label and the method, classifier, pretrained model or dataset its job record names. A job the filters or search hide is marked `outside the filtered tree`. Existing inputs are no relations: no job of the plan makes them. |

Placement lists the SLURM placement resources the job requests besides its
CPUs, GPUs and time limit, as `name=value` lines: `slurm_partition`,
`slurm_account`, `clusters`, `qos`, `constraint`, `gres`, `gpu_model`,
`mem_mb`, `mem_mb_per_cpu` and `slurm_extra`, each when set. A job requesting
none reads `none requested`.

Both panes are side by side in a terminal of 80 columns or more. In a
narrower terminal the controls take two columns and the focused pane fills
the terminal: `Enter` switches to the details, `Tab` from the tree to the
details and on to the relations, and `Tab` from the relations or `Escape`
back to the tree.

## Following relations

`Enter` on a relation inspects that job: its details and relations replace
the previous job's. If the tree shows the job, the tree's cursor moves to it,
expanding its groups. If the filters or search hide it, its details start
with `Outside the filtered tree`, and the filters, search, tree and counts
stay as they are. `Backspace` returns to the job the last relation was
followed from, with that relation selected; repeated, it retraces the
followed relations in reverse.

## Shell commands

`s` shows the selected job's shell command over the browser, as Snakemake
formats it from the job's evaluated params while planning, as a dry run
prints it. `c` copies it and `Escape` closes it. Nothing runs. When the rule
has no shell command, Snakemake cannot format it without running the job, or
the job's identity is unavailable, the preview says so instead.

## Keys

| Key | Action |
| --- | --- |
| `↑`, `↓` | Move through the tree, or scroll the details. |
| `→` | Expand the selected group. |
| `←` | Collapse the selected group, or move to its parent. |
| `Enter` | Show the selected group's summary or job's details. |
| `Tab` | Move from the tree to the details, the relations, and back to the tree. |
| `↑`, `↓`, `Enter` | On the relations, choose a relation and inspect its job. |
| `Backspace` | Return to the job the last relation was followed from. |
| `s` | Show the selected job's shell command; `c` copies it, `Escape` closes it. |
| `Escape` | Back to the tree. |
| `g` | Go to the grouping control. |
| `f` | Go to the filters, starting at the stage filter. |
| `←`, `→` | On a control, move to the previous or next one. |
| `Enter`, `↓` | On a control, open its options; `Enter` chooses one. |
| `/` | Search: typing narrows the tree; `Enter` or `Escape` goes back to it. |
| `c` | After a planning failure, copy the diagnostics to the clipboard. |
| `q` | Quit: exit status 1 after a planning failure, else 0. |

The footer lists the keys, the tree's border the arrow keys, and the
relations' border how to follow and return.
