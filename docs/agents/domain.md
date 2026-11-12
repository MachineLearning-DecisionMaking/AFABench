# Domain Docs

How the engineering skills should consume this repo's domain documentation
when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root. It is the glossary for AFABench and is
  the source of truth for naming. It reconciles the paper's vocabulary with
  the code's vocabulary and lists the synonyms to avoid.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in.
  The `/domain-modeling` skill (reached via `/grill-with-docs` and
  `/improve-codebase-architecture`) adds ADRs when a decision actually gets
  resolved.

Supporting background, in order of authority when they disagree:

1. `CONTEXT.md` (canonical terms).
2. The code's protocols in `src/afabench/core/types.py` (what the terms mean
   operationally).
3. The paper "AFABench: A Generic Framework for Benchmarking Active Feature
   Acquisition" (Schütz, Wu, Rezvan, Aronsson, Haghir Chehreghani, KDD '26,
   arXiv:2508.14734). Sections 2 and 3 define the problem and the episode
   components; Section 4.3 defines CUBE-NM.
4. `docs/reference/pipeline_configuration.md` for pipeline-level terms
   (method options, method sets, pretrain mappings, hard budgets, soft-budget
   parameters).

## File structure

Single-context repo:

```
/
├── CONTEXT.md
├── docs/adr/            ← architecture decision records
├── src/afabench/        ← library code
├── scripts/             ← pipeline entry points
├── conf/                ← Hydra configs
├── workflow/            ← Snakemake workflows
├── data/                ← one folder per dataset key (ADR 0010), prepared sources committed (ADR 0014)
└── logs/, output/       ← generated files (gitignored)
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor
proposal, a hypothesis, a test name), use the term as defined in
`CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

Known traps in this repo:

- **Action vs selection.** Action 0 is stop; selection i is action i + 1 and
  never includes stop. Never use them interchangeably.
- **Feature vs feature group vs selection.** With the direct Unmasker these
  coincide. With image patches or CUBE-NM they do not. A hard budget counts
  selection cost, not features.
- **Classifier vs predictor.** The paper says predictor; the code says
  classifier. Use classifier.
- **Built-in vs internal.** The paper says internal predictor; the code says
  built-in classifier. Use built-in.
- **Myopic vs greedy.** The paper and code say myopic; only the README table
  still says greedy. Use myopic.
- **Dataset realization vs seed.** The seed is the input; the realization is the
  generated, split dataset. ADRs written before the rename say dataset
  instance (`dataset_instance_index`); the code says
  `dataset_realization_index`.
- **Generation index vs split index.** The generation index is an
  instance's position before splitting; the split index is its position in
  one split. Neither is the episode id. Never write a bare "instance index".
  For synthetic datasets a generation index means nothing without its
  dataset realization.

If the concept you need isn't in the glossary yet, that's a signal: either
you're inventing language the project doesn't use (reconsider) or there's a
real gap (note it for `/domain-modeling`).

## Flag material architecture conflicts

Surface changes to consequential architectural decisions and explain their
trade-offs. For routine API/schema evolution, apply `AGENTS.md`'s development
policy rather than treating an ADR's compatibility language as a permanent
constraint or adding migration history to user docs.

## Decisions worth an ADR if reopened

Beyond the ADRs in `docs/adr/`, the following are documented design choices from the paper
(Sections 3 and 4.4) that satisfy the hard-to-reverse, surprising, and
trade-off criteria. Write an ADR before changing any of them:

- Evaluation is a single shared script for all methods; methods may train
  however they like but must conform to the episode protocol at evaluation.
- Hard-budget and soft-budget settings are evaluated separately and never
  mixed in one comparison.
- Headline results use a shared external classifier per dataset; built-in
  classifiers are reported separately.
- Methods keep the hyperparameters of their original implementations rather
  than being tuned per dataset, with documented exceptions.
- An action that would exceed the hard budget is overridden to stop, and the
  episode is marked as a forced stop.
