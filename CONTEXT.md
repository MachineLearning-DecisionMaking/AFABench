# AFABench

A benchmark framework for Active Feature Acquisition (AFA): comparing methods
that decide, per instance and one step at a time, which costly features to
acquire before predicting a class label. The vocabulary below follows the
accompanying paper (Schütz et al., KDD '26, arXiv:2508.14734) where the paper
and code agree, and the code where they differ. Paper-only synonyms are listed
under _Avoid_.

## Language

### Problem setting

**Active Feature Acquisition (AFA)**:
Sequentially choosing which features of a single instance to acquire, trading
predictive performance against acquisition cost. Decisions are instance-wise
and may condition on previously observed values.
_Avoid_: Dynamic feature selection, sequential feature selection, active
learning (that acquires labels, not features)

**Static feature selection**:
Choosing one global subset of features used for every instance. Included in the
benchmark only as a baseline paradigm.
_Avoid_: Global feature selection, feature importance method

**Instance**:
One data point: a full feature vector together with its label. In an image
dataset an instance is a whole image.
_Avoid_: Sample, example, row

**Feature**:
One scalar entry of an instance. For images, one pixel channel value.
_Avoid_: Attribute, variable, column

**Feature group**:
A set of features that is always revealed together by a single selection, for
example an image patch or the one-hot context columns of CUBE-NM. With the
direct Unmasker every feature group is a single feature.
_Avoid_: Patch (use only for images), block (reserved for CUBE-NM)

**Feature cost**:
The fixed, dataset-defined non-negative cost of acquiring one feature. Uniform
unit cost is the default. Dataset generation fixes the feature costs, from the
dataset's feature costs file or the dataset itself, and stores them in each
dataset bundle; later stages read them only from there.
_Avoid_: Price, acquisition fee

**Selection cost**:
The cost incurred by one selection, derived by the Unmasker from the feature
costs of the group it reveals.
_Avoid_: Action cost

**Accumulated cost**:
The sum of selection costs paid so far in an episode.
_Avoid_: Cumulative cost, spent budget

### Episode components

**Episode**:
The evaluation of one instance: starting from the initial feature mask, the
policy acts and the Unmasker reveals features step by step until a stop. Each
round is a **time step**.
_Avoid_: Trajectory, rollout, run

**Feature mask**:
A boolean indicator per feature of whether it has been observed.
_Avoid_: Observation mask, visibility mask

**Masked features**:
The instance's features with every unobserved feature replaced by zero.
_Avoid_: Partial observation, observed subset

**Initializer**:
The component that chooses the initial feature mask of an episode. A **cold
start** reveals nothing; a **warm start** reveals a fixed or random subset.
_Avoid_: Warm-up, initial policy

**Unmasker**:
The component that maps a selection to the feature group it reveals and
reports the selection costs. Changing the acquisition scheme of a dataset
means writing a new Unmasker, not a new method.
_Avoid_: Revealer, acquisition function, mask updater

**Direct Unmasker**:
The Unmasker where selection i reveals exactly feature i.
_Avoid_: Identity unmasker, default unmasker

**Image patch Unmasker**:
The Unmasker where each selection reveals one square patch of an image.
_Avoid_: Patch unmasker, grid unmasker

**Context Unmasker**:
The CUBE-NM Unmasker, which has one selection that reveals all one-hot context
features at once.
_Avoid_: CUBE-NM unmasker (as a concept; it is fine as a config name)

**Policy**:
The part of an AFA method that outputs an action given the current masked
features and feature mask.
_Avoid_: Agent, selector, acquirer

**Classifier**:
A model that predicts the label from masked features and a feature mask. The
benchmark is classification-only.
_Avoid_: Predictor (paper term), model (ambiguous)

**External classifier**:
A classifier pretrained independently of any method and shared by all methods
on a dataset realization during evaluation. Headline results use it.
_Avoid_: Shared classifier, common classifier

**Built-in classifier**:
A classifier that an AFA method trains and ships itself, often jointly with its
policy. Reported separately from external-classifier results.
_Avoid_: Internal predictor, internal classifier (paper terms), joint
classifier

**AFA method**:
A policy together with its optional built-in classifier, packaged so it can be
trained, saved, loaded, and evaluated.
_Avoid_: Algorithm, model, approach

### Actions and selections

**Action**:
The policy's output at a time step: either stop, or a request to reveal one
feature group. Action 0 is stop; action i > 0 is selection i - 1.
_Avoid_: Query (alone; a query instance is an instance, not an action),
move, decision

**Stop action**:
The action that ends the episode and triggers the final prediction.
_Avoid_: Terminate, halt, done

**Selection**:
A zero-indexed choice of feature group handed to the Unmasker. Selections never
include stop, so there is one fewer selection than action.
_Avoid_: Feature index (not the same once groups exist), choice

**Selection history**:
The ordered sequence of selections performed before a time step in an episode,
including repetitions but excluding stop and features revealed by the
Initializer.
_Avoid_: Action history (includes stop), acquired-feature set (loses order)

**Selection mask**:
A boolean indicator per selection of whether it has already been performed.
_Avoid_: Action mask, availability mask

**Forced stop**:
An episode ending because the next action would have exceeded the hard budget,
rather than because the policy chose to stop.
_Avoid_: Budget cutoff, truncation

**Forced acquisition**:
Overriding stop actions with the first available selection so that a policy
keeps acquiring up to the hard budget. Used in hard-budget evaluation.
_Avoid_: Ignore stop, no-stop mode

**Cheating method**:
A method or component that is handed the true label during an episode, for
benchmarking upper bounds only.
_Avoid_: Oracle (reserved for the oracle-based paradigm)

### Budgets

**Hard budget**:
A per-instance cap on accumulated selection cost. With unit costs it equals
the number of allowed selections, not the number of features. May differ
between training and evaluation: a method may train once under one hard
budget and be evaluated at several.
_Avoid_: Feature budget, max features, k

**Soft-budget parameter**:
The scalar that weights acquisition cost against prediction loss and so lets a
policy spend more on hard instances. May differ between training and
evaluation.
_Avoid_: Alpha, lambda, cost parameter, trade-off coefficient

**Hard-budget setting**:
The evaluation regime where every method acquires up to the same hard budget
and no soft-budget parameter is given.
_Avoid_: Fixed-budget setting, k-feature setting

**Soft-budget setting**:
The evaluation regime where no hard budget is enforced and each method stops
according to its soft-budget parameter.
_Avoid_: Early-stopping setting, cost-sensitive setting

### Method taxonomy

**Myopic method**:
A method that picks the next feature by maximizing one-step expected
information gain about the label per unit cost, with no lookahead.
_Avoid_: Greedy (README legacy), one-step method

**Non-myopic method**:
A method that optimizes long-term utility and may pay for uninformative
features now to acquire better ones later.
_Avoid_: Non-greedy, planning method, lookahead method

**Conditional mutual information (CMI)**:
The quantity myopic methods estimate: information a candidate feature carries
about the label given the observed features. Methods are split by whether they
estimate it **generatively** or **discriminatively**.
_Avoid_: Information gain (as a distinct concept), utility

**Model-free RL method**:
A non-myopic method that learns a policy directly from episodes of the AFA
Markov decision process.
_Avoid_: Policy-gradient method, Q-learning method (too specific)

**Model-based RL method**:
A non-myopic method that learns an explicit model of unobserved features and
plans with it.
_Avoid_: Planning-based RL

**Oracle-based method**:
A non-myopic method that scores candidate acquisitions with a lookahead oracle
instead of reinforcement learning.
_Avoid_: Non-RL method, search-based method

**Dummy method**:
A method with trivial behavior (random or sequential selection) used as a
sanity check and lower bound.
_Avoid_: Baseline (static methods are also baselines), toy method

**In-context method**:
An AFA method whose fit stages learn from tasks drawn from a task prior and
whose policy and classifier condition on a context set at evaluation, with
no per-dataset fitting on real labels.
_Avoid_: Meta-learning method, foundation model, meta-AFA method (paper
term)

**Task prior**:
The distribution that an in-context method's fit stages draw their
training tasks from. A pool task prior draws a task's instances from the
pretraining pool and labels them by a synthetic labelling function. The
TabICL task prior has no pool: it generates each task's features and
labels together, with the dataset's feature and class counts.
_Avoid_: Pretraining prior, synthetic prior (ambiguous about which half is
synthetic), BNN prior (one choice of labelling function)

**Pretraining pool**:
The instances that a pool task prior draws its instances from: the train
split's features, whose labels are discarded, or draws that share only the
dataset's feature count.
_Avoid_: Pretraining set, pretraining data

**Context set**:
The fixed set of labelled instances that an in-context method receives at
evaluation in place of fitting on real labels. It is drawn from the
validation split, so it is disjoint from the pretraining pool and from the
evaluated instances.
_Avoid_: Context (alone; CUBE-NM has a context feature), historical data,
shots, support set

**Query instance**:
An instance that an in-context method acts on and predicts for at
evaluation, as opposed to the context set it conditions on.
_Avoid_: Query (alone; reads as an action), test point

### Datasets

**Dataset key**:
The canonical snake_case string identifying a dataset throughout configs,
file names, and feature-cost files.
_Avoid_: Dataset name, dataset id

**Split**:
One of the train, validation, or test partitions of a dataset.
_Avoid_: Fold, subset

**Dataset realization**:
One seeded generation and split of a dataset. Benchmark results average over
several dataset realizations.
_Avoid_: Dataset instance (an instance is one data point), seed (the seed is
the input, the realization is the output), run

**Generation index**:
The position of an instance in the dataset produced by dataset generation,
before splitting. Together with the dataset realization it identifies the
instance. For a real-world dataset, generation keeps the source's order, so
the generation index also locates the instance in the source, in every
realization, unless the dataset drops source rows; then its source row does.
_Avoid_: Instance index, original index, row, sample id

**Provider**:
The organization that originally publishes a real-world dataset, such as
UCI or PhysioNet, as opposed to a mirror that copies its files.
_Avoid_: Host, upstream (ambiguous with vendored code)

**Prepared source**:
The file in a dataset's data folder that dataset generation reads, made
from the provider's raw files by source preparation. It is committed unless
the dataset's license forbids redistributing it.
_Avoid_: Raw data (the provider's files), processed data (the bundles)

**Source preparation**:
Turning a provider's raw files into a dataset's prepared source. It runs
rarely and reproduces the committed file exactly; dataset generation, which
runs on every pipeline run, starts from the prepared source.
_Avoid_: Preprocessing (also covers what dataset generation does)

**Source row**:
The row of a tabular instance in its source file, counting data rows from 0.
It equals the generation index unless the dataset drops rows from its source
(MiniBooNE drops placeholder and outlier rows); dataset bundles save it.
_Avoid_: Original index, row id

**Split index**:
The position of an instance within one split of one dataset realization.
_Avoid_: Instance index, episode id (an episode's order of evaluation need
not follow the split)

**CUBE**:
The synthetic tabular AFA dataset where each class makes three class-specific
features informative and the rest noise.
_Avoid_: Cube dataset (capitalize)

**CUBE-NM**:
The synthetic dataset introduced by this project, which prepends a **context
feature** that selects which **block** of CUBE-style features is informative.
Built to expose the gap between myopic and non-myopic methods.
_Avoid_: AFAContext (legacy class name), cube context

**CUBE-NUC**:
The CUBE variant with non-uniform feature costs: the informative features are
duplicated in a second block that costs twice as much.
_Avoid_: Cube non-uniform, costly cube

**PhysioNet Challenge 2012**:
The ICU in-hospital mortality classification dataset from the
PhysioNet/Computing in Cardiology Challenge 2012.
_Avoid_: PhysioNet (the hosting repository, not a specific dataset)

**MiniBooNE subset**:
MiniBooNE restricted to the 10 features used by the L2M paper, with the
same instances as MiniBooNE.
_Avoid_: MiniBooNE-10, small MiniBooNE

**Noiseless variant**:
A synthetic dataset generated with zero noise, so perfect accuracy is
attainable and the optimal policy is known.
_Avoid_: Clean variant, deterministic variant

### Software usage

**AFABench library**:
The reusable Python components for active feature acquisition that users
import into their own code, independently of the benchmark pipeline.
_Avoid_: Core framework, Python framework

**Benchmark pipeline**:
The configured orchestration of AFABench's pipeline stages to produce
benchmark outputs. It uses the AFABench library but is not required for
library use.
_Avoid_: Framework (when distinguishing it from the library)

### Pipeline

**Benchmark release**:
A versioned collection of benchmark results and reusable outputs, produced
with identified producing code revisions and pipeline configuration and
published by a maintainer under a release id. It is published from one release
package. Users choose it by its release id; it never holds smoke-test
outputs.
_Avoid_: Smoke release, latest results, pipeline run (a run need not be published)

**Output root**:
The folder one pipeline run writes all its artifacts and job records under.
A smoke test has its own output root, apart from the production one, so
its outputs never stand in for a real run's. A benchmark release downloaded
into an output root is a cache: a run reuses what it holds and produces
the rest, so the release never changes what the run outputs.
_Avoid_: Output folder, output directory, results dir

**Output snapshot**:
A verbatim copy of everything the pipeline has written under its output
root, taken so it can later be put back exactly where the workflow expects
it. It holds every file, with no selection.
_Avoid_: Backup, archive, export, release package

**Release package**:
The outputs prepared for publication as a benchmark release, retaining models
and results while excluding dataset bundles according to each dataset's release
policy and redistribution permission. Unlike a local output snapshot, it need
not contain every dataset bundle of the benchmark run.
_Avoid_: Backup, verbatim snapshot

**Release manifest**:
The JSON file beside a release package's output tree that identifies its
benchmark release, records the maintainers' dataset redistribution review
and the workflow configuration
that lays out the release, and indexes the package's included artifacts and dataset bundles excluded by
policy, with their exclusion reasons. The index is generated from the artifacts' provenance records,
never from the workflow configuration, so it is a cache of those records
rather than a second source of truth. It also records save-time package file
hashes for validation before publication (see
`docs/reference/release_manifest.md`).
_Avoid_: Metadata, release info, provenance record (that is per artifact)

**Payload category**:
One kind of reusable output a release manifest lists: raw or transformed
evaluation tables, the job duration table, or dataset, classifier,
pretrained-model or AFA-method bundles. A run reuses a published method's
evaluations only if that method's AFA-method bundles were downloaded too.
_Avoid_: Artifact type, output kind

**Shared prerequisite**:
A bundle that training or evaluating any method on a dataset needs and
that belongs to no single method: the dataset bundles, the external
classifier and the pretrained models.
_Avoid_: Shared input, common artifact

**Repository adopter**:
A user who forks the repository, adds their own method, and runs the
pipeline over a downloaded benchmark release, so only their method is
trained and evaluated while the published methods' outputs are reused
(see `docs/explanation/user_types.md`).
_Avoid_: Benchmark adopter, adopter

**Library user**:
A user who imports AFABench's library code, such as its evaluation
functions or models, into their own code base and never runs its pipeline
(see `docs/explanation/user_types.md`).
_Avoid_: Independent evaluator, package user

**Results-only user**:
A user who downloads a benchmark release's tables or plots without
installing AFABench (see `docs/explanation/user_types.md`).
_Avoid_: Results-only researcher, results-only journey

**Pipeline stage**:
One kind of pipeline job: **dataset generation**, **classifier training**,
**pretraining**, **training**, **evaluation**, **transformation**,
**aggregation** or **visualization**. Pretraining and training are optional
per method; evaluation is mandatory and shared by all methods. Classifier
training, pretraining, training and evaluation take a declared `cpu` or
`cuda` choice; the others always run on CPU. Code and execution configs name
each stage in snake case, e.g. `classifier_training`.
_Avoid_: Execution activity, activity, phase, step (reserved for time steps);
plotting (visualization covers every figure and view, not only plots)

**Fit stage**:
Either of the two pipeline stages that fit a model from a contract's inputs
and save it as a bundle: pretraining or training. Not classifier training,
which has no contract.
_Avoid_: Learning stage, training stage (reads as training only)

**Pretrained model**:
A named bundle produced in the pretraining stage and reusable across
methods, for example a partial VAE shared by EDDI and ODIN.
_Avoid_: Base model, backbone, checkpoint

**Method name**:
The pipeline-level identifier of one configured variant of an AFA method, such
as a method paired with its external or built-in classifier.
_Avoid_: Method id, method key

**Default hyperparameters**:
The values a method's pretraining or training uses on every dataset key
without a dataset override. Every hyperparameter has one, so a method runs
on any dataset.
_Avoid_: Base config, tabular config

**Dataset override**:
The hyperparameters a method uses on one dataset key in place of its
default hyperparameters, holding only the values that differ.
_Avoid_: Experiment, experiment config, dataset config

**Method set**:
A named group of method names plotted together. The `all` method set always
exists, holds every method name compared in a run, and cannot be configured.
Unrelated to the `all` pipeline profile.
_Avoid_: Method group, plot group

**Dataset set**:
A named group of datasets plotted together; each gets a `dataset_set-<name>`
plot folder when any of its datasets has evaluation results. The `all`
dataset set always exists, holds every dataset of a run, and cannot be
configured.
_Avoid_: Dataset group, plot group

**Bundle**:
The on-disk folder format in which datasets, classifiers, pretrained models,
and AFA methods are saved and loaded.
_Avoid_: Checkpoint, artifact (the umbrella term for bundles and evaluation
tables, not a synonym for bundle), pickle

**Artifact**:
Anything a pipeline job writes that can carry a provenance record: a bundle
or an evaluation table. Use it only when both are meant; say bundle or
evaluation table when only one is.
_Avoid_: Bundle (only one kind of artifact), result, file

**Provenance record**:
The typed description of how one bundle or evaluation table was produced:
code commit, resolved configuration, seed actually used, input bundles and
their content hashes, environment, and dataset identity. It is embedded in
the artifact itself (see `docs/adr/0002-provenance-recorded-in-artifacts.md`).
_Avoid_: Metadata (the free-form manifest field), lineage, run info

**Smoke test**:
A run mode where every stage executes as fast as possible to verify the
pipeline works end to end. It writes under its own output root, never the
production one.
_Avoid_: Dry run, quick mode

**Job duration**:
The measured wall-clock time of one pipeline job, from the start to the end of
its script.
_Avoid_: Runtime (the scheduler's requested time limit), job time, elapsed

**Job record**:
The description of one finished pipeline job: what it computed, its job
duration, and the hardware it was allocated. Unlike a provenance record it
describes the run, not the artifact, and is kept beside the artifact rather
than inside it.
_Avoid_: Time file, job log, provenance record (that is per artifact)

**W&B run**:
The Weights & Biases record of one attempt of one pipeline job that logs
metrics. A retried or rerun job gets another W&B run, so failed and
successful attempts never share one.
_Avoid_: Run (on its own), wandb run

**W&B group**:
The W&B runs of one pipeline invocation, collected under one name. A later
invocation joins an existing W&B group only when it names that group, so a
rerun after a failure starts a new group unless told otherwise.
_Avoid_: Run group, wandb group, experiment

**Job duration table**:
All job records of a pipeline's outputs collected into one table, one row
per job, with no aggregation.
_Avoid_: Time results, timing table

**Compute estimate**:
The predicted compute a pipeline invocation will consume before it runs,
derived from job durations of earlier runs and the allocation each job would
request, in CPU core-hours and GPU-hours.
_Avoid_: Compute cost (cost means feature, selection or accumulated cost),
compute budget (budget means hard budget or soft-budget parameter)

**Job type**:
What jobs share when they differ only in seeds, dataset realization, hard
budgets and soft-budget parameters: pipeline stage, name (method,
pretrained-model or classifier name) and dataset key.
_Avoid_: Job kind, job group

**Match level**:
How a compute estimate found a planned job's job durations: **exact**, from
job records with the same identity and device; **pooled**, from every job
record of its job type on the same device; or **unestimated**, from none, so
the job is left out of the totals.
_Avoid_: Match quality, fallback level

**Failure history**:
The failed and timed-out job records of a planned job's job type. They give
no job duration, but warn that the compute estimate of its jobs may be low.
_Avoid_: Timeout history, failure count

**Pipeline profile**:
The configuration that decides what a pipeline run contains, either grouped
by benchmark setting or supplied as one runnable benchmark config. It says
nothing about where jobs run or on which device; overrides narrow or change
it for one run.
_Avoid_: Preset, pipeline config, experiment config

**Execution file**:
The configuration that declares which pipeline stages and methods run on CPU
and which on GPU, independent of any computing site. A site profile then maps
each declared device to an allocation.
_Avoid_: Device config, hardware profile, execution profile

**Site profile**:
The configuration that places pipeline jobs on one computing site: its
scheduler settings and its allocations. Running locally needs no site
profile.
_Avoid_: Cluster config, cluster profile

**Pipeline launcher**:
The command that runs the pipeline from a pipeline profile or runnable
benchmark config, an optional site profile and overrides.
_Avoid_: Wrapper, runner

**Pipeline invocation**:
One command of the pipeline launcher, from start until Snakemake exits. A
pipeline run may take several invocations when reruns complete what an
earlier one left unfinished.
_Avoid_: Launch, session

**Pipeline plan**:
What a pipeline invocation would do, read from its Snakemake job graph
without running anything: every job of the requested targets' dependency
graph with its job status and Snakemake's reasons for it, which jobs each
needs, the allocation each would request and the shell command each
would run. Target rules, which have no action, are no jobs of it. A
compute estimate estimates the scheduled jobs of a pipeline plan.
_Avoid_: Dry run (Snakemake's printed listing of an invocation), job list

**Job status**:
What a pipeline invocation does with a job of its pipeline plan. **Run**:
the invocation schedules the job, including to replace outputs that exist
but are stale. **Reuse**: the invocation uses the job's existing outputs
without running it. Both come from Snakemake's need-run decision, never
from whether output files happen to exist: a missing output nothing needs
leaves a job reused. **Conditional reuse**: in the plan of a config
invocation of a downloaded release, the job's released outputs are reused
only if the dataset bundles it reads, which must be regenerated first,
match the release's recorded content; Snakemake alone would run it.
**Scheduled jobs**: the jobs of a pipeline plan whose status is Run or
Conditional reuse, which Snakemake schedules for the invocation; a compute
estimate estimates them.
_Avoid_: Cached, skipped, up to date (a reused job's description, not its
status), scheduled for one job's status (Run), verified reuse (Reuse)

**Interactive preview**:
The terminal browser of a pipeline plan that the pipeline launcher opens
with `--interactive-preview`: the jobs grouped by pipeline stage, method
and dataset, each with its job status, Snakemake's reasons for it, its
identity, paths and allocation, the jobs it needs and is needed by, which
can be followed, and its shell command on request, and each group counting
its jobs by status. Its grouping, filters, search and navigation reshape
the loaded pipeline plan only: it runs and submits nothing and changes no
invocation.
_Avoid_: TUI, job browser, dry-run viewer

**Pipeline profile override**:
A value given for one run that replaces a pipeline profile's value for the
same key, or merges into it key by key for a mapping.
_Avoid_: Config option, CLI config

**Allocation**:
The scheduler resources a site profile requests for a pipeline job, such as
account, partition and GPUs. A site profile has one for CPU jobs and may have
one for GPU jobs; a job's declared device selects which.
_Avoid_: Resources (Snakemake's term covers runtime and memory too), partition

**Contract**:
The fixed set of inputs the pipeline gives a pretraining or training script,
and the bundle it expects back at the save path. Each stage has its own
contract, and the two are independent: they take the same inputs today, but
neither is a subset of the other.
_Avoid_: Script interface, pipeline arguments, CLI args

**Pretraining contract**:
The contract for a pretraining script.

**Training contract**:
The contract for a training script.
