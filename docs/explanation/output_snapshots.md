# Output snapshots

An output snapshot is a copy of everything the pipeline wrote under its
output root, made so it can be put back where Snakemake expects it. A
benchmark release is published from a release package prepared from those
outputs. See [save or restore outputs](../how-to/run_pipeline.md#save-or-restore-outputs)
for local snapshots, or the [release guide](../how-to/manage_releases.md) for
publishing and reusing results.

## A snapshot copies everything

A snapshot copies the whole output root as it is, including stale files no
current rule would produce (#39). It does not select by dataset, method or
output category; selection happens when downloading a published release
([`download`](../reference/snapshot_command.md#selection)). Its bundles
and evaluation tables carry their own provenance records.

Saving with a release id instead prepares a release package: it omits
dataset bundles excluded by dataset policy or lacking explicit redistribution
permission. Its manifest indexes included artifacts and records the excluded
dataset bundles' identities and reasons separately. Models and results remain
included even when their dataset bundles are omitted. This keeps local
snapshots suitable for complete restoration without publishing datasets
that have not been approved for redistribution.

## Why a snapshot keeps mtimes

Snakemake decides whether an output is stale by comparing modification
times (mtimes): a file output by its own mtime, and a directory output by
the mtime of its `.snakemake_timestamp` file, or else of the directory
itself. A snapshot keeps the mtimes of every file and directory, including
the hidden `.snakemake_timestamp` files, so a tree in which no rule's
inputs were newer than its outputs is still so after being saved and
restored. Snakemake then reuses the restored outputs instead of producing
them again.

## Why a snapshot omits Snakemake metadata

A snapshot does not contain `.snakemake/metadata`, the record Snakemake
uses to rerun an output when its rule's code, parameters, input set or
software environment changed. Without it, a restored output is judged by
its mtime only. This is deliberate
([ADR 0003](../adr/0003-snapshots-omit-snakemake-metadata.md)):

- Every pipeline rule is a `shell:` rule, so Snakemake's code trigger
  hashes the shell command written in the `.smk` file, not the script it
  runs. That text changes for reasons unrelated to the output, such as a
  reformatted line.
- A repository adopter's fork moves on from the release's producing commit.
  With the metadata restored, such unrelated changes would make Snakemake
  rerun the published methods, the retraining a downloaded release exists
  to avoid, and an automatic compatibility check that #36 decided
  against.

The consequence: after restoring a snapshot into a checkout where a rule's
code or parameters changed, a dry run does not notice. Delete or
regenerate that rule's outputs yourself.
