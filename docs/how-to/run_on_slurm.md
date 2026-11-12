# Run on SLURM

Configure a site profile and submit from one authorized host with shared
repository, environment, input and output storage. Keep the controller alive
with `tmux` or your site's approved mechanism. Use one cluster per invocation.

## 1. Adapt a profile to your site

Copy the example matching your site's GPU request convention:

```shell
cp -r workflow/profiles/site/examples/mixed-gres workflow/profiles/site/my_site
# Or copy examples/mixed-gpus for --gpus requests.
```

Do not submit with the example accounts or partitions unchanged.
In `my_site/config.yaml`, update `execution_site_file` to
`workflow/profiles/site/my_site/site.yaml`.

In `site.yaml`, set your accounts and partitions:

```yaml
execution_site:
  cpu:
    slurm_partition: <cpu partition>
    slurm_account: <cpu account>
  gpu:
    slurm_partition: <gpu partition>
    slurm_account: <gpu account>
    gres: gpu:<type>:1
    # Alternatively: gpu: 1 and optionally gpu_model: <type>
```

Always provide a CPU allocation, including for GPU-method runs. Do not
request GPUs in it. Use exactly one GPU request convention in the GPU
allocation. Put other scheduler flags, such as `--qos=short`, in the
allocation's `slurm_extra`, not in `default-resources`.

If using the shipped `alvis` profile, verify that CPU-only submissions are
permitted before running; that behavior is unverified.

## 2. Set CPU, memory and walltime requests

Set `mem_mb` and `cpus_per_task` in the profile's `default-resources` or
`set-resources`. Keep partition, account and GPU requests in `site.yaml`.
Set hours-based walltime there using the
[walltime fields](../reference/requested_walltime.md#fields).
Remove profile `set-resources.*.runtime` values that would shadow that table.

## 3. Check requested walltime before submission

```shell
uv run workflow/run.py kdd26 --site my_site -i
```

Check each job's Device, CPUs, GPUs, Placement and Time limit in minutes
([job details](../reference/interactive_preview.md#panes)).
Use measured durations with a safety margin. For a one-run override, append
`-- --set-resources pretrain_model:runtime=240`.

Remove `-i` to submit. Keep the controller alive using the site's approved
mechanism. [Pipeline commands](run_pipeline.md).

## Check CPU/GPU choices

Choose the execution file using the [customization guide](customize_pipeline.md#choose-cpus-or-gpus).
It selects job devices; `site.yaml` supplies the corresponding allocations.
Dataset generation, transformation, aggregation and visualization always
use the CPU allocation. Check that these jobs request no GPUs and have
suitable CPU and memory limits; do not run processing on a login node
outside site policy.

## Run on a cluster of mixed architectures

Use a filesystem shared by every node. For each allocation in `site.yaml`,
name the venv matching its CPU architecture:

```yaml
execution_site:
  cpu:
    # Keep the partition and account entries.
    venv: .venv-x86_64
  gpu:
    # Keep the partition, account and GPU request entries.
    venv: .venv-aarch64
```

To inspect the plan before building site venvs, run this from the synced
checkout; replace `arrhenius` with your site name:

```shell
uv run workflow/run.py kdd26 --site arrhenius -i
```

Treat missing or stale site venv warnings as prerequisites for execution,
not blockers for previewing. Use `-n` for a plain-text dry run or `-e` for
an estimate instead.

Before executing jobs, submit the builds from the checkout:

```shell
uv run just build-venvs arrhenius
```

Append `--dry-run` to inspect build submissions first. Check
`workflow/venvs/build-<job id>.log` and wait for every build to finish.
Rebuild after `uv.lock` changes, with no pipeline jobs running. Allow roughly
85,000 files per venv when checking filesystem quotas, counting the checkout's
`.venv` that `uv run` syncs on the login node.

Set these fields in the site profile's `config.yaml`:

```yaml
shared-fs-usage: [persistence, input-output, sources, source-cache, storage-local-copies]
precommand: export PATH=$PWD/workflow/venvs/bin:$PATH
```

Check and submit:

```shell
uv run workflow/run.py kdd26 --site arrhenius -i
```

Remove `-i` to run. For individual commands on a node, use
`workflow/venvs/bin/python`.
