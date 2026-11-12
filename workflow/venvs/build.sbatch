#!/bin/bash
# Sync AFABench's locked environment into the venv of this node's architecture.
#
# Builds <checkout>/.venv-<arch>, where <arch> is `uname -m`, with uv from
# uv.lock (without the dev group), and records the uv.lock it synced from as
# .venv-<arch>/afabench.uv.lock, which the pipeline launcher checks before it
# submits (ADR 0015). On a cluster whose CPU and GPU nodes differ (Arrhenius:
# x86_64 and aarch64) submit it once to each. The account and partition
# differ per architecture, so they are given at submit time;
# workflow/venvs/submit_builds.py submits it with those of each allocation of
# a site profile that names a venv:
#
#   uv run workflow/venvs/submit_builds.py arrhenius
#
# which, for Arrhenius, submits from the checkout root:
#
#   sbatch -A <cpu-account> -p cpu --output=workflow/venvs/build-%j.log \
#     workflow/venvs/build.sbatch .venv-x86_64
#   sbatch -A <gpu-account> -p gpu --gpus=1 --output=workflow/venvs/build-%j.log \
#     workflow/venvs/build.sbatch .venv-aarch64
#
# The venv is synced in place: rebuild it while no job of the pipeline runs.

#SBATCH --job-name=afabench-venv
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

if [[ $# -ne 1 ]]; then
    cat >&2 <<'USAGE'
Usage: submit with sbatch from the checkout root, once per node architecture,
with the accounts and partitions of a site profile:

  uv run workflow/venvs/submit_builds.py <site>      (or: uv run just build-venvs <site>)

or by hand:

  sbatch -A <account> -p <partition> --output=workflow/venvs/build-%j.log \
      workflow/venvs/build.sbatch .venv-$(uname -m on that partition)

See "Run on a cluster of mixed architectures" in docs/how-to/run_on_slurm.md.
USAGE
    exit 2
fi
checkout="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$checkout"
arch="$(uname -m)"
venv=".venv-$arch"
if [[ "$1" != "$venv" ]]; then
    echo "This node is $arch, so it builds $venv, not $1: submit it to the partition of that venv's architecture" >&2
    exit 2
fi
record="$venv/afabench.uv.lock"

scratch="$(mktemp -d "${TMPDIR:-/tmp}/afabench-venv.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT
echo "host=$(hostname) arch=$arch venv=$checkout/$venv"
# The lock the venv is synced from, recorded only once the sync succeeded,
# so a failed or interrupted build leaves a venv the launcher refuses.
cp uv.lock "$scratch/uv.lock"
rm -f "$record"
# The uv cache may sit on another filesystem than the checkout (Arrhenius:
# /home and /nobackup), where uv cannot hardlink and would warn that it
# copies instead.
UV_PROJECT_ENVIRONMENT="$checkout/$venv" UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed \
    uv sync --frozen --no-dev
cp "$scratch/uv.lock" "$record.partial"
mv "$record.partial" "$record"
"$venv/bin/python" -m snakemake --version
"$venv/bin/python" -c 'import torch; print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")'
