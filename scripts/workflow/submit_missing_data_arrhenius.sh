#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: submit_missing_data_arrhenius.sh --profile NAME --cores N --mem-mb N \
       --time HH:MM:SS [--gpus N] [--gpu-workers N] [--device DEVICE] \
       [--mps] [--archive-dir DIR] [--remove-archived-source] \
       [--job-name NAME] [--afterok JOBID] [--afterany JOBID] \
       [--test-only] [-- SNAKEMAKE_ARGS...]

Submits one allocation after checking the live association run-minute cap and
asking Slurm for a non-submitting start-time forecast.
EOF
    exit 2
}

profile=""
cores=""
mem_mb=""
walltime=""
gpus=0
gpu_workers=""
mps=false
archive_dir=""
remove_archived_source=false
device=""
job_name=""
afterok=""
afterany=""
test_only=false
snakemake_args=()

while (($#)); do
    case "$1" in
        --profile) profile=${2:?}; shift 2 ;;
        --cores) cores=${2:?}; shift 2 ;;
        --mem-mb) mem_mb=${2:?}; shift 2 ;;
        --time) walltime=${2:?}; shift 2 ;;
        --gpus) gpus=${2:?}; shift 2 ;;
        --gpu-workers) gpu_workers=${2:?}; shift 2 ;;
        --mps) mps=true; shift ;;
        --archive-dir) archive_dir=${2:?}; shift 2 ;;
        --remove-archived-source) remove_archived_source=true; shift ;;
        --device) device=${2:?}; shift 2 ;;
        --job-name) job_name=${2:?}; shift 2 ;;
        --afterok) afterok=${2:?}; shift 2 ;;
        --afterany) afterany=${2:?}; shift 2 ;;
        --test-only) test_only=true; shift ;;
        --) shift; snakemake_args=("$@"); break ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

[[ -n ${profile} && -n ${cores} && -n ${mem_mb} && -n ${walltime} ]] || usage
[[ ${cores} =~ ^[1-9][0-9]*$ && ${mem_mb} =~ ^[1-9][0-9]*$ ]] || usage
[[ ${gpus} =~ ^[0-9]+$ ]] || usage
[[ -z ${gpu_workers} || ${gpu_workers} =~ ^[0-9]+$ ]] || usage
[[ -z ${afterok} || ${afterok} =~ ^[0-9]+$ ]] || usage
[[ -z ${afterany} || ${afterany} =~ ^[0-9]+$ ]] || usage
if ((gpus > 1)); then
    echo "only one physical GPU per allocation is supported" >&2
    exit 2
fi

partition=cpu
account=naiss2026-4-1278-cpu
resource=cpu
units=${cores}
if ((gpus > 0)); then
    partition=gpu
    account=naiss2026-4-1278-gpu
    resource=gres/gpu
    units=${gpus}
fi
if [[ -z ${device} ]]; then
    device=cpu
    ((gpus > 0)) && device=cuda
fi
gpu_workers=${gpu_workers:-${gpus}}
if ((gpus == 0)) && [[ ${device} == cuda* ]]; then
    echo "CUDA device requested without a GPU allocation" >&2
    exit 2
fi
if ((gpus == 0 && gpu_workers > 0)); then
    echo "GPU workers requested without a GPU allocation" >&2
    exit 2
fi
if [[ ${device} == cuda* && ${gpu_workers} -eq 0 ]]; then
    echo "CUDA execution requires at least one GPU worker" >&2
    exit 2
fi
if [[ ${mps} == true ]]; then
    [[ ${device} == cuda* && ${gpu_workers} -gt 1 ]] || {
        echo "--mps requires CUDA and at least two GPU workers" >&2
        exit 2
    }
fi
if [[ ${remove_archived_source} == true && -z ${archive_dir} ]]; then
    echo "--remove-archived-source requires --archive-dir" >&2
    exit 2
fi
if ((gpus > 0)) && [[ ${device} != cuda* ]]; then
    echo "GPU allocation requested for a non-CUDA device" >&2
    exit 2
fi
job_name=${job_name:-${profile#missing_data_}}

wall_minutes() {
    local value=$1 days=0 hours minutes seconds
    if [[ ${value} == *-* ]]; then
        days=${value%%-*}
        value=${value#*-}
    fi
    IFS=: read -r hours minutes seconds <<<"${value}"
    [[ ${days} =~ ^[0-9]+$ && ${hours} =~ ^[0-9]+$ && ${minutes} =~ ^[0-9]+$ && ${seconds} =~ ^[0-9]+$ ]] || return 1
    echo $((10#${days} * 1440 + 10#${hours} * 60 + 10#${minutes} + (10#${seconds} > 0)))
}

minutes=$(wall_minutes "${walltime}") || {
    echo "--time must be HH:MM:SS or D-HH:MM:SS" >&2
    exit 2
}
requested=$((units * minutes))

association_limits=$(sacctmgr -nP show assoc where account="${account}" \
    format=User,GrpTRESRunMins%100)
limit_spec=$(awk -F '|' '$1 == "" && $2 != "" {print $2; exit}' \
    <<<"${association_limits}")
limit=""
IFS=, read -ra limits <<<"${limit_spec}"
for item in "${limits[@]}"; do
    [[ ${item%%=*} == "${resource}" ]] && limit=${item#*=}
done
[[ ${limit} =~ ^[0-9]+$ ]] || {
    echo "could not resolve ${resource} run-minute cap for ${account}" >&2
    exit 1
}
if ((requested > limit)); then
    echo "refusing impossible request: ${requested} ${resource}-minutes exceeds association cap ${limit}" >&2
    exit 1
fi
echo "preflight: ${requested}/${limit} ${resource}-minutes (${account})"

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"
if [[ -n $(git status --short) ]]; then
    echo "refusing to submit from a dirty Git checkout" >&2
    exit 1
fi
mkdir -p extra/output/missing_data/controllers
export AFABENCH_REPO_ROOT=${repo_root}
usable_mem_mb=$((mem_mb * 9 / 10))
runner=(
    scripts/workflow/run_missing_data.sh
    --profile "${profile}"
    --device "${device}"
    --cores "${cores}"
    --mem-mb "${usable_mem_mb}"
    --gpu-workers "${gpu_workers}"
)
[[ ${mps} == true ]] && runner+=(--mps)
[[ -n ${archive_dir} ]] && runner+=(--archive-dir "${archive_dir}")
[[ ${remove_archived_source} == true ]] && runner+=(--remove-archived-source)
runner+=(-- "${snakemake_args[@]}")
sbatch_args=(
    --account="${account}"
    --partition="${partition}"
    --job-name="${job_name}"
    --cpus-per-task="${cores}"
    --mem="${mem_mb}M"
    --time="${walltime}"
    --chdir="${repo_root}"
    --output="extra/output/missing_data/controllers/%x-%j.out"
)
((gpus > 0)) && sbatch_args+=(--gpus="${gpus}")
[[ -n ${afterok} ]] && sbatch_args+=(--dependency="afterok:${afterok}")
[[ -n ${afterany} ]] && sbatch_args+=(--dependency="afterany:${afterany}")

sbatch --test-only "${sbatch_args[@]}" "${runner[@]}"
[[ ${test_only} == true ]] && exit 0
sbatch --parsable "${sbatch_args[@]}" "${runner[@]}"
