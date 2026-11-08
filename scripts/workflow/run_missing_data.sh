#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: run_missing_data.sh --profile NAME --device DEVICE [options] [-- SNAKEMAKE_ARGS...]

options:
  --cores N        concurrent CPU slots (default: allocation CPUs or 4)
  --mem-mb N       schedulable memory in MiB (default: 90% of allocation or 22000)
  --job-mem-mb N   per-rule memory reservation in MiB (default: 3000)
  --gpu-workers N  concurrent CUDA processes (default: 1 for CUDA, otherwise 0)
  --mps            share one CUDA device through a job-local NVIDIA MPS server
  --archive-dir D  archive the completed namespace into D
  --remove-archived-source
                   remove live namespace files after exact archive validation

The same command runs locally or inside one Slurm allocation. CUDA rules each
consume one `gpu` resource. On a large-memory accelerator this may exceed the
physical GPU count so several small, independent processes share one device.
EOF
    exit 2
}

profile=""
device=""
cores=""
mem_mb=""
job_mem_mb=""
gpu_workers=""
mps=false
archive_dir=""
remove_archived_source=false
snakemake_args=()

while (($#)); do
    case "$1" in
        --profile) profile=${2:?}; shift 2 ;;
        --device) device=${2:?}; shift 2 ;;
        --cores) cores=${2:?}; shift 2 ;;
        --mem-mb) mem_mb=${2:?}; shift 2 ;;
        --job-mem-mb) job_mem_mb=${2:?}; shift 2 ;;
        --gpu-workers|--gpu-slots) gpu_workers=${2:?}; shift 2 ;;
        --mps) mps=true; shift ;;
        --archive-dir) archive_dir=${2:?}; shift 2 ;;
        --remove-archived-source) remove_archived_source=true; shift ;;
        --) shift; snakemake_args=("$@"); break ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

[[ -n ${profile} && -n ${device} ]] || usage
[[ ${cores:-${SLURM_CPUS_PER_TASK:-4}} =~ ^[1-9][0-9]*$ ]] || {
    echo "--cores must be a positive integer" >&2
    exit 2
}
cores=${cores:-${SLURM_CPUS_PER_TASK:-4}}

if [[ -z ${mem_mb} ]]; then
    if [[ -n ${SLURM_MEM_PER_NODE:-} ]]; then
        mem_mb=$((SLURM_MEM_PER_NODE * 9 / 10))
    else
        mem_mb=22000
    fi
fi
[[ ${mem_mb} =~ ^[1-9][0-9]*$ ]] || {
    echo "--mem-mb must be a positive integer" >&2
    exit 2
}

job_mem_mb=${job_mem_mb:-3000}
[[ ${job_mem_mb} =~ ^[1-9][0-9]*$ ]] || {
    echo "--job-mem-mb must be a positive integer" >&2
    exit 2
}

if [[ -z ${gpu_workers} ]]; then
    gpu_workers=0
    [[ ${device} == cuda* ]] && gpu_workers=1
fi
[[ ${gpu_workers} =~ ^[0-9]+$ ]] || {
    echo "--gpu-workers must be a non-negative integer" >&2
    exit 2
}
if [[ ${device} == cuda* && ${gpu_workers} -eq 0 ]]; then
    echo "CUDA execution requires at least one GPU worker" >&2
    exit 2
fi
if [[ ${mps} == true ]]; then
    [[ ${device} == cuda* && ${gpu_workers} -gt 1 ]] || {
        echo "--mps requires CUDA and at least two GPU workers" >&2
        exit 2
    }
    [[ -n ${SLURM_JOB_ID:-} ]] || {
        echo "--mps is supported only inside a Slurm allocation" >&2
        exit 2
    }
fi
if [[ ${remove_archived_source} == true && -z ${archive_dir} ]]; then
    echo "--remove-archived-source requires --archive-dir" >&2
    exit 2
fi

repo_root=${AFABENCH_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
profile_dir="extra/workflow/profiles/config/${profile}"
cd "${repo_root}"
[[ -d ${profile_dir} ]] || {
    echo "unknown profile: ${profile}" >&2
    exit 2
}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export AFABENCH_DEVICE=${device}
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
uv_run=(uv run)

if [[ -n ${SLURM_JOB_ID:-} ]]; then
    # Arrhenius keeps one environment per architecture on project storage.
    # Sourcing inside the allocation selects aarch64 on GH200 and x86_64 on
    # CPU nodes.
    if [[ -r ${HOME}/.bashrc.d/afabench.sh ]]; then
        # shellcheck source=/dev/null
        source "${HOME}/.bashrc.d/afabench.sh"
    fi
    : "${SNIC_TMP:?Arrhenius did not provide job-local scratch}"
    export AFABENCH_SCRATCH_ROOT=${SNIC_TMP}
    uv_run+=(--no-sync)
    # A fresh directory per shell command prevents concurrent Hydra and
    # Lightning processes from sharing logs or checkpoints.
    export AFABENCH_SHELL_PREFIX='rule_tmp=$(mktemp -d "$AFABENCH_SCRATCH_ROOT/afabench-rule.XXXXXX"); export SNIC_TMP="$rule_tmp"; export TMPDIR="$rule_tmp"; '
else
    export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/afabench-uv-cache}
fi

run_id=${SLURM_JOB_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)-$$}
dry_run=false
for argument in "${snakemake_args[@]}"; do
    [[ ${argument} == -n || ${argument} == --dry-run ]] && dry_run=true
done

telemetry_pid=""
mps_started=false
stop_runtime_services() {
    if [[ -n ${telemetry_pid} ]]; then
        kill "${telemetry_pid}" 2>/dev/null || true
        wait "${telemetry_pid}" 2>/dev/null || true
        telemetry_pid=""
    fi
    if [[ ${mps_started} == true ]]; then
        echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
        mps_started=false
    fi
}
trap stop_runtime_services EXIT

if [[ ${mps} == true && ${dry_run} == false ]]; then
    export CUDA_MPS_PIPE_DIRECTORY="${SNIC_TMP}/afabench-mps-pipe"
    export CUDA_MPS_LOG_DIRECTORY="${SNIC_TMP}/afabench-mps-log"
    mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
    nvidia-cuda-mps-control -d
    mps_started=true
    echo "started job-local NVIDIA MPS server"
fi

if [[ ${device} == cuda* && ${dry_run} == false ]]; then
    "${uv_run[@]}" python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in this environment"; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'
fi

if [[ ${dry_run} == false ]]; then
    manifest_args=(
        --profile "${profile}" --run-id "${run_id}" --device "${device}"
        --cores "${cores}" --mem-mb "${mem_mb}"
        --job-mem-mb "${job_mem_mb}"
        --gpu-workers "${gpu_workers}"
    )
    [[ ${mps} == true ]] && manifest_args+=(--mps)
    [[ -n ${archive_dir} ]] && manifest_args+=(--archive-dir "${archive_dir}")
    [[ ${remove_archived_source} == true ]] && \
        manifest_args+=(--remove-archived-source)
    manifest_message=$("${uv_run[@]}" python scripts/workflow/write_run_manifest.py \
        "${manifest_args[@]}" --snakemake-args "${snakemake_args[@]}")
    echo "${manifest_message}"
    export AFABENCH_RUN_MANIFEST=${manifest_message##* }
    namespace=$(basename "$(dirname "${AFABENCH_RUN_MANIFEST}")")
fi

if [[ ${dry_run} == false && ${device} == cuda* ]]; then
    telemetry_dir="extra/output/missing_data/gpu_telemetry/${namespace}"
    telemetry_path="${telemetry_dir}/${run_id}.csv"
    mkdir -p "${telemetry_dir}"
    echo "timestamp,gpu_index,utilization_percent,memory_used_mb,memory_total_mb,power_draw_w,temperature_c" > "${telemetry_path}"
    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits --loop=5 >> "${telemetry_path}" &
    telemetry_pid=$!
    echo "recording GPU telemetry at ${telemetry_path}"
fi

set +e
# Reservations come from measured peak RSS on the GH200 timing gate: 3,187 MiB
# over every rule and 3,098 MiB for evaluation. A larger reservation only caps
# concurrency, which is the scarce resource once the allocation holds 72 cores.
"${uv_run[@]}" snakemake \
    --profile "${profile_dir}" all \
    --apptainer-prefix .snakemake/apptainer \
    --cores "${cores}" \
    --resources "mem_mb=${mem_mb}" "gpu=${gpu_workers}" \
    --default-resources "mem_mb=${job_mem_mb}" gpu=0 \
    --set-resources eval_missing_data_method:mem_mb=4000 \
    --rerun-incomplete \
    --printshellcmds \
    "${snakemake_args[@]}"
status=$?
set -e
stop_runtime_services
find .snakemake/metadata -type f -mmin +60 -delete 2>/dev/null || true
if [[ ${status} -eq 0 && ${dry_run} == false && -n ${archive_dir} ]]; then
    archive_args=(
        --namespace "${namespace}"
        --archive-dir "${archive_dir}"
    )
    [[ ${remove_archived_source} == true ]] && archive_args+=(--remove-source)
    scripts/workflow/archive_missing_data_namespace.sh "${archive_args[@]}"
fi
exit "${status}"
