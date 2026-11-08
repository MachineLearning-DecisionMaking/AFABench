#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: submit_missing_data.sh [--local DEVICE] [--dry-run] [PROFILE...]

Runs every missing-data profile of the paper and the rebuttal, or only the
named ones, in dependency order. By default each profile is one Slurm job;
a job waits for the profiles whose namespace it reads. With --local, the
profiles run one after another on this machine.

Slurm settings (environment): ACCOUNT, PARTITION, TIME, CPUS, MEM, GPUS,
RUN_ARGS (arguments to run_missing_data.sh).
EOF
    exit 2
}

# profile: profiles it depends on. Profiles of a group read the base
# namespace of that group, and the extra predictor ablation reuses the
# training views of the other imputers.
DEPENDENCIES=(
    "missing_data:"
)
for size in 500 1000 4000 16000; do
    DEPENDENCIES+=("rebuttal_size_sweep_n${size}:")
done
for size in 500 2000 8000 32000; do
    DEPENDENCIES+=("rebuttal_size_sweep_real_n${size}:")
done
for group in core_group induced_nonuniform induced_real; do
    base="rebuttal_imputers_${group}"
    DEPENDENCIES+=("${base}:")
    for experiment in sklearn_imputers constant_fill pvae_variants mice_label \
        multi_draw mar mnar_self view_classifier; do
        DEPENDENCIES+=("rebuttal_${experiment}_${group}:${base}")
    done
    DEPENDENCIES+=(
        "rebuttal_view_classifier_extra_${group}:${base} rebuttal_sklearn_imputers_${group} rebuttal_mice_label_${group} rebuttal_multi_draw_${group}"
    )
done

local_device=""
dry_run=false
selected=()
while (($#)); do
    case "$1" in
        --local) local_device=${2:?}; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage ;;
        -*) echo "unknown argument: $1" >&2; usage ;;
        *) selected+=("$1"); shift ;;
    esac
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"

for entry in "${DEPENDENCIES[@]}"; do
    profile=${entry%%:*}
    read -r -a needs <<< "${entry#*:}"
    if ((${#selected[@]})) && [[ " ${selected[*]} " != *" ${profile} "* ]]; then
        continue
    fi
    [[ -d extra/workflow/profiles/config/${profile} ]] || {
        echo "unknown profile: ${profile}" >&2
        exit 2
    }

    if [[ -n ${local_device} ]]; then
        command=(scripts/workflow/run_missing_data.sh --profile "${profile}"
            --device "${local_device}" -- --rerun-triggers mtime
            --rerun-incomplete)
        echo "${command[*]}"
        [[ ${dry_run} == true ]] || "${command[@]}"
        continue
    fi

    # Profiles share the repository, so each Snakemake skips the directory
    # lock; dependencies keep two runs from writing the same namespace.
    run="scripts/workflow/run_missing_data.sh --profile ${profile} ${RUN_ARGS:---device cuda --cores 72 --mem-mb 103680 --gpu-workers 34 --mps} -- --nolock --keep-going --rerun-triggers mtime --rerun-incomplete"
    after=()
    for need in ${needs[@]+"${needs[@]}"}; do
        id="job_${need}"
        [[ -n ${!id:-} ]] && after+=("afterok:${!id}")
    done
    sbatch_args=(
        --parsable
        --account="${ACCOUNT:-naiss2026-4-1278-gpu}"
        --partition="${PARTITION:-gpu}"
        --cpus-per-task="${CPUS:-72}"
        --mem="${MEM:-115200M}"
        --gpus="${GPUS:-1}"
        --time="${TIME:-1-00:00:00}"
        --chdir="${repo_root}"
        --output=extra/output/missing_data/controllers/%x-%j.out
        --job-name="${profile}"
    )
    ((${#after[@]})) && sbatch_args+=(--dependency="$(IFS=,; echo "${after[*]}")")
    if [[ ${dry_run} == true ]]; then
        echo "sbatch ${sbatch_args[*]} --wrap \"${run}\""
        printf -v "job_${profile}" '%s' "<${profile}>"
    else
        printf -v "job_${profile}" '%s' \
            "$(sbatch "${sbatch_args[@]}" --wrap "${run}")"
        id="job_${profile}"
        echo "${!id} ${profile}"
    fi
done
