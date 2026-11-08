#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: archive_missing_data_namespace.sh --namespace NAME --archive-dir DIR
       [--root DIR] [--remove-source]

Archives every directory named NAME below the missing-data output root. The
archive is validated before it is published. Source directories are retained
unless --remove-source is explicit.
EOF
    exit 2
}

namespace=""
archive_dir=""
root="extra/output/missing_data"
remove_source=false

while (($#)); do
    case "$1" in
        --namespace) namespace=${2:?}; shift 2 ;;
        --archive-dir) archive_dir=${2:?}; shift 2 ;;
        --root) root=${2:?}; shift 2 ;;
        --remove-source) remove_source=true; shift ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

[[ ${namespace} =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || {
    echo "--namespace must be a simple path component" >&2
    exit 2
}
[[ -n ${archive_dir} && -d ${root} ]] || usage

# GNU tar and coreutils, as on the cluster; Homebrew provides both on macOS.
tar=$(command -v gtar || command -v tar)
sha256=(sha256sum)
command -v sha256sum >/dev/null || sha256=(shasum -a 256)
root=$(cd "${root}" && pwd -P)
[[ ${root} != / ]] || {
    echo "refusing filesystem root as --root" >&2
    exit 2
}
mkdir -p "${archive_dir}"
archive_dir=$(cd "${archive_dir}" && pwd -P)
archive="${archive_dir}/missing-data-${namespace}.tar.zst"
manifest="${archive}.manifest.txt"
[[ ! -e ${archive} && ! -e ${manifest} ]] || {
    echo "archive already exists: ${archive}" >&2
    exit 1
}

roots=$(mktemp "${archive_dir}/.${namespace}.roots.XXXXXX")
temporary=$(mktemp "${archive_dir}/.${namespace}.archive.XXXXXX")
temporary_manifest=$(mktemp "${archive_dir}/.${namespace}.manifest.XXXXXX")
cleanup() {
    rm -f -- "${roots}" "${temporary}" "${temporary_manifest}"
}
trap cleanup EXIT

(
    cd "${root}"
    find . -type d -name "${namespace}" -prune -print0 | sort -z > "${roots}"
)
root_count=$(tr -cd '\0' < "${roots}" | wc -c)
((root_count > 0)) || {
    echo "no namespace directories found: ${namespace}" >&2
    exit 1
}

source_entries=0
while IFS= read -r -d '' source_root; do
    count=$(find "${root}/${source_root#./}" -print0 | tr -cd '\0' | wc -c)
    source_entries=$((source_entries + count))
done < "${roots}"

"${tar}" --create --zstd --file="${temporary}" --directory="${root}" \
    --null --files-from="${roots}"
zstd --test "${temporary}"
archive_entries=$("${tar}" --list --zstd --file="${temporary}" | wc -l)
[[ ${archive_entries} -eq ${source_entries} ]] || {
    echo "entry mismatch: source=${source_entries} archive=${archive_entries}" >&2
    exit 1
}
"${tar}" --compare --zstd --file="${temporary}" --directory="${root}"
checksum=$("${sha256[@]}" "${temporary}" | awk '{print $1}')

cat > "${temporary_manifest}" <<EOF
namespace=${namespace}
source_root=${root}
namespace_roots=${root_count}
entries=${source_entries}
sha256=${checksum}
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_revision=$(git -C "${root}" rev-parse HEAD 2>/dev/null || printf unknown)
EOF
mv "${temporary}" "${archive}"
mv "${temporary_manifest}" "${manifest}"

if [[ ${remove_source} == true ]]; then
    while IFS= read -r -d '' source_root; do
        [[ ${source_root} == ./summary/* ]] && continue
        rm -rf -- "${root}/${source_root#./}"
    done < "${roots}"
    find "${root}" -mindepth 1 -depth -type d -empty -delete
    if find "${root}" -path "${root}/summary" -prune -o \
        -type d -name "${namespace}" -print -quit | grep -q .; then
        echo "namespace remains after removal: ${namespace}" >&2
        exit 1
    fi
fi

printf 'archive=%s entries=%s sha256=%s source_removed=%s\n' \
    "${archive}" "${source_entries}" "${checksum}" "${remove_source}"
