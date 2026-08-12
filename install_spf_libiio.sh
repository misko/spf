#!/usr/bin/env bash
# Install one hardware-qualified SPF libiio host line from an immutable tag.

set -euo pipefail
umask 0022

series=0.25
python_bin="${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}"
python_bin="${python_bin:-python3}"
prefix=/usr/local
jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
keep_worktree=false

usage() {
    cat <<'EOF'
Usage: ./install_spf_libiio.sh [options]

Options:
  --series 0.25|0.26  Patched host line (default: 0.25)
  --python PATH       Python/venv receiving the binding (default: active venv)
  --prefix PATH       C library/tools prefix (default: /usr/local)
  --jobs N            Parallel build jobs
  --keep-worktree     Preserve the temporary source/build directory
  -h, --help          Show this help

Install OS build dependencies first; see docs/libiio_frame_metadata_install.md.
Run this script after `python -m pip install -e .`, so a later pip operation
cannot replace the patched binding with the unmodified PyPI copy.
EOF
}

while (($#)); do
    case "$1" in
    --series) series="${2:?missing value for --series}"; shift 2 ;;
    --python) python_bin="${2:?missing value for --python}"; shift 2 ;;
    --prefix) prefix="${2:?missing value for --prefix}"; shift 2 ;;
    --jobs) jobs="${2:?missing value for --jobs}"; shift 2 ;;
    --keep-worktree) keep_worktree=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

case "$series" in
0.25)
    source_ref=spf-frame-metadata-source/v0.25-final-v3
    source_commit=c26258bfa33098c2b215e19cf85d448e89499b1a
    expected_version=0.25
    expected_git=c26258b
    ;;
0.26)
    source_ref=spf-frame-metadata-source/v0.26-final-v3
    source_commit=d5695c3eaa9cec99cc6f7b2c91565555044b907a
    expected_version=0.26
    expected_git=d5695c3
    ;;
*)
    printf 'ERROR: --series must be 0.25 or 0.26, got %s\n' "$series" >&2
    exit 2
    ;;
esac

[[ "$prefix" == /* ]] || {
    printf 'ERROR: --prefix must be absolute: %s\n' "$prefix" >&2
    exit 2
}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || {
    printf 'ERROR: --jobs must be a positive integer: %s\n' "$jobs" >&2
    exit 2
}
command -v git >/dev/null || { printf 'ERROR: git is required\n' >&2; exit 1; }
command -v cmake >/dev/null || { printf 'ERROR: cmake is required\n' >&2; exit 1; }
"$python_bin" -m pip --version >/dev/null || {
    printf 'ERROR: Python with pip is required: %s\n' "$python_bin" >&2
    exit 1
}

worktree="$(mktemp -d "${TMPDIR:-/tmp}/spf-libiio-${series}.XXXXXX")"
cleanup() {
    if [[ "$keep_worktree" == true ]]; then
        printf 'Preserved build worktree: %s\n' "$worktree"
    else
        rm -rf -- "$worktree"
    fi
}
trap cleanup EXIT

source_dir="${worktree}/libiio"
build_dir="${worktree}/build"
printf 'Cloning misko/libiio tag %s...\n' "$source_ref"
git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$source_ref" \
    https://github.com/misko/libiio.git "$source_dir"
actual_commit="$(git -C "$source_dir" rev-parse HEAD)"
[[ "$actual_commit" == "$source_commit" ]] || {
    printf 'ERROR: source-lock mismatch: tag resolved to %s, expected %s\n' \
        "$actual_commit" "$source_commit" >&2
    exit 1
}

# libiio 0.25 groups iio_info/iio_attr under the legacy WITH_TESTS switch.
cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DINSTALL_UDEV_RULE=OFF \
    -DPYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE="$python_bin" \
    -DHAVE_DNS_SD=OFF \
    -DWITH_DOC=OFF \
    -DWITH_EXAMPLES=OFF \
    -DWITH_LOCAL_BACKEND=ON \
    -DWITH_NETWORK_BACKEND=ON \
    -DWITH_SERIAL_BACKEND=OFF \
    -DWITH_TESTS=ON \
    -DWITH_USB_BACKEND=ON
cmake --build "$build_dir" --parallel "$jobs"

install_cmd=(cmake --install "$build_dir")
if [[ -w "$prefix" || (! -e "$prefix" && -w "$(dirname "$prefix")") ]]; then
    "${install_cmd[@]}"
elif [[ "$(id -u)" == 0 ]]; then
    "${install_cmd[@]}"
else
    command -v sudo >/dev/null || {
        printf 'ERROR: %s requires root and sudo is unavailable\n' "$prefix" >&2
        exit 1
    }
    sudo "${install_cmd[@]}"
fi

library_dirs=("$prefix/lib" "$prefix/lib64")
library_path="$(IFS=:; printf '%s' "${library_dirs[*]}")"
export LD_LIBRARY_PATH="${library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ "$prefix" == /usr || "$prefix" == /usr/local ]]; then
    if [[ "$(id -u)" == 0 ]]; then
        ldconfig
    else
        sudo ldconfig
    fi
fi

# CMake installs the binding under its prefix. Install the same generated
# binding into the requested venv as a real pylibiio distribution as well.
"$python_bin" -m pip install --quiet --force-reinstall --no-deps \
    "$build_dir/bindings/python"

SPF_EXPECTED_IIO_VERSION="$expected_version" \
SPF_EXPECTED_IIO_GIT="$expected_git" \
"$python_bin" - <<'PY'
import os
import iio

expected_version = tuple(int(part) for part in os.environ["SPF_EXPECTED_IIO_VERSION"].split("."))
actual = tuple(iio.version)
assert actual[:2] == expected_version, (actual, expected_version)
assert str(actual[2]).startswith(os.environ["SPF_EXPECTED_IIO_GIT"]), actual
assert hasattr(iio, "MetadataBuffer"), "patched binding lacks MetadataBuffer"
print(f"PASS Python binding: {iio.__file__} version={actual}")
PY

tool="$prefix/bin/iio_info"
[[ -x "$tool" ]] || { printf 'ERROR: installed iio_info is absent: %s\n' "$tool" >&2; exit 1; }
"$tool" --version

printf '\nInstalled SPF libiio %s from %s (%s).\n' \
    "$series" "$source_ref" "$source_commit"
if [[ "$prefix" != /usr && "$prefix" != /usr/local ]]; then
    printf 'For new shells, export LD_LIBRARY_PATH=%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' \
        "$library_path"
fi
