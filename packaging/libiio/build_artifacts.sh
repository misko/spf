#!/usr/bin/env bash
# Build a Debian package and Python wheel from an immutable SPF libiio tag.

set -euo pipefail
umask 0022

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=versions.sh
source "${script_dir}/versions.sh"

series=0.25
output_dir="${PWD}/dist/libiio"
jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
keep_worktree=false

usage() {
    cat <<'EOF'
Usage: packaging/libiio/build_artifacts.sh [options]

Options:
  --series 0.25       Forward-only tandem host line (default: 0.25)
  --output-dir PATH    Artifact directory (default: dist/libiio)
  --jobs N             Parallel build jobs
  --keep-worktree      Preserve the temporary source/build directory
  -h, --help           Show this help

The build host must be Debian 12 on amd64 or arm64. The resulting native .deb
matches the build architecture; the pylibiio wheel is platform-independent.
EOF
}

while (($#)); do
    case "$1" in
    --series) series="${2:?missing value for --series}"; shift 2 ;;
    --output-dir) output_dir="${2:?missing value for --output-dir}"; shift 2 ;;
    --jobs) jobs="${2:?missing value for --jobs}"; shift 2 ;;
    --keep-worktree) keep_worktree=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

spf_libiio_select_version "$series"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || {
    printf 'ERROR: --jobs must be a positive integer: %s\n' "$jobs" >&2
    exit 2
}

for command_name in cmake dpkg dpkg-architecture dpkg-deb git python3 sha256sum; do
    command -v "$command_name" >/dev/null || {
        printf 'ERROR: required command is absent: %s\n' "$command_name" >&2
        exit 1
    }
done
python3 -c 'import setuptools, wheel' >/dev/null 2>&1 || {
    printf 'ERROR: python3-setuptools and python3-wheel are required to build the pylibiio artifact\n' >&2
    exit 1
}

architecture="$(dpkg --print-architecture)"
case "$architecture" in
amd64|arm64) ;;
*) printf 'ERROR: supported package architectures are amd64 and arm64, got %s\n' "$architecture" >&2; exit 1 ;;
esac

multiarch="$(dpkg-architecture -qDEB_HOST_MULTIARCH)"
package_version="${SPF_LIBIIO_EXPECTED_VERSION}+spfmeta${SPF_LIBIIO_METADATA_REVISION}-${SPF_LIBIIO_PACKAGE_REVISION}"
wheel_version="${SPF_LIBIIO_EXPECTED_VERSION}+spfmeta${SPF_LIBIIO_METADATA_REVISION}"

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
worktree="$(mktemp -d "${TMPDIR:-/tmp}/spf-libiio-package-${series}.XXXXXX")"
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
stage_dir="${worktree}/package"
python_build_dir="${worktree}/python-build"

printf 'Cloning misko/libiio tag %s...\n' "$SPF_LIBIIO_SOURCE_REF"
git -c advice.detachedHead=false clone --quiet --depth 1 \
    --branch "$SPF_LIBIIO_SOURCE_REF" https://github.com/misko/libiio.git "$source_dir"
actual_commit="$(git -C "$source_dir" rev-parse HEAD)"
[[ "$actual_commit" == "$SPF_LIBIIO_SOURCE_COMMIT" ]] || {
    printf 'ERROR: source-lock mismatch: tag resolved to %s, expected %s\n' \
        "$actual_commit" "$SPF_LIBIIO_SOURCE_COMMIT" >&2
    exit 1
}

# libiio 0.25 groups the command-line tools under the legacy WITH_TESTS flag.
# Python is deliberately packaged separately so it can be installed in a venv.
cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DCMAKE_INSTALL_LIBDIR="lib/${multiarch}" \
    -DINSTALL_UDEV_RULE=OFF \
    -DPYTHON_BINDINGS=OFF \
    -DHAVE_DNS_SD=OFF \
    -DWITH_DOC=OFF \
    -DWITH_EXAMPLES=OFF \
    -DWITH_IIOD=OFF \
    -DWITH_LOCAL_BACKEND=ON \
    -DWITH_NETWORK_BACKEND=ON \
    -DWITH_SERIAL_BACKEND=OFF \
    -DWITH_TESTS=ON \
    -DWITH_USB_BACKEND=ON
cmake --build "$build_dir" --parallel "$jobs"
DESTDIR="$stage_dir" cmake --install "$build_dir"

mkdir -p "$stage_dir/DEBIAN"
installed_size="$(du -sk "$stage_dir/usr" | awk '{print $1}')"
cat >"$stage_dir/DEBIAN/control" <<EOF
Package: spf-libiio
Version: ${package_version}
Section: libs
Priority: optional
Architecture: ${architecture}
Maintainer: SPF project <https://github.com/misko/spf>
Installed-Size: ${installed_size}
Depends: libc6 (>= 2.36), libusb-1.0-0, libxml2
Provides: libiio0 (= ${SPF_LIBIIO_EXPECTED_VERSION}), libiio-dev (= ${SPF_LIBIIO_EXPECTED_VERSION}), libiio-utils (= ${SPF_LIBIIO_EXPECTED_VERSION})
Conflicts: libiio0, libiio-dev, libiio-utils
Replaces: libiio0, libiio-dev, libiio-utils
Description: SPF libiio with request-driven tandem metadata support
 Hardware-qualified libiio ${SPF_LIBIIO_EXPECTED_VERSION} with versioned tandem
 session requests and sample-aligned gain metadata and events.
X-SPF-Source-Commit: ${SPF_LIBIIO_SOURCE_COMMIT}
EOF
cat >"$stage_dir/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
ldconfig
EOF
cat >"$stage_dir/DEBIAN/postrm" <<'EOF'
#!/bin/sh
set -e
ldconfig
EOF
chmod 0755 "$stage_dir/DEBIAN/postinst" "$stage_dir/DEBIAN/postrm"

deb_name="spf-libiio_${package_version}_${architecture}.deb"
dpkg-deb --root-owner-group --build "$stage_dir" "$output_dir/$deb_name"

# Configure the upstream single-module Python package, then distinguish the
# patched distribution from the unmodified PyPI release with a local version.
cmake -S "$source_dir/bindings/python" -B "$python_build_dir" \
    -DVERSION="$SPF_LIBIIO_EXPECTED_VERSION" \
    -DCMAKE_INSTALL_FULL_LIBDIR="/usr/lib/${multiarch}" \
    -DCMAKE_SHARED_LIBRARY_SUFFIX=.so
sed -i "s/version=\"${SPF_LIBIIO_EXPECTED_VERSION}\"/version=\"${wheel_version}\"/" \
    "$python_build_dir/setup.py"
# Upstream's custom install command refuses to assemble a wheel unless libiio
# is already installed on the build host. The wheel is pure Python and the
# matching staged .deb is intentionally not installed until the test step.
sed -i 's/cross_compiling = ("FALSE" == "TRUE")/cross_compiling = True/' \
    "$python_build_dir/setup.py"
grep -q 'cross_compiling = True' "$python_build_dir/setup.py" || {
    printf 'ERROR: could not disable upstream build-host libiio check\n' >&2
    exit 1
}
(
    cd "$python_build_dir"
    python3 setup.py --quiet bdist_wheel --dist-dir "$output_dir"
)

(
    cd "$output_dir"
    sha256sum "$deb_name" "pylibiio-${wheel_version}-py3-none-any.whl" >SHA256SUMS
)

printf '\nBuilt SPF libiio artifacts in %s:\n' "$output_dir"
printf '  %s\n' "$deb_name" "pylibiio-${wheel_version}-py3-none-any.whl" SHA256SUMS
