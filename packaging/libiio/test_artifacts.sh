#!/usr/bin/env bash
# Inspect built artifacts and, optionally, verify an installed runtime.

set -euo pipefail

bundle=
python_bin=
runtime=false

usage() {
    cat <<'EOF'
Usage: packaging/libiio/test_artifacts.sh --bundle DIR [--runtime --python PATH]

Without --runtime, validates checksums, package metadata/content, architecture,
and the wheel API. With --runtime, also checks the installed library and tools.
EOF
}

while (($#)); do
    case "$1" in
    --bundle) bundle="${2:?missing value for --bundle}"; shift 2 ;;
    --python) python_bin="${2:?missing value for --python}"; shift 2 ;;
    --runtime) runtime=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

[[ -n "$bundle" && -d "$bundle" ]] || { usage >&2; exit 2; }
bundle="$(cd "$bundle" && pwd)"
architecture="$(dpkg --print-architecture)"
deb="$(find "$bundle" -maxdepth 1 -type f -name "spf-libiio_*_${architecture}.deb" -print -quit)"
wheel="$(find "$bundle" -maxdepth 1 -type f -name 'pylibiio-*-py3-none-any.whl' -print -quit)"
[[ -n "$deb" ]] || { printf 'ERROR: no %s .deb in %s\n' "$architecture" "$bundle" >&2; exit 1; }
[[ -n "$wheel" ]] || { printf 'ERROR: no pylibiio wheel in %s\n' "$bundle" >&2; exit 1; }

(cd "$bundle" && sha256sum --check SHA256SUMS)
[[ "$(dpkg-deb --field "$deb" Package)" == spf-libiio ]]
[[ "$(dpkg-deb --field "$deb" Architecture)" == "$architecture" ]]
dpkg-deb --field "$deb" Version | grep -Eq '^0\.(25|26)\+spfmeta[0-9]+-[0-9]+$'
package_contents="$(dpkg-deb --contents "$deb")"
grep -Eq '/usr/lib/.*/libiio\.so\.0\.(25|26)$' <<<"$package_contents"
grep -Eq '/usr/bin/iio_info$' <<<"$package_contents"
python3 - "$wheel" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    source = archive.read("iio.py").decode()
assert "class MetadataBuffer" in source
PY

if [[ "$runtime" == true ]]; then
    [[ -n "$python_bin" ]] || { printf 'ERROR: --runtime requires --python\n' >&2; exit 2; }
    "$python_bin" - <<'PY'
import iio

assert iio.version[:2] in ((0, 25), (0, 26)), iio.version
assert hasattr(iio, "MetadataBuffer"), "patched binding lacks MetadataBuffer"
print(f"PASS Python binding: {iio.__file__} version={iio.version}")
PY
    iio_info --version
    iio_info -S
    library_path="$(ldconfig -p | awk '/libiio\.so\.0 / {print $NF; exit}')"
    [[ -n "$library_path" ]] || { printf 'ERROR: ldconfig cannot find libiio.so.0\n' >&2; exit 1; }
    if ldd "$library_path" | grep -q 'not found'; then
        printf 'ERROR: installed libiio has an unresolved shared-library dependency\n' >&2
        exit 1
    fi
fi

printf 'PASS SPF libiio artifact checks (%s)\n' "$architecture"
