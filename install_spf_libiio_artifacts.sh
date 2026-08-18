#!/usr/bin/env bash
# Install a prebuilt SPF libiio bundle into the OS and an SPF Python venv.

set -euo pipefail

bundle=
python_bin="${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}"
python_bin="${python_bin:-python3}"

usage() {
    cat <<'EOF'
Usage: ./install_spf_libiio_artifacts.sh --bundle DIR [--python PATH]

DIR must contain the matching spf-libiio Debian package, the pylibiio wheel,
and SHA256SUMS from one CI/release build. Defaults to the active venv's Python.
EOF
}

while (($#)); do
    case "$1" in
    --bundle) bundle="${2:?missing value for --bundle}"; shift 2 ;;
    --python) python_bin="${2:?missing value for --python}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

[[ -n "$bundle" && -d "$bundle" ]] || { usage >&2; exit 2; }
bundle="$(cd "$bundle" && pwd)"
architecture="$(dpkg --print-architecture)"
deb="$(find "$bundle" -maxdepth 1 -type f -name "spf-libiio_*_${architecture}.deb" -print -quit)"
wheel="$(find "$bundle" -maxdepth 1 -type f -name 'pylibiio-*-py3-none-any.whl' -print -quit)"
[[ -n "$deb" ]] || { printf 'ERROR: no %s package in %s\n' "$architecture" "$bundle" >&2; exit 1; }
[[ -n "$wheel" ]] || { printf 'ERROR: no pylibiio wheel in %s\n' "$bundle" >&2; exit 1; }
"$python_bin" -m pip --version >/dev/null
(cd "$bundle" && sha256sum --check SHA256SUMS)

apt_command=(apt-get install -y "$deb")
if [[ "$(id -u)" == 0 ]]; then
    "${apt_command[@]}"
else
    command -v sudo >/dev/null || { printf 'ERROR: installing the .deb requires root or sudo\n' >&2; exit 1; }
    sudo "${apt_command[@]}"
fi
"$python_bin" -m pip install --quiet --force-reinstall --no-deps "$wheel"

"$python_bin" - <<'PY'
import inspect
import iio

assert iio.version[:2] == (0, 25), iio.version
assert hasattr(iio, "MetadataBuffer"), "patched binding lacks MetadataBuffer"
assert "request" in inspect.signature(iio.MetadataBuffer).parameters, (
    "patched binding lacks request-driven tandem sessions"
)
print(f"PASS Python binding: {iio.__file__} version={iio.version}")
PY
iio_info --version
