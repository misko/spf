#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

variant="hex_7"
rotation="0"
spacing="23.75"
quality="production"
output=""

usage() {
    printf '%s\n' \
        "Usage: $(basename "$0") [options]" \
        "" \
        "Render the parametric antenna mount as a binary STL." \
        "" \
        "Options:" \
        "  --variant reference_5|hex_7  Layout to render (default: hex_7)" \
        "  --rotation 0|30              Outer-six rotation for hex_7 (default: 0)" \
        "  --spacing MM                 Centre-to-outer radius (default: 23.75)" \
        "  --quality draft|production   Tessellation quality (default: production)" \
        "  -o, --output PATH            Output STL path" \
        "  -h, --help                   Show this help"
}

require_value() {
    if (( $# < 2 )); then
        printf 'Missing value for %s\n' "$1" >&2
        usage >&2
        exit 2
    fi
}

while (( $# > 0 )); do
    case "$1" in
        --variant)
            require_value "$@"
            variant="$2"
            shift 2
            ;;
        --rotation)
            require_value "$@"
            rotation="$2"
            shift 2
            ;;
        --spacing)
            require_value "$@"
            spacing="$2"
            shift 2
            ;;
        --quality)
            require_value "$@"
            quality="$2"
            shift 2
            ;;
        -o|--output)
            require_value "$@"
            output="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$variant" in
    reference_5|hex_7) ;;
    *) printf 'Invalid variant: %s\n' "$variant" >&2; exit 2 ;;
esac

case "$rotation" in
    0|30) ;;
    *) printf 'Rotation must be 0 or 30 degrees, got: %s\n' "$rotation" >&2; exit 2 ;;
esac

case "$quality" in
    draft|production) ;;
    *) printf 'Quality must be draft or production, got: %s\n' "$quality" >&2; exit 2 ;;
esac

if [[ ! "$spacing" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf 'Spacing must be a positive number in millimetres, got: %s\n' "$spacing" >&2
    exit 2
fi

if ! command -v openscad >/dev/null 2>&1; then
    printf 'OpenSCAD is required but was not found on PATH.\n' >&2
    exit 127
fi

if [[ -z "$output" ]]; then
    spacing_tag="${spacing//./p}"
    output="${script_dir}/generated/antenna_mount_${variant}_s${spacing_tag}"
    if [[ "$variant" == "hex_7" ]]; then
        output+="_rot${rotation}"
    fi
    output+=".stl"
fi

mkdir -p -- "$(dirname -- "$output")"

openscad \
    --hardwarnings \
    --export-format binstl \
    -o "$output" \
    -D "variant=\"${variant}\"" \
    -D "antenna_spacing=${spacing}" \
    -D "outer_rotation=${rotation}" \
    -D "render_quality=\"${quality}\"" \
    "${script_dir}/antenna_mount.scad"

printf 'Wrote %s\n' "$output"
