#!/usr/bin/env bash
# Immutable, hardware-qualified SPF libiio source locks.
# shellcheck disable=SC2034  # Values are consumed by scripts that source this file.

SPF_LIBIIO_PACKAGE_REVISION=1

spf_libiio_select_version() {
    case "${1:-}" in
    0.25)
        SPF_LIBIIO_SOURCE_REF=iq-direct-async-v4-source/libiio-v1
        SPF_LIBIIO_SOURCE_COMMIT=5cb2389719d46d12463daa0371d1fda19eb25fa7
        SPF_LIBIIO_EXPECTED_VERSION=0.25
        SPF_LIBIIO_EXPECTED_GIT=5cb2389
        SPF_LIBIIO_METADATA_REVISION=6
        ;;
    *)
        printf 'ERROR: forward-only tandem libiio series must be 0.25, got %s\n' "${1:-}" >&2
        return 2
        ;;
    esac
}
