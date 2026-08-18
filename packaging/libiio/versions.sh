#!/usr/bin/env bash
# Immutable, hardware-qualified SPF libiio source locks.
# shellcheck disable=SC2034  # Values are consumed by scripts that source this file.

SPF_LIBIIO_PACKAGE_REVISION=1

spf_libiio_select_version() {
    case "${1:-}" in
    0.25)
        SPF_LIBIIO_SOURCE_REF=tandem-agc-v2-source/libiio-v8
        SPF_LIBIIO_SOURCE_COMMIT=9d7878dd53316e3879c3f154aeb06b27632fda4d
        SPF_LIBIIO_EXPECTED_VERSION=0.25
        SPF_LIBIIO_EXPECTED_GIT=9d7878d
        SPF_LIBIIO_METADATA_REVISION=4
        ;;
    *)
        printf 'ERROR: forward-only tandem libiio series must be 0.25, got %s\n' "${1:-}" >&2
        return 2
        ;;
    esac
}
