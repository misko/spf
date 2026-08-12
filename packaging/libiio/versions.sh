#!/usr/bin/env bash
# Immutable, hardware-qualified SPF libiio source locks.
# shellcheck disable=SC2034  # Values are consumed by scripts that source this file.

SPF_LIBIIO_PACKAGE_REVISION=1

spf_libiio_select_version() {
    case "${1:-}" in
    0.25)
        SPF_LIBIIO_SOURCE_REF=spf-frame-metadata-source/v0.25-final-v3
        SPF_LIBIIO_SOURCE_COMMIT=c26258bfa33098c2b215e19cf85d448e89499b1a
        SPF_LIBIIO_EXPECTED_VERSION=0.25
        SPF_LIBIIO_EXPECTED_GIT=c26258b
        SPF_LIBIIO_METADATA_REVISION=3
        ;;
    0.26)
        SPF_LIBIIO_SOURCE_REF=spf-frame-metadata-source/v0.26-final-v3
        SPF_LIBIIO_SOURCE_COMMIT=d5695c3eaa9cec99cc6f7b2c91565555044b907a
        SPF_LIBIIO_EXPECTED_VERSION=0.26
        SPF_LIBIIO_EXPECTED_GIT=d5695c3
        SPF_LIBIIO_METADATA_REVISION=3
        ;;
    *)
        printf 'ERROR: libiio series must be 0.25 or 0.26, got %s\n' "${1:-}" >&2
        return 2
        ;;
    esac
}
