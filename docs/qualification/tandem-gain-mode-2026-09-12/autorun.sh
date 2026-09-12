#!/bin/sh
# SPF gain-mode provider; hash-bound to the qualified v0.49 release.
(
    set -eu
    provider=/mnt/jffs2/spf-gain-mode/iiod
    echo '7b1abdfffcf1c9ebda365f4a81976d829d9d2c4c64b93301a98020579a3432af  /mnt/jffs2/spf-gain-mode/iiod' | sha256sum -c - || exit 1
    current=$(sha256sum /usr/sbin/iiod)
    current=${current%% *}
    [ "$current" != '7b1abdfffcf1c9ebda365f4a81976d829d9d2c4c64b93301a98020579a3432af' ] || exit 0
    if [ "$current" != '495c2508747ffe1375fcd947ee5825dde2c497072c53b9f5390fffa2fe5b318c' ]; then
        logger -t spf-gain-mode 'release executable changed; leaving it untouched'
        exit 1
    fi
    cp "$provider" /usr/sbin/iiod.spf-new
    chmod 755 /usr/sbin/iiod.spf-new
    mv /usr/sbin/iiod.spf-new /usr/sbin/iiod
    if [ -r /var/run/iiod-child.pid ]; then
        kill "$(cat /var/run/iiod-child.pid)"
    fi
)
