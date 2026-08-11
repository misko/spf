"""Compatibility shim: this module moved to :mod:`spf.direct_radio.sample_clock`.

Re-exported here because the direct-radio link was extracted into its own package so
consumers outside this repository can depend on it without the DOA research stack.
Import from ``spf.direct_radio.sample_clock`` in new code; the names are the same objects, so
``isinstance`` across the two paths still holds.
"""

from spf.direct_radio.sample_clock import *  # noqa: F401,F403
