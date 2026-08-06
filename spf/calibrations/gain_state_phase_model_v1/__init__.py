"""L26 gain-state phase model for the AD9361 dual-RX pair.

See ``README.md`` for the model description, physical backing, measured
performance, limitations, and the queued follow-up experiments.
"""

from .gain_tables import (
    BANDS,
    GainTables,
    HardwareState,
    band_for_lo,
    default_tables,
)
from .model import (
    GainStatePhaseModel,
    Prediction,
    UnsupportedGainState,
    wrap,
)

__all__ = [
    "BANDS",
    "GainTables",
    "GainStatePhaseModel",
    "HardwareState",
    "Prediction",
    "UnsupportedGainState",
    "band_for_lo",
    "default_tables",
    "wrap",
]
