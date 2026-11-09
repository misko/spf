"""Public gain selection and explicit, validated metadata session requests."""

from __future__ import annotations

import math
import struct

from spf.direct_radio.tandem_agc import TandemSessionRequestV1

GAIN_MODES = ("manual", "slow_attack", "fast_attack", "tandem")
GAIN_MODE_CODES = {mode: index for index, mode in enumerate(GAIN_MODES)}
LEGACY_METADATA_CAPABILITY = "iio,buffer-metadata-legacy-agc"


def validate_gain_modes(modes, channels=(0, 1), transport="iio") -> tuple[str, str]:
    modes = tuple(modes)
    if len(modes) != 2 or any(mode not in GAIN_MODES for mode in modes):
        raise ValueError(f"RX requires two gain modes from {GAIN_MODES}")
    if "tandem" in modes:
        if modes != ("tandem", "tandem") or tuple(channels) != (0, 1):
            raise ValueError("tandem requires both RX channels and the same mode")
        if transport != "iio":
            raise ValueError("tandem requires the IIO session transport")
    return modes


def tandem_request_for_frame(samples: int) -> TandemSessionRequestV1:
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError("tandem requires a positive integer RX buffer size")
    # Qualified common policy. Do not change AGC response time with frame size.
    request = TandemSessionRequestV1(cooldown_periods=16)
    # ABI 3 retains events across two refill windows.
    request.validate_frame_capacity(samples * 2)
    return request


def legacy_metadata_request(samples: int) -> bytes:
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples per channel must be a positive integer")
    interval = max(1024, (samples + 61) // 62)
    return struct.pack("<4sHHIHH", b"SPFL", 1, 16, interval, 64, 0)


def validate_manual_gains(sdr, modes, gains):
    """Validate against the active band's driver-provided dB range, read-only."""
    if len(gains) != 2:
        raise ValueError("RX gains must contain two dB values")
    for channel, mode in enumerate(modes):
        if mode != "manual":
            continue
        value = float(gains[channel])
        attr = sdr._ctrl.find_channel(f"voltage{channel}", False).attrs[
            "hardwaregain_available"
        ]
        minimum, step, maximum = map(float, attr.value.strip("[]").split())
        if (
            not math.isfinite(value)
            or step <= 0
            or not minimum <= value <= maximum
            or not math.isclose(
                (value - minimum) / step, round((value - minimum) / step), abs_tol=1e-8
            )
        ):
            raise ValueError(
                f"RX{channel + 1} manual gain must be in [{minimum}, {maximum}] dB in {step} dB steps"
            )
