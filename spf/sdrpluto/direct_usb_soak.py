"""Low-storage soak of the public PPlus direct-USB compatibility interface."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from spf.sdrpluto.direct_usb_protocol import RadioMetadataV2
from spf.sdrpluto.sdr_controller import PPlus, ReceiverConfig


def _rss_kib() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("VmRSS is absent from /proc/self/status")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--interval-seconds", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=524288)
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.interval_seconds < 0:
        parser.error("duration must be positive and interval must be nonnegative")

    uri = args.uri
    if not uri.startswith("pluto://"):
        uri = f"pluto://{uri}"
    config = ReceiverConfig(
        lo=5_766_000_000,
        rf_bandwidth=3_000_000,
        sample_rate=30_000_000,
        intermediate=100_000,
        uri=uri,
        buffer_size=args.samples,
        gains=[-3, -3],
        gain_control_modes=["slow_attack", "slow_attack"],
        enabled_channels=[0, 1],
        rx_spacing=0.05075,
        rx_buffers=4,
        filter_fir_en=1,
        rx_transport="direct_usb",
        direct_usb_protocol_version=2,
        direct_usb_serial=args.serial,
    )
    radio = PPlus(uri=uri, rx_config=config)
    frames = 0
    started = time.monotonic()
    first_rss = _rss_kib()
    maximum_rss = first_rss
    gain_min = np.full(2, np.inf)
    gain_max = np.full(2, -np.inf)
    rssi_min = np.full(2, np.inf)
    rssi_max = np.full(2, -np.inf)
    gain_duration_max = 0
    rssi_duration_max = 0
    endpoint_changed = np.zeros(2, dtype=np.uint64)
    try:
        radio.setup_rx_config()
        started = time.monotonic()
        deadline = started + args.duration_seconds
        next_frame = started
        while time.monotonic() < deadline:
            signal = radio.rx()
            gains = radio.gains()
            rssis = radio.rssis()
            metadata = radio._last_direct_metadata
            if not isinstance(metadata, RadioMetadataV2):
                raise RuntimeError("PPlus cache is not protocol v2 metadata")
            if not metadata.gain_metadata_valid or not metadata.rssi_metadata_valid:
                raise RuntimeError("invalid gain/RSSI metadata")
            if signal.shape != (2, args.samples) or signal.dtype != np.complex64:
                raise RuntimeError(
                    f"unexpected IQ shape/type: {signal.shape} {signal.dtype}"
                )
            if not np.isfinite(signal).all():
                raise RuntimeError("non-finite IQ")
            if not np.any(signal[0]) or not np.any(signal[1]):
                raise RuntimeError("all-zero IQ channel")
            if not np.isfinite(gains).all() or not np.isfinite(rssis).all():
                raise RuntimeError("non-finite compatibility metadata")
            gain_min = np.minimum(gain_min, gains)
            gain_max = np.maximum(gain_max, gains)
            rssi_min = np.minimum(rssi_min, rssis)
            rssi_max = np.maximum(rssi_max, rssis)
            gain_duration_max = max(
                gain_duration_max,
                metadata.gain_start_read_duration_ns,
                metadata.gain_end_read_duration_ns,
            )
            rssi_duration_max = max(
                rssi_duration_max,
                metadata.rssi_start_read_duration_ns,
                metadata.rssi_end_read_duration_ns,
            )
            endpoint_changed += np.logical_not(metadata.gain_endpoints_equal).astype(
                np.uint64
            )
            frames += 1
            maximum_rss = max(maximum_rss, _rss_kib())
            if frames % 100 == 0:
                print(
                    f"frames={frames} elapsed={time.monotonic() - started:.1f}s "
                    f"rss_kib={_rss_kib()}",
                    file=sys.stderr,
                    flush=True,
                )
            next_frame += args.interval_seconds
            remaining = next_frame - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
    finally:
        radio.close_rx()

    elapsed = time.monotonic() - started
    final_rss = _rss_kib()
    print(
        json.dumps(
            {
                "status": "pass",
                "frames": frames,
                "elapsed_seconds": elapsed,
                "frames_per_second": frames / elapsed,
                "samples_per_channel": args.samples,
                "gain_min_db": gain_min.tolist(),
                "gain_max_db": gain_max.tolist(),
                "rssi_min_db": rssi_min.tolist(),
                "rssi_max_db": rssi_max.tolist(),
                "gain_read_duration_max_ns": gain_duration_max,
                "rssi_read_duration_max_ns": rssi_duration_max,
                "endpoint_changed_frames": endpoint_changed.tolist(),
                "rss_initial_kib": first_rss,
                "rss_final_kib": final_rss,
                "rss_maximum_kib": maximum_rss,
                "rss_growth_kib": final_rss - first_rss,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
