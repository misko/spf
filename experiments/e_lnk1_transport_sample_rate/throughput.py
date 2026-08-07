"""E-LNK1 -- sustained RX throughput vs requested sample rate, per transport.

The question: at each sample rate, can the transport actually carry the samples?

Arms (see experiment_readme.md for why four and not two):

    A  direct-usb   USB 2.0   SPF bulk protocol   (production path)
    B  iio-usb      USB 2.0   libiio USB backend
    C  iio-rndis    USB 2.0   libiio over IP, via the usb0 gadget  ("not Ethernet")
    D  iio-eth      RJ45      libiio over IP, via real eth0

Metric is sustained achieved MS/s over a fixed wall-clock window of back-to-back
buffer reads, and the ratio achieved/requested. A ratio below 1 means the
transport could not keep up and the radio dropped samples.

Configuration is held identical across arms -- same LO, same gains, same buffer
size, same channel count -- so the only thing varying is how bytes reach the host.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# 2 RX channels x complex int16 = 8 bytes per sample instant.
BYTES_PER_SAMPLE_INSTANT = 8

RATES_HZ = [
    521_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 15_000_000,
    20_000_000, 30_000_000, 40_000_000, 50_000_000, 61_440_000,
]
LO_HZ = 5_766_000_000
GAIN_DB = 26
BUFFER_SAMPLES = 65_536          # SPF production value
KERNEL_BUFFERS = 4


def configure(sdr, rate_hz: int) -> int:
    sdr.rx_destroy_buffer()
    sdr.rx_enabled_channels = [0, 1]
    sdr.sample_rate = int(rate_hz)
    achieved = int(sdr.sample_rate)
    sdr.rx_rf_bandwidth = int(min(achieved * 0.8, 56_000_000))
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = GAIN_DB
    sdr.rx_hardwaregain_chan1 = GAIN_DB
    sdr.rx_lo = int(LO_HZ)
    sdr.rx_buffer_size = BUFFER_SAMPLES
    sdr._rxadc.set_kernel_buffers_count(KERNEL_BUFFERS)
    return achieved


def measure_iio(uri: str, rate_hz: int, seconds: float) -> dict:
    import adi

    sdr = adi.ad9361(uri=uri)
    try:
        achieved_rate = configure(sdr, rate_hz)
        sdr.rx()  # prime: first buffer carries setup cost
        latencies = []
        samples = 0
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            t0 = time.perf_counter()
            data = sdr.rx()
            latencies.append(time.perf_counter() - t0)
            samples += len(data[0])
        elapsed = time.perf_counter() - start
    finally:
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass
        del sdr
    return _summarize(rate_hz, achieved_rate, samples, elapsed, latencies)


def measure_direct_usb(serial: str, rate_hz: int, seconds: float) -> dict:
    import adi

    from spf.sdrpluto.direct_usb_receiver import PlutoDirectUsbReceiver

    # Sample rate is an IIO-side setting even when data comes over the bulk path.
    sdr = adi.ad9361(uri=f"usb:{_usb_uri_suffix(serial)}")
    try:
        achieved_rate = configure(sdr, rate_hz)
    finally:
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass
        del sdr

    with PlutoDirectUsbReceiver(serial=serial, protocol_version=2) as rx:
        # The gadget caps a finite request at `max_finite_frames`, so a sustained
        # window is several back-to-back requests. Each carries its own START/STOP;
        # that cost is real and belongs in the measurement.
        per_request = int(getattr(rx.capabilities, "max_finite_frames", 16))
        latencies = []
        samples = 0
        start = time.perf_counter()
        last = start
        while time.perf_counter() - start < seconds:
            for _frame in rx.stream_frames(
                samples_per_channel=BUFFER_SAMPLES,
                frame_count=per_request,
                queue_depth=2,
            ):
                now = time.perf_counter()
                latencies.append(now - last)
                last = now
                samples += BUFFER_SAMPLES
            if time.perf_counter() - start >= seconds:
                break
        elapsed = time.perf_counter() - start
    return _summarize(rate_hz, achieved_rate, samples, elapsed, latencies)


def _usb_uri_suffix(serial: str) -> str:
    import iio

    for uri, desc in iio.scan_contexts().items():
        if uri.startswith("usb:") and f"serial={serial}" in desc:
            return uri.split("usb:", 1)[1]
    raise RuntimeError(f"no USB-IIO context for {serial}")


def _summarize(requested, achieved_rate, samples, elapsed, latencies) -> dict:
    lat = np.asarray(latencies, dtype=np.float64) if latencies else np.zeros(1)
    throughput = samples / elapsed if elapsed > 0 else 0.0
    return {
        "requested_rate_hz": int(requested),
        "radio_rate_hz": int(achieved_rate),
        "sustained_rate_hz": float(throughput),
        "ratio_of_radio_rate": float(throughput / achieved_rate) if achieved_rate else 0.0,
        "mbytes_per_s": float(throughput * BYTES_PER_SAMPLE_INSTANT / 1e6),
        "buffers": int(len(latencies)),
        "seconds": float(elapsed),
        "latency_ms_p50": float(np.percentile(lat, 50) * 1e3),
        "latency_ms_p95": float(np.percentile(lat, 95) * 1e3),
        "latency_ms_max": float(lat.max() * 1e3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--eth-uri", default=None, help="ip:<addr> for the real eth0 arm")
    ap.add_argument("--rndis-uri", default=None, help="ip:<addr> for the usb0 gadget arm")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--rates", type=int, nargs="*", default=RATES_HZ)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    arms = {"direct-usb": None, "iio-usb": None}
    if args.rndis_uri:
        arms["iio-rndis"] = args.rndis_uri
    if args.eth_uri:
        arms["iio-eth"] = args.eth_uri
    if args.arms:
        arms = {k: v for k, v in arms.items() if k in args.arms}

    rows = []
    for rep in range(args.repetitions):
        for rate in args.rates:
            for arm, uri in arms.items():   # arms interleaved within each rate
                try:
                    if arm == "direct-usb":
                        row = measure_direct_usb(args.serial, rate, args.seconds)
                    elif arm == "iio-usb":
                        row = measure_iio(
                            f"usb:{_usb_uri_suffix(args.serial)}", rate, args.seconds
                        )
                    else:
                        row = measure_iio(uri, rate, args.seconds)
                    row.update(arm=arm, repetition=rep, status="ok")
                except Exception as error:
                    row = {
                        "arm": arm, "repetition": rep, "status": "error",
                        "requested_rate_hz": int(rate), "error": repr(error)[:200],
                    }
                rows.append(row)
                print(
                    f"rep{rep} {arm:11s} {rate/1e6:6.3f} MS/s requested -> "
                    + (
                        f"{row['sustained_rate_hz']/1e6:7.3f} MS/s sustained "
                        f"({row['ratio_of_radio_rate']*100:5.1f}% of radio rate, "
                        f"{row['mbytes_per_s']:6.1f} MB/s)"
                        if row["status"] == "ok" else f"ERROR {row['error'][:70]}"
                    ),
                    flush=True,
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
