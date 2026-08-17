#!/usr/bin/env python3
"""Focused libiio metadata RF burn-in without invoking pytest.

The fixture must provide at least 30 dB combined physical and commanded TX
attenuation from TX2 to RX1 and RX2. Every fresh session is identified by
hardware serial, mutes both TX channels on exit, retunes RX/TX LO and
bandwidth, and validates a cabled DDS tone plus the frame-associated metadata.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from spf.bench.dual_rx_phase import ToneQualityThresholds, analyze_common_tone
from spf.direct_radio.iio_metadata import IioMetadataRx
from spf.direct_radio.usb_protocol import MetadataFlags
from spf.scripts.mute_pluto_tx import validate_loopback_safety


DEFAULT_LOS_HZ = (
    868_000_000,
    915_000_000,
    1_280_000_000,
    2_412_000_000,
    4_000_000_000,
    5_804_000_000,
)
DEFAULT_BANDWIDTHS_HZ = (800_000, 1_500_000, 3_000_000)
TX_QUADRATURE_CALIBRATION_BANDWIDTH_HZ = 1_500_000
UNSAFE_METADATA_FLAGS = (
    MetadataFlags.DUMMY_GAINS
    | MetadataFlags.GAIN_READ_FAILED
    | MetadataFlags.RSSI_READ_FAILED
    | MetadataFlags.DEVICE_IIO_OVERFLOW
    | MetadataFlags.FPGA_EVENT_OVERFLOW
)


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def _mute(sdr) -> None:
    try:
        sdr.disable_dds()
    finally:
        try:
            sdr.tx_destroy_buffer()
        finally:
            sdr.tx_enabled_channels = []
            sdr.tx_hardwaregain_chan0 = -80
            sdr.tx_hardwaregain_chan1 = -80
            sdr.tx_cyclic_buffer = False


def _tone_analysis(signal, *, args) -> dict:
    return analyze_common_tone(
        signal,
        sample_rate_hz=args.sample_rate_hz,
        expected_tone_offset_hz=args.tone_hz,
        tone_search_width_hz=25_000,
        transient_samples=min(1_024, args.samples // 16),
        phase_segments=8,
        thresholds=ToneQualityThresholds(
            min_tone_snr_db=args.min_tone_snr_db,
            min_tone_dbfs=args.min_tone_dbfs,
            max_tone_dbfs=-1.0,
            min_coherence=args.min_coherence,
            max_within_capture_phase_std_deg=args.max_phase_std_deg,
        ),
    )


def _validate_frame(signal, metadata, *, args) -> dict:
    if signal.shape != (2, args.samples):
        raise RuntimeError(f"unexpected IQ shape {signal.shape}")
    if not metadata.gain_metadata_valid:
        raise RuntimeError("gain metadata is invalid")
    if not metadata.rssi_metadata_valid:
        raise RuntimeError("RSSI metadata is invalid")
    if metadata.flags & UNSAFE_METADATA_FLAGS:
        raise RuntimeError(f"unsafe metadata flags: 0x{int(metadata.flags):x}")
    if not metadata.gain_observations:
        raise RuntimeError("gain observation sequence is empty")
    if tuple(metadata.gain_db_start) != (args.rx_gain_db, args.rx_gain_db):
        raise RuntimeError(f"wrong starting gain {metadata.gain_db_start}")
    if tuple(metadata.gain_db_end) != (args.rx_gain_db, args.rx_gain_db):
        raise RuntimeError(f"wrong ending gain {metadata.gain_db_end}")
    if not np.isfinite(metadata.rssi_db_start).all():
        raise RuntimeError("non-finite starting RSSI")
    if not np.isfinite(metadata.rssi_db_end).all():
        raise RuntimeError("non-finite ending RSSI")

    analysis = _tone_analysis(signal, args=args)
    if not analysis["quality_valid"]:
        raise RuntimeError(f"tone quality failed: {analysis['quality_reasons']}")
    return {
        "buffer_sequence": metadata.buffer_sequence,
        "first_sample_sequence": metadata.first_sample_sequence,
        "gain_db_start": list(metadata.gain_db_start),
        "gain_db_end": list(metadata.gain_db_end),
        "rssi_db_start": list(metadata.rssi_db_start),
        "rssi_db_end": list(metadata.rssi_db_end),
        "gain_observation_count": len(metadata.gain_observations),
        "tone_frequency_hz": analysis["tone_frequency_hz"],
        "tone_frequency_error_hz": analysis["tone_frequency_error_hz"],
        "tone_dbfs": analysis["tone_dbfs"],
        "tone_snr_db": analysis["tone_snr_db"],
        "coherence": analysis["coherence"],
        "within_capture_phase_std_deg": analysis[
            "within_capture_phase_std_deg"
        ],
    }


def _capture_session(uri: str, lo_hz: int, bandwidth_hz: int, *, args) -> dict:
    import adi

    sdr = adi.ad9361(uri=uri)
    receiver = None
    started = time.monotonic()
    try:
        actual_serial = sdr._ctx.attrs.get("hw_serial")
        if actual_serial != args.serial:
            raise RuntimeError(
                f"serial mismatch for {uri}: {actual_serial} != {args.serial}"
            )
        capability = sdr._ctx.attrs.get("iio,buffer-metadata")
        if capability != "1":
            raise RuntimeError(f"metadata capability missing at {uri}: {capability!r}")

        _mute(sdr)
        sdr.rx_destroy_buffer()
        sdr.rx_enabled_channels = [0, 1]
        sdr.sample_rate = args.sample_rate_hz
        sdr.rx_rf_bandwidth = TX_QUADRATURE_CALIBRATION_BANDWIDTH_HZ
        sdr.tx_rf_bandwidth = TX_QUADRATURE_CALIBRATION_BANDWIDTH_HZ
        sdr.rx_lo = lo_hz
        sdr.tx_lo = lo_hz
        sdr.gain_control_mode_chan0 = "manual"
        sdr.gain_control_mode_chan1 = "manual"
        sdr.rx_hardwaregain_chan0 = args.rx_gain_db
        sdr.rx_hardwaregain_chan1 = args.rx_gain_db
        sdr._rxadc.set_kernel_buffers_count(args.kernel_buffers)

        # A forced calibration avoids a known AD9361 state where all TX
        # controls read back correctly but the analog RF path remains silent.
        sdr._ctrl.attrs["calib_mode"].value = "tx_quad"
        # The driver's TX-quadrature procedure can leave TX silent when it is
        # invoked at the 800 kHz test state.  Calibrate at the fixture's known
        # good 1.5 MHz state, then apply and verify the requested final filter.
        sdr.rx_rf_bandwidth = bandwidth_hz
        sdr.tx_rf_bandwidth = bandwidth_hz
        observed = {
            "sample_rate_hz": int(sdr.sample_rate),
            "rx_bandwidth_hz": int(sdr.rx_rf_bandwidth),
            "tx_bandwidth_hz": int(sdr.tx_rf_bandwidth),
            "rx_lo_hz": int(sdr.rx_lo),
            "tx_lo_hz": int(sdr.tx_lo),
            "kernel_buffers": int(sdr._rxadc.kernel_buffers_count),
        }
        expected = {
            "sample_rate_hz": args.sample_rate_hz,
            "rx_bandwidth_hz": bandwidth_hz,
            "tx_bandwidth_hz": bandwidth_hz,
            "rx_lo_hz": lo_hz,
            "tx_lo_hz": lo_hz,
            "kernel_buffers": args.kernel_buffers,
        }
        exact_keys = (
            "sample_rate_hz",
            "rx_bandwidth_hz",
            "tx_bandwidth_hz",
            "kernel_buffers",
        )
        exact_mismatch = any(observed[key] != expected[key] for key in exact_keys)
        lo_mismatch = any(
            abs(observed[key] - expected[key]) >= 10
            for key in ("rx_lo_hz", "tx_lo_hz")
        )
        if exact_mismatch or lo_mismatch:
            raise RuntimeError(f"RF state readback mismatch: {observed} != {expected}")
        sdr.tx_hardwaregain_chan0 = -80
        sdr.tx_hardwaregain_chan1 = args.tx_gain_db
        sdr.dds_single_tone(args.tone_hz, args.dds_scale, channel=1)
        time.sleep(args.settle_seconds)

        # Pluto+ shares the RX DMA with other paths.  Prime it once after the
        # retune/DDS arm; without this transition the first TX tone can remain
        # silent even though every AD9361 control reads back correctly.
        sdr.rx_buffer_size = args.samples
        priming_signal = np.asarray(sdr.rx())
        sdr.rx_destroy_buffer()
        if priming_signal.shape != (2, args.samples):
            raise RuntimeError(f"unexpected priming IQ shape {priming_signal.shape}")
        prime_analysis = _tone_analysis(priming_signal, args=args)
        tx_arm_attempt = 1
        while (
            not prime_analysis["quality_valid"]
            and tx_arm_attempt < args.max_tx_arm_attempts
        ):
            tx_arm_attempt += 1
            _mute(sdr)
            sdr.rx_rf_bandwidth = TX_QUADRATURE_CALIBRATION_BANDWIDTH_HZ
            sdr.tx_rf_bandwidth = TX_QUADRATURE_CALIBRATION_BANDWIDTH_HZ
            sdr._ctrl.attrs["calib_mode"].value = "tx_quad"
            sdr.rx_rf_bandwidth = bandwidth_hz
            sdr.tx_rf_bandwidth = bandwidth_hz
            sdr.tx_hardwaregain_chan0 = -80
            sdr.tx_hardwaregain_chan1 = args.tx_gain_db
            sdr.dds_single_tone(args.tone_hz, args.dds_scale, channel=1)
            time.sleep(args.settle_seconds)
            priming_signal = np.asarray(sdr.rx())
            sdr.rx_destroy_buffer()
            prime_analysis = _tone_analysis(priming_signal, args=args)
        if not prime_analysis["quality_valid"]:
            raise RuntimeError(
                "TX remained silent after "
                f"{tx_arm_attempt} arms: {prime_analysis['quality_reasons']}"
            )

        receiver = IioMetadataRx(
            sdr,
            sample_rate_hz=args.sample_rate_hz,
            samples_per_channel=args.samples,
        )
        receiver.open()
        frames = []
        for _ in range(args.frames):
            signal, metadata, capture_time = receiver.capture()
            frame = _validate_frame(signal, metadata, args=args)
            frame["capture_time_ns"] = capture_time[
                "sample_time_realtime_start_ns"
            ]
            frame["capture_time_uncertainty_ns"] = capture_time[
                "sample_time_uncertainty_ns"
            ]
            frames.append(frame)
        indices = [frame["buffer_sequence"] for frame in frames]
        if any(right <= left for left, right in zip(indices, indices[1:])):
            raise RuntimeError(f"non-increasing capture indices: {indices}")
        return {
            "uri": uri,
            "serial": actual_serial,
            "requested": {
                "lo_hz": lo_hz,
                "bandwidth_hz": bandwidth_hz,
            },
            "observed": observed,
            "elapsed_seconds": time.monotonic() - started,
            "tx_arm_attempt_count": tx_arm_attempt,
            "prime_tone_snr_db": prime_analysis["tone_snr_db"],
            "frames": frames,
        }
    finally:
        if receiver is not None:
            receiver.close()
        _mute(sdr)
        sdr.rx_destroy_buffer()


def _summarize(report: dict) -> dict:
    frames = [
        frame
        for session in report["sessions"]
        for frame in session["frames"]
    ]
    transports = {
        name: sum(session["transport"] == name for session in report["sessions"])
        for name in ("ip", "usb")
    }
    return {
        "session_count": len(report["sessions"]),
        "frame_count": len(frames),
        "recovered_attempt_failure_count": len(
            report["recovered_attempt_failures"]
        ),
        "maximum_session_attempts": max(
            session["attempt_count"] for session in report["sessions"]
        ),
        "maximum_tx_arm_attempts": max(
            session["tx_arm_attempt_count"] for session in report["sessions"]
        ),
        "tx_rearm_session_count": sum(
            session["tx_arm_attempt_count"] > 1 for session in report["sessions"]
        ),
        "transport_sessions": transports,
        "minimum_tone_snr_db": min(
            value for frame in frames for value in frame["tone_snr_db"]
        ),
        "minimum_tone_dbfs": min(
            value for frame in frames for value in frame["tone_dbfs"]
        ),
        "minimum_coherence": min(frame["coherence"] for frame in frames),
        "maximum_phase_std_deg": max(
            frame["within_capture_phase_std_deg"] for frame in frames
        ),
        "maximum_frequency_error_hz": max(
            abs(frame["tone_frequency_error_hz"]) for frame in frames
        ),
        "maximum_capture_time_uncertainty_ms": max(
            frame["capture_time_uncertainty_ns"] for frame in frames
        )
        / 1_000_000,
    }


def burn(args: argparse.Namespace) -> None:
    validate_loopback_safety(
        physical_attenuation_db=args.attenuation_db,
        strongest_tx_gain_db=args.tx_gain_db,
    )
    report = {
        "status": "running",
        "started_unix_ns": time.time_ns(),
        "serial": args.serial,
        "ip_uri": args.ip_uri,
        "usb_uri": args.usb_uri,
        "attenuation_db": args.attenuation_db,
        "sample_rate_hz": args.sample_rate_hz,
        "samples_per_channel": args.samples,
        "frames_per_session": args.frames,
        "kernel_buffers": args.kernel_buffers,
        "epochs": args.epochs,
        "los_hz": list(args.los_hz),
        "bandwidths_hz": list(args.bandwidths_hz),
        "sessions": [],
        "recovered_attempt_failures": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    combinations = [
        (lo_hz, bandwidth_hz)
        for lo_hz in args.los_hz
        for bandwidth_hz in args.bandwidths_hz
    ]
    try:
        for epoch in range(args.epochs):
            schedule = combinations.copy()
            random.Random(args.seed + epoch).shuffle(schedule)
            for cell, (lo_hz, bandwidth_hz) in enumerate(schedule):
                transport = "ip" if (epoch + cell) % 2 == 0 else "usb"
                uri = args.ip_uri if transport == "ip" else args.usb_uri
                session = None
                for attempt in range(1, args.max_session_attempts + 1):
                    try:
                        session = _capture_session(
                            uri, lo_hz, bandwidth_hz, args=args
                        )
                        break
                    except Exception as error:
                        failure = {
                            "epoch": epoch,
                            "cell": cell,
                            "transport": transport,
                            "uri": uri,
                            "lo_hz": lo_hz,
                            "bandwidth_hz": bandwidth_hz,
                            "attempt": attempt,
                            "error": f"{type(error).__name__}: {error}",
                        }
                        report["recovered_attempt_failures"].append(failure)
                        args.report.write_text(json.dumps(report, indent=2) + "\n")
                        print(
                            f"RETRY serial={args.serial[-8:]} epoch={epoch + 1} "
                            f"cell={cell + 1} transport={transport} "
                            f"attempt={attempt}/{args.max_session_attempts} "
                            f"error={failure['error']}",
                            flush=True,
                        )
                        if attempt == args.max_session_attempts:
                            raise
                if session is None:
                    raise AssertionError("session retry loop returned no result")
                session["attempt_count"] = attempt
                session.update({"epoch": epoch, "cell": cell, "transport": transport})
                report["sessions"].append(session)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"PASS serial={args.serial[-8:]} epoch={epoch + 1}/{args.epochs} "
                    f"cell={cell + 1}/{len(schedule)} transport={transport} "
                    f"lo_MHz={lo_hz / 1e6:g} bw_MHz={bandwidth_hz / 1e6:g} "
                    f"min_snr={min(value for frame in session['frames'] for value in frame['tone_snr_db']):.2f}",
                    flush=True,
                )
        report["status"] = (
            "pass_with_retries"
            if report["recovered_attempt_failures"]
            else "pass"
        )
        report["finished_unix_ns"] = time.time_ns()
        report["summary"] = _summarize(report)
    except BaseException as error:
        report["status"] = "fail"
        report["finished_unix_ns"] = time.time_ns()
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY " + json.dumps(report["summary"], sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--ip-uri", required=True)
    parser.add_argument("--usb-uri", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--attenuation-db", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--samples", type=int, default=65_536)
    parser.add_argument("--kernel-buffers", type=int, default=2)
    parser.add_argument("--max-session-attempts", type=int, default=3)
    parser.add_argument("--max-tx-arm-attempts", type=int, default=3)
    parser.add_argument("--sample-rate-hz", type=int, default=3_000_000)
    parser.add_argument("--tone-hz", type=int, default=100_000)
    parser.add_argument("--rx-gain-db", type=int, default=26)
    parser.add_argument("--tx-gain-db", type=float, default=-10.0)
    parser.add_argument("--dds-scale", type=float, default=0.25)
    parser.add_argument("--settle-seconds", type=float, default=0.25)
    parser.add_argument("--los-hz", type=_comma_separated_ints, default=DEFAULT_LOS_HZ)
    parser.add_argument(
        "--bandwidths-hz",
        type=_comma_separated_ints,
        default=DEFAULT_BANDWIDTHS_HZ,
    )
    parser.add_argument("--min-tone-snr-db", type=float, default=6.0)
    parser.add_argument("--min-tone-dbfs", type=float, default=-75.0)
    parser.add_argument("--min-coherence", type=float, default=0.90)
    parser.add_argument("--max-phase-std-deg", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    if (
        args.epochs <= 0
        or args.frames <= 0
        or args.max_session_attempts <= 0
        or args.max_tx_arm_attempts <= 0
        or args.samples < 16_384
    ):
        parser.error("epochs/frames must be positive and samples must be >= 16384")
    return args


if __name__ == "__main__":
    burn(parse_args())
