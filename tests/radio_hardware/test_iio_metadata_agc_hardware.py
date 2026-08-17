"""Large-frame libiio gain-history checks with a bounded cabled RF stimulus.

This module transmits on TX2.  It therefore requires ``--radio-hardware``,
``--radio-tx-loopback``, and an explicitly declared attenuation of at least
30 dB.  TX2 must be split to RX1 and RX2 on each selected radio.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time

import numpy as np
import pytest

from spf.bench.dual_rx_phase import ToneQualityThresholds, analyze_common_tone
from spf.direct_radio.iio_metadata import IioMetadataRx
from spf.direct_radio.usb_protocol import (
    GainObservationFlags,
    MetadataFlags,
    RadioMetadataV3,
)
from spf.scripts.mute_pluto_tx import mute_attached_plutos, validate_loopback_safety
from spf.scripts.resolve_pluto_ip import neighbor_candidates, resolve_pluto_ip


pytestmark = [pytest.mark.radio_hardware, pytest.mark.radio_tx_loopback]

SAMPLES_PER_CHANNEL = 524_288
SAMPLE_RATE_HZ = 3_000_000
RF_BANDWIDTH_HZ = 1_500_000
LO_HZ = 915_000_000
TONE_HZ = 100_000
STRONG_TX_GAIN_DB = -30.0
WEAK_TX_GAIN_DB = -60.0
MAX_CHANNEL_GAIN_DELTA_DB = 4.0
MAX_ENDPOINT_SEQUENCE_DELTA_DB = 4.0
MIN_WITHIN_FRAME_GAIN_RANGE_DB = 2.0
MIN_GAIN_DB_FOR_BOUNDED_STIMULUS = 0.0
MAX_CAPTURE_FRAMES = 8
UNSAFE_METADATA_FLAGS = (
    MetadataFlags.DUMMY_GAINS
    | MetadataFlags.GAIN_READ_FAILED
    | MetadataFlags.RSSI_READ_FAILED
    | MetadataFlags.DEVICE_IIO_OVERFLOW
    | MetadataFlags.FPGA_EVENT_OVERFLOW
    | MetadataFlags.GAIN_OBSERVATION_OVERFLOW
)


@pytest.fixture(autouse=True)
def tx_safety_guard(attached_plutos):
    serials = [radio.serial for radio in attached_plutos]
    mute_attached_plutos(serials=serials, expected_count=len(serials))
    try:
        yield
    finally:
        mute_attached_plutos(serials=serials, expected_count=len(serials))


def _usb_uri(serial: str) -> str:
    import iio

    matches = [
        uri
        for uri, description in iio.scan_contexts().items()
        if uri.startswith("usb:") and f"serial={serial}" in description
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one USB-IIO URI for {serial}, found {matches}")
    return matches[0]


def _lan_host(serial: str, interface: str, radio_lan_hosts: dict[str, str]) -> str:
    # IIO discovery and the neighbor table are deliberately resolved by the
    # immutable serial.  DHCP addresses are never associated by radio order.
    candidates = (
        (radio_lan_hosts[serial],)
        if radio_lan_hosts
        else neighbor_candidates(interface)
    )
    return resolve_pluto_ip(serial, candidates)


def _mute_sdr(sdr) -> None:
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


def _arm_verified_tone(sdr) -> dict:
    thresholds = ToneQualityThresholds(
        min_tone_snr_db=6.0,
        min_tone_dbfs=-75.0,
        max_tone_dbfs=-3.0,
        max_clipping_fraction=0.0,
        min_coherence=0.98,
        max_within_capture_phase_std_deg=5.0,
    )
    analysis = None
    for arm_attempt in range(1, 4):
        _mute_sdr(sdr)
        sdr._ctrl.attrs["calib_mode"].value = "tx_quad"
        sdr.tx_hardwaregain_chan0 = -80
        sdr.tx_hardwaregain_chan1 = STRONG_TX_GAIN_DB
        sdr.dds_single_tone(TONE_HZ, 0.25, channel=1)
        time.sleep(0.25)
        signal = np.asarray(sdr.rx())
        sdr.rx_destroy_buffer()
        analysis = analyze_common_tone(
            signal,
            sample_rate_hz=SAMPLE_RATE_HZ,
            expected_tone_offset_hz=TONE_HZ,
            tone_search_width_hz=25_000,
            transient_samples=1_024,
            phase_segments=8,
            thresholds=thresholds,
        )
        if analysis["quality_valid"]:
            analysis["arm_attempt_count"] = arm_attempt
            return analysis
    raise AssertionError(json.dumps(analysis, sort_keys=True))


def _remote_gain_pattern(host: str, serial: str) -> subprocess.Popen:
    if shutil.which("sshpass") is None:
        raise RuntimeError("sshpass is required for the radio-local AGC stimulus")
    command = f"""
set -eu
serial_path=/sys/kernel/config/usb_gadget/composite_gadget/strings/0x409/serialnumber
test "$(cat "$serial_path")" = {serial}
gain_path=/sys/bus/iio/devices/iio:device0/out_voltage1_hardwaregain
trap 'echo -80.000000 > "$gain_path"' EXIT HUP INT TERM
iteration=0
while test "$iteration" -lt 60; do
    echo {STRONG_TX_GAIN_DB:.6f} > "$gain_path"
    usleep 70000
    echo {WEAK_TX_GAIN_DB:.6f} > "$gain_path"
    usleep 70000
    iteration=$((iteration + 1))
done
"""
    return subprocess.Popen(
        [
            "sshpass",
            "-p",
            "analog",
            "ssh",
            "-o",
            "BatchMode=no",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "CheckHostIP=no",
            f"root@{host}",
            command,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_remote_pattern(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _clipping_fraction(signal: np.ndarray) -> np.ndarray:
    return np.mean(
        (np.abs(signal.real) >= 2_047) | (np.abs(signal.imag) >= 2_047),
        axis=1,
    )


def _compare_gain_history(signal: np.ndarray, metadata: RadioMetadataV3) -> dict:
    assert signal.shape == (2, SAMPLES_PER_CHANNEL)
    assert np.isfinite(signal).all()
    np.testing.assert_array_equal(_clipping_fraction(signal), [0.0, 0.0])

    assert metadata.samples_per_channel == SAMPLES_PER_CHANNEL
    assert metadata.gain_metadata_valid
    assert metadata.rssi_metadata_valid
    assert not metadata.flags & UNSAFE_METADATA_FLAGS
    assert metadata.gain_observations

    frame_start = metadata.first_sample_sequence
    frame_end = frame_start + metadata.samples_per_channel
    observations = metadata.gain_observations
    previous_before = None
    for observation in observations:
        required = (
            GainObservationFlags.VALID
            | GainObservationFlags.SAMPLE_INTERVAL_VALID
        )
        assert observation.flags & required == required
        assert observation.sample_sequence_before < frame_end
        assert observation.sample_sequence_after >= frame_start
        assert observation.sample_sequence_after >= observation.sample_sequence_before
        assert (
            observation.sample_sequence_after - observation.sample_sequence_before
            <= metadata.gain_observation_interval_samples
        )
        if previous_before is not None:
            assert observation.sample_sequence_before >= previous_before
        previous_before = observation.sample_sequence_before

    # A 524288-sample frame normally has 15-17 observations at the negotiated
    # 32768-sample interval.  Leave two periods of scheduling margin while
    # still rejecting a single observation spanning most of the frame.
    minimum_observations = max(
        3,
        metadata.samples_per_channel
        // metadata.gain_observation_interval_samples
        - 2,
    )
    assert len(observations) >= minimum_observations

    gains = np.asarray(
        [[item.rx1_gain_db, item.rx2_gain_db] for item in observations],
        dtype=np.float64,
    )
    start_gain = np.asarray(metadata.gain_db_start, dtype=np.float64)
    end_gain = np.asarray(metadata.gain_db_end, dtype=np.float64)
    assert np.isfinite(gains).all()
    assert np.isfinite(start_gain).all()
    assert np.isfinite(end_gain).all()

    # Both inputs receive the same split tone.  Their independent AGC loops
    # need not choose identical table entries, but large divergence is not a
    # meaningful result for this bounded, non-clipping stimulus.
    assert np.max(np.abs(gains[:, 0] - gains[:, 1])) <= MAX_CHANNEL_GAIN_DELTA_DB
    assert abs(start_gain[0] - start_gain[1]) <= MAX_CHANNEL_GAIN_DELTA_DB
    assert abs(end_gain[0] - end_gain[1]) <= MAX_CHANNEL_GAIN_DELTA_DB
    assert np.min(gains) >= MIN_GAIN_DB_FOR_BOUNDED_STIMULUS

    # Endpoint snapshots are independent reads, while the history is periodic.
    # Compare each endpoint to the nearest two sequence samples so a real AGC
    # transition at the frame boundary is not mislabeled as metadata failure.
    first_window = gains[: min(2, len(gains))]
    last_window = gains[-min(2, len(gains)) :]
    start_delta = np.min(np.abs(first_window - start_gain), axis=0)
    end_delta = np.min(np.abs(last_window - end_gain), axis=0)
    assert np.max(start_delta) <= MAX_ENDPOINT_SEQUENCE_DELTA_DB
    assert np.max(end_delta) <= MAX_ENDPOINT_SEQUENCE_DELTA_DB

    return {
        "buffer_sequence": metadata.buffer_sequence,
        "first_sample_sequence": frame_start,
        "gain_start": start_gain.tolist(),
        "gain_end": end_gain.tolist(),
        "gain_sequence": gains.tolist(),
        "gain_range_db": np.ptp(gains, axis=0).tolist(),
        "maximum_channel_delta_db": float(
            np.max(np.abs(gains[:, 0] - gains[:, 1]))
        ),
        "start_sequence_delta_db": start_delta.tolist(),
        "end_sequence_delta_db": end_delta.tolist(),
        "clipping_fraction": _clipping_fraction(signal).tolist(),
    }


def test_iio_large_frame_agc_history_matches_endpoints_and_channels(
    attached_plutos, radio_lan_hosts, pytestconfig, radio_report_dir
):
    attenuation = pytestconfig.getoption("--radio-tx-loopback-attenuation-db")
    validate_loopback_safety(
        physical_attenuation_db=attenuation,
        strongest_tx_gain_db=STRONG_TX_GAIN_DB,
    )

    interface = pytestconfig.getoption("--radio-direct-ip-ladder-interface")
    report = {
        "samples_per_channel": SAMPLES_PER_CHANNEL,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "lo_hz": LO_HZ,
        "bandwidth_hz": RF_BANDWIDTH_HZ,
        "strong_tx_gain_db": STRONG_TX_GAIN_DB,
        "weak_tx_gain_db": WEAK_TX_GAIN_DB,
        "attenuation_db": attenuation,
        "radios": [],
    }
    report_path = radio_report_dir / "iio_metadata_large_frame_agc.json"

    import adi

    for attached in attached_plutos:
        uri = _usb_uri(attached.serial)
        host = _lan_host(attached.serial, interface, radio_lan_hosts)
        sdr = adi.ad9361(uri=uri)
        receiver = None
        stimulus = None
        try:
            assert sdr._ctx.attrs.get("hw_serial") == attached.serial
            assert sdr._ctx.attrs.get("iio,buffer-metadata") == "1"
            _mute_sdr(sdr)
            sdr.rx_enabled_channels = [0, 1]
            sdr.sample_rate = SAMPLE_RATE_HZ
            sdr.rx_rf_bandwidth = RF_BANDWIDTH_HZ
            sdr.tx_rf_bandwidth = RF_BANDWIDTH_HZ
            sdr.rx_lo = LO_HZ
            sdr.tx_lo = LO_HZ
            sdr.rx_buffer_size = 65_536
            sdr.gain_control_mode_chan0 = "manual"
            sdr.gain_control_mode_chan1 = "manual"
            sdr.rx_hardwaregain_chan0 = 26
            sdr.rx_hardwaregain_chan1 = 26
            tone_analysis = _arm_verified_tone(sdr)

            sdr.gain_control_mode_chan0 = "slow_attack"
            sdr.gain_control_mode_chan1 = "slow_attack"
            sdr.tx_hardwaregain_chan1 = WEAK_TX_GAIN_DB
            sdr.rx_buffer_size = SAMPLES_PER_CHANNEL
            sdr._rxadc.set_kernel_buffers_count(2)
            time.sleep(0.75)

            receiver = IioMetadataRx(
                sdr,
                sample_rate_hz=SAMPLE_RATE_HZ,
                samples_per_channel=SAMPLES_PER_CHANNEL,
            )
            receiver.open()
            stimulus = _remote_gain_pattern(host, attached.serial)
            frames = []
            qualifying = []
            for _ in range(MAX_CAPTURE_FRAMES):
                signal, metadata, _capture_time = receiver.capture()
                assert isinstance(metadata, RadioMetadataV3)
                comparison = _compare_gain_history(signal, metadata)
                frames.append(comparison)
                if all(
                    value >= MIN_WITHIN_FRAME_GAIN_RANGE_DB
                    for value in comparison["gain_range_db"]
                ):
                    qualifying.append(comparison)
            assert qualifying, {
                "reason": "bounded TX steps did not move both AGC channels",
                "gain_ranges": [item["gain_range_db"] for item in frames],
            }

            report["radios"].append(
                {
                    "serial": attached.serial,
                    "usb_uri": uri,
                    "lan_host": host,
                    "tone_analysis": tone_analysis,
                    "frames": frames,
                    "qualifying_frame_count": len(qualifying),
                }
            )
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            _stop_remote_pattern(stimulus)
            if receiver is not None:
                receiver.close()
            _mute_sdr(sdr)
            sdr.rx_destroy_buffer()

    assert len(report["radios"]) == len(attached_plutos)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
