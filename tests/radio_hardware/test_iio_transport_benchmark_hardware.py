"""Standard libiio USB and IP/TCP sample-rate/throughput matrix.

The test is selected explicitly as a file and remains gated by
``--radio-hardware``.  It never enables TX.  High RF rates are measured and
classified rather than assumed sustainable: frame metadata exposes how much
of the continuously advancing FPGA sample timeline the host actually receives.
"""

from __future__ import annotations

import errno
import gc
import json
import time

import pytest

from spf.direct_radio.usb_protocol import MetadataFlags, RadioMetadataV3
from spf.scripts.resolve_pluto_ip import neighbor_candidates, resolve_pluto_ip
from spf.scripts.direct_ip_parallel_ladder import parse_sample_rate_ladder


pytestmark = pytest.mark.radio_hardware

KERNEL_BUFFERS = 2
WARMUP_FRAMES = 2
MINIMUM_METADATA_RETENTION = 0.65
MINIMUM_EXPECTED_TRANSPORT_MBPS = {"usb": 15.0, "tcp": 30.0}
MINIMUM_CONTINUOUS_METADATA_RATE_HZ = {"usb": 2_000_000, "tcp": 3_000_000}
UNSAFE_FLAGS = (
    MetadataFlags.DUMMY_GAINS
    | MetadataFlags.GAIN_READ_FAILED
    | MetadataFlags.RSSI_READ_FAILED
    | MetadataFlags.DEVICE_IIO_OVERFLOW
    | MetadataFlags.FPGA_EVENT_OVERFLOW
    | MetadataFlags.GAIN_OBSERVATION_OVERFLOW
)


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


def _refill_metadata(buffer, *, startup: bool) -> bytes:
    for retry in range(65):
        try:
            return buffer.refill()
        except OSError as error:
            if startup and error.errno == errno.EAGAIN and retry < 64:
                continue
            raise
    raise AssertionError("unreachable")


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


def _capture_cell(
    *,
    uri: str,
    serial: str,
    transport: str,
    mode: str,
    sample_rate_hz: int,
    samples: int,
    frames: int,
) -> dict:
    import adi
    import iio

    sdr = adi.ad9361(uri=uri)
    buffer = None
    try:
        assert sdr._ctx.attrs.get("hw_serial") == serial
        if mode == "metadata":
            assert sdr._ctx.attrs.get("iio,buffer-metadata") == "1"
        sdr._ctx.set_timeout(20_000)
        _mute(sdr)
        sdr.rx_destroy_buffer()
        sdr.rx_enabled_channels = [0, 1]
        sdr.sample_rate = sample_rate_hz
        actual_rate = int(sdr.sample_rate)
        assert actual_rate == sample_rate_hz
        bandwidth_hz = min(max(sample_rate_hz, 200_000), 20_000_000)
        sdr.rx_rf_bandwidth = bandwidth_hz
        sdr.gain_control_mode_chan0 = "manual"
        sdr.gain_control_mode_chan1 = "manual"
        sdr.rx_hardwaregain_chan0 = 26
        sdr.rx_hardwaregain_chan1 = 26

        device = sdr._rxadc
        for channel in device.channels:
            if channel.scan_element:
                channel.enabled = True
        device.set_kernel_buffers_count(KERNEL_BUFFERS)
        assert device.kernel_buffers_count == KERNEL_BUFFERS
        if mode == "metadata":
            buffer = iio.MetadataBuffer(device, samples, 64 * 1024)
        else:
            buffer = iio.Buffer(device, samples)

        expected_iq_bytes = samples * 8
        for warmup in range(WARMUP_FRAMES):
            if mode == "metadata":
                _refill_metadata(buffer, startup=warmup == 0)
            else:
                buffer.refill()
            assert len(buffer.read()) == expected_iq_bytes

        metadata_records = []
        started_ns = time.perf_counter_ns()
        for frame_index in range(frames):
            if mode == "metadata":
                raw_metadata = _refill_metadata(buffer, startup=False)
                metadata = RadioMetadataV3.unpack(raw_metadata)
                assert metadata.samples_per_channel == samples
                assert metadata.iq_payload_bytes == expected_iq_bytes
                assert metadata.gain_metadata_valid
                assert metadata.rssi_metadata_valid
                assert metadata.gain_observations
                assert not metadata.flags & UNSAFE_FLAGS
                metadata_records.append(metadata)
            else:
                buffer.refill()
            assert len(buffer.read()) == expected_iq_bytes
        elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1_000_000_000

        iq_bytes = frames * expected_iq_bytes
        result = {
            "transport": transport,
            "uri": uri,
            "mode": mode,
            "sample_rate_hz": sample_rate_hz,
            "continuous_payload_requirement_MBps": sample_rate_hz * 8 / 1_000_000,
            "samples_per_channel": samples,
            "frames": frames,
            "kernel_buffers": KERNEL_BUFFERS,
            "elapsed_seconds": elapsed_seconds,
            "payload_MBps": iq_bytes / elapsed_seconds / 1_000_000,
            "payload_MiBps": iq_bytes / elapsed_seconds / (1024 * 1024),
            "delivered_sample_rate_sps": frames * samples / elapsed_seconds,
        }
        if metadata_records:
            capture_indices = [item.buffer_sequence for item in metadata_records]
            first_samples = [item.first_sample_sequence for item in metadata_records]
            assert all(
                right > left for left, right in zip(capture_indices, capture_indices[1:])
            )
            assert all(
                right > left for left, right in zip(first_samples, first_samples[1:])
            )
            source_span_samples = first_samples[-1] + samples - first_samples[0]
            coverage_ratio = frames * samples / source_span_samples
            result.update(
                {
                    "capture_index_first": capture_indices[0],
                    "capture_index_last": capture_indices[-1],
                    "capture_index_gap_count": sum(
                        right - left - 1
                        for left, right in zip(
                            capture_indices, capture_indices[1:]
                        )
                    ),
                    "source_span_samples": source_span_samples,
                    "captured_sample_coverage_ratio": coverage_ratio,
                    "host_delivery_fraction_of_source_rate": (
                        result["delivered_sample_rate_sps"] / sample_rate_hz
                    ),
                    "continuous_sustainable": bool(
                        coverage_ratio >= 0.98
                        and result["delivered_sample_rate_sps"]
                        >= sample_rate_hz * 0.90
                    ),
                }
            )
        return result
    finally:
        del buffer
        gc.collect()
        _mute(sdr)
        sdr.rx_destroy_buffer()


def test_iio_usb_tcp_sample_rate_and_throughput_matrix(
    attached_plutos, radio_lan_hosts, pytestconfig, radio_report_dir
):
    rates = parse_sample_rate_ladder(
        pytestconfig.getoption("--radio-iio-rate-ladder")
    )
    frames = pytestconfig.getoption("--radio-iio-rate-frames")
    samples = pytestconfig.getoption("--radio-iio-rate-samples")
    if frames < 4:
        pytest.fail("--radio-iio-rate-frames must be at least 4")
    if samples < 16_384:
        pytest.fail("--radio-iio-rate-samples must be at least 16384")

    import adi
    import iio

    interface = pytestconfig.getoption("--radio-direct-ip-ladder-interface")
    report = {
        "host_libiio_version": list(iio.version),
        "rates_hz": list(rates),
        "frames_per_cell": frames,
        "samples_per_channel": samples,
        "minimum_continuous_metadata_rate_hz": MINIMUM_CONTINUOUS_METADATA_RATE_HZ,
        "radios": [],
    }
    report_path = radio_report_dir / "iio_usb_tcp_rate_matrix.json"

    for attached in attached_plutos:
        usb_uri = _usb_uri(attached.serial)
        candidates = (
            (radio_lan_hosts[attached.serial],)
            if radio_lan_hosts
            else neighbor_candidates(interface)
        )
        host = resolve_pluto_ip(attached.serial, candidates)
        radio_report = {
            "serial": attached.serial,
            "usb_uri": usb_uri,
            "tcp_uri": f"ip:{host}",
            "cells": [],
        }
        report["radios"].append(radio_report)

        restore_sdr = adi.ad9361(uri=usb_uri)
        original = {
            "sample_rate": int(restore_sdr.sample_rate),
            "rx_bandwidth": int(restore_sdr.rx_rf_bandwidth),
            "mode0": str(restore_sdr.gain_control_mode_chan0),
            "mode1": str(restore_sdr.gain_control_mode_chan1),
            "gain0": float(restore_sdr.rx_hardwaregain_chan0),
            "gain1": float(restore_sdr.rx_hardwaregain_chan1),
        }
        del restore_sdr
        try:
            for rate in rates:
                for transport, uri in (("usb", usb_uri), ("tcp", f"ip:{host}")):
                    for mode in ("ordinary", "metadata"):
                        cell = _capture_cell(
                            uri=uri,
                            serial=attached.serial,
                            transport=transport,
                            mode=mode,
                            sample_rate_hz=rate,
                            samples=samples,
                            frames=frames,
                        )
                        radio_report["cells"].append(cell)
                        report_path.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            restore_sdr = adi.ad9361(uri=usb_uri)
            try:
                _mute(restore_sdr)
                restore_sdr.sample_rate = original["sample_rate"]
                restore_sdr.rx_rf_bandwidth = original["rx_bandwidth"]
                restore_sdr.gain_control_mode_chan0 = original["mode0"]
                restore_sdr.gain_control_mode_chan1 = original["mode1"]
                if original["mode0"] == "manual":
                    restore_sdr.rx_hardwaregain_chan0 = original["gain0"]
                if original["mode1"] == "manual":
                    restore_sdr.rx_hardwaregain_chan1 = original["gain1"]
            finally:
                _mute(restore_sdr)

        by_key = {
            (cell["transport"], cell["sample_rate_hz"], cell["mode"]): cell
            for cell in radio_report["cells"]
        }
        for rate in rates:
            for transport in ("usb", "tcp"):
                ordinary = by_key[(transport, rate, "ordinary")]
                metadata = by_key[(transport, rate, "metadata")]
                expected_floor = min(
                    rate * 8 / 1_000_000,
                    MINIMUM_EXPECTED_TRANSPORT_MBPS[transport],
                )
                assert ordinary["payload_MBps"] >= expected_floor * 0.65
                assert metadata["payload_MBps"] >= expected_floor * 0.65
                assert (
                    metadata["payload_MBps"] / ordinary["payload_MBps"]
                    >= MINIMUM_METADATA_RETENTION
                )
                if rate <= MINIMUM_CONTINUOUS_METADATA_RATE_HZ[transport]:
                    assert metadata["continuous_sustainable"], (
                        f"{transport} metadata did not continuously sustain "
                        f"the qualified {rate:g} sample/s cell: {metadata}"
                    )

    report_path.write_text(json.dumps(report, indent=2) + "\n")
