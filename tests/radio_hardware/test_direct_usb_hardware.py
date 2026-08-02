"""Quick, explicit, receive-only direct-USB hardware gates.

These tests never enable TX. Run them only with ``--radio-hardware``.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from spf.sdrpluto.direct_usb_protocol import (
    CapabilityFlags,
    MetadataFeatures,
    RadioMetadataV2,
    RuntimeState,
    RuntimeStatusFlags,
)
from spf.sdrpluto.direct_usb_receiver import (
    PlutoDirectUsbReceiver,
    iq_payload_to_complex64,
)


pytestmark = pytest.mark.radio_hardware

V2_REQUIRED_FEATURES = (
    MetadataFeatures.GAIN_ENDPOINT_SNAPSHOTS
    | MetadataFeatures.HEADER_CRC32
    | MetadataFeatures.SAMPLE_SEQUENCE
    | MetadataFeatures.GAIN_DB_ENDPOINTS
    | MetadataFeatures.RSSI_ENDPOINT_SNAPSHOTS
)


def _rss_kib() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("VmRSS is absent from /proc/self/status")


def _validate_frame(frame, samples: int) -> dict:
    metadata = frame.metadata
    assert isinstance(metadata, RadioMetadataV2)
    assert metadata.samples_per_channel == samples
    assert metadata.iq_payload_bytes == samples * 8
    assert metadata.gain_metadata_valid
    assert metadata.rssi_metadata_valid
    assert np.isfinite(metadata.gain_db_start).all()
    assert np.isfinite(metadata.gain_db_end).all()
    assert np.isfinite(metadata.rssi_db_start).all()
    assert np.isfinite(metadata.rssi_db_end).all()
    signal = iq_payload_to_complex64(frame.iq_payload, samples)
    assert signal.shape == (2, samples)
    assert signal.dtype == np.complex64
    assert np.isfinite(signal).all()
    assert np.any(signal[0] != 0)
    assert np.any(signal[1] != 0)
    return {
        "stream_id": metadata.stream_id,
        "buffer_sequence": metadata.buffer_sequence,
        "sample_sequence": metadata.first_sample_sequence,
        "gain_db_start": list(metadata.gain_db_start),
        "gain_db_end": list(metadata.gain_db_end),
        "rssi_db_start": list(metadata.rssi_db_start),
        "rssi_db_end": list(metadata.rssi_db_end),
        "channel_rms": [
            float(np.sqrt(np.mean(np.abs(channel).astype(np.float64) ** 2)))
            for channel in signal
        ],
    }


def test_identity_and_v2_capabilities(attached_plutos):
    for radio in attached_plutos:
        with PlutoDirectUsbReceiver(
            serial=radio.serial, protocol_version=2
        ) as receiver:
            assert receiver.identity.serial == radio.serial
            assert receiver.identity.port_path == radio.port_path
            capabilities = receiver.capabilities
            assert capabilities.protocol_min <= 2 <= capabilities.protocol_max
            assert (
                capabilities.supported_features & V2_REQUIRED_FEATURES
                == V2_REQUIRED_FEATURES
            )
            assert capabilities.capability_flags & CapabilityFlags.FINITE_RX
            assert capabilities.capability_flags & CapabilityFlags.HARDWARE_IDENTITY
            assert capabilities.capability_flags & CapabilityFlags.STATUS
            identity = receiver.query_hardware_identity()
            assert identity.gadget_build_id
            status = receiver.query_runtime_status()
            assert status.lifecycle_state == RuntimeState.IDLE
            assert status.flags & RuntimeStatusFlags.BOOT_ID_VALID
            assert status.flags & RuntimeStatusFlags.PROCESS_NONCE_VALID
            assert any(status.boot_id)
            assert any(status.process_nonce)


def test_runtime_status_accounts_for_completed_frames(attached_plutos, pytestconfig):
    samples = pytestconfig.getoption("--radio-samples")
    frames_per_request = pytestconfig.getoption("--radio-frames-per-request")
    assert frames_per_request > 1
    for radio in attached_plutos:
        with PlutoDirectUsbReceiver(
            serial=radio.serial, protocol_version=2
        ) as receiver:
            before = receiver.query_runtime_status()
            frames = list(
                receiver.stream_frames(
                    samples_per_channel=samples,
                    frame_count=frames_per_request,
                    queue_depth=1,
                )
            )
            after = receiver.query_runtime_status()
        assert len(frames) == frames_per_request
        assert after.lifecycle_state == RuntimeState.IDLE
        assert after.process_nonce == before.process_nonce
        assert after.start_count == before.start_count + 1
        assert (
            after.completed_frame_count
            == before.completed_frame_count + frames_per_request
        )
        assert after.last_completed_sequence == frames_per_request - 1
        assert after.dropped_frame_count == before.dropped_frame_count
        assert after.iio_refill_error_count == before.iio_refill_error_count
        assert after.usb_submit_error_count == before.usb_submit_error_count
        assert after.short_write_count == before.short_write_count
        assert after.buffer_starvation_count == before.buffer_starvation_count
        assert after.stop_timeout_count == before.stop_timeout_count


def test_contiguous_multiframe_request(attached_plutos, pytestconfig):
    samples = pytestconfig.getoption("--radio-samples")
    frames_per_request = pytestconfig.getoption("--radio-frames-per-request")
    assert samples > 0
    assert frames_per_request > 1
    # Legacy capture preallocates every transfer. Keep this parser comparison
    # below the Pi's common 16 MiB usbfs ceiling; the rolling tests exercise the
    # full configured group with one resident transfer per radio.
    legacy_frames = min(frames_per_request, 3)
    for radio in attached_plutos:
        with PlutoDirectUsbReceiver(
            serial=radio.serial, protocol_version=2
        ) as receiver:
            capture = receiver.capture(
                samples_per_channel=samples,
                frame_count=legacy_frames,
            )
        assert len(capture.frames) == legacy_frames
        assert len({frame.metadata.stream_id for frame in capture.frames}) == 1
        assert [frame.metadata.buffer_sequence for frame in capture.frames] == list(
            range(legacy_frames)
        )
        assert [frame.metadata.first_sample_sequence for frame in capture.frames] == [
            index * samples for index in range(legacy_frames)
        ]
        for frame in capture.frames:
            _validate_frame(frame, samples)


def test_repeated_production_frame_lifecycle(
    attached_plutos, pytestconfig, radio_report_dir
):
    samples = pytestconfig.getoption("--radio-samples")
    cycles = pytestconfig.getoption("--radio-cycles")
    max_growth_kib = int(pytestconfig.getoption("--radio-max-rss-growth-mib") * 1024)
    assert cycles > 0
    report = {"samples_per_channel": samples, "cycles": cycles, "radios": []}
    initial_rss = _rss_kib()
    for radio in attached_plutos:
        frames = []
        with PlutoDirectUsbReceiver(
            serial=radio.serial, protocol_version=2
        ) as receiver:
            for _ in range(cycles):
                capture = receiver.capture(samples_per_channel=samples, frame_count=1)
                assert len(capture.frames) == 1
                frames.append(_validate_frame(capture.frames[0], samples))
        report["radios"].append(
            {
                **dataclasses.asdict(radio),
                "first_stream_id": frames[0]["stream_id"],
                "last_stream_id": frames[-1]["stream_id"],
                "channel_rms_last": frames[-1]["channel_rms"],
            }
        )
    final_rss = _rss_kib()
    report["rss_initial_kib"] = initial_rss
    report["rss_final_kib"] = final_rss
    report["rss_growth_kib"] = final_rss - initial_rss
    (radio_report_dir / "repeated_lifecycle.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    assert final_rss - initial_rss <= max_growth_kib


def _simultaneous_capture(serial: str, samples: int, frames: int) -> dict:
    with PlutoDirectUsbReceiver(serial=serial, protocol_version=2) as receiver:
        capture = receiver.capture(samples_per_channel=samples, frame_count=frames)
    for frame in capture.frames:
        _validate_frame(frame, samples)
    return {
        "serial": serial,
        "frames": len(capture.frames),
        "elapsed_seconds": capture.elapsed_seconds,
    }


def _simultaneous_rolling_stream(serial: str, samples: int, frames: int) -> dict:
    with PlutoDirectUsbReceiver(serial=serial, protocol_version=2) as receiver:
        received = list(
            receiver.stream_frames(
                samples_per_channel=samples,
                frame_count=frames,
                queue_depth=1,
            )
        )
    for frame in received:
        _validate_frame(frame, samples)
    return {
        "serial": serial,
        "frames": len(received),
        "stream_ids": len({frame.metadata.stream_id for frame in received}),
        "sequences": [frame.metadata.buffer_sequence for frame in received],
    }


def test_simultaneous_radios(attached_plutos, pytestconfig, radio_report_dir):
    if len(attached_plutos) < 2:
        pytest.skip("simultaneous capture requires at least two selected radios")
    samples = pytestconfig.getoption("--radio-samples")
    # Production asks for one finite frame per radio. Do not multiply the
    # multi-frame parser test by the radio count here: two radios x three
    # 4 MiB frames exceeds Linux's common 16 MiB usbfs transfer-memory limit
    # before the gadget receives START. Per-radio queued-depth coverage lives
    # in test_contiguous_multiframe_request above.
    frames = 1
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(attached_plutos)
    ) as executor:
        futures = [
            executor.submit(_simultaneous_capture, radio.serial, samples, frames)
            for radio in attached_plutos
        ]
        results = [future.result() for future in futures]
    assert {result["serial"] for result in results} == {
        radio.serial for radio in attached_plutos
    }
    (radio_report_dir / "simultaneous_capture.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


def test_simultaneous_rolling_streams(attached_plutos, pytestconfig, radio_report_dir):
    if len(attached_plutos) < 2:
        pytest.skip("simultaneous streaming requires at least two selected radios")
    samples = pytestconfig.getoption("--radio-samples")
    frames = pytestconfig.getoption("--radio-frames-per-request")
    assert frames > 1
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(attached_plutos)
    ) as executor:
        futures = [
            executor.submit(
                _simultaneous_rolling_stream,
                radio.serial,
                samples,
                frames,
            )
            for radio in attached_plutos
        ]
        results = [future.result() for future in futures]
    assert {result["serial"] for result in results} == {
        radio.serial for radio in attached_plutos
    }
    for result in results:
        assert result["frames"] == frames
        assert result["stream_ids"] == 1
        assert result["sequences"] == list(range(frames))
    (radio_report_dir / "simultaneous_rolling_stream.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
