import json
import math
from pathlib import Path

import numpy as np
import yaml

from spf.bench.dual_rx_phase import ToneQualityThresholds
from spf.calibrations.dual_rx_gain_frequency.config import (
    CalibrationConfig,
    build_schedule,
    group_schedule_by_frequency,
)
from spf.calibrations.dual_rx_gain_frequency.hardware import make_cyclic_tone
from spf.calibrations.dual_rx_gain_frequency.hardware import DirectUsbLoopbackRadio
from spf.calibrations.dual_rx_gain_frequency.model import (
    fit_additive_surface,
    fit_dataset,
)
from spf.calibrations.dual_rx_gain_frequency.report import (
    build_analysis_summary,
    render_markdown,
)
from spf.calibrations.dual_rx_gain_frequency.runner import run_calibration
from spf.calibrations.dual_rx_gain_frequency.validate import validate_dataset
from spf.sdrpluto.direct_usb_protocol import MetadataFlags
from spf.sdrpluto.sdr_controller import PlutoRxBuffer, SdrDeviceIdentity


FIRMWARE = {
    "release-tag": "test-release",
    "asset-name": "test.dfu",
    "image-url": "https://example.invalid/test.dfu",
    "image-sha256": "a" * 64,
    "firmware-git-sha": "b" * 40,
    "gadget-git-sha": "c" * 40,
    "boot-mode": "ram",
}


def small_config(**overrides):
    values = {
        "frequencies_hz": (2_400_000_000, 2_450_000_000),
        "gains_db": (0, 10),
        "repetitions": 3,
        "sample_rate_hz": 2_000_000,
        "bandwidth_hz": 1_000_000,
        "buffer_size": 4_096,
        "tone_offset_hz": 100_000,
        "tone_search_width_hz": 10_000,
        "transient_samples": 128,
        "phase_segments": 4,
        "settle_seconds": 0,
        "frequency_settle_seconds": 0,
        "discard_frames_after_gain": 0,
        "max_retries": 0,
        "tx_gain_db": -30,
        "tx_digital_amplitude": 4_096,
        "quality": ToneQualityThresholds(
            min_tone_snr_db=10,
            min_tone_dbfs=-70,
            max_tone_dbfs=-3,
            max_clipping_fraction=0,
            min_coherence=0.98,
            max_within_capture_phase_std_deg=5,
        ),
    }
    values.update(overrides)
    return CalibrationConfig(**values)


def synthetic_signal(config, phase, seed=0):
    rng = np.random.default_rng(seed)
    sample_index = np.arange(config.buffer_size)
    common = 0.31
    result = []
    for channel_phase in (common + phase / 2, common - phase / 2):
        result.append(
            300
            * np.exp(
                1j
                * (
                    2
                    * np.pi
                    * config.tone_offset_hz
                    * sample_index
                    / config.sample_rate_hz
                    + channel_phase
                )
            )
        )
    noise = rng.normal(size=(2, config.buffer_size)) + 1j * rng.normal(
        size=(2, config.buffer_size)
    )
    return (np.asarray(result) + noise).astype(np.complex64)


def fake_frame(config, gain1, gain2, *, stream_id=1, seed=0):
    phase = 0.01 * gain1 - 0.02 * gain2
    signal = synthetic_signal(config, phase, seed)
    flags = (
        MetadataFlags.START_VALID
        | MetadataFlags.END_VALID
        | MetadataFlags.SAMPLE_SEQUENCE_VALID
        | MetadataFlags.GAIN_FULL_TABLE_MODE
        | MetadataFlags.GAIN_DB_VALUES
        | MetadataFlags.RSSI_START_VALID
        | MetadataFlags.RSSI_END_VALID
    )
    return PlutoRxBuffer(
        signal_matrix=signal,
        rssis=np.array([40.25, 41.0]),
        gains=np.array([gain1, gain2], dtype=np.float64),
        gain_index_start=np.array([0xFF, 0xFF], dtype=np.uint8),
        gain_index_end=np.array([0xFF, 0xFF], dtype=np.uint8),
        gain_metadata_valid=True,
        gain_endpoints_equal=np.array([True, True]),
        gain_metadata_flags=int(flags),
        stream_id=stream_id,
        buffer_sequence=0,
        sample_sequence=0,
        gain_start_read_duration_ns=100,
        gain_end_read_duration_ns=110,
        first_gain_change_sample=np.array([-1, -1], dtype=np.int32),
        iq_power_dbfs=np.array([-20, -20], dtype=np.float32),
        gain_db_start=np.array([gain1, gain2], dtype=np.float32),
        gain_db_end=np.array([gain1, gain2], dtype=np.float32),
        rssi_db_start=np.array([40, 40.75], dtype=np.float32),
        rssi_db_end=np.array([40.25, 41], dtype=np.float32),
        rssi_metadata_valid=True,
        rssi_start_read_duration_ns=120,
        rssi_end_read_duration_ns=130,
    )


def fake_identity(serial):
    return SdrDeviceIdentity(
        sdr_family="pluto",
        serial=serial,
        receiver_uri=f"usb:{serial}",
        rx_transport="direct_usb",
        usb_vendor_id=0x0456,
        usb_product_id=0xB673,
        usb_bus=1,
        usb_address=2 if serial == "SERIAL-A" else 3,
        usb_port_path=(1,) if serial == "SERIAL-A" else (2,),
        direct_usb_interface=6,
        direct_usb_bulk_in_endpoint=0x83,
        direct_usb_bulk_out_endpoint=0x04,
        direct_usb_protocol_version=2,
        direct_usb_protocol_min=1,
        direct_usb_protocol_max=2,
        direct_usb_supported_features=0x37,
        direct_usb_capability_flags=1,
    )


def write_ready_manifest(path, serials):
    manifest = {
        "ready_manifest_version": 1,
        "firmware": {
            "release_tag": FIRMWARE["release-tag"],
            "asset_name": FIRMWARE["asset-name"],
            "image_url": FIRMWARE["image-url"],
            "image_sha256": FIRMWARE["image-sha256"],
            "firmware_git_sha": FIRMWARE["firmware-git-sha"],
            "gadget_git_sha": FIRMWARE["gadget-git-sha"],
            "boot_mode": FIRMWARE["boot-mode"],
        },
        "radios": [
            {
                "serial": serial,
                "firmware_verified": True,
                "protocol_min": 1,
                "protocol_max": 2,
                "supported_features": 0x37,
                "capability_flags": 1,
            }
            for serial in serials
        ],
    }
    path.write_text(json.dumps(manifest))


def write_config(path, config):
    document = {
        "data-version": 7,
        "pluto-firmware": FIRMWARE,
        "calibration": {
            **config.as_json(),
            "quality": config.as_json()["quality"],
        },
    }
    path.write_text(yaml.safe_dump(document))


class FakeLoopbackRadio:
    instances = {}

    def __init__(self, serial, config):
        self.serial = serial
        self.config = config
        self.frequency = None
        self.gains = (0, 0)
        self.stream_id = 0
        self.tone_active = False
        FakeLoopbackRadio.instances[serial] = self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def available_gains(self):
        return tuple(range(-3, 72))

    def identity(self):
        return fake_identity(self.serial)

    def configure_frequency(self, frequency):
        self.frequency = frequency
        self.tone_active = True

    def set_gains(self, gain1, gain2):
        self.gains = (gain1, gain2)

    def set_tx_gain(self, gain):
        self.tx_gain = gain

    def discard(self, frame_count):
        self.stream_id += frame_count

    def capture(self):
        self.stream_id += 1
        return fake_frame(
            self.config,
            *self.gains,
            stream_id=self.stream_id,
            seed=self.stream_id,
        )

    def stop_tone(self):
        self.tone_active = False

    def close(self):
        self.stop_tone()


def test_schedule_has_complete_separated_epochs_and_deterministic_pair_orders():
    config = small_config()
    schedule = build_schedule(config)
    assert len(schedule) == 3 * 2 * 2**2
    assert schedule == build_schedule(config)
    blocks = group_schedule_by_frequency(schedule)
    assert len(blocks) == 3 * 2
    assert [block[0].epoch for block in blocks] == [0, 0, 1, 1, 2, 2]
    for epoch in range(3):
        epoch_entries = [entry for entry in schedule if entry.epoch == epoch]
        assert {
            (entry.frequency_index, entry.gain_rx1_db, entry.gain_rx2_db)
            for entry in epoch_entries
        } == {
            (frequency, gain1, gain2)
            for frequency in range(2)
            for gain1 in (0, 10)
            for gain2 in (0, 10)
        }
    assert blocks[1][0].lo_frequency_hz != blocks[2][0].lo_frequency_hz
    assert blocks[3][0].lo_frequency_hz != blocks[4][0].lo_frequency_hz


def test_cyclic_tone_ends_on_an_integer_period():
    config = small_config()
    tone = make_cyclic_tone(config)
    phase_step = 2 * np.pi * config.tone_offset_hz / config.sample_rate_hz
    expected_next = tone[-1] * np.exp(1j * phase_step)
    np.testing.assert_allclose(tone[0], expected_next, rtol=1e-5, atol=1e-3)


class FakePrimingSdr:
    def __init__(self, config):
        self.config = config
        self.calls = []
        self.tx_hardwaregain_chan0 = -80
        self.tx_hardwaregain_chan1 = -80
        self.tx_enabled_channels = []
        self.tx_cyclic_buffer = False

    def rx(self):
        self.calls.append("rx")
        return np.zeros((2, self.config.buffer_size), dtype=np.complex64)

    def rx_destroy_buffer(self):
        self.calls.append("rx_destroy")

    def tx_destroy_buffer(self):
        self.calls.append("tx_destroy")

    def tx(self, tone):
        self.calls.append("tx")


def test_tone_primes_iio_rx_before_arming_cyclic_tx_and_updates_gain_in_place():
    config = small_config()
    radio = DirectUsbLoopbackRadio.__new__(DirectUsbLoopbackRadio)
    radio.config = config
    radio.sdr = FakePrimingSdr(config)
    radio._tone = make_cyclic_tone(config)
    radio._tone_active = False
    radio._active_tx_gain = None

    radio.start_tone(tx_channel=1, tx_gain_db=-20)
    assert radio.sdr.calls == [
        "rx",
        "rx_destroy",
        "tx_destroy",
        "tx",
        "rx",
        "rx_destroy",
    ]
    assert radio.sdr.tx_enabled_channels == [1]
    assert radio.sdr.tx_hardwaregain_chan1 == -20

    radio.set_tx_gain(-35)
    assert radio.sdr.tx_hardwaregain_chan1 == -35
    assert radio.sdr.calls == [
        "rx",
        "rx_destroy",
        "tx_destroy",
        "tx",
        "rx",
        "rx_destroy",
    ]


def test_two_radio_runner_writes_valid_v7_and_fits_model(tmp_path, monkeypatch):
    config = small_config(frequencies_hz=(2_400_000_000,))
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    ready_path = tmp_path / "ready.json"
    serials = ("SERIAL-A", "SERIAL-B")
    write_ready_manifest(ready_path, serials)
    monkeypatch.setenv("SPF_DIRECT_USB_READY_FILE", str(ready_path))

    result = run_calibration(
        config_path=config_path,
        output_dir=tmp_path / "output",
        ready_manifest_path=ready_path,
        serials=serials,
        radio_factory=FakeLoopbackRadio,
    )
    assert result["status"] == "complete"
    assert result["completed_measurements"] == 24

    for serial in serials:
        dataset = tmp_path / "output" / serial / "calibration.v7.zarr"
        report = validate_dataset(
            dataset,
            config=config,
            expected_serial=serial,
            recompute_iq=True,
        )
        assert report["status"] == "pass"
        assert report["completed_frames"] == 12
        model = fit_dataset(dataset, config=config)
        assert model["serial"] == serial
        assert model["quality_valid_observations"] == 12
        assert model["cross_validation_metrics"]["circular_p95_deg"] < 0.2
        summary = build_analysis_summary(report, model)
        assert summary["serial"] == serial
        assert summary["passing_cells"] == 4
        assert len(summary["frequency_summary"]) == 1
        assert "Leave-one-epoch-out circular MAE" in render_markdown(summary)

    resumed = run_calibration(
        config_path=config_path,
        output_dir=tmp_path / "output",
        ready_manifest_path=ready_path,
        serials=serials,
        radio_factory=FakeLoopbackRadio,
    )
    assert resumed["completed_measurements"] == 24


def test_additive_circular_model_recovers_wrapped_gain_effects():
    gain_count = 4
    g1, g2 = np.meshgrid(np.arange(gain_count), np.arange(gain_count), indexing="ij")
    rx1 = np.array([0.0, 0.2, 0.5, 1.0])
    rx2 = np.array([0.0, -0.3, -0.8, -1.4])
    phase = (2.9 + rx1[g1] + rx2[g2] + np.pi) % (2 * np.pi) - np.pi
    fitted = fit_additive_surface(phase.ravel(), g1.ravel(), g2.ravel(), gain_count)
    assert np.max(np.abs(fitted["residual_rad"])) < 1e-6
