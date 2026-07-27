import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from spf.bench.dual_rx_phase import ToneQualityThresholds
from spf.calibrations.dual_rx_gain_frequency.config import (
    CalibrationConfig,
    build_schedule,
    group_schedule_by_frequency,
)
from spf.calibrations.dual_rx_gain_frequency.dc_offset import (
    decode_rf_dc_correction_words,
    signed_10bit,
)
from spf.calibrations.dual_rx_gain_frequency.hardware import make_cyclic_tone
from spf.calibrations.dual_rx_gain_frequency.hardware import DirectUsbLoopbackRadio
from spf.calibrations.dual_rx_gain_frequency.model import (
    fit_additive_surface,
    fit_dataset,
    fit_frequency_delay,
    fit_grouped_additive_surface,
    predict_phase_offset,
)
from spf.calibrations.dual_rx_gain_frequency.report import (
    build_analysis_summary,
    render_markdown,
    write_analysis_bundle,
)
from spf.calibrations.dual_rx_gain_frequency.runner import run_calibration
from spf.calibrations.dual_rx_gain_frequency.validate import validate_dataset
from spf.sdrpluto.direct_usb_protocol import MetadataFlags
from spf.sdrpluto.sdr_controller import PlutoRxBuffer, SdrDeviceIdentity
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store


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

    def configure_frequency(self, frequency, *, start_tone=True):
        self.frequency = frequency
        if start_tone:
            self.start_tone()

    def start_tone(
        self,
        tx_channel=1,
        tx_gain_db=None,
        *,
        prime_after_arm=False,
    ):
        self.tone_active = True
        self.prime_after_arm = prime_after_arm

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


class FakeNegotiatingLoopbackRadio(FakeLoopbackRadio):
    active_contexts = 0
    max_active_contexts = 0

    def __init__(self, serial, config):
        super().__init__(serial, config)
        self.closed = False
        type(self).active_contexts += 1
        type(self).max_active_contexts = max(
            type(self).max_active_contexts,
            type(self).active_contexts,
        )

    def capture(self):
        frame = super().capture()
        if self.serial == "SERIAL-B" and not self.prime_after_arm:
            rng = np.random.default_rng(self.stream_id)
            frame.signal_matrix = (
                rng.normal(size=frame.signal_matrix.shape)
                + 1j * rng.normal(size=frame.signal_matrix.shape)
            ).astype(np.complex64)
        return frame

    def close(self):
        if self.closed:
            return
        self.closed = True
        super().close()
        type(self).active_contexts -= 1


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


def test_frequency_delay_fit_uses_rx1_minus_rx2_physical_sign():
    frequency_hz = np.asarray(
        [5_766_000_000, 5_804_000_000, 5_838_000_000, 5_866_000_000]
    )
    reference_hz = float(np.mean(frequency_hz))
    expected_delay_seconds = 0.75e-9
    phase_rad = (
        -0.9 - 2 * np.pi * (frequency_hz - reference_hz) * expected_delay_seconds
    )

    fitted = fit_frequency_delay(frequency_hz, phase_rad)

    assert fitted["descriptive_delay_seconds"] == pytest.approx(expected_delay_seconds)
    assert fitted["equivalent_free_space_path_m"] == pytest.approx(
        expected_delay_seconds * 299_792_458.0
    )
    assert fitted["fit_residual_metrics"]["circular_max_deg"] < 1e-9
    assert [point["frequency_hz"] for point in fitted["frequency_points"]] == sorted(
        frequency_hz.tolist()
    )


def test_rf_dc_correction_register_decoder_flags_documented_stuck_word():
    expected = {
        "rx1_q": 0x123,
        "rx1_i": 0x234,
        "rx2_q": 0x345,
        "rx2_i": 0x200,
    }
    registers = {
        0x174: expected["rx1_q"] & 0xFF,
        0x175: ((expected["rx1_i"] & 0x3F) << 2) | (expected["rx1_q"] >> 8),
        0x176: ((expected["rx2_q"] & 0x0F) << 4) | (expected["rx1_i"] >> 6),
        0x177: ((expected["rx2_i"] & 0x03) << 6) | (expected["rx2_q"] >> 4),
        0x178: expected["rx2_i"] >> 2,
    }

    decoded = decode_rf_dc_correction_words(registers)

    assert {name: value["raw"] for name, value in decoded.items()} == expected
    assert decoded["rx2_i"] == {
        "raw": 0x200,
        "signed": -512,
        "is_documented_stuck_value": True,
    }
    assert all(
        not value["is_documented_stuck_value"]
        for name, value in decoded.items()
        if name != "rx2_i"
    )
    assert signed_10bit(0x1FF) == 511
    assert signed_10bit(0x3FF) == -1


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

    def disable_dds(self):
        self.calls.append("disable_dds")

    def dds_single_tone(self, frequency, scale, channel):
        self.calls.append(("dds_single_tone", frequency, scale, channel))


def test_dds_tone_supports_negotiated_post_arm_prime_and_gain_updates():
    config = small_config()
    radio = DirectUsbLoopbackRadio.__new__(DirectUsbLoopbackRadio)
    radio.config = config
    radio.sdr = FakePrimingSdr(config)
    radio._tone_active = False
    radio._active_tx_gain = None

    radio.start_tone(
        tx_channel=1,
        tx_gain_db=-20,
        prime_after_arm=True,
    )
    assert radio.sdr.calls == [
        "disable_dds",
        "tx_destroy",
        ("dds_single_tone", 100_000, 0.125, 1),
        "rx",
        "rx_destroy",
    ]
    assert radio.sdr.tx_enabled_channels == []
    assert radio.sdr.tx_cyclic_buffer is False
    assert radio.sdr.tx_hardwaregain_chan1 == -20

    radio.set_tx_gain(-35)
    assert radio.sdr.tx_hardwaregain_chan1 == -35
    assert radio.sdr.calls == [
        "disable_dds",
        "tx_destroy",
        ("dds_single_tone", 100_000, 0.125, 1),
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
    FakeNegotiatingLoopbackRadio.active_contexts = 0
    FakeNegotiatingLoopbackRadio.max_active_contexts = 0

    result = run_calibration(
        config_path=config_path,
        output_dir=tmp_path / "output",
        ready_manifest_path=ready_path,
        serials=serials,
        radio_factory=FakeNegotiatingLoopbackRadio,
    )
    assert result["status"] == "complete"
    assert result["completed_measurements"] == 24
    assert FakeNegotiatingLoopbackRadio.active_contexts == 0
    assert FakeNegotiatingLoopbackRadio.max_active_contexts == 1

    for serial in serials:
        dataset = tmp_path / "output" / serial / "calibration.v7.zarr"
        zarr = zarr_open_from_lmdb_store(str(dataset), mode="r")
        assert zarr.attrs["lmdb_write_policy"] == "map_async_block_sync"
        zarr.store.close()
        report = validate_dataset(
            dataset,
            config=config,
            expected_serial=serial,
            recompute_iq=True,
        )
        assert report["status"] == "pass"
        assert report["completed_frames"] == 12
        model = fit_dataset(dataset, config=config)
        assert model["schema_version"] == 2
        assert model["serial"] == serial
        assert model["quality_valid_observations"] == 12
        assert model["cross_validation_metrics"]["circular_p95_deg"] < 0.2
        predicted = predict_phase_offset(
            model,
            frequency_hz=2_400_000_000,
            gain_rx1_db=0,
            gain_rx2_db=10,
        )
        assert (
            abs(math.atan2(math.sin(predicted + 0.2), math.cos(predicted + 0.2))) < 0.01
        )
        with pytest.raises(ValueError, match="unsupported RF frequency"):
            predict_phase_offset(
                model,
                frequency_hz=2_401_000_000,
                gain_rx1_db=0,
                gain_rx2_db=10,
            )
        model["frequency_models"][0]["supported_gain_pair"][0][1] = False
        with pytest.raises(ValueError, match="unvalidated ordered gain pair"):
            predict_phase_offset(
                model,
                frequency_hz=2_400_000_000,
                gain_rx1_db=0,
                gain_rx2_db=10,
            )
        model["frequency_models"][0]["supported_gain_pair"][0][1] = True
        assert (
            model["frequency_models"][0]["cross_validation_metrics"]["circular_p95_deg"]
            < 0.2
        )
        comparisons = model["model_comparisons"]
        assert comparisons["additive"]["n_observations"] == 12
        assert comparisons["additive_vs_cell_interaction"]["n_observations"] == 12
        assert (
            comparisons["additive_vs_cell_interaction"]["recommended_model"]
            == "additive"
        )
        assert comparisons["additive_vs_gain_difference_only"]["n_observations"] == 12
        assert (
            comparisons["additive_vs_gain_difference_only"]["recommended_model"]
            == "additive"
        )
        assert comparisons["unanchored_vs_one_frame_anchor"]["n_observations"] == 9
        assert (
            comparisons["unanchored_vs_one_frame_anchor"]["recommended_model"]
            == "unanchored"
        )
        assert (
            comparisons["frequency_specific_vs_shared_gain_curves"]["n_observations"]
            == 12
        )
        assert (
            comparisons["frequency_specific_vs_shared_gain_curves"]["recommended_model"]
            == "frequency_shared_gain_curves"
        )
        assert [
            tier["minimum_both_channel_tone_snr_db"]
            for tier in comparisons["confidence_tiers"]
        ] == [-10.0, 0.0, 10.0]
        summary = build_analysis_summary(report, model)
        assert summary["schema_version"] == 2
        assert summary["serial"] == serial
        assert summary["passing_cells"] == 4
        assert len(summary["frequency_summary"]) == 1
        markdown = render_markdown(summary)
        assert "Leave-one-epoch-out circular MAE" in markdown
        assert "Paired model comparisons" in markdown
        assert "Signal-confidence tiers" in markdown
        if serial == "SERIAL-A":
            validation_path = tmp_path / "validation.json"
            model_path = tmp_path / "model.json"
            validation_path.write_text(json.dumps(report))
            model_path.write_text(json.dumps(model))
            bundle = write_analysis_bundle(
                validation_path=validation_path,
                model_path=model_path,
                output_dir=tmp_path / "analysis",
            )
            assert set(bundle["plot_files"]) == {
                "phase_surface_2400000000.png",
                "fitted_gain_effects.png",
                "additive_residual_2400000000.png",
            }
            assert all(
                (tmp_path / "analysis" / filename).is_file()
                for filename in bundle["plot_files"]
            )
        preflights = [
            json.loads(line)
            for line in (tmp_path / "output" / serial / "preflight.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(preflights) == 3
        if serial == "SERIAL-A":
            assert all(item["attempt"] == 1 for item in preflights)
            assert all(item["prime_after_arm"] is False for item in preflights)
        else:
            assert all(item["attempt"] == 2 for item in preflights)
            assert all(item["prime_after_arm"] is True for item in preflights)
            failures = [
                json.loads(line)
                for line in (tmp_path / "output" / serial / "preflight_failures.jsonl")
                .read_text()
                .splitlines()
            ]
            assert len(failures) == 3
            assert all(item["prime_after_arm"] is False for item in failures)

    from spf.calibrations.dual_rx_gain_frequency import dataset as dataset_module

    real_open = dataset_module.zarr_open_from_lmdb_store
    resume_open_calls = []

    def tracked_open(*args, **kwargs):
        resume_open_calls.append((args, kwargs))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(dataset_module, "zarr_open_from_lmdb_store", tracked_open)
    resumed = run_calibration(
        config_path=config_path,
        output_dir=tmp_path / "output",
        ready_manifest_path=ready_path,
        serials=serials,
        radio_factory=FakeNegotiatingLoopbackRadio,
    )
    assert resumed["completed_measurements"] == 24
    writable_resume_calls = [
        kwargs for _, kwargs in resume_open_calls if kwargs.get("mode") == "rw"
    ]
    assert len(writable_resume_calls) == 2
    assert all(kwargs["map_async"] is True for kwargs in writable_resume_calls)


def test_additive_circular_model_recovers_wrapped_gain_effects():
    gain_count = 4
    g1, g2 = np.meshgrid(np.arange(gain_count), np.arange(gain_count), indexing="ij")
    rx1 = np.array([0.0, 0.2, 0.5, 1.0])
    rx2 = np.array([0.0, -0.3, -0.8, -1.4])
    phase = (2.9 + rx1[g1] + rx2[g2] + np.pi) % (2 * np.pi) - np.pi
    fitted = fit_additive_surface(phase.ravel(), g1.ravel(), g2.ravel(), gain_count)
    assert np.max(np.abs(fitted["residual_rad"])) < 1e-6


def test_grouped_additive_model_recovers_shared_curves_and_group_intercepts():
    gain_count = 4
    group_count = 3
    group, g1, g2 = np.meshgrid(
        np.arange(group_count),
        np.arange(gain_count),
        np.arange(gain_count),
        indexing="ij",
    )
    intercept = np.array([2.9, -2.8, 0.7])
    rx1 = np.array([0.0, 0.2, 0.5, 1.0])
    rx2 = np.array([0.0, -0.3, -0.8, -1.4])
    phase = (intercept[group] + rx1[g1] + rx2[g2] + np.pi) % (2 * np.pi) - np.pi
    fitted = fit_grouped_additive_surface(
        phase.ravel(),
        g1.ravel(),
        g2.ravel(),
        group.ravel(),
        gain_count,
        group_count,
    )
    assert np.max(np.abs(fitted["residual_rad"])) < 1e-6
