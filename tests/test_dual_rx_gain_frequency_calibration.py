import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from spf.bench.dual_rx_phase import ToneQualityThresholds, wrap_phase
from spf.calibrations.dual_rx_gain_frequency.additive_cross import (
    analyze_additive_cross_dataset,
    compare_additive_cross_results,
)
from spf.calibrations.dual_rx_gain_frequency.comparative_analysis import (
    DEFAULT_STAGE_BOUNDARIES_DB,
    HIGH_BAND_GAIN_MIN_DB,
    HIGH_BAND_LNA_MIXER_BYTE_BY_GAIN,
    _derive_stage_boundaries,
    _transfer,
    write_comparative_bundle,
)
from spf.calibrations.dual_rx_gain_frequency.config import (
    CalibrationConfig,
    build_schedule,
    group_schedule_by_frequency,
)
from spf.calibrations.dual_rx_gain_frequency.cross_radio import (
    compare_radio_models,
    render_cross_radio_markdown,
)
from spf.calibrations.dual_rx_gain_frequency.dc_diagnostic import (
    run_rf_dc_recovery,
    run_rx2_dc_diagnostic,
)
from spf.calibrations.dual_rx_gain_frequency.dc_offset import (
    decode_rf_dc_correction_words,
    signed_10bit,
)
from spf.calibrations.dual_rx_gain_frequency.dc_report import (
    write_rf_dc_evidence_report,
)
from spf.calibrations.dual_rx_gain_frequency.hardware import (
    DirectUsbLoopbackRadio,
    make_cyclic_tone,
)
from spf.calibrations.dual_rx_gain_frequency.low_cost_calibration import (
    _adaptation_delta,
    _evaluate_per_frequency_strategy,
)
from spf.calibrations.dual_rx_gain_frequency.model import (
    fit_additive_surface,
    fit_dataset,
    fit_frequency_delay,
    fit_grouped_additive_surface,
    predict_phase_offset,
)
from spf.calibrations.dual_rx_gain_frequency.model_matrix import (
    INDEPENDENT_GAIN_MODEL,
    MODEL_SPECS,
    SYMMETRIC_GAIN_MODEL,
    _fit,
    _fit_summary,
    _frequency_scaling_structure,
    _held_out_gain_pair_comparison,
    _predict,
    _radio_arrays_from_paths,
    _symmetric_curve_rad,
    _symmetric_model_comparison,
)
from spf.calibrations.dual_rx_gain_frequency.report import (
    build_analysis_summary,
    render_markdown,
    write_analysis_bundle,
)
from spf.calibrations.dual_rx_gain_frequency.runner import (
    _capture_after_discard,
    load_calibration_document,
    run_calibration,
)
from spf.calibrations.dual_rx_gain_frequency.validate import validate_dataset
from spf.hardware_fingerprint import stable_identity_sha256
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
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
        "ready_manifest_version": 2,
        "host_boot_id": "BOOT-A",
        "fingerprint_session_id": "SESSION-A",
        "firmware": {
            "release_tag": FIRMWARE["release-tag"],
            "asset_name": FIRMWARE["asset-name"],
            "image_url": FIRMWARE["image-url"],
            "image_sha256": FIRMWARE["image-sha256"],
            "firmware_git_sha": FIRMWARE["firmware-git-sha"],
            "gadget_git_sha": FIRMWARE["gadget-git-sha"],
            "boot_mode": FIRMWARE["boot-mode"],
        },
        "radios": [],
    }
    for serial in serials:
        identity = fake_identity(serial)
        stable_identity = {
            "pluto_serial": serial,
            "spi_nor_unique_id_hmac_sha256": (
                "a" * 64 if serial == "SERIAL-A" else "b" * 64
            ),
        }
        manifest["radios"].append(
            {
                "serial": serial,
                "firmware_verified": True,
                "protocol_min": 1,
                "protocol_max": 2,
                "supported_features": 0x37,
                "capability_flags": 1,
                "hardware_fingerprint": {
                    "schema": "spf.hardware_compatibility_fingerprint",
                    "schema_version": 1,
                    "fingerprint_timing": "post_firmware_before_recording",
                    "acquisition_binding": True,
                    "passive_observation": True,
                    "tx_operations_performed": False,
                    "host_boot_id": "BOOT-A",
                    "fingerprint_session_id": "SESSION-A",
                    "hmac_key_id": "c" * 16,
                    "stable_identity": stable_identity,
                    "stable_fingerprint_sha256": stable_identity_sha256(
                        stable_identity
                    ),
                    "attachment": {
                        "usb_bus": identity.usb_bus,
                        "usb_address": identity.usb_address,
                        "usb_port_path": ".".join(
                            str(part) for part in identity.usb_port_path
                        ),
                    },
                    "compatibility": {
                        "status": "compatible",
                        "failures": [],
                    },
                },
            }
        )
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
        self.rf_dc_calibrations = 0
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

    def run_rf_dc_calibration(self):
        assert self.tone_active is False
        self.rf_dc_calibrations += 1

    def close(self):
        self.stop_tone()


def test_capture_after_discard_prefers_one_batched_request():
    class BatchedRadio:
        def __init__(self):
            self.calls = []

        def capture_after_discard(self, frame_count):
            self.calls.append(("batch", frame_count))
            return "recorded-frame"

        def discard(self, frame_count):
            self.calls.append(("discard", frame_count))

        def capture(self):
            self.calls.append(("capture",))
            return "legacy-frame"

    radio = BatchedRadio()
    assert _capture_after_discard(radio, 2) == "recorded-frame"
    assert radio.calls == [("batch", 2)]


def test_capture_after_discard_preserves_legacy_adapter_fallback():
    class LegacyRadio:
        def __init__(self):
            self.calls = []

        def discard(self, frame_count):
            self.calls.append(("discard", frame_count))

        def capture(self):
            self.calls.append(("capture",))
            return "legacy-frame"

    radio = LegacyRadio()
    assert _capture_after_discard(radio, 2) == "legacy-frame"
    assert radio.calls == [("discard", 2), ("capture",)]


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


def test_additive_cross_schedule_separates_training_and_held_out_pairs():
    config = small_config(
        frequencies_hz=(2_400_000_000,),
        gains_db=(0, 10, 20),
        schedule_design="additive_cross",
        schedule_reference_gain_db=10,
        held_out_gain_pairs=((0, 20), (20, 0)),
    )

    assert config.training_gain_pairs == (
        (0, 10),
        (10, 10),
        (20, 10),
        (10, 0),
        (10, 20),
    )
    assert config.gain_pairs == config.training_gain_pairs + ((0, 20), (20, 0))
    assert config.measurements_per_radio == 3 * 7
    blocks = group_schedule_by_frequency(build_schedule(config))
    assert len(blocks) == 3
    assert all(
        {(entry.gain_rx1_db, entry.gain_rx2_db) for entry in block}
        == set(config.gain_pairs)
        for block in blocks
    )
    assert build_schedule(config) == build_schedule(config)


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


def test_gain_dependent_branch_delay_model_predicts_unseen_frequency():
    frequencies_hz = np.asarray(
        [850_000_000, 1_300_000_000, 2_450_000_000, 5_800_000_000],
        dtype=np.int64,
    )
    gains_db = np.asarray([0, 20, 40], dtype=np.int64)
    reference_frequency_hz = float(np.mean(frequencies_hz))
    phase_effect_rx1 = np.asarray([-0.08, 0.0, 0.11])
    phase_effect_rx2 = np.asarray([0.04, 0.0, -0.06])
    delay_rx1_seconds = np.asarray([-7e-12, 0.0, 13e-12])
    delay_rx2_seconds = np.asarray([5e-12, 0.0, -9e-12])
    base_delay_seconds = 31e-12
    rows = []
    for frequency_index, frequency_hz in enumerate(frequencies_hz):
        for gain1_index, gain1_db in enumerate(gains_db):
            for gain2_index, gain2_db in enumerate(gains_db):
                differential_delay = (
                    base_delay_seconds
                    + delay_rx1_seconds[gain1_index]
                    + delay_rx2_seconds[gain2_index]
                )
                phase = (
                    0.27
                    + phase_effect_rx1[gain1_index]
                    + phase_effect_rx2[gain2_index]
                    - 2
                    * np.pi
                    * (frequency_hz - reference_frequency_hz)
                    * differential_delay
                )
                rows.append(
                    (
                        frequency_index,
                        frequency_hz,
                        gain1_index,
                        gain2_index,
                        gain1_db,
                        gain2_db,
                        phase,
                    )
                )
    values = np.asarray(rows)
    data = {
        "radio": np.zeros(len(rows), dtype=np.int64),
        "epoch": np.zeros(len(rows), dtype=np.int64),
        "frequency": values[:, 0].astype(np.int64),
        "frequency_hz": values[:, 1].astype(np.int64),
        "gain1": values[:, 2].astype(np.int64),
        "gain2": values[:, 3].astype(np.int64),
        "gain1_db": values[:, 4].astype(np.int64),
        "gain2_db": values[:, 5].astype(np.int64),
        "phase": values[:, 6],
    }
    spec = next(
        model
        for model in MODEL_SPECS
        if model.name == "branch_gain_delay_lut_per_radio"
    )
    train = {key: value[data["frequency"] != 3] for key, value in data.items()}
    test = {key: value[data["frequency"] == 3] for key, value in data.items()}
    fitted = _fit(
        spec,
        train,
        gain_count=3,
        frequency_count=4,
        reference_gain=1,
        reference_frequency_hz=reference_frequency_hz,
    )
    prediction, supported = _predict(
        fitted,
        test,
        gain_count=3,
        frequency_count=4,
        reference_gain=1,
    )

    assert supported.all()
    assert np.max(np.abs(wrap_phase(test["phase"] - prediction))) < 1e-7
    base_slope = fitted.beta[fitted.feature_names.index("frequency_rad_per_ghz")]
    fitted_base_delay = -base_slope / (2 * np.pi * 1e9)
    assert fitted_base_delay == pytest.approx(base_delay_seconds, abs=1e-16)


def test_model_matrix_frequency_specific_antisymmetric_gain_lut():
    gain_values = np.asarray([-1, 26, 62], dtype=np.int64)
    frequency_values = np.asarray([2_412_000_000, 2_467_100_000], dtype=np.int64)
    intercept = np.asarray([0.2, -0.3])
    gain_effect = np.asarray(
        [
            [-0.11, 0.0, 0.17],
            [0.08, 0.0, -0.14],
        ]
    )
    rows = []
    for frequency_index, frequency_hz in enumerate(frequency_values):
        for gain1_index, gain1_db in enumerate(gain_values):
            for gain2_index, gain2_db in enumerate(gain_values):
                rows.append(
                    (
                        frequency_index,
                        frequency_hz,
                        gain1_index,
                        gain2_index,
                        gain1_db,
                        gain2_db,
                        intercept[frequency_index]
                        + gain_effect[frequency_index, gain1_index]
                        - gain_effect[frequency_index, gain2_index],
                    )
                )
    values = np.asarray(rows)
    data = {
        "radio": np.zeros(len(rows), dtype=np.int64),
        "epoch": np.zeros(len(rows), dtype=np.int64),
        "frequency": values[:, 0].astype(np.int64),
        "frequency_hz": values[:, 1].astype(np.int64),
        "gain1": values[:, 2].astype(np.int64),
        "gain2": values[:, 3].astype(np.int64),
        "gain1_db": values[:, 4].astype(np.int64),
        "gain2_db": values[:, 5].astype(np.int64),
        "phase": values[:, 6],
    }
    spec = next(
        model
        for model in MODEL_SPECS
        if model.name == "frequency_specific_antisymmetric_gain_per_radio"
    )
    fitted = _fit(
        spec,
        data,
        gain_count=3,
        frequency_count=2,
        reference_gain=1,
        reference_frequency_hz=float(np.mean(frequency_values)),
    )
    prediction, supported = _predict(
        fitted,
        data,
        gain_count=3,
        frequency_count=2,
        reference_gain=1,
    )

    assert supported.all()
    assert fitted.feature_names == [
        "frequency[0].intercept",
        "frequency[0].gain_phase[0]",
        "frequency[0].gain_phase[2]",
        "frequency[1].intercept",
        "frequency[1].gain_phase[0]",
        "frequency[1].gain_phase[2]",
    ]
    assert np.max(np.abs(wrap_phase(data["phase"] - prediction))) < 1e-9
    summary = _fit_summary(
        spec,
        data,
        radio_count=1,
        gain_count=3,
        frequency_count=2,
        reference_gain=1,
        reference_frequency_hz=float(np.mean(frequency_values)),
    )
    assert summary["total_parameter_count"] == 2 * 3
    recovered_curve = _symmetric_curve_rad(
        summary["fits"][0],
        frequency_index=0,
        gain_count=3,
        reference_gain=1,
    )
    np.testing.assert_allclose(
        recovered_curve,
        gain_effect[0] - gain_effect[0, 1],
        atol=1e-9,
    )


@pytest.mark.parametrize(
    (
        "model_name",
        "frequency_values",
        "table_gain_effect",
        "frequency_scaled",
        "parameter_count",
    ),
    (
        (
            "frequency_lut_symmetric_gain_per_radio",
            (915_000_000, 2_412_000_000),
            ((-0.11, 0.0, 0.17),),
            False,
            2,
        ),
        (
            "frequency_lut_frequency_scaled_symmetric_gain_per_radio",
            (915_000_000, 2_412_000_000),
            ((-0.11, 0.0, 0.17),),
            True,
            2,
        ),
        (
            "frequency_lut_gain_table_symmetric_gain_per_radio",
            (915_000_000, 2_412_000_000, 5_804_000_000),
            (
                (-0.11, 0.0, 0.17),
                (0.08, 0.0, -0.14),
                (-0.04, 0.0, 0.21),
            ),
            False,
            6,
        ),
        (
            "frequency_lut_gain_table_frequency_scaled_symmetric_gain_per_radio",
            (915_000_000, 2_412_000_000, 5_804_000_000),
            (
                (-0.11, 0.0, 0.17),
                (0.08, 0.0, -0.14),
                (-0.04, 0.0, 0.21),
            ),
            True,
            6,
        ),
    ),
)
def test_model_matrix_frequency_scaled_symmetric_gain_lut(
    model_name,
    frequency_values,
    table_gain_effect,
    frequency_scaled,
    parameter_count,
):
    gain_values = np.asarray([-1, 26, 62], dtype=np.int64)
    frequency_values = np.asarray(frequency_values, dtype=np.int64)
    table_gain_effect = np.asarray(table_gain_effect)
    intercept = np.linspace(-0.2, 0.3, frequency_values.size)
    rows = []
    for frequency_index, frequency_hz in enumerate(frequency_values):
        table = (
            0
            if frequency_hz <= 1_300_000_000
            else (1 if frequency_hz <= 4_000_000_000 else 2)
        )
        effect = table_gain_effect[min(table, table_gain_effect.shape[0] - 1)]
        frequency_scale = frequency_hz / 1e9 if frequency_scaled else 1.0
        for gain1_index, gain1_db in enumerate(gain_values):
            for gain2_index, gain2_db in enumerate(gain_values):
                rows.append(
                    (
                        frequency_index,
                        frequency_hz,
                        gain1_index,
                        gain2_index,
                        gain1_db,
                        gain2_db,
                        intercept[frequency_index]
                        + frequency_scale
                        * (effect[gain1_index] - effect[gain2_index]),
                    )
                )
    values = np.asarray(rows)
    data = {
        "radio": np.zeros(len(rows), dtype=np.int64),
        "epoch": np.zeros(len(rows), dtype=np.int64),
        "frequency": values[:, 0].astype(np.int64),
        "frequency_hz": values[:, 1].astype(np.int64),
        "gain1": values[:, 2].astype(np.int64),
        "gain2": values[:, 3].astype(np.int64),
        "gain1_db": values[:, 4].astype(np.int64),
        "gain2_db": values[:, 5].astype(np.int64),
        "phase": values[:, 6],
    }
    spec = next(model for model in MODEL_SPECS if model.name == model_name)
    fitted = _fit(
        spec,
        data,
        gain_count=3,
        frequency_count=frequency_values.size,
        reference_gain=1,
        reference_frequency_hz=float(np.mean(frequency_values)),
    )
    prediction, supported = _predict(
        fitted,
        data,
        gain_count=3,
        frequency_count=frequency_values.size,
        reference_gain=1,
    )

    assert supported.all()
    assert np.max(np.abs(wrap_phase(data["phase"] - prediction))) < 1e-9
    assert fitted.beta.size == frequency_values.size + parameter_count


def test_frequency_scaled_lut_analysis_scores_held_out_pairs_and_rank():
    gain_values = np.asarray([-1, 26, 62], dtype=np.int64)
    frequency_values = np.asarray(
        [915_000_000, 2_412_000_000, 3_900_000_000, 5_804_000_000],
        dtype=np.int64,
    )
    gain_delay = np.asarray([-0.08, 0.0, 0.13])
    rows = []
    for frequency_index, frequency_hz in enumerate(frequency_values):
        frequency_ghz = frequency_hz / 1e9
        for gain1_index, gain1_db in enumerate(gain_values):
            for gain2_index, gain2_db in enumerate(gain_values):
                rows.append(
                    (
                        frequency_index,
                        frequency_hz,
                        gain1_index,
                        gain2_index,
                        gain1_db,
                        gain2_db,
                        0.1 * frequency_index
                        + frequency_ghz
                        * (gain_delay[gain1_index] - gain_delay[gain2_index]),
                    )
                )
    values = np.asarray(rows)
    data = {
        "radio": np.zeros(len(rows), dtype=np.int64),
        "epoch": np.zeros(len(rows), dtype=np.int64),
        "frequency": values[:, 0].astype(np.int64),
        "frequency_hz": values[:, 1].astype(np.int64),
        "gain1": values[:, 2].astype(np.int64),
        "gain2": values[:, 3].astype(np.int64),
        "gain1_db": values[:, 4].astype(np.int64),
        "gain2_db": values[:, 5].astype(np.int64),
        "phase": values[:, 6],
    }
    common = {
        "radio_count": 1,
        "gain_count": 3,
        "frequency_count": 4,
        "reference_gain": 1,
        "reference_frequency_hz": float(np.mean(frequency_values)),
    }
    model_metadata = {
        spec.name: {
            "label": spec.label,
            "total_parameter_count": 1,
        }
        for spec in MODEL_SPECS
        if spec.name
        in {
            "frequency_lut_symmetric_gain_per_radio",
            "frequency_lut_frequency_scaled_symmetric_gain_per_radio",
            "frequency_lut_gain_table_symmetric_gain_per_radio",
            "frequency_lut_gain_table_frequency_scaled_symmetric_gain_per_radio",
            SYMMETRIC_GAIN_MODEL,
            INDEPENDENT_GAIN_MODEL,
        }
    }
    held_out = _held_out_gain_pair_comparison(
        data=data,
        held_out_gain_pairs=((-1, 62), (62, -1)),
        models=model_metadata,
        **common,
    )

    assert held_out["available"] is True
    assert held_out["quality_valid_observations"] == 8
    scaled = held_out["models"][
        "frequency_lut_frequency_scaled_symmetric_gain_per_radio"
    ]
    assert scaled["circular_mae_deg"] < 1e-8

    symmetric_spec = next(
        spec for spec in MODEL_SPECS if spec.name == SYMMETRIC_GAIN_MODEL
    )
    symmetric_summary = _fit_summary(symmetric_spec, data, **common)
    structure = _frequency_scaling_structure(
        models={SYMMETRIC_GAIN_MODEL: symmetric_summary},
        provenance=[{"radio_index": 0, "serial": "TEST-SERIAL"}],
        frequencies_hz=frequency_values.tolist(),
        gains_db=gain_values.tolist(),
        reference_gain_db=26,
    )
    assert (
        structure["mean_across_radios"]["all"][
            "forced_frequency_energy_fraction"
        ]
        == pytest.approx(1.0)
    )
    assert (
        structure["mean_across_radios"]["all"]["best_rank1_energy_fraction"]
        == pytest.approx(1.0)
    )


def test_symmetric_model_comparison_always_reports_independent_gap():
    def metric(mae, rmse, p95, maximum):
        return {
            "circular_mae_deg": mae,
            "circular_rmse_deg": rmse,
            "circular_p95_deg": p95,
            "circular_max_deg": maximum,
        }

    symmetric = metric(0.9, 1.4, 3.0, 13.0)
    independent = metric(0.7, 1.1, 2.5, 12.0)
    symmetric["per_radio"] = {"0": metric(0.8, 1.3, 2.8, 11.0)}
    independent["per_radio"] = {"0": metric(0.6, 1.0, 2.3, 10.0)}
    result = _symmetric_model_comparison(
        {
            SYMMETRIC_GAIN_MODEL: {
                "total_parameter_count": 64,
                "leave_one_epoch_out": symmetric,
            },
            INDEPENDENT_GAIN_MODEL: {
                "total_parameter_count": 127,
                "leave_one_epoch_out": independent,
            },
        },
        [{"radio_index": 0, "serial": "TEST-SERIAL"}],
    )

    assert result["default_parsimonious_model"] == SYMMETRIC_GAIN_MODEL
    assert result["accuracy_reference_model"] == INDEPENDENT_GAIN_MODEL
    assert result["parameter_count"] == {
        "symmetric": 64,
        "independent": 127,
        "reduction": 63,
        "reduction_fraction": pytest.approx(63 / 127),
    }
    assert result["leave_one_epoch_out"]["symmetric_minus_independent"] == {
        "circular_mae_deg": pytest.approx(0.2),
        "circular_rmse_deg": pytest.approx(0.3),
        "circular_p95_deg": pytest.approx(0.5),
        "circular_max_deg": pytest.approx(1.0),
    }


def test_compact_stage_boundaries_are_reproducibly_derived_from_gain_table():
    assert len(HIGH_BAND_LNA_MIXER_BYTE_BY_GAIN) == 73
    assert (
        _derive_stage_boundaries(
            HIGH_BAND_LNA_MIXER_BYTE_BY_GAIN,
            minimum_gain_db=HIGH_BAND_GAIN_MIN_DB,
        )
        == DEFAULT_STAGE_BOUNDARIES_DB
        == (-6, 6, 16, 23, 26, 41)
    )


def test_low_cost_calibration_one_and_two_value_adjustments():
    frequencies = np.asarray([1.0e9, 2.0e9, 3.0e9])
    constant = _adaptation_delta(
        frequencies,
        anchor_frequencies_hz=(2_000_000_000,),
        anchor_residuals_rad=(0.3,),
    )
    linear = _adaptation_delta(
        frequencies,
        anchor_frequencies_hz=(1_000_000_000, 3_000_000_000),
        anchor_residuals_rad=(0.1, 0.5),
    )

    assert constant == pytest.approx([0.3, 0.3, 0.3])
    assert linear == pytest.approx([0.1, 0.3, 0.5])


def test_low_cost_two_value_adjustment_uses_nearest_circular_branch():
    frequencies = np.asarray([1.0e9, 2.0e9, 3.0e9])
    adjustment = _adaptation_delta(
        frequencies,
        anchor_frequencies_hz=(1_000_000_000, 3_000_000_000),
        anchor_residuals_rad=(math.radians(170), math.radians(-170)),
    )

    assert np.degrees(adjustment) == pytest.approx([170, 180, 190])


def test_low_cost_one_value_per_frequency_removes_frequency_baseline():
    target = {
        "phase": np.asarray([0.2, 0.2, -0.3, -0.3]),
        "frequency_hz": np.asarray([1_000, 1_000, 2_000, 2_000]),
        "gain1_db": np.asarray([10, 20, 10, 20]),
        "gain2_db": np.asarray([10, 10, 10, 10]),
    }
    prepared = [
        {
            "radio_index": 0,
            "serial": "test",
            "target": target,
            "supported": np.ones(4, dtype=bool),
            "prediction": np.zeros(4),
            "base_residual": target["phase"],
        }
    ]

    result = _evaluate_per_frequency_strategy(
        prepared=prepared,
        frequencies_hz=(1_000, 2_000),
        reference_gain_db=10,
    )

    assert result["evaluated_observations"] == 2
    assert result["circular_mae_deg"] == pytest.approx(0)


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
        table_shared = comparisons[
            "frequency_specific_vs_gain_table_shared_gain_curves"
        ]
        assert table_shared["n_observations"] == 12
        assert table_shared["recommended_model"] == "gain_table_shared_gain_curves"
        assert table_shared["gain_table_boundaries_hz"] == [
            1_300_000_000,
            4_000_000_000,
        ]
        assert table_shared["nominal_parameter_counts"] == {
            "frequency_specific_gain_curves": 3,
            "gain_table_shared_gain_curves": 7,
        }
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
                "model_fit_2400000000.png",
            }
            assert all(
                (tmp_path / "analysis" / filename).is_file()
                for filename in bundle["plot_files"]
            )
            report_markdown = (tmp_path / "analysis" / "REPORT.md").read_text()
            assert "## Model fit plots" in report_markdown
            assert "model_fit_2400000000.png" in report_markdown
            assert "solid lines are additive-model predictions" in report_markdown
        preflights = [
            json.loads(line)
            for line in (tmp_path / "output" / serial / "preflight.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(preflights) == 3
        assert all(
            item["rf_dc_calibration_policy"] == "before_each_frequency_block"
            for item in preflights
        )
        assert all(item["rf_dc_calibration_before_tone"] is True for item in preflights)
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

    comparative = write_comparative_bundle(
        config_path=config_path,
        artifact_root=tmp_path / "output",
        output_dir=tmp_path / "comparative",
    )
    assert Path(comparative["comparison"]).is_file()
    assert Path(comparative["report"]).is_file()
    assert set(comparative["calibrations"]) == set(serials)
    comparison = json.loads(Path(comparative["comparison"]).read_text())
    assert comparison["schema_version"] == 2
    assert set(comparison["radios"]) == set(serials)
    assert comparison["minimum_complete_epochs_per_fitted_radio_frequency"] == 3
    assert set(comparison["cross_radio_transfer_summary"]) == {
        "SERIAL-A->SERIAL-B",
        "SERIAL-B->SERIAL-A",
    }
    assert all(
        len(radio["analysis_input_sha256"]) == 64
        for radio in comparison["radios"].values()
    )
    for calibration_path in comparative["calibrations"].values():
        calibration = json.loads(Path(calibration_path).read_text())
        assert calibration["deployable"] is True
        assert (
            calibration["frequency_models"][0]["production_supported_pair_count"] == 4
        )
    comparative_report = Path(comparative["report"]).read_text()
    assert "Errors explained by competing models" in comparative_report
    assert "New (“unseen”) radio" in comparative_report


def test_additive_cross_fit_scores_pairs_excluded_from_training(tmp_path, monkeypatch):
    config = small_config(
        frequencies_hz=(2_400_000_000,),
        gains_db=(0, 10, 20),
        schedule_design="additive_cross",
        schedule_reference_gain_db=10,
        held_out_gain_pairs=((0, 20), (20, 0)),
    )
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    ready_path = tmp_path / "ready.json"
    write_ready_manifest(ready_path, ("SERIAL-A",))
    monkeypatch.setenv("SPF_DIRECT_USB_READY_FILE", str(ready_path))

    result = run_calibration(
        config_path=config_path,
        output_dir=tmp_path / "output",
        ready_manifest_path=ready_path,
        serials=("SERIAL-A",),
        radio_factory=FakeLoopbackRadio,
    )
    assert result["status"] == "complete"
    dataset = tmp_path / "output" / "SERIAL-A" / "calibration.v7.zarr"
    validation = validate_dataset(dataset, config=config, expected_serial="SERIAL-A")
    assert validation["status"] == "pass"
    assert validation["expected_cells"] == 7
    assert sum(cell["role"] == "held_out" for cell in validation["cells"]) == 2

    analysis = analyze_additive_cross_dataset(dataset, config=config)
    assert analysis["training_pairs_per_frequency"] == 5
    assert analysis["held_out_pairs_per_frequency"] == 2
    assert (
        analysis["frequency_results"][0]["reference_cell_quality_valid_observations"]
        == config.repetitions
    )
    metrics = analysis["overall_held_out_independent_rx_metrics"]
    assert metrics["n_observations"] == 6
    assert metrics["circular_p95_deg"] < 0.2
    second = json.loads(json.dumps(analysis))
    second["serial"] = "SERIAL-B"
    comparison = compare_additive_cross_results([analysis, second])
    assert (
        comparison["held_out_frequency_specific_radio_shared_curve_metrics"][
            "circular_p95_deg"
        ]
        < 0.2
    )
    assert comparison["frequency_comparisons"][0]["pairwise_radio_curve_comparisons"][
        0
    ]["curve_rms_difference_deg"] == pytest.approx(0)
    assert comparison["schema_version"] == 2
    assert len(comparison["held_out_directional_cross_radio_transfers"]) == 2
    assert all(
        row["metrics"]["circular_p95_deg"] < 0.2
        for row in comparison["held_out_directional_cross_radio_transfers"]
    )

    loaded_config, matrix_data, provenance = _radio_arrays_from_paths(
        config_path=config_path,
        dataset_paths=(dataset,),
    )
    assert loaded_config == config
    assert len(provenance) == 1
    assert matrix_data["phase"].size == config.measurements_per_radio
    assert matrix_data["frequency"].dtype == np.int64
    assert matrix_data["gain1"].dtype == np.int64
    assert matrix_data["gain2"].dtype == np.int64

    full_cell = next(spec for spec in MODEL_SPECS if spec.kind == "full_cell")
    summary = _fit_summary(
        full_cell,
        matrix_data,
        radio_count=1,
        gain_count=len(config.gains_db),
        frequency_count=len(config.frequencies_hz),
        reference_gain=config.gains_db.index(config.schedule_reference_gain_db),
        reference_frequency_hz=float(config.frequencies_hz[0]),
    )
    assert summary["total_parameter_count"] == len(config.gain_pairs)


def test_rx2_dc_diagnostic_writes_matched_full_iq_without_v7_mutation(tmp_path):
    config = small_config(
        frequencies_hz=(2_400_000_000,),
        gains_db=(0, 10),
        discard_frames_after_gain=0,
    )
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    output = tmp_path / "dc_diagnostic"

    result = run_rx2_dc_diagnostic(
        config_path=config_path,
        serial="SERIAL-A",
        output_dir=output,
        frequency_hz=2_400_000_000,
        gain_rx1_db=0,
        gain_rx2_values_db=(0, 10),
        frames_per_state=1,
        radio_factory=FakeLoopbackRadio,
        dc_reader=lambda radio: {
            "fake": {
                "tone_active": radio.tone_active,
                "gains": list(radio.gains),
            }
        },
        sleep=lambda _: None,
    )

    assert result["status"] == "complete"
    assert result["record_count"] == result["expected_record_count"] == 4
    assert len(result["on_off_comparisons"]) == 2
    assert len(list((output / "frames").glob("*.npy"))) == 4
    for iq_path in (output / "frames").glob("*.npy"):
        iq = np.load(iq_path, allow_pickle=False)
        assert iq.shape == (2, config.buffer_size)
        assert iq.dtype == np.complex64
    records = [
        json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()
    ]
    assert {record["tx2_enabled"] for record in records} == {False, True}
    assert all(record["frame_metadata"]["gain_metadata_valid"] for record in records)
    assert all(
        record["handoff"]["rf_dc_calibration_before_tone"] is False
        for record in records
        if record["tx2_enabled"]
    )
    assert not (output / "calibration.v7.zarr").exists()


def test_rf_dc_recovery_snapshots_lut_and_never_arms_tx(tmp_path):
    config = small_config(
        frequencies_hz=(2_400_000_000,),
        gains_db=(0, 10),
    )
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    output = tmp_path / "rf_dc_recovery.json"

    result = run_rf_dc_recovery(
        config_path=config_path,
        serial="SERIAL-A",
        output_path=output,
        frequency_hz=2_400_000_000,
        gain_rx1_db=0,
        gain_rx2_values_db=(0, 10),
        radio_factory=FakeLoopbackRadio,
        dc_reader=lambda radio: {
            "fake": {
                "tone_active": radio.tone_active,
                "calibrations": radio.rf_dc_calibrations,
                "gains": list(radio.gains),
            }
        },
        sleep=lambda _: None,
    )

    assert output.is_file()
    assert result["status"] == "complete"
    assert result["tx2_enabled"] is False
    assert all(
        snapshot["correction_banks"]["fake"]["tone_active"] is False
        for snapshot in result["before"] + result["after"]
    )
    assert all(
        snapshot["correction_banks"]["fake"]["calibrations"] == 0
        for snapshot in result["before"]
    )
    assert all(
        snapshot["correction_banks"]["fake"]["calibrations"] == 1
        for snapshot in result["after"]
    )


def test_rf_dc_report_is_deterministic_and_hashes_full_evidence(tmp_path):
    def state(tx2_enabled, gain, dc, clipping, valid):
        return {
            "tx2_enabled": tx2_enabled,
            "gain_rx1_db": 26,
            "gain_rx2_db": gain,
            "frames": 3,
            "quality_valid_frames": valid,
            "median_tone_dbfs": [-30.0, -30.0],
            "median_dc_dbfs": [-70.0, dc],
            "median_clipping_fraction": [0.0, clipping],
            "maximum_clipping_fraction": [0.0, clipping],
        }

    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()
    (before_dir / "frame.npy").write_bytes(b"full IQ before")
    (after_dir / "frame.npy").write_bytes(b"full IQ after")
    common = {
        "status": "complete",
        "serial": "SERIAL-A",
        "frequency_hz": 2_400_000_000,
        "on_off_comparisons": [
            {"gain_rx1_db": 26, "gain_rx2_db": 48},
        ],
    }
    (before_dir / "summary.json").write_text(
        json.dumps(
            {
                **common,
                "state_summaries": [
                    state(False, 48, -5.0, 0.1, 0),
                    state(True, 48, -5.0, 0.1, 0),
                ],
            }
        )
    )
    (after_dir / "summary.json").write_text(
        json.dumps(
            {
                **common,
                "state_summaries": [
                    state(False, 48, -70.0, 0.0, 0),
                    state(True, 48, -70.0, 0.0, 3),
                ],
            }
        )
    )
    recovery_path = tmp_path / "recovery.json"
    correction_before = {
        "A": {
            "correction_words": {
                "rx2_i": {
                    "signed": -511,
                    "is_documented_stuck_value": False,
                },
                "rx2_q": {
                    "signed": -512,
                    "is_documented_stuck_value": True,
                },
            }
        }
    }
    correction_after = {
        "A": {
            "correction_words": {
                "rx2_i": {
                    "signed": 10,
                    "is_documented_stuck_value": False,
                },
                "rx2_q": {
                    "signed": -120,
                    "is_documented_stuck_value": False,
                },
            }
        }
    }
    recovery_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "serial": "SERIAL-A",
                "frequency_hz": 2_400_000_000,
                "gain_rx2_values_db": [48],
                "operation": "Linux IIO calib_mode=rf_dc_offs",
                "operation_started_unix_seconds": 10.0,
                "operation_completed_unix_seconds": 10.08,
                "tx2_enabled": False,
                "before": [
                    {
                        "gain_rx2_db": 48,
                        "correction_banks": correction_before,
                    }
                ],
                "after": [
                    {
                        "gain_rx2_db": 48,
                        "correction_banks": correction_after,
                    }
                ],
            }
        )
    )
    output1 = tmp_path / "report1"
    output2 = tmp_path / "report2"
    first = write_rf_dc_evidence_report(
        before_dir=before_dir,
        recovery_path=recovery_path,
        after_dir=after_dir,
        output_dir=output1,
    )
    second = write_rf_dc_evidence_report(
        before_dir=before_dir,
        recovery_path=recovery_path,
        after_dir=after_dir,
        output_dir=output2,
    )

    assert first == second
    assert first["conclusions"]["rf_dc_recovery_passed"] is True
    assert first["before_failed_rx2_gains_db"] == [48]
    assert first["after_failed_rx2_gains_db"] == []
    assert (output1 / "evidence.json").read_bytes() == (
        output2 / "evidence.json"
    ).read_bytes()
    assert (output1 / "REPORT.md").read_bytes() == (output2 / "REPORT.md").read_bytes()
    assert first["input_evidence"]["before_diagnostic"]["file_count"] == 2


def test_committed_cross_band_configs_cover_three_gain_tables():
    config_dir = (
        Path(__file__).parents[1]
        / "spf"
        / "calibrations"
        / "dual_rx_gain_frequency"
        / "configs"
    )
    expected_frequencies = (
        868_000_000,
        915_000_000,
        1_280_000_000,
        1_320_000_000,
        2_412_000_000,
        2_467_000_000,
        3_990_000_000,
        4_010_000_000,
        5_766_000_000,
        5_804_000_000,
        5_838_000_000,
        5_866_000_000,
    )
    _, pilot = load_calibration_document(config_dir / "pilot_cross_band.yaml")
    _, survey = load_calibration_document(config_dir / "survey_cross_band.yaml")

    assert pilot.frequencies_hz == survey.frequencies_hz == expected_frequencies
    assert pilot.gains_db == (-1, 26, 62)
    assert survey.gains_db == (
        -1,
        0,
        3,
        5,
        6,
        15,
        16,
        17,
        23,
        25,
        26,
        27,
        33,
        34,
        41,
        52,
        62,
    )
    assert pilot.measurements_per_radio == 324
    assert survey.measurements_per_radio == 10_404
    assert survey.tx_gain_db == 0
    for config in (pilot, survey):
        blocks = group_schedule_by_frequency(build_schedule(config))
        assert len(blocks) == 36
        assert all(len(block) == len(config.gains_db) ** 2 for block in blocks)
        assert all(
            {(entry.gain_rx1_db, entry.gain_rx2_db) for entry in block}
            == {
                (gain1, gain2) for gain1 in config.gains_db for gain2 in config.gains_db
            }
            for block in blocks
        )


def test_committed_all_gain_cross_covers_complete_midband_gain_axis():
    config_path = (
        Path(__file__).parents[1]
        / "spf"
        / "calibrations"
        / "dual_rx_gain_frequency"
        / "configs"
        / "all_gain_cross_2p4.yaml"
    )
    _, config = load_calibration_document(config_path)

    assert config.frequencies_hz == (2_412_000_000, 2_467_000_000)
    assert config.gains_db == tuple(range(-3, 72))
    assert config.schedule_design == "additive_cross"
    assert config.schedule_reference_gain_db == 26
    assert len(config.training_gain_pairs) == 149
    assert len(config.held_out_gain_pairs) == 56
    assert config.measurements_per_radio == 1_230


def test_historical_exact_lo_cross_preserves_integer_centres():
    config_path = (
        Path(__file__).parents[1]
        / "spf"
        / "calibrations"
        / "dual_rx_gain_frequency"
        / "configs"
        / "historical_exact_lo_cross_2p4.yaml"
    )
    _, config = load_calibration_document(config_path)

    assert config.frequencies_hz == (2_411_950_000, 2_467_100_000)
    assert config.gains_db == tuple(range(-3, 72))
    assert config.schedule_design == "additive_cross"
    assert config.schedule_reference_gain_db == 26
    assert len(config.training_gain_pairs) == 149
    assert len(config.held_out_gain_pairs) == 56
    assert config.measurements_per_radio == 1_230


def test_frequency_scout_densely_covers_gain_table_boundaries():
    config_path = (
        Path(__file__).parents[1]
        / "spf"
        / "calibrations"
        / "dual_rx_gain_frequency"
        / "configs"
        / "frequency_scout_cross_band.yaml"
    )
    _, scout = load_calibration_document(config_path)

    assert scout.frequencies_hz[0] == 433_000_000
    assert scout.frequencies_hz[-1] == 5_900_000_000
    assert len(scout.frequencies_hz) == 47
    assert scout.gains_db == (-1, 26, 62)
    assert scout.measurements_per_radio == 47 * 9 * 3
    assert {
        1_299_000_000,
        1_300_000_000,
        1_301_000_000,
        3_999_000_000,
        4_000_000_000,
        4_001_000_000,
    }.issubset(scout.frequencies_hz)
    assert scout.tx_gain_db == 0
    blocks = group_schedule_by_frequency(build_schedule(scout))
    assert len(blocks) == 47 * 3
    assert all(len(block) == 9 for block in blocks)


def test_cross_radio_delay_report_recovers_known_difference():
    frequencies = (900_000_000, 1_200_000_000, 2_000_000_000, 3_000_000_000)
    delay_a = 35e-12
    delay_b = 10e-12

    def model(serial, delay):
        return {
            "serial": serial,
            "frequency_models": [
                {
                    "status": "fit",
                    "frequency_hz": frequency,
                    "intercept_rad": -2 * np.pi * frequency * delay + 0.2,
                }
                for frequency in frequencies
            ],
            "frequency_intercept_delay_model": {
                "descriptive_delay_seconds": delay,
                "equivalent_free_space_path_m": delay * 299_792_458.0,
                "fit_residual_metrics": {
                    "circular_mae_deg": 0.0,
                    "circular_rmse_deg": 0.0,
                    "circular_p95_deg": 0.0,
                    "circular_max_deg": 0.0,
                },
            },
        }

    result = compare_radio_models(
        model("SERIAL-A", delay_a),
        model("SERIAL-B", delay_b),
    )
    assert result["schema_version"] == 1
    assert result["common_frequency_count"] == 4
    assert result["global_difference_fit"]["effective_delay_seconds"] == pytest.approx(
        delay_a - delay_b
    )
    assert (
        result["global_difference_fit"]["residual_metrics"]["circular_max_deg"] < 1e-9
    )
    markdown = render_cross_radio_markdown(result)
    assert "literal PCB trace length" in markdown
    assert "25.00 ps" in markdown


def test_cross_radio_transfer_distinguishes_whole_run_and_epoch_anchors():
    config = small_config(gains_db=(0, 26))
    gain_pairs = ((0, 0), (0, 26), (26, 0), (26, 26))
    rx1_effect = {0: 0.0, 26: 0.12}
    rx2_effect = {0: 0.0, 26: -0.08}
    epoch_shift = {0: -0.3, 1: 0.0, 2: 0.3}

    def observations(*, shifted):
        gain1 = []
        gain2 = []
        phase = []
        epochs = []
        for epoch in range(3):
            for value1, value2 in gain_pairs:
                gain1.append(value1)
                gain2.append(value2)
                epochs.append(epoch)
                phase.append(
                    0.4
                    + rx1_effect[value1]
                    + rx2_effect[value2]
                    + (epoch_shift[epoch] if shifted else 0.0)
                )
        return {
            "gain1": np.asarray(gain1),
            "gain2": np.asarray(gain2),
            "phase": np.asarray(phase),
            "epoch": np.asarray(epochs),
        }

    result = _transfer(
        observations(shifted=False),
        observations(shifted=True),
        config=config,
    )
    whole_run = result["single_26_db_equal_gain"]["source_shape_plus_anchor"]
    per_epoch = result["one_26_db_anchor_per_epoch"]
    assert whole_run["circular_mae_deg"] > 10
    assert per_epoch["anchored_epochs"] == [0, 1, 2]
    assert per_epoch["anchor_observations"] == 3
    assert per_epoch["scored_observations"] == 9
    assert per_epoch["source_shape_plus_epoch_anchor"]["circular_max_deg"] < 1e-9


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
