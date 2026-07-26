import json
import math

import numpy as np
import pytest

from spf.bench.dual_rx_phase import (
    PHASE_CONVENTION,
    SweepConfig,
    ToneQualityThresholds,
    analyze_common_tone,
    build_schedule,
    circular_stats,
    generate_report,
    parse_gain_available,
    resolve_pluto_uri,
    run_sweep,
    select_gain_values,
    wrap_phase,
)


def synthetic_tone(
    *,
    phase_difference=0.8,
    sample_rate=2_000_000,
    tone_frequency=100_000,
    samples=16_384,
    amplitude_rx1=300.0,
    amplitude_rx2=240.0,
    noise_std=2.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    sample_index = np.arange(samples)
    common_phase = 0.37
    rx1_phase = common_phase + phase_difference / 2
    rx2_phase = common_phase - phase_difference / 2
    rx1 = amplitude_rx1 * np.exp(
        1j * (2 * np.pi * tone_frequency * sample_index / sample_rate + rx1_phase)
    )
    rx2 = amplitude_rx2 * np.exp(
        1j * (2 * np.pi * tone_frequency * sample_index / sample_rate + rx2_phase)
    )
    noise = noise_std * (
        rng.normal(size=(2, samples)) + 1j * rng.normal(size=(2, samples))
    )
    dc = np.array([[8 + 4j], [-5 + 2j]])
    return np.vstack((rx1, rx2)) + noise + dc


def test_gain_range_parser_and_selection():
    assert parse_gain_available("[-3 1 71]") == list(range(-3, 72))
    assert parse_gain_available("[-10 2 62]") == list(range(-10, 63, 2))
    with pytest.raises(ValueError, match="non-integral"):
        parse_gain_available("[-3 0.5 71]")
    available = list(range(-3, 72))
    assert select_gain_values(available, gain_start=0, gain_end=20, gain_step=10) == [
        0,
        10,
        20,
    ]
    assert select_gain_values(available, explicit=[20, -3, 20]) == [20, -3]
    with pytest.raises(ValueError, match="unavailable"):
        select_gain_values(available, explicit=[72])


def test_schedule_is_complete_repeatable_and_randomized_per_repetition():
    schedule = build_schedule([0, 10, 20], 2, 2, seed=42, randomize=True)
    assert len(schedule) == 3**2 * 2 * 2
    for repetition in range(2):
        pairs = [
            (gain_rx1, gain_rx2)
            for rep, gain_rx1, gain_rx2, capture in schedule
            if rep == repetition and capture == 0
        ]
        assert set(pairs) == {
            (gain_rx1, gain_rx2) for gain_rx1 in (0, 10, 20) for gain_rx2 in (0, 10, 20)
        }
    first_order = [entry[1:3] for entry in schedule if entry[0] == 0]
    second_order = [entry[1:3] for entry in schedule if entry[0] == 1]
    assert first_order != second_order
    assert schedule == build_schedule([0, 10, 20], 2, 2, 42, True)


def test_common_tone_analysis_recovers_phase_sign_frequency_and_amplitude():
    expected_phase = 0.9
    signal = synthetic_tone(
        phase_difference=expected_phase,
        tone_frequency=103_250,
        amplitude_rx1=320,
        amplitude_rx2=160,
    )
    result = analyze_common_tone(
        signal,
        sample_rate_hz=2_000_000,
        expected_tone_offset_hz=100_000,
        tone_search_width_hz=10_000,
        transient_samples=256,
        phase_segments=8,
    )
    assert PHASE_CONVENTION == "angle(rx1) - angle(rx2)"
    assert float(wrap_phase(result["phase_difference_rad"] - expected_phase)) == (
        pytest.approx(0, abs=2e-3)
    )
    assert result["tone_frequency_hz"] == pytest.approx(103_250, abs=30)
    assert result["amplitude_ratio_db_rx1_over_rx2"] == pytest.approx(
        20 * math.log10(2), abs=0.1
    )
    assert result["coherence"] > 0.999
    assert result["within_capture_phase_std_deg"] < 0.1
    assert result["quality_valid"]
    assert result["quality_reasons"] == []


def test_quality_gate_labels_clipping_and_excessive_tone_level():
    signal = synthetic_tone(
        amplitude_rx1=2_500,
        amplitude_rx2=2_500,
        noise_std=1,
    )
    result = analyze_common_tone(
        signal,
        sample_rate_hz=2_000_000,
        expected_tone_offset_hz=100_000,
        tone_search_width_hz=10_000,
        thresholds=ToneQualityThresholds(max_tone_dbfs=-3),
    )
    assert not result["quality_valid"]
    assert "rx1_tone_too_strong" in result["quality_reasons"]
    assert "rx2_tone_too_strong" in result["quality_reasons"]
    assert "rx1_clipping" in result["quality_reasons"]
    assert "rx2_clipping" in result["quality_reasons"]


def test_circular_aggregation_handles_wrap_boundary():
    values = np.deg2rad([179, -179, 178, -178])
    result = circular_stats(values)
    assert abs(abs(math.degrees(result["mean_rad"])) - 180) < 0.1
    assert math.degrees(result["circular_std_rad"]) < 2


def test_serial_uri_resolution_is_stable_and_ambiguous_default_fails():
    contexts = {
        "usb:1.4.5": "Pluto, serial=AAA",
        "usb:1.9.5": "Pluto, serial=BBB",
        "ip:pluto.local": "Pluto, serial=AAA",
    }
    scan = lambda: contexts
    assert resolve_pluto_uri(serial="BBB", scan_contexts=scan) == "usb:1.9.5"
    assert resolve_pluto_uri(uri="usb:9.9.9", scan_contexts=scan) == "usb:9.9.9"
    with pytest.raises(ValueError, match="multiple or zero"):
        resolve_pluto_uri(scan_contexts=scan)
    with pytest.raises(ValueError, match="either URI or serial"):
        resolve_pluto_uri(uri="usb:1.4.5", serial="AAA", scan_contexts=scan)


class FakeDualRxRadio:
    def __init__(self, config, gains, fail_first_key=None):
        self.config = config
        self.gains = tuple(gains)
        self.capture_calls = 0
        self.set_calls = []
        self.closed = False
        self.fail_first_key = fail_first_key
        self.failed = False

    def identity(self):
        return {
            "uri": "fake:",
            "context_attrs": {"hw_serial": "FAKE-SERIAL", "fw_version": "fake"},
            "tracking": {},
        }

    def available_gains(self):
        return [0, 10]

    def set_gains(self, gain_rx1, gain_rx2):
        self.gains = (gain_rx1, gain_rx2)
        self.set_calls.append(self.gains)
        return self.gains

    def read_gains(self):
        return self.gains

    def read_gain_indices(self):
        return self.gains[0] + 3, self.gains[1] + 3

    def capture(self):
        self.capture_calls += 1
        if self.fail_first_key == self.gains and not self.failed:
            self.failed = True
            raise OSError("synthetic one-shot USB error")
        phase = 0.01 * self.gains[0] - 0.02 * self.gains[1]
        return synthetic_tone(
            phase_difference=phase,
            samples=self.config.buffer_size,
            sample_rate=self.config.sample_rate_hz,
            tone_frequency=self.config.expected_tone_offset_hz,
            seed=self.capture_calls,
        )

    def close(self):
        self.closed = True


class UnstableFakeDualRxRadio(FakeDualRxRadio):
    def capture(self):
        self.capture_calls += 1
        measurement = (self.capture_calls - 1) // (self.config.flush_buffers + 1)
        phase = -0.5 if measurement % 2 == 0 else 0.5
        return synthetic_tone(
            phase_difference=phase,
            samples=self.config.buffer_size,
            sample_rate=self.config.sample_rate_hz,
            tone_frequency=self.config.expected_tone_offset_hz,
            seed=self.capture_calls,
        )


class DriftingGainIndexFakeDualRxRadio(FakeDualRxRadio):
    def __init__(self, config, gains):
        super().__init__(config, gains)
        self.index_reads = 0

    def read_gain_indices(self):
        self.index_reads += 1
        offset = 0 if self.index_reads % 2 else 1
        return self.gains[0] + 3 + offset, self.gains[1] + 3


def small_config():
    return SweepConfig(
        lo_hz=2_400_000_000,
        sample_rate_hz=2_000_000,
        bandwidth_hz=1_000_000,
        expected_tone_offset_hz=100_000,
        tone_search_width_hz=10_000,
        buffer_size=4_096,
        transient_samples=128,
        phase_segments=4,
        repetitions=2,
        captures_per_pair=1,
        randomize_pairs=True,
        settle_seconds=0,
        flush_buffers=1,
        max_retries=1,
    )


def test_sweep_is_resumable_retries_errors_and_writes_reports(tmp_path):
    config = small_config()
    radio = FakeDualRxRadio(config, (0, 0), fail_first_key=(10, 0))
    report = run_sweep(
        radio,
        config,
        [0, 10],
        tmp_path,
        sleep=lambda _: None,
    )
    assert report["status"] == "pass"
    assert report["expected_measurements"] == 8
    assert report["completed_measurements"] == 8
    assert report["quality_valid_measurements"] == 8
    assert report["error_attempts"] == 1
    assert report["valid_cells"] == 4
    assert report["passing_cells"] == 4
    assert report["source_frequency_hz"] == 2_400_100_000
    assert report["radio_firmware_version"] == "fake"
    assert report["reference_gain_db"] == 10
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "observations.jsonl").exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.csv").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "phase_delta_deg.csv").exists()
    assert (tmp_path / "phase_circular_std_deg.csv").exists()
    assert (tmp_path / "phase_sweep_heatmaps.png").exists()
    observations = [
        json.loads(line)
        for line in (tmp_path / "observations.jsonl").read_text().splitlines()
    ]
    assert sum(item["status"] == "ok" for item in observations) == 8
    assert sum(item["status"] == "error" for item in observations) == 1
    successful = [item for item in observations if item["status"] == "ok"]
    assert {item["schedule_index"] for item in successful} == set(range(8))
    assert all("gain_index_pre_capture" in item for item in successful)
    assert all(
        item["gain_index_pre_capture"] == item["gain_index_after"]
        for item in successful
    )
    capture_calls = radio.capture_calls

    resumed = run_sweep(
        radio,
        config,
        [0, 10],
        tmp_path,
        sleep=lambda _: None,
    )
    assert resumed["completed_measurements"] == 8
    assert radio.capture_calls == capture_calls


def test_report_regeneration_preserves_summary(tmp_path):
    config = small_config()
    radio = FakeDualRxRadio(config, (0, 0))
    original = run_sweep(radio, config, [0, 10], tmp_path, sleep=lambda _: None)
    regenerated = generate_report(tmp_path)
    for key in (
        "status",
        "run_signature",
        "completed_measurements",
        "quality_valid_measurements",
        "reference_phase_rad",
    ):
        assert regenerated[key] == original[key]


def test_fully_captured_sweep_fails_when_no_cell_meets_quality_gate(tmp_path):
    config = SweepConfig(
        **{
            **small_config().__dict__,
            "quality": ToneQualityThresholds(min_tone_dbfs=0.0),
        }
    )
    radio = FakeDualRxRadio(config, (0, 0))
    report = run_sweep(radio, config, [0, 10], tmp_path, sleep=lambda _: None)
    assert report["completed_measurements"] == report["expected_measurements"]
    assert report["quality_valid_measurements"] == 0
    assert report["passing_cells"] == 0
    assert report["status"] == "fail_quality"


def test_fully_captured_sweep_fails_repeatability_gate(tmp_path):
    config = small_config()
    radio = UnstableFakeDualRxRadio(config, (0, 0))
    report = run_sweep(radio, config, [0], tmp_path, sleep=lambda _: None)
    assert report["quality_valid_measurements"] == 2
    assert report["cells"][0]["phase_circular_std_deg"] > 5
    assert not report["cells"][0]["repeatability_pass"]
    assert report["passing_cells"] == 0
    assert report["status"] == "fail_quality"


def test_gain_index_drift_is_retried_and_never_accepted(tmp_path):
    config = small_config()
    radio = DriftingGainIndexFakeDualRxRadio(config, (0, 0))
    report = run_sweep(radio, config, [0], tmp_path, sleep=lambda _: None)
    assert report["status"] == "partial"
    assert report["completed_measurements"] == 0
    assert report["error_attempts"] == 4
    observations = [
        json.loads(line)
        for line in (tmp_path / "observations.jsonl").read_text().splitlines()
    ]
    assert all(item["status"] == "error" for item in observations)
    assert all("raw gain-table index changed" in item["error"] for item in observations)


def test_resume_rejects_changed_configuration(tmp_path):
    config = small_config()
    radio = FakeDualRxRadio(config, (0, 0))
    run_sweep(radio, config, [0, 10], tmp_path, sleep=lambda _: None)
    changed = SweepConfig(**{**config.__dict__, "repetitions": 3})
    with pytest.raises(ValueError, match="manifest does not match"):
        run_sweep(radio, changed, [0, 10], tmp_path, sleep=lambda _: None)
