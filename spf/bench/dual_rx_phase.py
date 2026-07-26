"""Dual-RX common-tone phase characterization across manual gain pairs.

The phase convention is the one used by :func:`spf.rf.get_phase_diff`:

    phase_difference = angle(RX1) - angle(RX2)

All angles on disk are radians unless a field name explicitly ends in
``_deg``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np


SCHEMA_VERSION = 1
PHASE_CONVENTION = "angle(rx1) - angle(rx2)"
GAIN_AVAILABLE_RE = re.compile(
    r"^\[\s*(-?\d+(?:\.\d+)?)\s+" r"(\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\]$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wrap_phase(value):
    """Wrap radians into [-pi, pi)."""

    return (np.asarray(value) + np.pi) % (2 * np.pi) - np.pi


def circular_stats(values: Iterable[float]) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return {
            "mean_rad": None,
            "circular_std_rad": None,
            "resultant_length": None,
        }
    resultant = np.mean(np.exp(1j * values))
    length = float(np.clip(np.abs(resultant), 0.0, 1.0))
    circular_std = float(np.sqrt(-2.0 * np.log(max(length, 1e-15))))
    return {
        "mean_rad": float(np.angle(resultant)),
        "circular_std_rad": circular_std,
        "resultant_length": length,
    }


def parse_gain_available(value: str) -> list[int]:
    """Parse the IIO ``hardwaregain_available`` range.

    AD936x exposes ranges such as ``[-3 1 71]``. This experiment deliberately
    supports only integral-dB gain states because pyadi's manual gain interface
    and the current full tables are integral.
    """

    match = GAIN_AVAILABLE_RE.match(value.strip())
    if match is None:
        raise ValueError(f"unsupported hardwaregain_available value: {value!r}")
    start_float, step_float, end_float = (float(group) for group in match.groups())
    if not all(number.is_integer() for number in (start_float, step_float, end_float)):
        raise ValueError(f"non-integral manual gain range is unsupported: {value!r}")
    start, step, end = int(start_float), int(step_float), int(end_float)
    if step <= 0 or end < start or (end - start) % step:
        raise ValueError(f"invalid manual gain range: {value!r}")
    return list(range(start, end + 1, step))


def select_gain_values(
    available: Iterable[int],
    gain_start: int | None = None,
    gain_end: int | None = None,
    gain_step: int | None = None,
    explicit: Iterable[int] | None = None,
) -> list[int]:
    available = sorted(set(int(value) for value in available))
    if not available:
        raise ValueError("radio reported no manual gain values")
    if explicit is not None:
        selected = [int(value) for value in explicit]
        if not selected:
            raise ValueError("explicit gain list is empty")
        unavailable = sorted(set(selected) - set(available))
        if unavailable:
            raise ValueError(f"requested unavailable gains: {unavailable}")
        return list(dict.fromkeys(selected))
    start = available[0] if gain_start is None else gain_start
    end = available[-1] if gain_end is None else gain_end
    step = 1 if gain_step is None else gain_step
    if step <= 0 or end < start:
        raise ValueError("gain range requires positive step and end >= start")
    selected = list(range(start, end + 1, step))
    unavailable = sorted(set(selected) - set(available))
    if unavailable:
        raise ValueError(f"gain range includes unavailable values: {unavailable[:10]}")
    return selected


def build_schedule(
    gains: Iterable[int],
    repetitions: int,
    captures_per_pair: int,
    seed: int,
    randomize: bool = True,
) -> list[tuple[int, int, int, int]]:
    """Return ``(repetition, gain_rx1, gain_rx2, capture_index)`` entries."""

    gains = list(gains)
    if not gains:
        raise ValueError("at least one gain is required")
    if repetitions <= 0 or captures_per_pair <= 0:
        raise ValueError("repetitions and captures_per_pair must be positive")
    schedule = []
    base_pairs = [(gain_rx1, gain_rx2) for gain_rx1 in gains for gain_rx2 in gains]
    for repetition in range(repetitions):
        pairs = base_pairs.copy()
        if randomize:
            random.Random(seed + repetition).shuffle(pairs)
        for gain_rx1, gain_rx2 in pairs:
            for capture_index in range(captures_per_pair):
                schedule.append((repetition, gain_rx1, gain_rx2, capture_index))
    return schedule


@dataclass(frozen=True)
class ToneQualityThresholds:
    min_tone_snr_db: float = 15.0
    min_tone_dbfs: float = -70.0
    max_tone_dbfs: float = -3.0
    max_clipping_fraction: float = 0.0
    min_coherence: float = 0.98
    max_within_capture_phase_std_deg: float = 5.0


@dataclass(frozen=True)
class SweepConfig:
    lo_hz: int
    sample_rate_hz: int = 2_000_000
    bandwidth_hz: int = 1_000_000
    expected_tone_offset_hz: float = 100_000.0
    tone_search_width_hz: float = 25_000.0
    buffer_size: int = 65_536
    transient_samples: int = 1_024
    phase_segments: int = 8
    repetitions: int = 3
    captures_per_pair: int = 1
    random_seed: int = 20260726
    randomize_pairs: bool = True
    settle_seconds: float = 0.025
    flush_buffers: int = 2
    max_retries: int = 1
    min_quality_valid_per_cell: int = 2
    max_across_repeat_phase_std_deg: float = 5.0
    enable_phase_inversion_mitigation: bool = True
    enable_qec_tracking: bool = True
    source_power_dbm: float | None = None
    setup_label: str = ""
    notes: str = ""
    quality: ToneQualityThresholds = ToneQualityThresholds()

    def validate(self) -> None:
        if self.lo_hz <= 0 or self.sample_rate_hz <= 0 or self.bandwidth_hz <= 0:
            raise ValueError("LO, sample rate, and bandwidth must be positive")
        if self.buffer_size <= 0 or self.transient_samples < 0:
            raise ValueError("buffer and transient sizes are invalid")
        if self.transient_samples >= self.buffer_size:
            raise ValueError("transient_samples must be smaller than buffer_size")
        if self.phase_segments < 2:
            raise ValueError("phase_segments must be at least two")
        if self.buffer_size - self.transient_samples < self.phase_segments * 32:
            raise ValueError("too few samples per phase segment")
        if self.tone_search_width_hz <= 0:
            raise ValueError("tone_search_width_hz must be positive")
        nyquist = self.sample_rate_hz / 2
        if abs(self.expected_tone_offset_hz) + self.tone_search_width_hz >= nyquist:
            raise ValueError("tone search band must fit strictly inside Nyquist")
        if self.repetitions <= 0 or self.captures_per_pair <= 0:
            raise ValueError("repetition counts must be positive")
        expected_per_cell = self.repetitions * self.captures_per_pair
        if not 1 <= self.min_quality_valid_per_cell <= expected_per_cell:
            raise ValueError(
                "min_quality_valid_per_cell must be between one and the "
                "planned measurements per cell"
            )
        if self.max_across_repeat_phase_std_deg <= 0:
            raise ValueError("max_across_repeat_phase_std_deg must be positive")
        if self.settle_seconds < 0 or self.flush_buffers < 0 or self.max_retries < 0:
            raise ValueError("settle, flush, and retry values cannot be negative")

    def as_json(self) -> dict:
        result = asdict(self)
        return result


def _parabolic_peak(power: np.ndarray, index: int) -> float:
    if index <= 0 or index >= power.size - 1:
        return float(index)
    left, center, right = np.log(np.maximum(power[index - 1 : index + 2], 1e-30))
    denominator = left - 2 * center + right
    if abs(denominator) < 1e-15:
        return float(index)
    delta = 0.5 * (left - right) / denominator
    return float(index + np.clip(delta, -0.5, 0.5))


def _matched_amplitude(signal: np.ndarray, frequency_hz: float, sample_rate_hz: int):
    sample_index = np.arange(signal.shape[-1], dtype=np.float64)
    oscillator = np.exp(-2j * np.pi * frequency_hz * sample_index / sample_rate_hz)
    return np.mean(signal * oscillator, axis=-1)


def analyze_common_tone(
    signal_matrix: np.ndarray,
    *,
    sample_rate_hz: int,
    expected_tone_offset_hz: float,
    tone_search_width_hz: float,
    transient_samples: int = 0,
    phase_segments: int = 8,
    thresholds: ToneQualityThresholds = ToneQualityThresholds(),
    adc_full_scale: float = 2048.0,
) -> dict:
    """Measure common-tone phase and quality from one two-channel IQ buffer."""

    signal_matrix = np.asarray(signal_matrix)
    if signal_matrix.ndim != 2 or signal_matrix.shape[0] != 2:
        raise ValueError(
            f"signal_matrix must have shape (2, samples), got {signal_matrix.shape}"
        )
    if signal_matrix.shape[1] - transient_samples < phase_segments * 32:
        raise ValueError("not enough samples after transient removal")
    if not np.isfinite(signal_matrix).all():
        raise ValueError("signal_matrix contains non-finite values")

    raw = signal_matrix[:, transient_samples:].astype(np.complex128, copy=False)
    dc = np.mean(raw, axis=1)
    signal = raw - dc[:, None]
    sample_count = signal.shape[1]
    window = np.hanning(sample_count)
    spectrum = np.fft.fft(signal * window[None, :], axis=1)
    frequencies = np.fft.fftfreq(sample_count, d=1.0 / sample_rate_hz)
    search_mask = np.abs(frequencies - expected_tone_offset_hz) <= tone_search_width_hz
    search_indices = np.flatnonzero(search_mask)
    if search_indices.size < 3:
        raise ValueError("tone search band contains fewer than three FFT bins")
    combined_power = np.sum(np.abs(spectrum[:, search_indices]) ** 2, axis=0)
    local_peak_index = int(np.argmax(combined_power))
    refined_local_index = _parabolic_peak(combined_power, local_peak_index)
    first_bin = int(search_indices[0])
    refined_bin = first_bin + refined_local_index
    tone_frequency_hz = float(
        frequencies[first_bin]
        + (refined_bin - first_bin) * sample_rate_hz / sample_count
    )

    amplitudes = _matched_amplitude(signal, tone_frequency_hz, sample_rate_hz)
    sample_index = np.arange(sample_count, dtype=np.float64)
    fitted_tone = amplitudes[:, None] * np.exp(
        2j * np.pi * tone_frequency_hz * sample_index / sample_rate_hz
    )
    residual = signal - fitted_tone
    tone_power = np.abs(amplitudes) ** 2
    residual_power = np.mean(np.abs(residual) ** 2, axis=1)
    numerical_floor = np.finfo(np.float64).tiny
    with np.errstate(divide="ignore"):
        tone_snr_db = 10 * np.log10(
            np.maximum(tone_power, numerical_floor)
            / np.maximum(residual_power, numerical_floor)
        )
        tone_dbfs = 20 * np.log10(
            np.maximum(np.abs(amplitudes), numerical_floor) / adc_full_scale
        )
        dc_dbfs = 20 * np.log10(
            np.maximum(np.abs(dc), numerical_floor) / adc_full_scale
        )
        amplitude_ratio_db = 20 * np.log10(
            max(np.abs(amplitudes[0]), numerical_floor)
            / max(np.abs(amplitudes[1]), numerical_floor)
        )

    segment_length = sample_count // phase_segments
    segment_amplitudes = []
    segment_phases = []
    for segment_index in range(phase_segments):
        start = segment_index * segment_length
        end = start + segment_length
        segment = signal[:, start:end]
        absolute_sample_index = np.arange(start, end, dtype=np.float64)
        oscillator = np.exp(
            -2j * np.pi * tone_frequency_hz * absolute_sample_index / sample_rate_hz
        )
        segment_amplitude = np.mean(segment * oscillator[None, :], axis=1)
        segment_amplitudes.append(segment_amplitude)
        segment_phases.append(
            float(np.angle(segment_amplitude[0] * np.conj(segment_amplitude[1])))
        )
    segment_amplitudes = np.asarray(segment_amplitudes)
    phase_stats = circular_stats(segment_phases)
    cross = np.mean(segment_amplitudes[:, 0] * np.conj(segment_amplitudes[:, 1]))
    denominator = np.mean(np.abs(segment_amplitudes[:, 0]) ** 2) * np.mean(
        np.abs(segment_amplitudes[:, 1]) ** 2
    )
    coherence = (
        float(np.clip(np.abs(cross) ** 2 / denominator, 0.0, 1.0))
        if denominator > 0
        else 0.0
    )
    clipping_fraction = np.mean(
        (np.abs(raw.real) >= adc_full_scale - 1)
        | (np.abs(raw.imag) >= adc_full_scale - 1),
        axis=1,
    )

    reasons = []
    for channel_index in range(2):
        if tone_snr_db[channel_index] < thresholds.min_tone_snr_db:
            reasons.append(f"rx{channel_index + 1}_tone_snr_low")
        if tone_dbfs[channel_index] < thresholds.min_tone_dbfs:
            reasons.append(f"rx{channel_index + 1}_tone_too_weak")
        if tone_dbfs[channel_index] > thresholds.max_tone_dbfs:
            reasons.append(f"rx{channel_index + 1}_tone_too_strong")
        if clipping_fraction[channel_index] > thresholds.max_clipping_fraction:
            reasons.append(f"rx{channel_index + 1}_clipping")
    if coherence < thresholds.min_coherence:
        reasons.append("cross_channel_coherence_low")
    phase_std_deg = math.degrees(phase_stats["circular_std_rad"])
    if phase_std_deg > thresholds.max_within_capture_phase_std_deg:
        reasons.append("within_capture_phase_unstable")
    if local_peak_index in (0, search_indices.size - 1):
        reasons.append("tone_peak_at_search_edge")

    return {
        "phase_difference_rad": phase_stats["mean_rad"],
        "phase_difference_deg": math.degrees(phase_stats["mean_rad"]),
        "within_capture_phase_std_rad": phase_stats["circular_std_rad"],
        "within_capture_phase_std_deg": phase_std_deg,
        "within_capture_resultant_length": phase_stats["resultant_length"],
        "segment_phase_rad": [float(value) for value in segment_phases],
        "tone_frequency_hz": tone_frequency_hz,
        "tone_frequency_error_hz": tone_frequency_hz - expected_tone_offset_hz,
        "tone_dbfs": [float(value) for value in tone_dbfs],
        "tone_snr_db": [float(value) for value in tone_snr_db],
        "dc_dbfs": [float(value) for value in dc_dbfs],
        "amplitude_ratio_db_rx1_over_rx2": float(amplitude_ratio_db),
        "coherence": coherence,
        "clipping_fraction": [float(value) for value in clipping_fraction],
        "quality_valid": not reasons,
        "quality_reasons": reasons,
    }


def resolve_pluto_uri(
    *, uri: str | None = None, serial: str | None = None, scan_contexts=None
) -> str:
    if uri and serial:
        raise ValueError("specify either URI or serial, not both")
    if uri:
        return uri
    if scan_contexts is None:
        import iio

        scan_contexts = iio.scan_contexts
    contexts = scan_contexts()
    usb_contexts = {
        candidate_uri: description
        for candidate_uri, description in contexts.items()
        if candidate_uri.startswith("usb:")
    }
    if serial:
        matches = [
            candidate_uri
            for candidate_uri, description in usb_contexts.items()
            if f"serial={serial}" in description
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one USB Pluto with serial {serial}, found {matches}"
            )
        return matches[0]
    if len(usb_contexts) != 1:
        raise ValueError(
            "multiple or zero USB Plutos found; select one with --serial or --uri: "
            f"{sorted(usb_contexts)}"
        )
    return next(iter(usb_contexts))


class DualRxRadio(Protocol):
    def identity(self) -> dict:
        ...

    def available_gains(self) -> list[int]:
        ...

    def set_gains(self, gain_rx1: int, gain_rx2: int) -> tuple[int, int]:
        ...

    def read_gains(self) -> tuple[int, int]:
        ...

    def read_gain_indices(self) -> tuple[int, int]:
        ...

    def capture(self) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


class PlutoDualRxRadio:
    """Thin pyadi adapter for one Pluto+/AD9361 in standard USB-IIO mode."""

    def __init__(self, uri: str, config: SweepConfig, adi_module=None):
        config.validate()
        if adi_module is None:
            import adi as adi_module

        self.uri = uri
        self.config = config
        self.sdr = adi_module.ad9361(uri=uri)
        self._configure()

    def _configure(self) -> None:
        sdr = self.sdr
        sdr.rx_destroy_buffer()
        sdr.rx_enabled_channels = [0, 1]
        sdr.sample_rate = int(self.config.sample_rate_hz)
        sdr.rx_rf_bandwidth = int(self.config.bandwidth_hz)
        sdr.rx_lo = int(self.config.lo_hz)
        sdr.gain_control_mode_chan0 = "manual"
        sdr.gain_control_mode_chan1 = "manual"
        sdr.rx_buffer_size = int(self.config.buffer_size)
        sdr._rxadc.set_kernel_buffers_count(1)

        if self.config.enable_phase_inversion_mitigation:
            debug_attr = "adi,rx1-rx2-phase-inversion-enable"
            sdr._ctrl.debug_attrs[debug_attr].value = "1"
            register_0x22 = sdr._ctrl.reg_read(0x22)
            sdr._ctrl.reg_write(0x22, register_0x22 | (1 << 6))
            if not (sdr._ctrl.reg_read(0x22) & (1 << 6)):
                raise RuntimeError("failed to enable AD936x phase inversion mitigation")

        for channel_name in ("voltage0", "voltage1"):
            channel = sdr._ctrl.find_channel(channel_name, is_output=False)
            if "quadrature_tracking_en" in channel.attrs:
                channel.attrs["quadrature_tracking_en"].value = (
                    "1" if self.config.enable_qec_tracking else "0"
                )
        tx_lo = sdr._ctrl.find_channel("altvoltage1", is_output=True)
        tx_lo.attrs["powerdown"].value = "1"
        configured = self.configured_radio()
        expected = {
            "rx_enabled_channels": [0, 1],
            "sample_rate_hz": int(self.config.sample_rate_hz),
            "bandwidth_hz": int(self.config.bandwidth_hz),
            "lo_hz": int(self.config.lo_hz),
            "gain_control_modes": ["manual", "manual"],
        }
        mismatches = {
            name: {"expected": value, "actual": configured[name]}
            for name, value in expected.items()
            if configured[name] != value
        }
        if mismatches:
            raise RuntimeError(f"Pluto configuration readback mismatch: {mismatches}")

    def configured_radio(self) -> dict:
        return {
            "rx_enabled_channels": [
                int(channel) for channel in self.sdr.rx_enabled_channels
            ],
            "sample_rate_hz": int(self.sdr.sample_rate),
            "bandwidth_hz": int(self.sdr.rx_rf_bandwidth),
            "lo_hz": int(self.sdr.rx_lo),
            "gain_control_modes": [
                str(self.sdr.gain_control_mode_chan0),
                str(self.sdr.gain_control_mode_chan1),
            ],
        }

    def identity(self) -> dict:
        attrs = dict(self.sdr._ctx.attrs)
        tracking = {}
        for channel_name in ("voltage0", "voltage1"):
            channel = self.sdr._ctrl.find_channel(channel_name, is_output=False)
            tracking[channel_name] = {
                name: channel.attrs[name].value
                for name in (
                    "bb_dc_offset_tracking_en",
                    "rf_dc_offset_tracking_en",
                    "quadrature_tracking_en",
                )
                if name in channel.attrs
            }
        return {
            "uri": self.uri,
            "context_attrs": attrs,
            "configured_radio": self.configured_radio(),
            "tracking": tracking,
            "phase_inversion_debug_attr": self.sdr._ctrl.debug_attrs[
                "adi,rx1-rx2-phase-inversion-enable"
            ].value,
            "register_0x22": int(self.sdr._ctrl.reg_read(0x22)),
        }

    def available_gains(self) -> list[int]:
        ranges = []
        for channel_name in ("voltage0", "voltage1"):
            channel = self.sdr._ctrl.find_channel(channel_name, is_output=False)
            ranges.append(
                parse_gain_available(channel.attrs["hardwaregain_available"].value)
            )
        if ranges[0] != ranges[1]:
            raise RuntimeError(
                f"RX1/RX2 manual gain ranges differ: {ranges[0]} != {ranges[1]}"
            )
        return ranges[0]

    def read_gains(self) -> tuple[int, int]:
        return (
            int(round(float(self.sdr.rx_hardwaregain_chan0))),
            int(round(float(self.sdr.rx_hardwaregain_chan1))),
        )

    def read_gain_indices(self) -> tuple[int, int]:
        return (
            int(self.sdr._ctrl.reg_read(0x2B0) & 0x7F),
            int(self.sdr._ctrl.reg_read(0x2B5) & 0x7F),
        )

    def set_gains(self, gain_rx1: int, gain_rx2: int) -> tuple[int, int]:
        self.sdr.rx_hardwaregain_chan0 = int(gain_rx1)
        self.sdr.rx_hardwaregain_chan1 = int(gain_rx2)
        readback = self.read_gains()
        if readback != (gain_rx1, gain_rx2):
            raise RuntimeError(
                f"manual gain readback {readback} != {(gain_rx1, gain_rx2)}"
            )
        return readback

    def capture(self) -> np.ndarray:
        return np.vstack(self.sdr.rx()).astype(np.complex64, copy=False)

    def close(self) -> None:
        self.sdr.rx_destroy_buffer()


def _stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _manifest_signature(config: dict, gain_values: list[int], identity: dict) -> str:
    serial = identity.get("context_attrs", {}).get("hw_serial")
    signed = {"config": config, "gain_values": gain_values, "serial": serial}
    return hashlib.sha256(_stable_json(signed).encode()).hexdigest()


def software_provenance() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    provenance = {
        "python": sys.version,
        "numpy": np.__version__,
        "spf_git_sha": None,
        "spf_git_dirty": None,
    }
    try:
        provenance["spf_git_sha"] = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        provenance["spf_git_dirty"] = (
            subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                check=False,
                stdout=subprocess.PIPE,
                text=True,
                stderr=subprocess.DEVNULL,
            ).stdout.strip()
            != ""
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return provenance


def _measurement_key(
    repetition: int, gain_rx1: int, gain_rx2: int, capture_index: int
) -> str:
    return f"{repetition}:{gain_rx1}:{gain_rx2}:{capture_index}"


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(_stable_json(record) + "\n")
        output.flush()
        os.fsync(output.fileno())


def load_observations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    observations = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                observations.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {error}"
                ) from error
    return observations


def initialize_manifest(
    output_dir: Path,
    *,
    config: SweepConfig,
    gain_values: list[int],
    identity: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    config_json = config.as_json()
    signature = _manifest_signature(config_json, gain_values, identity)
    expected = len(gain_values) ** 2 * config.repetitions * config.captures_per_pair
    proposed = {
        "schema": "spf.dual_rx_phase_sweep",
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "phase_convention": PHASE_CONVENTION,
        "config": config_json,
        "gain_values_db": gain_values,
        "radio": identity,
        "software": software_provenance(),
        "run_signature": signature,
        "expected_measurements": expected,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("run_signature") != signature:
            raise ValueError(
                "existing output manifest does not match this radio/config/gain set"
            )
        return existing
    manifest_path.write_text(json.dumps(proposed, indent=2, sort_keys=True) + "\n")
    return proposed


def run_sweep(
    radio: DualRxRadio,
    config: SweepConfig,
    gain_values: list[int],
    output_dir: Path,
    *,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[int, int, tuple[int, int, int, int]], None] | None = None,
) -> dict:
    """Run or resume a sweep and return the generated report."""

    config.validate()
    output_dir = Path(output_dir)
    manifest = initialize_manifest(
        output_dir,
        config=config,
        gain_values=gain_values,
        identity=radio.identity(),
    )
    observations_path = output_dir / "observations.jsonl"
    previous = load_observations(observations_path)
    successful_keys = {
        observation["measurement_key"]
        for observation in previous
        if observation.get("status") == "ok"
    }
    attempts_by_key = Counter(
        observation.get("measurement_key") for observation in previous
    )
    schedule = build_schedule(
        gain_values,
        config.repetitions,
        config.captures_per_pair,
        config.random_seed,
        config.randomize_pairs,
    )

    for schedule_index, entry in enumerate(schedule):
        repetition, gain_rx1, gain_rx2, capture_index = entry
        key = _measurement_key(*entry)
        if key in successful_keys:
            if progress:
                progress(schedule_index + 1, len(schedule), entry)
            continue
        for retry_index in range(config.max_retries + 1):
            attempt = attempts_by_key[key] + 1
            attempts_by_key[key] += 1
            started = time.monotonic()
            base = {
                "schema_version": SCHEMA_VERSION,
                "schedule_index": schedule_index,
                "measurement_key": key,
                "repetition": repetition,
                "capture_index": capture_index,
                "gain_rx1_db": gain_rx1,
                "gain_rx2_db": gain_rx2,
                "attempt": attempt,
                "started_utc": utc_now(),
            }
            try:
                before = radio.set_gains(gain_rx1, gain_rx2)
                sleep(config.settle_seconds)
                for _ in range(config.flush_buffers):
                    radio.capture()
                readback_before = radio.read_gains()
                if readback_before != (gain_rx1, gain_rx2):
                    raise RuntimeError(
                        f"pre-capture gain readback {readback_before} drifted"
                    )
                gain_indices_before = radio.read_gain_indices()
                signal_matrix = radio.capture()
                readback_after = radio.read_gains()
                if readback_after != (gain_rx1, gain_rx2):
                    raise RuntimeError(
                        f"post-capture gain readback {readback_after} drifted"
                    )
                gain_indices_after = radio.read_gain_indices()
                if gain_indices_after != gain_indices_before:
                    raise RuntimeError(
                        "raw gain-table index changed around manual-gain capture: "
                        f"{gain_indices_before} -> {gain_indices_after}"
                    )
                analysis = analyze_common_tone(
                    signal_matrix,
                    sample_rate_hz=config.sample_rate_hz,
                    expected_tone_offset_hz=config.expected_tone_offset_hz,
                    tone_search_width_hz=config.tone_search_width_hz,
                    transient_samples=config.transient_samples,
                    phase_segments=config.phase_segments,
                    thresholds=config.quality,
                )
                record = {
                    **base,
                    "status": "ok",
                    "gain_readback_before_db": list(before),
                    "gain_readback_pre_capture_db": list(readback_before),
                    "gain_readback_after_db": list(readback_after),
                    "gain_index_pre_capture": list(gain_indices_before),
                    "gain_index_after": list(gain_indices_after),
                    "elapsed_seconds": time.monotonic() - started,
                    "analysis": analysis,
                }
                append_jsonl(observations_path, record)
                successful_keys.add(key)
                break
            except Exception as error:
                append_jsonl(
                    observations_path,
                    {
                        **base,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                if retry_index >= config.max_retries:
                    break
        if progress:
            progress(schedule_index + 1, len(schedule), entry)

    return generate_report(output_dir, manifest=manifest)


def _latest_successful_observations(observations: list[dict]) -> dict[str, dict]:
    result = {}
    for observation in observations:
        if observation.get("status") == "ok":
            result[observation["measurement_key"]] = observation
    return result


def _write_matrix_csv(
    path: Path, gains: list[int], cells: dict, field_name: str
) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["gain_rx1_db\\gain_rx2_db", *gains])
        for gain_rx1 in gains:
            writer.writerow(
                [
                    gain_rx1,
                    *[
                        cells.get((gain_rx1, gain_rx2), {}).get(field_name, "")
                        for gain_rx2 in gains
                    ],
                ]
            )


def _write_heatmaps(output_dir: Path, gains: list[int], cells: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    phase = np.full((len(gains), len(gains)), np.nan)
    stability = np.full_like(phase, np.nan)
    valid_fraction = np.full_like(phase, np.nan)
    for row, gain_rx1 in enumerate(gains):
        for column, gain_rx2 in enumerate(gains):
            cell = cells.get((gain_rx1, gain_rx2))
            if cell:
                phase[row, column] = cell["phase_delta_from_reference_deg"]
                stability[row, column] = cell["phase_circular_std_deg"]
                valid_fraction[row, column] = cell["quality_valid_fraction"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    images = [
        axes[0].imshow(
            phase,
            origin="lower",
            aspect="auto",
            cmap="twilight",
            vmin=-180,
            vmax=180,
        ),
        axes[1].imshow(
            stability,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0,
        ),
        axes[2].imshow(
            valid_fraction,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
        ),
    ]
    titles = (
        "Phase delta from reference (deg)",
        "Across-repeat circular std (deg)",
        "Quality-valid fraction",
    )
    maximum_ticks = 12
    tick_stride = max(1, math.ceil(len(gains) / maximum_ticks))
    tick_positions = list(range(0, len(gains), tick_stride))
    if tick_positions[-1] != len(gains) - 1:
        tick_positions.append(len(gains) - 1)
    tick_labels = [gains[position] for position in tick_positions]
    for axis, image, title in zip(axes, images, titles):
        axis.set_title(title)
        axis.set_xlabel("RX2 manual gain (dB)")
        axis.set_ylabel("RX1 manual gain (dB)")
        axis.set_xticks(tick_positions, tick_labels)
        axis.set_yticks(tick_positions, tick_labels)
        figure.colorbar(image, ax=axis)
    figure.savefig(output_dir / "phase_sweep_heatmaps.png", dpi=160)
    plt.close(figure)


def generate_report(output_dir: Path, *, manifest: dict | None = None) -> dict:
    output_dir = Path(output_dir)
    if manifest is None:
        manifest = json.loads((output_dir / "manifest.json").read_text())
    gains = [int(value) for value in manifest["gain_values_db"]]
    observations = load_observations(output_dir / "observations.jsonl")
    latest = _latest_successful_observations(observations)
    grouped = defaultdict(list)
    for observation in latest.values():
        grouped[(observation["gain_rx1_db"], observation["gain_rx2_db"])].append(
            observation
        )

    reference_gain = gains[len(gains) // 2]
    reference_observations = [
        observation
        for observation in grouped.get((reference_gain, reference_gain), [])
        if observation["analysis"]["quality_valid"]
    ]
    if not reference_observations:
        for gain in gains:
            reference_observations = [
                observation
                for observation in grouped.get((gain, gain), [])
                if observation["analysis"]["quality_valid"]
            ]
            if reference_observations:
                reference_gain = gain
                break
    reference_stats = circular_stats(
        observation["analysis"]["phase_difference_rad"]
        for observation in reference_observations
    )
    reference_phase = reference_stats["mean_rad"]

    cells = {}
    expected_per_cell = (
        manifest["config"]["repetitions"] * manifest["config"]["captures_per_pair"]
    )
    for gain_rx1 in gains:
        for gain_rx2 in gains:
            cell_observations = grouped.get((gain_rx1, gain_rx2), [])
            valid = [
                observation
                for observation in cell_observations
                if observation["analysis"]["quality_valid"]
            ]
            stats = circular_stats(
                observation["analysis"]["phase_difference_rad"] for observation in valid
            )
            reasons = Counter(
                reason
                for observation in cell_observations
                for reason in observation["analysis"]["quality_reasons"]
            )
            phase_delta = (
                float(wrap_phase(stats["mean_rad"] - reference_phase))
                if stats["mean_rad"] is not None and reference_phase is not None
                else None
            )

            def median_field(name, channel=None):
                if not cell_observations:
                    return None
                values = [
                    observation["analysis"][name]
                    if channel is None
                    else observation["analysis"][name][channel]
                    for observation in cell_observations
                ]
                return float(np.median(values))

            gain_indices_rx1 = sorted(
                {
                    int(observation["gain_index_after"][0])
                    for observation in cell_observations
                    if observation.get("gain_index_after") is not None
                }
            )
            gain_indices_rx2 = sorted(
                {
                    int(observation["gain_index_after"][1])
                    for observation in cell_observations
                    if observation.get("gain_index_after") is not None
                }
            )
            repeatability_pass = bool(
                len(valid) >= manifest["config"]["min_quality_valid_per_cell"]
                and stats["circular_std_rad"] is not None
                and math.degrees(stats["circular_std_rad"])
                <= manifest["config"]["max_across_repeat_phase_std_deg"]
            )
            cells[(gain_rx1, gain_rx2)] = {
                "gain_rx1_db": gain_rx1,
                "gain_rx2_db": gain_rx2,
                "gain_indices_rx1": gain_indices_rx1,
                "gain_indices_rx2": gain_indices_rx2,
                "n_expected": expected_per_cell,
                "n_ok": len(cell_observations),
                "n_quality_valid": len(valid),
                "quality_valid_fraction": len(valid) / expected_per_cell,
                "repeatability_pass": repeatability_pass,
                "phase_mean_rad": stats["mean_rad"],
                "phase_mean_deg": (
                    math.degrees(stats["mean_rad"])
                    if stats["mean_rad"] is not None
                    else None
                ),
                "phase_delta_from_reference_rad": phase_delta,
                "phase_delta_from_reference_deg": (
                    math.degrees(phase_delta) if phase_delta is not None else None
                ),
                "phase_circular_std_rad": stats["circular_std_rad"],
                "phase_circular_std_deg": (
                    math.degrees(stats["circular_std_rad"])
                    if stats["circular_std_rad"] is not None
                    else None
                ),
                "resultant_length": stats["resultant_length"],
                "median_within_capture_phase_std_deg": median_field(
                    "within_capture_phase_std_deg"
                ),
                "median_coherence": median_field("coherence"),
                "median_tone_dbfs_rx1": median_field("tone_dbfs", 0),
                "median_tone_dbfs_rx2": median_field("tone_dbfs", 1),
                "median_tone_snr_db_rx1": median_field("tone_snr_db", 0),
                "median_tone_snr_db_rx2": median_field("tone_snr_db", 1),
                "quality_reasons": dict(sorted(reasons.items())),
            }

    cell_rows = [cells[pair] for pair in sorted(cells)]
    error_count = sum(
        observation.get("status") == "error" for observation in observations
    )
    expected_measurements = int(manifest["expected_measurements"])
    quality_valid_measurements = sum(
        observation["analysis"]["quality_valid"] for observation in latest.values()
    )
    complete_measurements = len(latest)
    valid_cells = sum(row["n_quality_valid"] > 0 for row in cell_rows)
    minimum_valid = int(manifest["config"]["min_quality_valid_per_cell"])
    maximum_repeat_std = float(manifest["config"]["max_across_repeat_phase_std_deg"])
    passing_cells = sum(row["repeatability_pass"] for row in cell_rows)
    phase_values = [
        row["phase_delta_from_reference_deg"]
        for row in cell_rows
        if row["phase_delta_from_reference_deg"] is not None
    ]
    if complete_measurements != expected_measurements:
        status = "partial"
    elif passing_cells == len(cell_rows):
        status = "pass"
    else:
        status = "fail_quality"
    report = {
        "schema": "spf.dual_rx_phase_sweep.report",
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "status": status,
        "phase_convention": PHASE_CONVENTION,
        "run_signature": manifest["run_signature"],
        "radio_uri": manifest["radio"].get("uri"),
        "radio_serial": manifest["radio"].get("context_attrs", {}).get("hw_serial"),
        "radio_firmware_version": manifest["radio"]
        .get("context_attrs", {})
        .get("fw_version"),
        "spf_git_sha": manifest.get("software", {}).get("spf_git_sha"),
        "spf_git_dirty": manifest.get("software", {}).get("spf_git_dirty"),
        "source_frequency_hz": (
            manifest["config"]["lo_hz"] + manifest["config"]["expected_tone_offset_hz"]
        ),
        "source_power_dbm": manifest["config"].get("source_power_dbm"),
        "setup_label": manifest["config"].get("setup_label", ""),
        "notes": manifest["config"].get("notes", ""),
        "gain_values_db": gains,
        "gain_pair_count": len(gains) ** 2,
        "expected_measurements": expected_measurements,
        "completed_measurements": complete_measurements,
        "quality_valid_measurements": quality_valid_measurements,
        "error_attempts": error_count,
        "valid_cells": valid_cells,
        "passing_cells": passing_cells,
        "min_quality_valid_per_cell": minimum_valid,
        "max_across_repeat_phase_std_deg": maximum_repeat_std,
        "total_cells": len(gains) ** 2,
        "reference_gain_db": (reference_gain if reference_phase is not None else None),
        "reference_phase_rad": reference_phase,
        "reference_phase_deg": (
            math.degrees(reference_phase) if reference_phase is not None else None
        ),
        "phase_delta_span_deg": (
            max(phase_values) - min(phase_values) if phase_values else None
        ),
        "cells": cell_rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "report.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=cell_rows[0].keys())
        writer.writeheader()
        writer.writerows(cell_rows)
    _write_matrix_csv(
        output_dir / "phase_delta_deg.csv",
        gains,
        cells,
        "phase_delta_from_reference_deg",
    )
    _write_matrix_csv(
        output_dir / "phase_circular_std_deg.csv",
        gains,
        cells,
        "phase_circular_std_deg",
    )
    _write_heatmaps(output_dir, gains, cells)

    markdown = [
        "# Dual-RX manual-gain phase sweep",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Pluto serial: `{report['radio_serial']}`",
        f"- Pluto URI: `{report['radio_uri']}`",
        f"- Pluto firmware: `{report['radio_firmware_version']}`",
        (
            f"- SPF commit: `{report['spf_git_sha']}` "
            f"(dirty={report['spf_git_dirty']})"
        ),
        f"- Source frequency: {report['source_frequency_hz']:.0f} Hz",
        f"- Source power: {report['source_power_dbm']} dBm",
        f"- Setup label: `{report['setup_label']}`",
        f"- Notes: {report['notes'] or '(none)'}",
        f"- Phase convention: `{PHASE_CONVENTION}`",
        f"- Gain states: {len(gains)} ({gains[0]} through {gains[-1]} dB)",
        f"- Gain pairs: {report['gain_pair_count']}",
        (
            "- Measurements: "
            f"{complete_measurements}/{expected_measurements} complete; "
            f"{quality_valid_measurements} quality-valid"
        ),
        f"- Cells with at least one valid measurement: {valid_cells}/{len(cells)}",
        (
            "- Passing cells: "
            f"{passing_cells}/{len(cells)} "
            f"(requires at least {minimum_valid} valid measurements and "
            f"≤{maximum_repeat_std:g}° repeat circular std per cell)"
        ),
        f"- Error attempts: {error_count}",
    ]
    if reference_phase is not None:
        markdown.extend(
            [
                (
                    f"- Reference: ({reference_gain}, {reference_gain}) dB = "
                    f"{math.degrees(reference_phase):.3f}°"
                ),
                f"- Valid-cell phase-delta span: {report['phase_delta_span_deg']:.3f}°",
            ]
        )
    else:
        markdown.append("- Reference: unavailable (no quality-valid equal-gain cell)")
    invalid_reason_totals = Counter()
    for row in cell_rows:
        invalid_reason_totals.update(row["quality_reasons"])
    markdown.extend(["", "## Quality exclusions", ""])
    if invalid_reason_totals:
        markdown.extend(
            f"- `{reason}`: {count}"
            for reason, count in invalid_reason_totals.most_common()
        )
    else:
        markdown.append("- None")
    markdown.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `manifest.json`: immutable radio/config/gain provenance",
            "- `observations.jsonl`: append-only resumable measurements",
            "- `report.json`: machine-readable summary and all cells",
            "- `report.csv`: one row per gain pair",
            "- `phase_delta_deg.csv`: RX1-gain × RX2-gain phase matrix",
            "- `phase_circular_std_deg.csv`: across-repeat stability matrix",
            "- `phase_sweep_heatmaps.png`: phase, stability, and validity",
            "",
            "Equal endpoints or a good aggregate do not prove absence of a",
            "within-buffer gain transition in AGC mode. This experiment uses",
            "fixed manual gain and verifies readback around every capture.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(markdown))
    return report
