"""Configuration and deterministic scheduling for gain/frequency calibration."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Iterable

from spf.bench.dual_rx_phase import ToneQualityThresholds


@dataclass(frozen=True)
class ScheduleEntry:
    """One frame in a serial-specific V7 calibration dataset."""

    record_index: int
    epoch: int
    frequency_index: int
    lo_frequency_hz: int
    gain_rx1_db: int
    gain_rx2_db: int

    @property
    def key(self) -> str:
        return (
            f"{self.epoch}:{self.frequency_index}:{self.lo_frequency_hz}:"
            f"{self.gain_rx1_db}:{self.gain_rx2_db}"
        )


@dataclass(frozen=True)
class CalibrationConfig:
    frequencies_hz: tuple[int, ...]
    # The normal AD9361 full table above 4 GHz exposes [-10 1 62]. Configs
    # targeting other bands must provide their own explicit common gain set.
    gains_db: tuple[int, ...] = tuple(range(-10, 63))
    repetitions: int = 3
    sample_rate_hz: int = 30_000_000
    bandwidth_hz: int = 3_000_000
    buffer_size: int = 65_536
    tone_offset_hz: float = 100_000.0
    tone_search_width_hz: float = 25_000.0
    transient_samples: int = 1_024
    phase_segments: int = 8
    settle_seconds: float = 0.025
    frequency_settle_seconds: float = 0.5
    discard_frames_after_gain: int = 1
    max_retries: int = 1
    random_seed: int = 20260727
    tx_gain_db: float = -30.0
    tx_gain_policy: str = "fixed"
    tx_reference_rx_gain_db: int = 34
    tx_min_gain_db: float = -80.0
    tx_max_gain_db: float = 0.0
    tx_digital_amplitude: float = 8_192.0
    min_quality_valid_per_cell: int = 2
    max_across_repeat_phase_std_deg: float = 5.0
    quality: ToneQualityThresholds = field(default_factory=ToneQualityThresholds)
    setup_label: str = ""
    notes: str = ""

    def validate(self) -> None:
        if not self.frequencies_hz:
            raise ValueError("at least one frequency is required")
        if len(set(self.frequencies_hz)) != len(self.frequencies_hz):
            raise ValueError("frequencies must be unique")
        if any(frequency <= 0 for frequency in self.frequencies_hz):
            raise ValueError("frequencies must be positive")
        if not self.gains_db:
            raise ValueError("at least one gain is required")
        if len(set(self.gains_db)) != len(self.gains_db):
            raise ValueError("gains must be unique")
        if self.repetitions != 3:
            raise ValueError("this calibration requires exactly three epochs")
        if self.sample_rate_hz <= 0 or self.bandwidth_hz <= 0:
            raise ValueError("sample rate and bandwidth must be positive")
        if self.buffer_size <= 0 or self.transient_samples < 0:
            raise ValueError("buffer and transient sizes are invalid")
        if self.transient_samples >= self.buffer_size:
            raise ValueError("transient samples must be smaller than the buffer")
        if self.phase_segments < 2:
            raise ValueError("at least two phase segments are required")
        if self.buffer_size - self.transient_samples < self.phase_segments * 32:
            raise ValueError("too few samples per phase segment")
        nyquist = self.sample_rate_hz / 2
        if abs(self.tone_offset_hz) + self.tone_search_width_hz >= nyquist:
            raise ValueError("tone search band must fit inside Nyquist")
        if self.settle_seconds < 0 or self.frequency_settle_seconds < 0:
            raise ValueError("settling times cannot be negative")
        if self.discard_frames_after_gain < 0 or self.max_retries < 0:
            raise ValueError("discard and retry counts cannot be negative")
        if not 1 <= self.min_quality_valid_per_cell <= self.repetitions:
            raise ValueError("minimum valid count must fit within the epochs")
        if self.max_across_repeat_phase_std_deg <= 0:
            raise ValueError("repeat phase threshold must be positive")
        if not 0 < self.tx_digital_amplitude <= 2**14:
            raise ValueError("TX digital amplitude must be in (0, 16384]")
        if self.tx_gain_policy not in ("fixed", "adaptive_max_rx_gain"):
            raise ValueError(f"unsupported TX gain policy: {self.tx_gain_policy}")
        if self.tx_min_gain_db > self.tx_max_gain_db:
            raise ValueError("TX gain limits are reversed")
        if not self.tx_min_gain_db <= self.tx_gain_db <= self.tx_max_gain_db:
            raise ValueError("reference TX gain is outside the configured limits")

    @property
    def measurements_per_radio(self) -> int:
        return self.repetitions * len(self.frequencies_hz) * len(self.gains_db) ** 2

    def as_json(self) -> dict:
        result = asdict(self)
        result["frequencies_hz"] = list(self.frequencies_hz)
        result["gains_db"] = list(self.gains_db)
        return result

    def tx_gain_for(self, gain_rx1_db: int, gain_rx2_db: int) -> float:
        """Choose a tracked source level while keeping the stronger RX near target."""

        if self.tx_gain_policy == "fixed":
            return float(self.tx_gain_db)
        strongest_rx_gain = max(gain_rx1_db, gain_rx2_db)
        requested = self.tx_gain_db - (strongest_rx_gain - self.tx_reference_rx_gain_db)
        return float(min(self.tx_max_gain_db, max(self.tx_min_gain_db, requested)))

    @property
    def signature(self) -> str:
        encoded = json.dumps(
            self.as_json(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _frequency_order(
    frequencies: tuple[int, ...],
    epoch: int,
    seed: int,
    previous_last: int | None,
) -> list[tuple[int, int]]:
    indexed = list(enumerate(frequencies))
    random.Random(seed + epoch * 1_000_003).shuffle(indexed)
    if (
        len(indexed) > 1
        and previous_last is not None
        and indexed[0][1] == previous_last
    ):
        indexed = indexed[1:] + indexed[:1]
    return indexed


def _pair_seed(seed: int, epoch: int, frequency_index: int) -> int:
    payload = f"{seed}:{epoch}:{frequency_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def build_schedule(config: CalibrationConfig) -> list[ScheduleEntry]:
    """Build three separated epochs of complete frequency/gain Cartesian scans."""

    config.validate()
    schedule: list[ScheduleEntry] = []
    previous_last_frequency = None
    gains = tuple(config.gains_db)
    base_pairs = [(gain1, gain2) for gain1 in gains for gain2 in gains]
    for epoch in range(config.repetitions):
        frequency_order = _frequency_order(
            config.frequencies_hz,
            epoch,
            config.random_seed,
            previous_last_frequency,
        )
        for frequency_index, frequency_hz in frequency_order:
            pairs = base_pairs.copy()
            random.Random(
                _pair_seed(config.random_seed, epoch, frequency_index)
            ).shuffle(pairs)
            for gain_rx1, gain_rx2 in pairs:
                schedule.append(
                    ScheduleEntry(
                        record_index=len(schedule),
                        epoch=epoch,
                        frequency_index=frequency_index,
                        lo_frequency_hz=frequency_hz,
                        gain_rx1_db=gain_rx1,
                        gain_rx2_db=gain_rx2,
                    )
                )
        previous_last_frequency = frequency_order[-1][1]
    return schedule


def group_schedule_by_frequency(
    schedule: Iterable[ScheduleEntry],
) -> list[list[ScheduleEntry]]:
    """Group adjacent schedule entries into epoch/frequency blocks."""

    groups: list[list[ScheduleEntry]] = []
    for entry in schedule:
        if (
            not groups
            or groups[-1][0].epoch != entry.epoch
            or groups[-1][0].frequency_index != entry.frequency_index
        ):
            groups.append([])
        groups[-1].append(entry)
    return groups
