"""Deterministic scheduling helpers for the mixed USB/IP radio burn-in."""

from __future__ import annotations

import dataclasses
import random


@dataclasses.dataclass(frozen=True, slots=True)
class BurnStep:
    epoch: int
    frequency_hz: int
    transports: tuple[str, str, str]
    gain_modes: tuple[str, str, str]


def parse_frequency_list(value: str) -> tuple[int, ...]:
    """Parse unique positive frequencies with optional K/M/G suffixes."""

    frequencies = []
    scale = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    for raw_token in value.split(","):
        token = raw_token.strip().lower()
        if not token:
            raise ValueError("frequency list contains an empty value")
        multiplier = scale.get(token[-1], 1)
        number = token[:-1] if multiplier != 1 else token
        try:
            frequency = int(round(float(number) * multiplier))
        except ValueError as error:
            raise ValueError(f"invalid frequency: {raw_token!r}") from error
        if frequency <= 0:
            raise ValueError("frequencies must be positive")
        frequencies.append(frequency)
    if not frequencies:
        raise ValueError("at least one frequency is required")
    if len(set(frequencies)) != len(frequencies):
        raise ValueError("frequencies must be unique")
    return tuple(frequencies)


def build_burn_schedule(
    frequencies_hz: tuple[int, ...], *, epochs: int, seed: int = 20260810
) -> tuple[BurnStep, ...]:
    """Return interleaved frequency and transport transitions.

    Every epoch uses a deterministic shuffle instead of immediately repeating
    one frequency. Adjacent transport sessions alternate USB/IP and each cell
    changes manual -> slow-attack -> manual gain state.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not frequencies_hz or len(set(frequencies_hz)) != len(frequencies_hz):
        raise ValueError("frequencies must be non-empty and unique")
    schedule = []
    previous_frequency = None
    for epoch in range(epochs):
        order = list(frequencies_hz)
        random.Random(seed + epoch).shuffle(order)
        if len(order) > 1 and order[0] == previous_frequency:
            order[0], order[1] = order[1], order[0]
        for index, frequency_hz in enumerate(order):
            usb_first = (epoch + index) % 2 == 0
            schedule.append(
                BurnStep(
                    epoch=epoch,
                    frequency_hz=frequency_hz,
                    transports=("usb", "ip", "usb")
                    if usb_first
                    else ("ip", "usb", "ip"),
                    gain_modes=("manual_26", "slow_attack", "manual_41"),
                )
            )
        previous_frequency = order[-1]
    return tuple(schedule)
