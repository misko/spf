from pathlib import Path

from spf.mavlink.mavlink_controller import tones
from spf.mavlink_radio_collection import (
    READINESS_TONE_INTERVAL_SECONDS,
    maybe_play_readiness_wait_tone,
)


class FakeDrone:
    def __init__(self):
        self.played = []

    def buzzer(self, tone):
        self.played.append(tone)


def test_readiness_tone_plays_at_fifteen_second_intervals(tmp_path: Path):
    drone = FakeDrone()
    disable_path = tmp_path / "disable_annoying_tones"

    next_at = maybe_play_readiness_wait_tone(
        drone, now=14.9, next_tone_at=15.0, disable_path=disable_path
    )
    assert next_at == 15.0
    assert drone.played == []

    next_at = maybe_play_readiness_wait_tone(
        drone, now=15.0, next_tone_at=next_at, disable_path=disable_path
    )
    assert next_at == 15.0 + READINESS_TONE_INTERVAL_SECONDS
    assert drone.played == [tones["readiness-wait"]]

    next_at = maybe_play_readiness_wait_tone(
        drone, now=30.0, next_tone_at=next_at, disable_path=disable_path
    )
    assert next_at == 30.0 + READINESS_TONE_INTERVAL_SECONDS
    assert drone.played == [tones["readiness-wait"], tones["readiness-wait"]]


def test_disable_annoying_tones_flag_suppresses_readiness_chirp(tmp_path: Path):
    drone = FakeDrone()
    disable_path = tmp_path / "disable_annoying_tones"
    disable_path.touch()

    next_at = maybe_play_readiness_wait_tone(
        drone, now=15.0, next_tone_at=15.0, disable_path=disable_path
    )

    assert next_at == 30.0
    assert drone.played == []
