"""`ardu_cli rc` — the live RC_CHANNELS listener used to debug a radio bind.

The three outcomes are what the operator is actually trying to distinguish, and
they are what the exit code encodes:

    no frames at all  -> the receiver is not reaching the FC (wiring/protocol)
    frames, no motion -> the link works; the problem is at the transmitter
    frames + motion   -> the radio path is good end to end

Getting these backwards would send someone to rewire a rover whose wiring is
fine, so each one is pinned here.
"""

import argparse
from types import SimpleNamespace

import pytest

from spf.ardupilot import ardu_cli


BASELINE = {
    1: 1495,
    2: 1495,
    3: 1515,
    4: 1492,
    5: 1100,
    6: 1500,
    7: 1500,
    8: 1100,
    9: 1500,
    10: 1200,
    11: 1500,
    12: 1500,
}


def rc_frame(values, chancount=12, rssi=200):
    message = SimpleNamespace(chancount=chancount, rssi=rssi)
    for channel in range(1, ardu_cli.RC_MAX_CHANNELS + 1):
        setattr(message, f"chan{channel}_raw", values.get(channel, 0))
    return message


class FakeRCConnection:
    def __init__(self, frames):
        self.frames = list(frames)
        self.closed = False

    def recv_match(self, **_kwargs):
        return self.frames.pop(0) if self.frames else None

    def close(self):
        self.closed = True


def run_rc(monkeypatch, frames, **overrides):
    connection = FakeRCConnection(frames)
    monkeypatch.setattr(
        ardu_cli, "_connect", lambda _args: (connection, None, "/dev/fake")
    )
    monkeypatch.setattr(ardu_cli, "_request_message", lambda *a, **k: None)
    settings = {
        "duration": 0.4,
        "threshold": 15,
        "rate_hz": 10.0,
        "all": False,
        "quiet_tick": 99.0,
        "json": False,
        "json_output": None,
    }
    settings.update(overrides)
    return ardu_cli.command_rc(argparse.Namespace(**settings)), connection


def test_silence_is_a_failure_and_blames_the_receiver_to_fc_path(monkeypatch, capsys):
    code, _ = run_rc(monkeypatch, [])
    output = capsys.readouterr().out
    assert code == 1
    assert "no RC frames at all" in output
    # Must not send the operator to the transmitter: the FC saw nothing, so the
    # fault is downstream of the bind.
    assert "RC_PROTOCOLS" in output


def test_static_frames_fail_and_blame_the_transmitter(monkeypatch, capsys):
    code, _ = run_rc(monkeypatch, [rc_frame(BASELINE) for _ in range(4)])
    output = capsys.readouterr().out
    assert code == 1
    assert "nothing moved" in output
    assert "RxNum" in output


def test_switch_movement_passes_and_names_the_channel(monkeypatch, capsys):
    moved = dict(BASELINE)
    moved[8] = 1900
    code, _ = run_rc(
        monkeypatch, [rc_frame(BASELINE), rc_frame(BASELINE), rc_frame(moved)]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "CH8" in output
    assert "1100 -> 1900" in output
    # The role label is the point of the tool: a number alone does not tell an
    # operator which physical switch they just flipped.
    assert "FLIGHT MODE" in output


def test_movement_below_threshold_is_not_reported(monkeypatch, capsys):
    jittered = dict(BASELINE)
    jittered[1] = BASELINE[1] + 3
    code, _ = run_rc(monkeypatch, [rc_frame(BASELINE), rc_frame(jittered)])
    assert code == 1
    assert "nothing moved" in capsys.readouterr().out


def test_connection_is_closed_even_though_the_listener_loops(monkeypatch):
    _code, connection = run_rc(monkeypatch, [rc_frame(BASELINE)])
    assert connection.closed


def test_json_reports_moved_channels_and_roles(monkeypatch, capsys, tmp_path):
    moved = dict(BASELINE)
    moved[5] = 1900
    code, _ = run_rc(
        monkeypatch, [rc_frame(BASELINE), rc_frame(moved)], json=True
    )
    payload = capsys.readouterr().out
    assert code == 0
    assert '"moved": [' in payload
    assert '"rc_received": true' in payload


def test_padded_channels_past_chancount_are_ignored(monkeypatch):
    """A 8-channel receiver must not report CH9-18 as real zero-valued channels."""
    eight = {channel: BASELINE[channel] for channel in range(1, 9)}
    snapshot = ardu_cli._rc_snapshot(rc_frame(eight, chancount=8))
    assert sorted(snapshot) == list(range(1, 9))


@pytest.mark.parametrize("channel", [5, 8, 9])
def test_safety_critical_channels_are_labelled(channel):
    """An unlabelled arm/mode/shutdown channel makes the tool useless for its job."""
    assert ardu_cli.RC_CHANNEL_ROLES[channel]
