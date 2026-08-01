from pathlib import Path

import numpy as np
import pytest

from spf.scripts import validate_restart_soak as soak


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, key):
        return self.values[key]


class _Receiver:
    def __init__(self, *, serial, fingerprint, session, port, timestamps):
        self.attrs = {
            "sdr_serial": serial,
            "usb_port_path": port,
            "hardware_fingerprint_v1": {
                "stable_fingerprint_sha256": fingerprint,
                "fingerprint_session_id": session,
            },
        }
        self.system_timestamp = _Array(timestamps)

    def __getitem__(self, key):
        return getattr(self, key)


class _Store:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Capture:
    def __init__(self, receiver, status="complete"):
        self.attrs = {"capture_status": status}
        self.receivers = {"r0": receiver}
        self.store = _Store()

    def __getitem__(self, key):
        if key == "receivers":
            return self.receivers
        return self.receivers[key.removeprefix("receivers/")]


def _capture(session, *, serial="SERIAL", fingerprint="FINGERPRINT", port=(1, 2)):
    return _Capture(
        _Receiver(
            serial=serial,
            fingerprint=fingerprint,
            session=session,
            port=port,
            timestamps=[0.0, 5.0, 10.0],
        )
    )


def test_restart_soak_requires_multiple_fresh_fingerprint_sessions(monkeypatch):
    captures = {Path("one"): _capture("SESSION_1"), Path("two"): _capture("SESSION_2")}
    monkeypatch.setattr(
        soak,
        "zarr_open_from_lmdb_store",
        lambda path, **kwargs: captures[Path(path)],
    )

    result = soak.validate(list(captures), minimum_sessions=2, minimum_seconds=20)

    assert result["status"] == "pass"
    assert result["total_capture_seconds"] == 20
    assert result["radio_identity_bindings"][0]["serial"] == "SERIAL"
    assert all(capture.store.closed for capture in captures.values())


def test_restart_soak_reports_insufficient_duration_without_claiming_pass(monkeypatch):
    capture = _capture("SESSION_1")
    monkeypatch.setattr(
        soak, "zarr_open_from_lmdb_store", lambda path, **kwargs: capture
    )

    result = soak.validate([Path("one")], minimum_sessions=2, minimum_seconds=20)

    assert result["status"] == "incomplete"


@pytest.mark.parametrize(
    "second",
    [
        _capture("SESSION_2", serial="OTHER"),
        _capture("SESSION_2", fingerprint="OTHER"),
        _capture("SESSION_2", port=(1, 3)),
    ],
)
def test_restart_soak_fails_closed_on_durable_identity_change(monkeypatch, second):
    captures = {Path("one"): _capture("SESSION_1"), Path("two"): second}
    monkeypatch.setattr(
        soak,
        "zarr_open_from_lmdb_store",
        lambda path, **kwargs: captures[Path(path)],
    )

    with pytest.raises(ValueError, match="durable radio identity changed"):
        soak.validate(list(captures), minimum_sessions=2, minimum_seconds=20)


def test_restart_soak_rejects_reused_fingerprint_session(monkeypatch):
    captures = {Path("one"): _capture("REUSED"), Path("two"): _capture("REUSED")}
    monkeypatch.setattr(
        soak,
        "zarr_open_from_lmdb_store",
        lambda path, **kwargs: captures[Path(path)],
    )

    with pytest.raises(ValueError, match="reused across restarts"):
        soak.validate(list(captures), minimum_sessions=2, minimum_seconds=20)
