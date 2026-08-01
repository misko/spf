from types import SimpleNamespace

import pytest

from spf.sdrpluto.direct_usb_receiver import (
    DirectUsbRecoveryAttestationError,
    RecoveryAttestationDifference,
)
from tests.radio_hardware import test_interrupted_collection_hardware as harness


def _attestation_error(field):
    return DirectUsbRecoveryAttestationError(
        (RecoveryAttestationDifference(field, "expected", "observed"),)
    )


def test_fresh_probe_retries_only_missing_configuration_attestor(monkeypatch):
    sessions = []

    class Probe:
        def __init__(self, *, serial, protocol_version):
            self.identity = SimpleNamespace(serial=serial, port_path=(1, 2))
            self.index = len(sessions)
            sessions.append(self)
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

        def capture(self, **_kwargs):
            if self.index == 0:
                raise _attestation_error("iio_rx_configuration")
            return SimpleNamespace(frames=[object()])

    monkeypatch.setattr(harness, "PlutoDirectUsbReceiver", Probe)
    radio = SimpleNamespace(serial="serial-1", port_path=(1, 2))

    assert harness._capture_from_fresh_probe_session(radio) == 2
    assert len(sessions) == 2
    assert all(session.closed for session in sessions)


def test_fresh_probe_does_not_retry_identity_attestation_failure(monkeypatch):
    sessions = []

    class Probe:
        def __init__(self, *, serial, protocol_version):
            self.identity = SimpleNamespace(serial=serial, port_path=(1, 2))
            sessions.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def capture(self, **_kwargs):
            raise _attestation_error("usb_serial")

    monkeypatch.setattr(harness, "PlutoDirectUsbReceiver", Probe)
    radio = SimpleNamespace(serial="serial-1", port_path=(1, 2))

    with pytest.raises(DirectUsbRecoveryAttestationError):
        harness._capture_from_fresh_probe_session(radio)
    assert len(sessions) == 1
