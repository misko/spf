from types import SimpleNamespace

import pytest

from spf.scripts import pluto_gadget_build_id
from spf.sdrpluto.direct_usb_protocol import HardwareIdentityFlags


def _receiver_with_identity(monkeypatch, *, flags, gadget_build_id):
    identity = SimpleNamespace(flags=flags, gadget_build_id=gadget_build_id)

    class FakeReceiver:
        def __init__(self, **kwargs):
            assert kwargs == {"serial": "SERIAL_A", "protocol_version": 2}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def query_hardware_identity(self):
            return identity

    monkeypatch.setattr(pluto_gadget_build_id, "PlutoDirectUsbReceiver", FakeReceiver)


def test_reads_valid_passive_gadget_build_id(monkeypatch):
    expected = "27a7eed7b6abcaf1b9c78f7978bf743a7a315325"
    _receiver_with_identity(
        monkeypatch,
        flags=HardwareIdentityFlags.GADGET_BUILD_ID_VALID,
        gadget_build_id=expected,
    )

    assert pluto_gadget_build_id.read_gadget_build_id("SERIAL_A") == expected


def test_fails_closed_when_gadget_build_identity_is_not_valid(monkeypatch):
    _receiver_with_identity(
        monkeypatch,
        flags=HardwareIdentityFlags(0),
        gadget_build_id="",
    )

    with pytest.raises(RuntimeError, match="gadget build identity is unavailable"):
        pluto_gadget_build_id.read_gadget_build_id("SERIAL_A")
