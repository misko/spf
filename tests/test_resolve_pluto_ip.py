import pytest

from spf.scripts.resolve_pluto_ip import resolve_pluto_ip


def test_resolve_pluto_ip_follows_serial_when_preferred_address_changes():
    serials = {
        "192.168.1.163": None,
        "192.168.1.182": "SERIAL-A",
        "192.168.1.200": "SERIAL-OTHER",
    }

    assert resolve_pluto_ip("SERIAL-A", serials, probe=serials.get) == "192.168.1.182"


def test_resolve_pluto_ip_rejects_ambiguous_serial():
    serials = {
        "192.168.1.182": "SERIAL-A",
        "192.168.1.183": "SERIAL-A",
    }
    with pytest.raises(RuntimeError, match="exactly one"):
        resolve_pluto_ip("SERIAL-A", serials, probe=serials.get)
