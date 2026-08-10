import subprocess

import pytest

from spf.scripts.resolve_pluto_ip import neighbor_candidates, resolve_pluto_ip


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


def test_neighbor_candidates_excludes_addresses_routed_over_usb(monkeypatch):
    def fake_run(command, **kwargs):
        if command == ["ip", "-4", "neigh", "show", "dev", "eth0"]:
            stdout = (
                "192.168.1.165 lladdr 00:11:22:33:44:55 REACHABLE\n"
                "192.168.2.1 lladdr 66:77:88:99:aa:bb STALE\n"
            )
        elif command == ["ip", "-4", "route", "get", "192.168.1.165"]:
            stdout = "192.168.1.165 dev eth0 src 192.168.1.153\n"
        elif command == ["ip", "-4", "route", "get", "192.168.2.1"]:
            stdout = "192.168.2.1 dev eth2 src 192.168.2.10\n"
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert neighbor_candidates("eth0") == ("192.168.1.165",)
