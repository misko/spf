from types import SimpleNamespace

import pytest

from spf.scripts.mute_pluto_tx import mute_attached_plutos


class FakeSdr:
    def __init__(self, serial):
        self._ctx = SimpleNamespace(attrs={"hw_serial": serial})
        self.tx_hardwaregain_chan0 = -12
        self.tx_hardwaregain_chan1 = -9
        self.tx_enabled_channels = [1]
        self.tx_cyclic_buffer = True
        self.calls = []

    def disable_dds(self):
        self.calls.append("disable_dds")

    def tx_destroy_buffer(self):
        self.calls.append("tx_destroy_buffer")


class FakeAdi:
    def __init__(self, radios):
        self.radios = radios

    def ad9361(self, *, uri):
        return self.radios[uri]


def test_mute_attached_plutos_is_verified_and_serial_selective():
    radios = {
        "usb:1.2.3": FakeSdr("SERIAL-A"),
        "usb:1.2.4": FakeSdr("SERIAL-B"),
    }
    muted = mute_attached_plutos(
        serials=["SERIAL-B"],
        expected_count=1,
        adi_module=FakeAdi(radios),
        scan_contexts=lambda: {
            "usb:1.2.3": "Pluto A",
            "usb:1.2.4": "Pluto B",
            "ip:192.168.2.1": "duplicate network context",
        },
    )

    assert [item.serial for item in muted] == ["SERIAL-B"]
    assert radios["usb:1.2.3"].tx_hardwaregain_chan1 == -9
    selected = radios["usb:1.2.4"]
    assert selected.tx_hardwaregain_chan0 == -80
    assert selected.tx_hardwaregain_chan1 == -80
    assert selected.tx_enabled_channels == []
    assert selected.tx_cyclic_buffer is False
    assert selected.calls == ["disable_dds", "tx_destroy_buffer"]


def test_mute_attached_plutos_fails_closed_on_missing_radio():
    radios = {"usb:1.2.3": FakeSdr("SERIAL-A")}
    with pytest.raises(RuntimeError, match="SERIAL-MISSING"):
        mute_attached_plutos(
            serials=["SERIAL-MISSING"],
            expected_count=1,
            adi_module=FakeAdi(radios),
            scan_contexts=lambda: {"usb:1.2.3": "Pluto A"},
        )
