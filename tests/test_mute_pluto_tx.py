from types import SimpleNamespace

import pytest

from spf.scripts.mute_pluto_tx import mute_attached_plutos, mute_sdr_tx


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


class OrderSensitiveFakeSdr:
    """Model pyadi's enabled-channel lookup for per-channel TX properties."""

    def __init__(self):
        self._tx1 = -12
        self._tx2 = -9
        self.tx_enabled_channels = [0, 1]
        self.tx_cyclic_buffer = True
        self.calls = []

    @property
    def tx_hardwaregain_chan0(self):
        return self._tx1

    @tx_hardwaregain_chan0.setter
    def tx_hardwaregain_chan0(self, value):
        if 0 in self.tx_enabled_channels:
            self._tx1 = value

    @property
    def tx_hardwaregain_chan1(self):
        return self._tx2

    @tx_hardwaregain_chan1.setter
    def tx_hardwaregain_chan1(self, value):
        if 1 in self.tx_enabled_channels:
            self._tx2 = value

    def disable_dds(self):
        self.calls.append("disable_dds")

    def tx_destroy_buffer(self):
        self.calls.append("tx_destroy_buffer")


def test_mute_sdr_tx_attenuates_before_disabling_channel_lookup():
    sdr = OrderSensitiveFakeSdr()

    assert mute_sdr_tx(sdr) == (-80.0, -80.0)
    assert sdr.tx_enabled_channels == []
    assert sdr.tx_cyclic_buffer is False


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
