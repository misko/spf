from tests.radio_hardware import conftest as hardware


class _FakeDevice:
    def __init__(self, *, address, serial=None):
        self._address = address
        self._serial = serial

    def getVendorID(self):
        return hardware.PLUTO_VENDOR_ID

    def getProductID(self):
        return hardware.PLUTO_PRODUCT_ID

    def getDeviceAddress(self):
        return self._address

    def getSerialNumber(self):
        assert self._address != 0, "address-zero ghost must not be opened"
        return self._serial

    def getBusNumber(self):
        return 3

    def getPortNumberList(self):
        return [10, 2]


class _FakeContext:
    def open(self):
        return None

    def close(self):
        return None

    def getDeviceIterator(self, *, skip_on_error):
        assert skip_on_error
        return iter(
            (
                _FakeDevice(address=0),
                _FakeDevice(address=49, serial="winbond-0123456789abcdef"),
            )
        )


def test_discovery_ignores_reserved_address_zero_libusb_ghost(monkeypatch):
    monkeypatch.setattr(hardware.usb1, "USBContext", _FakeContext)

    radios = hardware._discover_attached_plutos()

    assert len(radios) == 1
    assert radios[0].serial == "winbond-0123456789abcdef"
    assert radios[0].address == 49
