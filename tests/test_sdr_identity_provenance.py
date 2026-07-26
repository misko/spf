from types import SimpleNamespace

import pytest

from spf.data_collector import DataCollector, SDR_IDENTITY_VERSION
from spf.dataset.v4_data import v4rx_keys, v4rx_new_dataset
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store
from spf.sdrpluto import sdr_controller
from spf.sdrpluto.sdr_controller import PPlus, SdrDeviceIdentity


SERIAL_A = "104000707f0700120f001a0095f2dbee49"
SERIAL_B = "104000f6ad020002fdff3a00bba2f096a1"


class _IdentityPPlus:
    def __init__(self, identity):
        self._identity = identity

    def receiver_identity(self):
        return self._identity


def _identity(
    serial,
    port_path,
    *,
    uri,
    transport="iio",
    direct=False,
):
    return SdrDeviceIdentity(
        sdr_family="pluto",
        serial=serial,
        receiver_uri=uri,
        rx_transport=transport,
        usb_vendor_id=0x0456,
        usb_product_id=0xB673,
        usb_bus=1,
        usb_address=8 if port_path == (1, 1) else 9,
        usb_port_path=port_path,
        direct_usb_interface=6 if direct else None,
        direct_usb_bulk_in_endpoint=0x89 if direct else None,
        direct_usb_bulk_out_endpoint=0x07 if direct else None,
        direct_usb_protocol_version=2 if direct else None,
        direct_usb_capability_flags=0x3F if direct else None,
    )


def _collector_with_identities(tmp_path, identities):
    zarr_path = tmp_path / "identity.zarr"
    collector = object.__new__(DataCollector)
    collector.yaml_config = {"receivers": [{} for _ in identities]}
    collector.receiver_pplus = {
        f"receiver-{idx}": _IdentityPPlus(identity)
        for idx, identity in enumerate(identities)
    }
    collector.data_filename = str(zarr_path)
    collector.zarr = v4rx_new_dataset(
        filename=str(zarr_path),
        timesteps=1,
        buffer_size=8,
        n_receivers=len(identities),
        config=collector.yaml_config,
        chunk_size=1,
    )
    return collector, zarr_path


def test_v4_records_two_pluto_identities_without_changing_array_schema(tmp_path):
    identities = [
        _identity(SERIAL_A, (1, 2), uri="usb:1.9.5"),
        _identity(
            SERIAL_B,
            (1, 1),
            uri="usb:1.8.5",
            transport="direct_usb",
            direct=True,
        ),
    ]
    collector, zarr_path = _collector_with_identities(tmp_path, identities)

    collector._record_receiver_identities()
    collector.zarr.store.close()

    zarr = zarr_open_from_lmdb_store(str(zarr_path))
    try:
        assert zarr.attrs["sdr_identity_version"] == SDR_IDENTITY_VERSION
        assert set(zarr.receivers.r0.keys()) == set(v4rx_keys())
        assert set(zarr.receivers.r1.keys()) == set(v4rx_keys())

        r0_attrs = dict(zarr.receivers.r0.attrs)
        assert r0_attrs == {
            "iio_uri_at_capture": "usb:1.9.5",
            "rx_transport": "iio",
            "sdr_family": "pluto",
            "sdr_identity_version": 1,
            "sdr_serial": SERIAL_A,
            "usb_address_at_capture": 9,
            "usb_bus_at_capture": 1,
            "usb_port_path": [1, 2],
            "usb_product_id": 0xB673,
            "usb_vendor_id": 0x0456,
        }

        r1_attrs = dict(zarr.receivers.r1.attrs)
        assert r1_attrs["sdr_serial"] == SERIAL_B
        assert r1_attrs["usb_port_path"] == [1, 1]
        assert r1_attrs["rx_transport"] == "direct_usb"
        assert r1_attrs["direct_usb_serial"] == SERIAL_B
        assert r1_attrs["direct_usb_bus"] == 1
        assert r1_attrs["direct_usb_port_path"] == [1, 1]
        assert r1_attrs["direct_usb_interface"] == 6
        assert r1_attrs["direct_usb_bulk_in_endpoint"] == 0x89
        assert r1_attrs["direct_usb_bulk_out_endpoint"] == 0x07
        assert r1_attrs["gain_metadata_protocol_version"] == 2
        assert r1_attrs["gain_metadata_capability_flags"] == 0x3F
    finally:
        zarr.store.close()


@pytest.mark.parametrize(
    "identities,error",
    [
        (
            [
                _identity(SERIAL_A, (1, 1), uri="usb:1.8.5"),
                _identity(SERIAL_A, (1, 2), uri="usb:1.9.5"),
            ],
            "one Pluto serial",
        ),
        (
            [
                _identity(SERIAL_A, (1, 1), uri="usb:1.8.5"),
                _identity(SERIAL_B, (1, 1), uri="usb:1.9.5"),
            ],
            "one Pluto USB physical path",
        ),
    ],
)
def test_two_receiver_identity_must_be_a_bijection(identities, error):
    collector = object.__new__(DataCollector)
    collector.yaml_config = {"receivers": [{}, {}]}
    collector.receiver_pplus = {
        f"receiver-{idx}": _IdentityPPlus(identity)
        for idx, identity in enumerate(identities)
    }
    collector.data_filename = None

    with pytest.raises(RuntimeError, match=error):
        collector._record_receiver_identities()


def test_pluto_identity_requires_a_serial():
    collector = object.__new__(DataCollector)
    collector.yaml_config = {"receivers": [{}]}
    collector.receiver_pplus = {
        "receiver-0": _IdentityPPlus(
            _identity(None, (1, 1), uri="usb:1.8.5")
        )
    }
    collector.data_filename = None

    with pytest.raises(RuntimeError, match="non-empty serial"):
        collector._record_receiver_identities()


def test_usb_iio_identity_resolves_serial_to_local_physical_path(monkeypatch):
    pplus = object.__new__(PPlus)
    pplus.uri = "usb:1.9.5"
    pplus.rx_config = SimpleNamespace(rx_transport="iio")
    pplus.direct_rx = None
    pplus.sdr = SimpleNamespace(
        _ctx=SimpleNamespace(attrs={"hw_serial": SimpleNamespace(value=SERIAL_A)})
    )
    monkeypatch.setattr(
        sdr_controller,
        "_find_local_pluto_usb_device",
        lambda serial: (1, 9, (1, 2)) if serial == SERIAL_A else None,
    )

    identity = pplus.receiver_identity()

    assert identity.serial == SERIAL_A
    assert identity.usb_bus == 1
    assert identity.usb_address == 9
    assert identity.usb_port_path == (1, 2)
    assert identity.receiver_uri == "usb:1.9.5"
    del pplus.sdr


def test_explicit_direct_serial_cannot_override_the_iio_radio():
    pplus = object.__new__(PPlus)
    pplus.rx_config = SimpleNamespace(
        direct_usb_protocol_version=2,
        direct_usb_frame_count_per_request=1,
        direct_usb_serial=SERIAL_B,
        direct_usb_port_path=None,
    )
    pplus._iio_hardware_serial = lambda: SERIAL_A

    with pytest.raises(RuntimeError, match="does not match the USB-IIO radio"):
        pplus._open_direct_rx()
