import dataclasses
import struct
import types

import pytest
from test_iio_metadata import _metadata

from spf.direct_radio.gain_control import (
    GAIN_MODES,
    legacy_metadata_request,
    tandem_request_for_frame,
    validate_gain_modes,
)
from spf.direct_radio.iio_metadata import IioMetadataRx
from spf.direct_radio.metadata_v6 import RadioMetadataV6, _seal
from spf.direct_radio.usb_protocol import MetadataFeatures, ProtocolError


def v6_record(tandem=True, missing=0):
    metadata = _metadata()
    if tandem:
        raw = bytearray(metadata.pack())
        features = 0xFFF
    else:
        base = dataclasses.replace(
            metadata.base,
            gain_event_capacity=0,
            features=MetadataFeatures(int(metadata.base.features) & ~8),
        )
        raw = bytearray(base.pack())
        raw[124:124] = bytes(56)
        features = 0xEF7
    struct.pack_into("<HHI", raw, 4, 6, len(raw), features)
    struct.pack_into("<i", raw, 164, 42000)
    struct.pack_into("<Q", raw, 116, missing)
    if missing:
        flags = struct.unpack_from("<I", raw, 12)[0] | (1 << 23) | (1 << 11)
        struct.pack_into("<I", raw, 12, flags)
    return _seal(raw)


@pytest.mark.parametrize("mode", GAIN_MODES)
def test_public_modes(mode):
    assert validate_gain_modes([mode, mode]) == (mode, mode)


@pytest.mark.parametrize(
    "modes,channels,transport",
    [
        (["tandem", "manual"], [0, 1], "iio"),
        (["tandem", "tandem"], [0], "iio"),
        (["tandem", "tandem"], [0, 1], "direct_usb"),
        (["typo", "typo"], [0, 1], "iio"),
    ],
)
def test_invalid_modes(modes, channels, transport):
    with pytest.raises(ValueError):
        validate_gain_modes(modes, channels, transport)


def test_fixed_policy_accepts_rover_frames_and_rejects_oversized_frames():
    assert tandem_request_for_frame(524288).cooldown_periods == 16
    with pytest.raises(ValueError, match="event capacity"):
        tandem_request_for_frame(1048576)


def test_metadata_default_is_explicitly_legacy_without_writes():
    sdr = types.SimpleNamespace(
        _ctx=types.SimpleNamespace(
            attrs={
                "iio,buffer-metadata": "3",
                "iio,buffer-metadata-legacy-agc": "1",
            }
        )
    )
    rx = IioMetadataRx(sdr, sample_rate_hz=2500000, samples_per_channel=524288)
    assert rx._tandem_request == legacy_metadata_request(524288)
    assert rx._requested_tandem is None
    sdr._ctx.attrs.pop("iio,buffer-metadata-legacy-agc")
    with pytest.raises(ValueError, match="legacy AGC"):
        IioMetadataRx(sdr, sample_rate_hz=2500000, samples_per_channel=524288)


@pytest.mark.parametrize("tandem", [False, True])
def test_v6_preserves_metadata_and_ownership(tandem):
    metadata = RadioMetadataV6.unpack(v6_record(tandem))
    assert (metadata.tandem is not None) == tandem
    assert metadata.ownership_epoch == (7 if tandem else None)
    assert metadata.ad9361_temperature_mdeg_c == 42000
    assert metadata.gain_db_end == (20, 21)
    assert metadata.first_sample_sequence == 1000000
    assert metadata.rssi_metadata_valid


@pytest.mark.parametrize("offset,value", [(4, 7), (8, 0), (168, 1), (54, 1)])
def test_v6_rejects_bad_contract_even_with_valid_crc(offset, value):
    raw = bytearray(v6_record())
    raw[offset] = value
    with pytest.raises(ProtocolError):
        RadioMetadataV6.unpack(_seal(raw))


def test_legacy_cannot_claim_a_tandem_lease():
    raw = bytearray(v6_record(False))
    raw[124] = 1
    with pytest.raises(ProtocolError, match="claims tandem"):
        RadioMetadataV6.unpack(_seal(raw))


def test_gap_count_preserved_and_crc_checked():
    raw = v6_record(missing=1234)
    assert RadioMetadataV6.unpack(raw).missing_samples_before == 1234
    corrupt = bytearray(raw)
    corrupt[-1] ^= 1
    with pytest.raises(ProtocolError, match="CRC"):
        RadioMetadataV6.unpack(corrupt)


@pytest.fixture
def radio_session(monkeypatch):
    import numpy as np

    from spf.direct_radio import iio_metadata
    from spf.sdrpluto.sdr_controller import PPlus

    calls = []
    fault = {"at": None, "restoration": False}

    class Device:
        _ctrl = types.SimpleNamespace(
            find_channel=lambda *args: types.SimpleNamespace(
                attrs={
                    "hardwaregain_available": types.SimpleNamespace(value="[-3 1 73]")
                }
            )
        )
        gain_control_mode_chan0 = "manual"
        gain_control_mode_chan1 = "manual"
        rx_hardwaregain_chan0 = 20
        rx_hardwaregain_chan1 = 23
        owned = False

        def __setattr__(self, name, value):
            if name.startswith(("gain_control_mode", "rx_hardwaregain")):
                assert not self.owned, "gain write while the provider owns the radio"
                calls.append((name, value))
            object.__setattr__(self, name, value)

        def rx_destroy_buffer(self):
            calls.append("destroy")

    class Session:
        def __init__(self, sdr, *, tandem_request, **kwargs):
            self.sdr = sdr
            self.request = tandem_request
            self.is_open = False
            self.prior = None

        def open(self):
            calls.append("open")
            if fault["at"] == "open":
                fault["at"] = None
                raise OSError("injected open failure")
            if fault["restoration"] and self.request is None:
                raise OSError("injected restore failure")
            self.is_open = True
            if self.request is not None:
                self.prior = (
                    self.sdr.gain_control_mode_chan0,
                    self.sdr.gain_control_mode_chan1,
                )
                self.sdr.gain_control_mode_chan0 = "manual"
                self.sdr.gain_control_mode_chan1 = "manual"
                self.sdr.owned = True

        def capture(self):
            assert self.is_open
            if fault["at"] == "capture":
                fault["at"] = None
                raise OSError("injected capture failure")
            return (
                np.zeros((2, 1024)),
                RadioMetadataV6.unpack(v6_record(self.request is not None)),
                {},
            )

        def close(self):
            calls.append("close")
            self.is_open = False
            self.sdr.owned = False
            if self.prior:
                self.sdr.gain_control_mode_chan0, self.sdr.gain_control_mode_chan1 = (
                    self.prior
                )
                self.prior = None

    monkeypatch.setattr(iio_metadata, "IioMetadataRx", Session)
    radio = PPlus.__new__(PPlus)
    radio.rx_config = types.SimpleNamespace(
        gain_control_modes=["manual", "manual"],
        gains=[20, 23],
        enabled_channels=[0, 1],
        rx_transport="iio",
        sample_rate=2500000,
        buffer_size=1024,
    )
    radio.sdr = Device()
    radio._iio_metadata_rx = Session(radio.sdr, tandem_request=None)
    radio._iio_metadata_rx.open()
    calls.clear()
    return radio, calls, fault


def test_switching_releases_ownership_before_legacy_gain_writes(radio_session):
    radio, calls, _ = radio_session
    radio.set_gain_mode("tandem")
    assert radio.sdr.owned
    assert radio.rx_config.gain_control_modes == ["tandem", "tandem"]
    radio.set_gain_mode("fast_attack")
    assert not radio.sdr.owned
    assert radio.sdr.gain_control_mode_chan0 == "fast_attack"
    radio.set_gain_mode("manual", gains=[17, 29])
    assert (radio.sdr.rx_hardwaregain_chan0, radio.sdr.rx_hardwaregain_chan1) == (
        17,
        29,
    )


@pytest.mark.parametrize("stage", ["open", "capture"])
def test_failure_restores_both_manual_channels_and_capture(radio_session, stage):
    radio, _, fault = radio_session
    fault["at"] = stage
    with pytest.raises(OSError, match="injected"):
        radio.set_gain_mode("tandem")
    assert radio.rx_config.gain_control_modes == ["manual", "manual"]
    assert (
        radio.sdr.gain_control_mode_chan0
        == radio.sdr.gain_control_mode_chan1
        == "manual"
    )
    assert (radio.sdr.rx_hardwaregain_chan0, radio.sdr.rx_hardwaregain_chan1) == (
        20,
        23,
    )
    assert radio._iio_metadata_rx.is_open
    assert not radio.sdr.owned


def test_invalid_mode_preserves_active_session(radio_session):
    radio, calls, _ = radio_session
    previous_session = radio._iio_metadata_rx
    with pytest.raises(ValueError):
        radio.set_gain_mode(["tandem", "fast_attack"])
    assert calls == []
    assert radio._iio_metadata_rx is previous_session


@pytest.mark.parametrize("gains", [[20, 74], [20, -4], [20, 20.5]])
def test_invalid_manual_gain_preserves_active_session(radio_session, gains):
    radio, calls, _ = radio_session
    previous_session = radio._iio_metadata_rx
    with pytest.raises(ValueError, match="manual gain"):
        radio.set_gain_mode("manual", gains=gains)
    assert calls == []
    assert radio._iio_metadata_rx is previous_session


def test_failed_restoration_stops_rx(radio_session):
    radio, _, fault = radio_session
    fault.update(at="capture", restoration=True)
    with pytest.raises(RuntimeError, match="RX stopped"):
        radio.set_gain_mode("tandem")
    assert radio.rx_config is None
    assert radio._iio_metadata_rx is None
    with pytest.raises(RuntimeError, match="not configured"):
        radio.rx()


def test_reapplying_healthy_mode_does_not_restart_session(radio_session):
    radio, calls, _ = radio_session
    radio.set_gain_mode("tandem")
    session = radio._iio_metadata_rx
    calls.clear()
    radio.set_gain_mode("tandem")
    assert radio._iio_metadata_rx is session
    assert calls == []


def test_invalid_gain_or_rssi_metadata_stops_rx(radio_session, monkeypatch):
    radio, _, _ = radio_session
    radio.rx_with_metadata()

    def reject_metadata(metadata):
        raise RuntimeError("invalid gain metadata")

    monkeypatch.setattr(radio, "_cache_direct_legacy_values", reject_metadata)
    with pytest.raises(RuntimeError, match="invalid gain metadata"):
        radio.rx_with_metadata()
    assert radio._iio_metadata_rx is None
    assert radio.rx_config is None


def test_v7_persists_ownership_and_gaps_with_legacy_store_compatibility(
    radio_session, tmp_path, monkeypatch
):
    from spf.data_collector import DroneDataCollectorRaw, DroneDataCollectorRawV7
    from spf.dataset.v7_data import v7rx_new_dataset

    radio, _, _ = radio_session
    frame = dataclasses.replace(
        radio.rx_with_metadata(), tandem_ownership_epoch=77, missing_samples_before=1234
    )
    zarr = v7rx_new_dataset(
        str(tmp_path / "capture.zarr"),
        timesteps=2,
        buffer_size=1024,
        n_receivers=1,
        config={},
    )
    # Exercise the metadata writer without requiring a vehicle navigation fix.
    monkeypatch.setattr(
        DroneDataCollectorRaw, "write_to_record_matrix", lambda *args: None
    )
    writer = DroneDataCollectorRawV7.__new__(DroneDataCollectorRawV7)
    writer.data_filename = str(tmp_path / "capture.zarr")
    writer.zarr = zarr
    try:
        writer.write_to_record_matrix(0, 0, frame)
        receiver = zarr["receivers/r0"]
        assert receiver["tandem_ownership_epoch"][0] == 77
        assert receiver["missing_samples_before"][0] == 1234
        assert receiver["requested_gain_mode_rx1"][0] == 0
        assert receiver["requested_gain_mode_rx2"][0] == 0
        assert zarr.attrs["gain_control_schema_version"] == 1
        writer.write_to_record_matrix(
            0, 1, dataclasses.replace(frame, tandem_ownership_epoch=None)
        )
        assert receiver["tandem_ownership_epoch"][1] == 0
        del receiver["tandem_ownership_epoch"]
        del receiver["missing_samples_before"]
        writer.write_to_record_matrix(0, 0, frame)
    finally:
        zarr.store.close()
