"""RF-DC tracking-loop and calibration-policy control (E-CAL1 arm 2 enablement).

Two mechanisms hide inside "the RF-DC machinery" and these tests keep them
separate:

  1. the one-shot initialization calibration -- ``calib_mode = rf_dc_offs``,
     gated by ``rf_dc_calibration_policy``
  2. the continuous tracking loop            -- ``rf_dc_offset_tracking_en``

The load-bearing test here is
``test_silently_ignored_write_is_rejected``: the AD9361 driver can accept an
attribute write and not apply it, and an unverified write would let a capture
claim the tracking loop was off when it was never touched. That produces a null
result for the wrong reason -- indistinguishable from a real null on exactly the
question arm 2 exists to answer.
"""

from __future__ import annotations

import json

import pytest

from spf.calibrations.dual_rx_gain_frequency.config import RF_DC_CALIBRATION_POLICIES
from spf.calibrations.dual_rx_gain_frequency.hardware import DirectUsbLoopbackRadio
from spf.calibrations.dual_rx_gain_frequency.runner import run_calibration
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

from tests.test_dual_rx_gain_frequency_calibration import (
    FakeLoopbackRadio,
    small_config,
    write_config,
    write_ready_manifest,
)


TRACKING_ATTR = "rf_dc_offset_tracking_en"


# --------------------------------------------------------------------------
# A minimal libiio/pyadi stand-in, only as deep as ``_configure_static`` goes.
# --------------------------------------------------------------------------


class FakeAttr:
    def __init__(self, value="1"):
        self.value = value


class StickyAttr(FakeAttr):
    """Accepts a write and keeps the old value, as the driver sometimes does."""

    def __init__(self, value="1"):
        self._value = value
        self.writes = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self.writes.append(new_value)


class FakeChannel:
    def __init__(self, attrs):
        self.attrs = attrs


class FakeCtrl:
    def __init__(self, channels):
        self._channels = channels
        self.debug_attrs = {"adi,rx1-rx2-phase-inversion-enable": FakeAttr("0")}
        self.attrs = {"calib_mode": FakeAttr("")}
        self._registers = {0x22: 0}

    def find_channel(self, name, is_output=False):
        return self._channels[name]

    def reg_read(self, address):
        return self._registers.get(address, 0)

    def reg_write(self, address, value):
        self._registers[address] = value


class FakeSdr:
    def __init__(self, serial, channels, *, on_retune=None):
        self._ctx = FakeContext(serial)
        self._ctrl = FakeCtrl(channels)
        self._rxadc = FakeRxAdc()
        self.rx_lo = 0
        self.tx_lo = 0
        self._on_retune = on_retune

    # ``_configure_static`` / ``stop_tone`` touch these and ignore the values.
    def rx_destroy_buffer(self):
        pass

    def tx_destroy_buffer(self):
        pass

    def disable_dds(self):
        pass

    def __setattr__(self, name, value):
        if name == "rx_lo" and getattr(self, "_on_retune", None) is not None:
            self._on_retune()
        super().__setattr__(name, value)


class FakeContext:
    def __init__(self, serial):
        self.attrs = {"hw_serial": serial}


class FakeRxAdc:
    def set_kernel_buffers_count(self, count):
        pass


class FakeCapabilities:
    protocol_min = 1
    protocol_max = 2
    supported_features = 55
    capability_flags = 13


class FakeDirectIdentity:
    def __init__(self, serial):
        self.serial = serial
        self.bus = 1
        self.address = 2
        self.port_path = (1, 1)
        self.interface = 6
        self.bulk_in_endpoint = 1
        self.bulk_out_endpoint = 2


class FakeDirectReceiver:
    def __init__(self, serial, protocol_version=2, **options):
        self.identity = FakeDirectIdentity(serial)
        self.capabilities = FakeCapabilities()
        self.protocol_version = protocol_version
        self.options = options

    def open(self):
        pass

    def close(self):
        pass


def build_radio(config, *, tracking_attr_factory=FakeAttr, serial="SERIAL-A", **kwargs):
    """Construct a ``DirectUsbLoopbackRadio`` over the fake stack."""

    channels = {
        name: FakeChannel(
            {
                "quadrature_tracking_en": FakeAttr("0"),
                TRACKING_ATTR: tracking_attr_factory(),
            }
        )
        for name in ("voltage0", "voltage1")
    }
    holder = {}

    class FakeAdiModule:
        @staticmethod
        def ad9361(uri):
            sdr = FakeSdr(serial, channels, **kwargs)
            holder["sdr"] = sdr
            return sdr

    radio = DirectUsbLoopbackRadio(
        serial,
        config,
        adi_module=FakeAdiModule,
        direct_receiver_class=FakeDirectReceiver,
        scan_contexts=lambda: {"usb:1.2.5": f"serial={serial}"},
    )
    return radio, channels, holder["sdr"]


def test_loopback_adapter_forwards_protocol_v3_receiver_options():
    config = small_config()
    holder = {}

    class RecordingReceiver(FakeDirectReceiver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder["receiver"] = self

    channels = {
        name: FakeChannel(
            {
                "quadrature_tracking_en": FakeAttr("0"),
                TRACKING_ATTR: FakeAttr(),
            }
        )
        for name in ("voltage0", "voltage1")
    }

    class FakeAdiModule:
        @staticmethod
        def ad9361(uri):
            return FakeSdr("SERIAL-A", channels)

    radio = DirectUsbLoopbackRadio(
        "SERIAL-A",
        config,
        adi_module=FakeAdiModule,
        direct_receiver_class=RecordingReceiver,
        scan_contexts=lambda: {"usb:1.2.5": "serial=SERIAL-A"},
        direct_protocol_version=3,
        direct_receiver_options={
            "gain_observation_interval_samples": 2048,
            "gain_observation_capacity": 256,
        },
    )
    try:
        assert holder["receiver"].protocol_version == 3
        assert holder["receiver"].options == {
            "gain_observation_interval_samples": 2048,
            "gain_observation_capacity": 256,
        }
        assert radio.identity().direct_usb_protocol_version == 3
    finally:
        radio.close()


# --------------------------------------------------------------------------
# Config: the policy enum widened, it did not open
# --------------------------------------------------------------------------


def test_never_policy_is_accepted_and_garbage_still_fails_closed():
    assert set(RF_DC_CALIBRATION_POLICIES) == {"before_each_frequency_block", "never"}
    small_config(rf_dc_calibration_policy="never").validate()
    with pytest.raises(ValueError, match="unsupported RF-DC calibration policy"):
        small_config(rf_dc_calibration_policy="whenever_it_feels_like_it").validate()


def test_tracking_flag_rejects_non_boolean():
    with pytest.raises(ValueError, match="must be a bool or null"):
        small_config(rf_dc_offset_tracking_en="0").validate()


def test_unset_tracking_leaves_the_run_signature_untouched():
    """Configs written before this knob existed must keep their signature.

    ``_run_signature`` hashes ``as_json()``, and the signature guards dataset
    resume. A config that does not pin the tracking loop did not shape the
    capture, so it must not shape the signature either.
    """

    baseline = small_config()
    assert "rf_dc_offset_tracking_en" not in baseline.as_json()

    pinned = small_config(rf_dc_offset_tracking_en=False)
    assert pinned.as_json()["rf_dc_offset_tracking_en"] is False
    # Pinning the knob *is* a different run, and must be a different signature.
    assert pinned.signature != baseline.signature


# --------------------------------------------------------------------------
# Hardware: the write, and the readback that can fail
# --------------------------------------------------------------------------


def test_unset_tracking_never_writes_the_attribute():
    config = small_config()
    radio, channels, _ = build_radio(config, tracking_attr_factory=StickyAttr)
    try:
        for channel in channels.values():
            assert channel.attrs[TRACKING_ATTR].writes == []
    finally:
        radio.close()


@pytest.mark.parametrize("requested,expected", [(False, "0"), (True, "1")])
def test_pinned_tracking_is_written_to_both_channels(requested, expected):
    config = small_config(rf_dc_offset_tracking_en=requested)
    radio, channels, _ = build_radio(config)
    try:
        observed = radio.apply_rf_dc_offset_tracking()
        assert observed == {"voltage0": expected, "voltage1": expected}
        for channel in channels.values():
            assert channel.attrs[TRACKING_ATTR].value == expected
    finally:
        radio.close()


def test_silently_ignored_write_is_rejected():
    """A write the chip accepts but does not apply must be a hard error.

    Without this, arm 2 would report "disabling tracking changed nothing" when
    tracking was never disabled -- a false null indistinguishable from a real
    one.
    """

    config = small_config(rf_dc_offset_tracking_en=False)
    with pytest.raises(RuntimeError, match="readback does not match"):
        build_radio(config, tracking_attr_factory=StickyAttr)


def test_missing_attribute_is_rejected_rather_than_skipped():
    config = small_config(rf_dc_offset_tracking_en=False)

    def no_attribute():
        return None

    channels = {
        name: FakeChannel({"quadrature_tracking_en": FakeAttr("0")})
        for name in ("voltage0", "voltage1")
    }

    class FakeAdiModule:
        @staticmethod
        def ad9361(uri):
            return FakeSdr("SERIAL-A", channels)

    with pytest.raises(RuntimeError, match="does not expose"):
        DirectUsbLoopbackRadio(
            "SERIAL-A",
            config,
            adi_module=FakeAdiModule,
            direct_receiver_class=FakeDirectReceiver,
            scan_contexts=lambda: {"usb:1.2.5": "serial=SERIAL-A"},
        )


def test_driver_reassertion_on_retune_is_repaired():
    """An LO retune can restore the driver's default; the pin must survive it.

    ``apply_rf_dc_offset_tracking`` re-asserts and then verifies, so the correct
    behaviour is repair, not merely detection. The undetectable case -- a write
    the chip ignores -- is covered by
    ``test_silently_ignored_write_is_rejected``.
    """

    config = small_config(rf_dc_offset_tracking_en=False)
    channels = {
        name: FakeChannel(
            {"quadrature_tracking_en": FakeAttr("0"), TRACKING_ATTR: FakeAttr("1")}
        )
        for name in ("voltage0", "voltage1")
    }

    def driver_restores_default():
        for channel in channels.values():
            channel.attrs[TRACKING_ATTR].value = "1"

    class FakeAdiModule:
        @staticmethod
        def ad9361(uri):
            return FakeSdr("SERIAL-A", channels, on_retune=driver_restores_default)

    radio = DirectUsbLoopbackRadio(
        "SERIAL-A",
        config,
        adi_module=FakeAdiModule,
        direct_receiver_class=FakeDirectReceiver,
        scan_contexts=lambda: {"usb:1.2.5": "serial=SERIAL-A"},
    )
    try:
        # The retune fires the driver's restore, and the retune path re-pins.
        radio.configure_frequency(2_400_000_000, start_tone=False)
        for channel in channels.values():
            assert channel.attrs[TRACKING_ATTR].value == "0"
    finally:
        radio.close()


def test_driver_reassertion_by_rf_dc_calibration_is_repaired():
    config = small_config(rf_dc_offset_tracking_en=False)
    radio, channels, _ = build_radio(config)
    try:
        # The one-shot calibration re-enables the tracking loop behind our back.
        for channel in channels.values():
            channel.attrs[TRACKING_ATTR].value = "1"
        radio.run_rf_dc_calibration()
        for channel in channels.values():
            assert channel.attrs[TRACKING_ATTR].value == "0"
    finally:
        radio.close()


# --------------------------------------------------------------------------
# Runner and dataset provenance
# --------------------------------------------------------------------------


def _run(tmp_path, config, monkeypatch, serials=("SERIAL-A",)):
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    ready_path = tmp_path / "ready.json"
    write_ready_manifest(ready_path, serials)
    monkeypatch.setenv("SPF_DIRECT_USB_READY_FILE", str(ready_path))
    FakeLoopbackRadio.instances = {}
    result = run_calibration(
        config_path=config_path,
        output_dir=tmp_path / "output",
        ready_manifest_path=ready_path,
        serials=serials,
        radio_factory=FakeLoopbackRadio,
    )
    assert result["status"] == "complete"
    return result


def test_never_policy_suppresses_the_one_shot_calibration(tmp_path, monkeypatch):
    config = small_config(
        frequencies_hz=(2_400_000_000,), rf_dc_calibration_policy="never"
    )
    _run(tmp_path, config, monkeypatch)
    radio = FakeLoopbackRadio.instances["SERIAL-A"]
    assert radio.rf_dc_calibrations == 0


def test_default_policy_runs_calibration_and_records_tracking_state(
    tmp_path, monkeypatch
):
    """Two assertions share one capture.

    Each V7 dataset reserves a 128 GiB LMDB address-space map, and the suite
    already runs close enough to that ceiling that extra datasets tip unrelated
    tests into ``lmdb.MemoryError``. One run, both checks.
    """

    config = small_config(frequencies_hz=(2_400_000_000,))
    _run(tmp_path, config, monkeypatch)
    radio = FakeLoopbackRadio.instances["SERIAL-A"]
    assert radio.rf_dc_calibrations > 0

    dataset = tmp_path / "output" / "SERIAL-A" / "calibration.v7.zarr"
    zarr = zarr_open_from_lmdb_store(str(dataset), mode="r")
    try:
        assert zarr.attrs["rf_dc_calibration_policy"] == "before_each_frequency_block"
        assert zarr.attrs["rf_dc_offset_tracking_en_requested"] == "unset"
        # The fake radio exposes no readback path, so "observed" is explicitly
        # null rather than silently absent.
        assert json.loads(zarr.attrs["rf_dc_offset_tracking_en_observed"]) is None
    finally:
        zarr.store.close()
