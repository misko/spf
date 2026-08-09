from __future__ import annotations

import pytest

from spf.calibrations.dual_rx_gain_frequency.hardware import DirectUsbLoopbackRadio

from tests.test_dual_rx_gain_frequency_calibration import small_config


class FakeAttribute:
    def __init__(self, events, *, fail: bool = False):
        self.events = events
        self.fail = fail
        self._value = "auto"

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.events.append(("calib_mode", value))
        if self.fail:
            raise OSError("calibration failed")
        self._value = value


class FakeControl:
    def __init__(self, events, *, calibration_fails: bool = False):
        self.attrs = {
            "calib_mode": FakeAttribute(events, fail=calibration_fails),
        }


class FakeToneSdr:
    def __init__(self, *, calibration_fails: bool = False):
        self.events = []
        self._ctrl = FakeControl(self.events, calibration_fails=calibration_fails)
        self.tx_hardwaregain_chan0 = -80.0
        self.tx_hardwaregain_chan1 = -80.0
        self.tx_enabled_channels = []
        self.tx_cyclic_buffer = False

    def disable_dds(self):
        self.events.append(("disable_dds",))

    def tx_destroy_buffer(self):
        self.events.append(("tx_destroy_buffer",))

    def dds_single_tone(self, frequency, scale, channel):
        self.events.append(("dds_single_tone", frequency, scale, channel))


def _radio(fake_sdr: FakeToneSdr) -> DirectUsbLoopbackRadio:
    radio = object.__new__(DirectUsbLoopbackRadio)
    radio.sdr = fake_sdr
    radio.config = small_config(tone_offset_hz=100_000, tx_gain_db=-10.0)
    radio._tone_active = False
    radio._active_tx_gain = None
    radio._prime_iio_rx_dma = lambda: fake_sdr.events.append(("prime_rx_dma",))
    return radio


def test_start_tone_calibrates_tx_before_programming_and_rx_priming():
    sdr = FakeToneSdr()
    radio = _radio(sdr)

    radio.start_tone(tx_channel=1, tx_gain_db=-10.0, prime_after_arm=True)

    significant = [
        event[0]
        for event in sdr.events
        if event[0] in ("calib_mode", "dds_single_tone", "prime_rx_dma")
    ]
    assert significant == [
        "calib_mode",
        "dds_single_tone",
        "prime_rx_dma",
    ]
    assert sdr._ctrl.attrs["calib_mode"].value == "tx_quad"
    assert sdr.tx_hardwaregain_chan0 == -80
    assert sdr.tx_hardwaregain_chan1 == -10.0
    assert radio._tone_active


def test_failed_tx_quadrature_calibration_fails_closed_and_mutes_both_outputs():
    sdr = FakeToneSdr(calibration_fails=True)
    radio = _radio(sdr)

    with pytest.raises(OSError, match="calibration failed"):
        radio.start_tone(tx_channel=1, tx_gain_db=-10.0, prime_after_arm=True)

    assert not radio._tone_active
    assert sdr.tx_hardwaregain_chan0 == -80
    assert sdr.tx_hardwaregain_chan1 == -80
    assert not any(event[0] == "dds_single_tone" for event in sdr.events)
