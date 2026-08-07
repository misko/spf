"""Prove the RF-DC tracking pin reaches silicon (E-CAL1 arm 2 enablement).

These tests **write** AD9361 configuration, so they need both
``--radio-hardware`` and ``--radio-rf-dc-tracking``:

    pytest tests/radio_hardware/test_rf_dc_tracking_hardware.py \
      --radio-hardware --radio-rf-dc-tracking --radio-expected-count 2

TX is never enabled -- reading and writing these attributes needs no transmit,
so this stays in the receive-only hardware family.

Every test restores the pre-test tracking state. That is not politeness: these
are the same radios the readiness manifest blesses and every capture uses, and a
test that left RF-DC tracking disabled would silently degrade every subsequent
dataset on the bench with no error anywhere.

The unit tier (``tests/test_dual_rx_gain_frequency_rf_dc_tracking.py``) covers
the logic against fakes. What only real hardware can answer is whether the
driver *accepts and applies* the write on this firmware.
"""

from __future__ import annotations

import pytest

from spf.calibrations.dual_rx_gain_frequency.config import CalibrationConfig
from spf.calibrations.dual_rx_gain_frequency.dc_offset import inspect_radio_rf_dc
from spf.calibrations.dual_rx_gain_frequency.hardware import DirectUsbLoopbackRadio


pytestmark = [pytest.mark.radio_hardware, pytest.mark.radio_rf_dc_tracking]

TRACKING_ATTR = "rf_dc_offset_tracking_en"
PROBE_LO_LOW_BAND_HZ = 2_400_000_000
# Above the 4 GHz gain-table band edge, so the retune crosses a table boundary.
PROBE_LO_HIGH_BAND_HZ = 5_100_000_000


def probe_config(tracking: bool | None) -> CalibrationConfig:
    """A minimal valid config; only the RF-DC fields matter to these tests."""

    return CalibrationConfig(
        frequencies_hz=(PROBE_LO_LOW_BAND_HZ,),
        gains_db=(10, 20),
        repetitions=1,
        min_quality_valid_per_cell=1,
        sample_rate_hz=30_000_000,
        bandwidth_hz=3_000_000,
        buffer_size=65_536,
        settle_seconds=0,
        frequency_settle_seconds=0,
        rf_dc_offset_tracking_en=tracking,
    )


def observed_via_independent_path(serial: str) -> dict[str, str]:
    """Read the tracking state through a *different* code path than the writer.

    ``inspect_radio_rf_dc`` opens its own IIO context and reads the attribute
    itself, so a bug in the writer's own readback cannot mask a failed write.
    """

    snapshot = inspect_radio_rf_dc(serial=serial)
    return {
        channel_name: channel[TRACKING_ATTR]
        for channel_name, channel in snapshot["channels"].items()
        if TRACKING_ATTR in channel
    }


def pin_and_release(serial: str, tracking: bool | None, action=None) -> None:
    """Pin the tracking loop, optionally act, then release the device.

    The writer holds the USB-IIO interface exclusively, so verification must
    happen after the radio is closed. That is the stricter check anyway: it
    proves the state persists once the capture process lets go.
    """

    radio = DirectUsbLoopbackRadio(serial, probe_config(tracking))
    try:
        radio.apply_rf_dc_offset_tracking()
        if action is not None:
            action(radio)
    finally:
        radio.close()


@pytest.fixture
def restore_tracking_state(attached_plutos):
    """Restore whatever the chip reported before the test, whatever happens."""

    original = {
        pluto.serial: observed_via_independent_path(pluto.serial)
        for pluto in attached_plutos
    }
    try:
        yield original
    finally:
        for serial, channels in original.items():
            values = set(channels.values())
            if len(values) != 1:
                continue
            pin_and_release(serial, next(iter(values)) == "1")


def test_disabling_tracking_reaches_silicon(attached_plutos, restore_tracking_state):
    for pluto in attached_plutos:
        assert restore_tracking_state[
            pluto.serial
        ], f"{pluto.serial}: firmware does not expose {TRACKING_ATTR}"
        pin_and_release(pluto.serial, False)
        observed = observed_via_independent_path(pluto.serial)
        assert set(observed.values()) == {"0"}, observed


def test_enabling_tracking_reaches_silicon(attached_plutos, restore_tracking_state):
    """Symmetry: the disable test cannot pass by the attribute being stuck."""

    for pluto in attached_plutos:
        pin_and_release(pluto.serial, False)
        assert set(observed_via_independent_path(pluto.serial).values()) == {"0"}
        pin_and_release(pluto.serial, True)
        observed = observed_via_independent_path(pluto.serial)
        assert set(observed.values()) == {"1"}, observed


def test_pin_survives_the_one_shot_rf_dc_calibration(
    attached_plutos, restore_tracking_state
):
    """The two RF-DC mechanisms must stay independent."""

    for pluto in attached_plutos:
        pin_and_release(
            pluto.serial, False, action=lambda radio: radio.run_rf_dc_calibration()
        )
        observed = observed_via_independent_path(pluto.serial)
        assert set(observed.values()) == {"0"}, observed


def test_pin_survives_a_band_crossing_retune(attached_plutos, restore_tracking_state):
    for pluto in attached_plutos:
        pin_and_release(
            pluto.serial,
            False,
            action=lambda radio: radio.configure_frequency(
                PROBE_LO_HIGH_BAND_HZ, start_tone=False
            ),
        )
        observed = observed_via_independent_path(pluto.serial)
        assert set(observed.values()) == {"0"}, observed
