"""Guards for the destructive RC channels: CH9 shutdown, CH7 reboot, CH10 magcal.

CH9 is level-triggered and clockless on purpose, and nothing here sets a clock
for it -- that is the property under test. The interlock this replaced required
a >2s continuous hold measured with time.monotonic() at the moment each frame
was *processed*, so it silently required an RC stream faster than ~0.67 Hz and a
receive loop that never stalled. Both dependencies are invisible from the
cockpit. The rate-independence test below is the one that would have caught it.
"""

import logging
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from spf.mavlink.mavlink_controller import Drone


def _drone(*, armed=False, motor_active=False):
    drone = Drone.__new__(Drone)
    drone.armed = armed
    drone.motor_active = motor_active
    drone.mav_mode = "ROVER_MODE_GUIDED"
    drone.disable_distance_finder = False
    drone.distance_finder = object()
    drone.run_compass_calibration = Mock()
    drone.reboot = Mock()
    drone.send_status = Mock()
    drone.buzzer = Mock()
    drone.request_motion_stop = Mock()
    drone.wait_for_abort_hold = Mock(return_value=True)
    drone.disarm = Mock()
    drone._pump_one_heartbeat = Mock(return_value=True)
    return drone


def _rc_message(*, ch9_raw, ch7_raw=0, ch10_raw=0, ch12_raw=0):
    return SimpleNamespace(
        chan7_raw=ch7_raw,
        chan9_raw=ch9_raw,
        chan10_raw=ch10_raw,
        chan12_raw=ch12_raw,
    )


def _mock_poweroff(monkeypatch, **kwargs):
    kwargs.setdefault("return_value", SimpleNamespace(returncode=0))
    poweroff = Mock(**kwargs)
    monkeypatch.setattr("spf.mavlink.mavlink_controller.subprocess.run", poweroff)
    return poweroff


def _press(drone, *, released_first=True, frames=1, value=1800):
    """The gesture: a release, then the press."""
    if released_first:
        drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000))
    for _ in range(frames):
        drone.handle_RC_CHANNELS(_rc_message(ch9_raw=value))


# ------------------------------------------------------------------ CH9 ---


def test_single_frame_after_a_release_powers_off(monkeypatch):
    """One frame above threshold IS the gesture. No hold, no toggle count."""
    poweroff = _mock_poweroff(monkeypatch)
    drone = _drone()

    _press(drone)

    poweroff.assert_called_once_with(["sudo", "poweroff"], check=False)


def test_high_from_the_first_frame_is_ignored_until_released(monkeypatch, caplog):
    """The one failure mode a pure level test cannot ship without.

    A receiver on failsafe Hold reports the last values it saw, and the capture
    process restarts on every capture iteration -- so acting on a switch that
    was already high at connect is a shutdown boot-loop, not a rare edge.
    """
    poweroff = _mock_poweroff(monkeypatch)
    drone = _drone()
    caplog.set_level(logging.WARNING)

    for _ in range(10):
        drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1800))

    poweroff.assert_not_called()
    stale = [m for m in caplog.messages if "has been high since connect" in m]
    assert len(stale) == 1, "the stale-high warning must be logged once, not per frame"

    _press(drone)
    poweroff.assert_called_once()


def test_failsafe_zero_reads_as_released_not_as_a_press(monkeypatch):
    """chan9_raw == 0 is "no such channel"/failsafe, never an actuation."""
    poweroff = _mock_poweroff(monkeypatch)
    drone = _drone()

    for _ in range(5):
        drone.handle_RC_CHANNELS(_rc_message(ch9_raw=0))

    poweroff.assert_not_called()


@pytest.mark.parametrize("frames", [1, 3, 20])
def test_a_held_switch_powers_off_exactly_once(monkeypatch, frames):
    poweroff = _mock_poweroff(monkeypatch)
    drone = _drone()

    _press(drone, frames=frames)

    poweroff.assert_called_once()


def test_release_rearms_for_a_second_press(monkeypatch):
    poweroff = _mock_poweroff(monkeypatch)
    drone = _drone()

    _press(drone)
    _press(drone)

    assert poweroff.call_count == 2


# --------------------------------------------------- state is not a veto ---


def test_armed_rover_still_powers_off_and_is_safed_first(monkeypatch):
    """The regression that made this switch useless in the field.

    `permitted = not armed and not motor_active` was a veto, and the planner
    arms the vehicle for the whole of a capture -- so the switch did nothing
    exactly when an operator reached for it.
    """
    poweroff = _mock_poweroff(monkeypatch)
    drone = _drone(armed=True, motor_active=True)
    # .armed is cleared by handle_HEARTBEAT, so it can only change when a
    # heartbeat is pumped -- never by the disarm command returning.
    drone._pump_one_heartbeat = Mock(
        side_effect=lambda **_: setattr(drone, "armed", False) or True
    )

    _press(drone)

    drone.request_motion_stop.assert_called_once()
    drone.wait_for_abort_hold.assert_called_once()
    drone.disarm.assert_called_once()
    drone._pump_one_heartbeat.assert_called()
    poweroff.assert_called_once_with(["sudo", "poweroff"], check=False)


def test_disarm_confirmation_pumps_heartbeats_rather_than_sleeping(monkeypatch):
    """handle_RC_CHANNELS runs ON the receive loop thread.

    .armed is set by handle_HEARTBEAT on that same thread, so a wait that
    merely sleeps blocks the only loop that could ever observe the disarm --
    it could not do anything but time out and log a false "still armed".
    """
    _mock_poweroff(monkeypatch)
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.RC_SHUTDOWN_DISARM_TIMEOUT_SECONDS", 5.0
    )
    drone = _drone(armed=True)
    pumped = []

    def pump(**_kwargs):
        pumped.append(1)
        if len(pumped) == 3:  # the third heartbeat carries the disarmed bit
            drone.armed = False
        return True

    drone._pump_one_heartbeat = Mock(side_effect=pump)

    _press(drone)

    assert len(pumped) == 3, "the wait must advance by heartbeats, not by time"
    assert drone.armed is False


def test_an_unreadable_link_stops_the_disarm_wait_early(monkeypatch):
    """A dead link must not burn the whole timeout before powering off."""
    poweroff = _mock_poweroff(monkeypatch)
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.RC_SHUTDOWN_DISARM_TIMEOUT_SECONDS", 30.0
    )
    drone = _drone(armed=True)
    drone._pump_one_heartbeat = Mock(return_value=False)

    _press(drone)

    drone._pump_one_heartbeat.assert_called_once()
    poweroff.assert_called_once()


def test_motion_stop_precedes_disarm(monkeypatch):
    """Ordering matters: the planner re-arms within 0.1s of a bare disarm."""
    _mock_poweroff(monkeypatch)
    order = []
    drone = _drone(armed=True)
    drone.request_motion_stop = Mock(side_effect=lambda **_: order.append("stop"))
    drone.wait_for_abort_hold = Mock(side_effect=lambda **_: order.append("hold"))
    drone.disarm = Mock(
        side_effect=lambda: (order.append("disarm"), setattr(drone, "armed", False))
    )

    _press(drone)

    assert order == ["stop", "hold", "disarm"]


def test_a_disarm_that_never_confirms_still_powers_off(monkeypatch, caplog):
    """Bounded, not blocking. Refusing to halt is the failure being removed."""
    poweroff = _mock_poweroff(monkeypatch)
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.RC_SHUTDOWN_DISARM_TIMEOUT_SECONDS", 0.2
    )
    drone = _drone(armed=True)  # .armed never clears
    caplog.set_level(logging.ERROR)

    _press(drone)

    poweroff.assert_called_once()
    assert any("still armed" in message for message in caplog.messages)


def test_a_failed_hold_still_powers_off(monkeypatch):
    poweroff = _mock_poweroff(monkeypatch)
    drone = _drone(armed=False)
    drone.request_motion_stop = Mock(side_effect=RuntimeError("no MAVLink"))

    _press(drone)

    poweroff.assert_called_once()


def test_a_failed_disarm_command_still_powers_off(monkeypatch):
    poweroff = _mock_poweroff(monkeypatch)
    drone = _drone(armed=True)
    drone.disarm = Mock(side_effect=RuntimeError("connection changed"))

    _press(drone)

    poweroff.assert_called_once()


# ----------------------------------------------------- rate independence ---


@pytest.mark.parametrize("frames_per_press", [1, 2, 5, 40])
def test_behaviour_is_identical_at_every_stream_rate(monkeypatch, frames_per_press):
    """The property the timed interlock did not have.

    Frame count stands in for stream rate: a 0.5 Hz link, a 10 Hz link, and a
    stalled loop draining forty buffered frames at once must all do the same
    thing. Under the hold interlock, 0.5 Hz could never fire at all and a burst
    drain collapsed a genuine 2s hold to milliseconds.
    """
    poweroff = _mock_poweroff(monkeypatch)
    drone = _drone()

    _press(drone, frames=frames_per_press)

    poweroff.assert_called_once()


# ----------------------------------------------------- command plumbing ---


def test_shutdown_owns_its_rc_message(monkeypatch):
    """An accepted shutdown must not also reboot the FC or start a magcal."""
    poweroff = _mock_poweroff(monkeypatch)
    drone = _drone()

    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000))
    drone.handle_RC_CHANNELS(
        _rc_message(ch9_raw=1800, ch7_raw=1800, ch10_raw=1800, ch12_raw=1800)
    )

    poweroff.assert_called_once()
    drone.run_compass_calibration.assert_not_called()
    drone.reboot.assert_not_called()
    assert drone.disable_distance_finder is False


def test_operator_gets_confirmation_before_the_pi_goes_away(monkeypatch):
    _mock_poweroff(monkeypatch)
    drone = _drone()

    _press(drone)

    drone.buzzer.assert_called_once()
    drone.send_status.assert_called_once()


def test_a_failed_status_text_does_not_stop_the_poweroff(monkeypatch):
    poweroff = _mock_poweroff(monkeypatch)
    drone = _drone()
    drone.send_status = Mock(side_effect=RuntimeError("link down"))

    _press(drone)

    poweroff.assert_called_once()


def test_nonzero_poweroff_return_code_is_logged(monkeypatch, caplog):
    _mock_poweroff(monkeypatch, return_value=SimpleNamespace(returncode=7))
    drone = _drone()
    caplog.set_level(logging.ERROR)

    _press(drone)

    assert "RC shutdown command failed with return code 7" in caplog.messages


@pytest.mark.parametrize(
    "command_error",
    [
        pytest.param(OSError("poweroff executable unavailable"), id="oserror"),
        pytest.param(
            subprocess.SubprocessError("poweroff execution failed"),
            id="subprocess-error",
        ),
    ],
)
def test_poweroff_exception_is_contained(monkeypatch, caplog, command_error):
    _mock_poweroff(monkeypatch, side_effect=command_error)
    drone = _drone()
    caplog.set_level(logging.ERROR)

    _press(drone)

    assert any(
        message.startswith("RC shutdown command failed:") for message in caplog.messages
    )
    # The receive loop survives, so later channels keep working.
    for _ in range(3):
        drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch12_raw=1800))
    assert drone.disable_distance_finder is True


# -------------------------------------------------------- CH7 and CH10 ---


def test_ch7_resting_mid_position_does_not_reboot(monkeypatch):
    """The latent hazard the shared latch closes.

    `elif msg.chan7_raw > 1000: reboot(); sys.exit(1)` fired for any resting
    value in (1000, 1500] -- a centred 3-position switch reads 1500 -- with
    nothing to latch it, so it re-fired on every frame. The only thing keeping
    it quiet was CH7 happening to rest at <=1000.
    """
    _mock_poweroff(monkeypatch)
    drone = _drone()

    for _ in range(10):
        drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch7_raw=1500))

    drone.reboot.assert_not_called()


def test_ch7_reboots_once_per_press(monkeypatch):
    _mock_poweroff(monkeypatch)
    drone = _drone()

    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch7_raw=0))
    for _ in range(5):
        drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch7_raw=1800))

    drone.reboot.assert_called_once_with(force=True)


def test_ch10_starts_one_magcal_per_press(monkeypatch):
    _mock_poweroff(monkeypatch)
    drone = _drone()

    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch10_raw=0))
    for _ in range(5):
        drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch10_raw=1800))

    drone.run_compass_calibration.assert_called_once()


# ------------------------------------------------------------ CH12 ---
#
# The ultrasonic switch is NOT a destructive action and keeps its
# consecutive-sample debounce; these pin that the rewrite left it alone.


def test_ultrasonic_rc_is_ignored_when_capture_disabled_the_sensor(monkeypatch, caplog):
    _mock_poweroff(monkeypatch)
    drone = _drone()
    drone.distance_finder = None
    drone.disable_distance_finder = True

    for _ in range(3):
        drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch12_raw=0))

    assert drone.disable_distance_finder is True
    assert "ULTRASONIC" not in caplog.text


def test_ultrasonic_rc_requires_three_consistent_samples(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _mock_poweroff(monkeypatch)
    drone = _drone()

    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch12_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch12_raw=900))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch12_raw=1800))
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch12_raw=1800))
    assert drone.disable_distance_finder is False
    drone.handle_RC_CHANNELS(_rc_message(ch9_raw=1000, ch12_raw=1800))

    assert drone.disable_distance_finder is True
    assert caplog.text.count("DISABLE ULTRASONIC") == 1
    assert "ch12_raw=1800" in caplog.text
