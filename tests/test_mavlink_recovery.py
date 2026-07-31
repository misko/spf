import threading
import time
from types import SimpleNamespace

import pytest

from spf.mavlink.mavlink_controller import (
    Drone,
    MavlinkConnectionError,
    connect_with_heartbeat,
    resolve_ardupilot_serial,
)


class FakeMessage(SimpleNamespace):
    def get_type(self):
        return self.message_type


def heartbeat(custom_mode=0):
    return FakeMessage(
        message_type="HEARTBEAT",
        system_status=3,
        custom_mode=custom_mode,
        base_mode=0,
    )


class FakeMav:
    def __init__(self, *, tune_error=None):
        self.arm_commands = []
        self.tune_commands = []
        self.tune_error = tune_error

    def command_long_send(self, *args):
        self.arm_commands.append(args)

    def play_tune_send(self, *args):
        if self.tune_error is not None:
            raise self.tune_error
        self.tune_commands.append(args)


class ReconnectConnection:
    target_system = 1
    target_component = 1

    def __init__(
        self,
        *,
        receive_error=None,
        reconnect_heartbeat=None,
        tune_error=None,
    ):
        self.mav = FakeMav(tune_error=tune_error)
        self.receive_error = receive_error
        self.reconnect_heartbeat = reconnect_heartbeat
        self.closed = False

    def recv_match(self, *, blocking, timeout):
        if self.receive_error is not None:
            error, self.receive_error = self.receive_error, None
            raise error
        return None

    def wait_heartbeat(self, *, blocking=True, timeout):
        return self.reconnect_heartbeat

    def close(self):
        self.closed = True


class ParameterConnection:
    def __init__(self):
        self.fetches = 0
        self.starting_sizes = []
        self.drone = None

    def param_fetch_all(self):
        self.starting_sizes.append(len(self.drone.params))
        self.fetches += 1
        count = 3
        values = [("A", 1.0), ("B", 2.0)]
        if self.fetches == 2:
            values.append(("C", 3.0))
        for index, (name, value) in enumerate(values):
            self.drone.handle_PARAM_VALUE(
                FakeMessage(
                    param_id=name,
                    param_value=value,
                    param_index=index,
                    param_count=count,
                )
            )


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_transient_serial_read_failure_reopens_and_requires_new_heartbeat():
    disconnected = ReconnectConnection(receive_error=OSError("USB disappeared"))
    reconnected = ReconnectConnection(reconnect_heartbeat=heartbeat())
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return reconnected

    drone = Drone(
        disconnected,
        fake=True,
        connection_factory=factory,
        reconnect_attempts=1,
        reconnect_backoff=0,
        reconnect_heartbeat_timeout=0.01,
    )
    drone.gps[:] = [1.0, 2.0]
    drone.ekf_healthy = True
    drone.sensors_health = ["MAV_SYS_STATUS_SENSOR_GPS"]
    drone.drone_ready = True
    drone.start()

    assert wait_until(lambda: drone.connection is reconnected)
    assert factory_calls == [True]
    assert disconnected.closed
    assert drone.connection_healthy
    assert drone.last_heartbeat > 0
    assert not drone.drone_ready
    assert not drone.ekf_healthy
    assert drone.sensors_health == []


def test_failed_reconnect_fails_closed_for_arming():
    disconnected = ReconnectConnection(receive_error=OSError("USB disappeared"))

    def factory():
        raise OSError("still absent")

    drone = Drone(
        disconnected,
        fake=True,
        connection_factory=factory,
        reconnect_attempts=1,
        reconnect_backoff=0,
        reconnect_heartbeat_timeout=0.01,
    ).start()

    assert wait_until(lambda: drone.connection_failure is not None)
    with pytest.raises(MavlinkConnectionError, match="not healthy"):
        drone.arm()
    assert disconnected.mav.arm_commands == []


def test_unhealthy_buzzer_refuses_to_use_stale_connection():
    disconnected = ReconnectConnection()
    drone = Drone(
        disconnected,
        fake=True,
        connection_factory=lambda: ReconnectConnection(),
    )

    assert not drone.buzzer(b"tone")
    assert disconnected.mav.tune_commands == []


def test_buzzer_send_error_does_not_kill_caller_or_trigger_reconnect():
    disconnected = ReconnectConnection(tune_error=OSError("USB disappeared"))
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return ReconnectConnection(reconnect_heartbeat=heartbeat())

    drone = Drone(
        disconnected,
        fake=True,
        connection_factory=factory,
    )
    drone.process_message(heartbeat())

    assert not drone.buzzer(b"tone")
    assert drone.connection is disconnected
    assert drone.connection_failure is None
    assert factory_calls == []


def test_mode_wait_survives_reconnect_and_buzzes_replacement_connection():
    disconnected = ReconnectConnection()
    replacement = ReconnectConnection(reconnect_heartbeat=heartbeat(custom_mode=0))
    factory_entered = threading.Event()
    release_factory = threading.Event()

    def factory():
        factory_entered.set()
        assert release_factory.wait(timeout=1)
        return replacement

    drone = Drone(
        disconnected,
        fake=True,
        connection_factory=factory,
        reconnect_attempts=1,
        reconnect_backoff=0,
        reconnect_heartbeat_timeout=0.1,
    )
    drone.process_message(heartbeat(custom_mode=15))

    recovery_result = []
    recovery_thread = threading.Thread(
        target=lambda: recovery_result.append(
            drone._recover_connection(OSError("USB disappeared"))
        )
    )
    recovery_thread.start()
    assert factory_entered.wait(timeout=1)

    with pytest.raises(MavlinkConnectionError, match="not healthy"):
        drone.arm()
    assert disconnected.mav.arm_commands == []
    assert replacement.mav.arm_commands == []

    planner_errors = []
    buzzer_started = threading.Event()
    buzzer = drone.buzzer

    def observed_buzzer(tone):
        buzzer_started.set()
        return buzzer(tone)

    drone.buzzer = observed_buzzer

    def wait_for_manual_mode():
        try:
            drone._wait_for_mode(
                "ROVER_MODE_MANUAL",
                b"tone",
                "waiting for manual mode",
                poll_interval=0.001,
            )
        except Exception as error:
            planner_errors.append(error)

    planner_thread = threading.Thread(target=wait_for_manual_mode)
    planner_thread.start()
    assert buzzer_started.wait(timeout=1)

    # The mode-wait buzzer must not touch the closed connection while the
    # reconnect owns the connection swap.
    assert disconnected.closed
    assert disconnected.mav.tune_commands == []

    release_factory.set()
    recovery_thread.join(timeout=1)
    planner_thread.join(timeout=1)

    assert not recovery_thread.is_alive()
    assert not planner_thread.is_alive()
    assert recovery_result == [True]
    assert planner_errors == []
    assert replacement.mav.tune_commands
    assert drone.connection is replacement
    assert drone.connection_healthy


def test_incomplete_parameter_fetch_is_restarted_from_scratch():
    connection = ParameterConnection()
    drone = Drone(connection, fake=True)
    connection.drone = drone

    assert drone.update_all_parameters(
        timeout=0.01,
        max_attempts=2,
        retry_backoff=0,
    )
    assert connection.fetches == 2
    assert connection.starting_sizes == [0, 0]
    assert set(drone.params) == {"A", "B", "C"}


def test_initial_connection_retries_are_bounded_and_require_heartbeat():
    connections = [
        ReconnectConnection(reconnect_heartbeat=None),
        ReconnectConnection(reconnect_heartbeat=None),
        ReconnectConnection(reconnect_heartbeat=heartbeat()),
    ]
    calls = []

    def factory():
        calls.append(True)
        return connections[len(calls) - 1]

    with pytest.raises(MavlinkConnectionError, match="after 2 attempts"):
        connect_with_heartbeat(
            factory,
            attempts=2,
            heartbeat_timeout=0.01,
            retry_backoff=0,
        )

    assert len(calls) == 2
    assert connections[0].closed
    assert connections[1].closed
    assert not connections[2].closed


def test_tty_name_is_promoted_to_matching_stable_by_id(monkeypatch):
    stable = "/dev/serial/by-id/usb-ArduPilot_Pixhawk1"
    realpaths = {
        "/dev/ttyACM1": "/dev/ttyACM1",
        stable: "/dev/ttyACM1",
    }
    monkeypatch.setattr(
        "spf.mavlink.mavlink_controller.os.path.realpath",
        lambda path: realpaths[path],
    )

    assert (
        resolve_ardupilot_serial(
            "/dev/ttyACM1",
            available_pilots=[stable],
        )
        == stable
    )
