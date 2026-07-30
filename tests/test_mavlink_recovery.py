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


def heartbeat():
    return FakeMessage(
        message_type="HEARTBEAT",
        system_status=3,
        custom_mode=0,
        base_mode=0,
    )


class FakeMav:
    def __init__(self):
        self.arm_commands = []

    def command_long_send(self, *args):
        self.arm_commands.append(args)


class ReconnectConnection:
    target_system = 1
    target_component = 1

    def __init__(self, *, receive_error=None, reconnect_heartbeat=None):
        self.mav = FakeMav()
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
