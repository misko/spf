from types import SimpleNamespace

from pymavlink import mavutil

from spf.mavlink.check_prearm import PREARM_COMMAND, PrearmResult, run_prearm_checks


class FakeMessage(SimpleNamespace):
    def get_type(self):
        return self.message_type


class FakeMav:
    def __init__(self):
        self.commands = []

    def command_long_send(self, *args):
        self.commands.append(args)


class FakeConnection:
    target_system = 1
    target_component = 1

    def __init__(self, messages):
        self.messages = list(messages)
        self.mav = FakeMav()

    def recv_match(self, *, blocking, type=None, timeout=None):
        if not blocking:
            return None
        if not self.messages:
            return None
        return self.messages.pop(0)


def command_ack(result=mavutil.mavlink.MAV_RESULT_ACCEPTED):
    return FakeMessage(
        message_type="COMMAND_ACK",
        command=PREARM_COMMAND,
        result=result,
    )


def sys_status(healthy):
    return FakeMessage(
        message_type="SYS_STATUS",
        onboard_control_sensors_health=(
            mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK if healthy else 0
        ),
    )


def statustext(text):
    return FakeMessage(message_type="STATUSTEXT", text=text)


def test_prearm_pass_uses_ack_and_health_bit():
    connection = FakeConnection([command_ack(), sys_status(True)])

    result = run_prearm_checks(connection, timeout_s=0.01)

    assert result == PrearmResult(
        command_result=mavutil.mavlink.MAV_RESULT_ACCEPTED,
        healthy=True,
        messages=(),
    )
    assert result.passed
    assert connection.mav.commands[0][2] == PREARM_COMMAND


def test_prearm_failure_collects_unique_failure_messages():
    connection = FakeConnection(
        [
            command_ack(),
            statustext(b"PreArm: Compass not calibrated\x00"),
            statustext("PreArm: Compass not calibrated"),
            statustext("Unrelated status"),
            statustext("PreArm: GPS 1: Bad fix"),
            sys_status(False),
        ]
    )

    result = run_prearm_checks(connection, timeout_s=0.01)

    assert result.conclusive
    assert not result.passed
    assert result.messages == (
        "PreArm: Compass not calibrated",
        "PreArm: GPS 1: Bad fix",
    )


def test_rejected_command_is_inconclusive_even_with_health_bit():
    connection = FakeConnection(
        [
            command_ack(mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED),
            sys_status(True),
        ]
    )

    result = run_prearm_checks(connection, timeout_s=0.01)

    assert not result.conclusive
    assert not result.passed
