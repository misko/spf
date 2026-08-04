"""Regression for a pymavlink defect that killed `magcal start` on Rover 4.

The shim is shared: ardu_cli reaches magcal via the CLI, and the production
collector reaches it from Taranis CH10, so both apply it at import.

pymavlink stores a message with no instance value through a path that leaves
``_instances`` at its ``None`` default, then assumes on the next message of the
same type that it is a dict. The result is a TypeError deep inside recv_match,
and because the poisoned entry persists, the connection can never be read
again -- mid-calibration, on a real vehicle.
"""

import copy
from types import SimpleNamespace

import pytest
from pymavlink import mavutil

from spf.mavlink.pymavlink_compat import harden_pymavlink_instance_messages

# Mirrors what ardu_cli and mavlink_controller both do at import.
harden_pymavlink_instance_messages()


def message(instance_field, value, instances=None):
    msg = SimpleNamespace()
    msg._instance_field = instance_field
    msg._instances = instances
    if instance_field is not None:
        setattr(msg, instance_field, value)
    return msg


def test_the_underlying_pymavlink_defect_is_real():
    """If this ever stops raising, upstream fixed it and the shim can go."""
    original = getattr(mavutil.add_message, "_spf_original", None)
    assert original is not None, "hardening was not applied"

    messages = {}
    # First MAG_CAL_PROGRESS with no instance value: stored with _instances=None.
    original(messages, "MAG_CAL_PROGRESS", message("compass_id", None))
    # Next one carries compass_id=0 and assumes a dict is already there.
    with pytest.raises(TypeError):
        original(messages, "MAG_CAL_PROGRESS", message("compass_id", 0))


def test_hardened_add_message_survives_the_same_sequence():
    messages = {}
    mavutil.add_message(messages, "MAG_CAL_PROGRESS", message("compass_id", None))
    mavutil.add_message(messages, "MAG_CAL_PROGRESS", message("compass_id", 0))
    mavutil.add_message(messages, "MAG_CAL_PROGRESS", message("compass_id", 1))

    assert messages["MAG_CAL_PROGRESS"]._instances.keys() == {0, 1}
    assert "MAG_CAL_PROGRESS[0]" in messages
    assert "MAG_CAL_PROGRESS[1]" in messages


def test_normal_instance_tracking_is_unchanged():
    """The shim must not disturb the ordinary path."""
    messages = {}
    mavutil.add_message(messages, "RAW_IMU", message("id", 0))
    mavutil.add_message(messages, "RAW_IMU", message("id", 1))

    assert messages["RAW_IMU"]._instances.keys() == {0, 1}


def test_messages_without_an_instance_field_are_untouched():
    messages = {}
    plain = message(None, None)
    mavutil.add_message(messages, "HEARTBEAT", plain)

    assert messages["HEARTBEAT"] is plain


def test_hardening_is_idempotent():
    """Import order must not wrap the wrapper on every import."""
    before = mavutil.add_message
    harden_pymavlink_instance_messages()
    assert mavutil.add_message is before


def test_copy_semantics_still_hold():
    """add_message stores copies, not the live object, for instance types."""
    messages = {}
    first = message("compass_id", 0)
    mavutil.add_message(messages, "MAG_CAL_REPORT", first)
    assert messages["MAG_CAL_REPORT"] is not first
    assert isinstance(copy.copy(first), SimpleNamespace)
