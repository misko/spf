"""Compatibility shims for defects in the installed pymavlink.

Shared by the CLI and the production collector: both read MAVLink from a
live vehicle, and both can be handed a compass calibration -- ardu_cli via
`magcal start`, mavlink_controller via Taranis CH10.
"""

from __future__ import annotations

from pymavlink import mavutil


def harden_pymavlink_instance_messages() -> None:
    """Work around a pymavlink defect that makes a link unusable mid-magcal.

    ``pymavlink.mavutil.add_message`` stores a message that has no instance
    value through a simple path which leaves ``_instances`` at its ``None``
    default, and then assumes on every later message of that type that
    ``_instances`` is a dict::

        messages[mtype]._instances[instance_value] = msg
        TypeError: 'NoneType' object does not support item assignment

    Once a type is poisoned the state persists, so every subsequent message of
    that type raises and the connection cannot be read from again.

    Only four types on a rover's wire declare an instance field -- RAW_IMU
    ('id'), SERVO_OUTPUT_RAW ('port'), MAG_CAL_PROGRESS and MAG_CAL_REPORT
    (both 'compass_id'). The mag ones appear only while a calibration is
    running, which is why this crashes `magcal start` on Rover 4 and nothing
    else. Verified against pymavlink 2.4.49; the same code is present in
    2.4.42, so this is not version-specific.

    Repairs the entry rather than the library: dropping the un-tracked entry
    makes the original re-create it with a real dict.
    """
    if getattr(mavutil.add_message, "_spf_hardened", False):
        return
    original = mavutil.add_message

    def add_message(messages, mtype, msg):
        existing = messages.get(mtype)
        if existing is not None and getattr(existing, "_instances", None) is None:
            del messages[mtype]
        return original(messages, mtype, msg)

    add_message._spf_hardened = True
    add_message._spf_original = original
    mavutil.add_message = add_message
