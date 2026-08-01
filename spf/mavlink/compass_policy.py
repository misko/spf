#!/usr/bin/env python3
"""Evaluate the Rover fleet's device-ID-aware compass policy.

ArduPilot may enumerate a compass in a different numbered slot after a hardware
or firmware change.  The policy therefore identifies the external GPS compass
by device ID and ``COMPASS_EXTERN*`` instead of assuming it is always slot 1.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Mapping


EXPECTED_EXTERNAL_COMPASS_DEVICE_ID = 658953
DEFAULT_COMPASS_CAL_FIT = 16.0
PROJECT_TARGET_EXTERNAL_OFFSET_NORM_MG = 300.0
PREFERRED_MAX_EXTERNAL_OFFSET_NORM_MG = 500.0
MIN_CONFIGURED_OFFSET_LIMIT_MG = 500.0
MAX_CONFIGURED_OFFSET_LIMIT_MG = 3000.0
CONFIGURED_COMPASS_SLOTS = (1, 2, 3)
EXTRA_COMPASS_SLOTS = (4, 5, 6, 7, 8)


class UnsafeCompassRepairError(ValueError):
    """The live inventory cannot identify one external compass safely."""


@dataclass(frozen=True)
class CompassInstance:
    """One ArduPilot compass slot."""

    slot: int
    device_id: int
    external: bool
    used_for_yaw: bool
    offset_x_mg: float
    offset_y_mg: float
    offset_z_mg: float

    @property
    def detected(self) -> bool:
        return self.device_id != 0

    @property
    def offset_norm_mg(self) -> float:
        return math.sqrt(
            self.offset_x_mg**2 + self.offset_y_mg**2 + self.offset_z_mg**2
        )


@dataclass(frozen=True)
class CompassPolicyReport:
    """Fail-closed result of evaluating the fleet compass policy."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    instances: tuple[CompassInstance, ...]
    external_compass: CompassInstance | None
    external_offset_limit_mg: float
    priorities: tuple[int, int, int]
    extra_devices: tuple[tuple[int, int], ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "external_compass": (
                _instance_to_dict(self.external_compass)
                if self.external_compass is not None
                else None
            ),
            "external_offset_limit_mg": self.external_offset_limit_mg,
            "instances": [_instance_to_dict(instance) for instance in self.instances],
            "priorities": list(self.priorities),
            "extra_devices": [
                {
                    "slot": slot,
                    "device_id": device_id,
                    **decode_compass_device_id(device_id),
                }
                for slot, device_id in self.extra_devices
            ],
        }


def _instance_to_dict(instance: CompassInstance) -> dict:
    return {
        "slot": instance.slot,
        "device_id": instance.device_id,
        "external": instance.external,
        "used_for_yaw": instance.used_for_yaw,
        "offset_x_mg": instance.offset_x_mg,
        "offset_y_mg": instance.offset_y_mg,
        "offset_z_mg": instance.offset_z_mg,
        "offset_norm_mg": instance.offset_norm_mg,
        **decode_compass_device_id(instance.device_id),
    }


def decode_compass_device_id(device_id: int) -> dict[str, int | None]:
    """Decode ArduPilot's bus-aware 24-bit device identifier."""
    if not device_id:
        return {
            "bus_type": None,
            "bus": None,
            "address": None,
            "device_type": None,
        }
    return {
        "bus_type": device_id & 0x07,
        "bus": (device_id >> 3) & 0x1F,
        "address": (device_id >> 8) & 0xFF,
        "device_type": (device_id >> 16) & 0xFF,
    }


def _slot_suffix(slot: int) -> str:
    return "" if slot == 1 else str(slot)


def _external_key(slot: int) -> str:
    return "COMPASS_EXTERNAL" if slot == 1 else f"COMPASS_EXTERN{slot}"


def _number(params: Mapping[str, float], key: str, errors: list[str]) -> float:
    if key not in params:
        errors.append(f"missing required parameter {key}")
        return 0.0
    try:
        return float(params[key])
    except (TypeError, ValueError):
        errors.append(f"parameter {key} is not numeric: {params[key]!r}")
        return 0.0


def _read_instances(
    params: Mapping[str, float], errors: list[str]
) -> tuple[CompassInstance, ...]:
    instances = []
    for slot in CONFIGURED_COMPASS_SLOTS:
        suffix = _slot_suffix(slot)
        instances.append(
            CompassInstance(
                slot=slot,
                device_id=int(_number(params, f"COMPASS_DEV_ID{suffix}", errors)),
                external=bool(_number(params, _external_key(slot), errors)),
                used_for_yaw=bool(_number(params, f"COMPASS_USE{suffix}", errors)),
                offset_x_mg=_number(params, f"COMPASS_OFS{suffix}_X", errors),
                offset_y_mg=_number(params, f"COMPASS_OFS{suffix}_Y", errors),
                offset_z_mg=_number(params, f"COMPASS_OFS{suffix}_Z", errors),
            )
        )
    return tuple(instances)


def evaluate_compass_policy(
    params: Mapping[str, float],
    *,
    expected_external_device_id: int = EXPECTED_EXTERNAL_COMPASS_DEVICE_ID,
) -> CompassPolicyReport:
    """Evaluate a parameter snapshot without changing the flight controller."""
    errors: list[str] = []
    warnings: list[str] = []
    instances = _read_instances(params, errors)
    extra_devices = tuple(
        (slot, int(float(params.get(f"COMPASS_DEV_ID{slot}", 0))))
        for slot in EXTRA_COMPASS_SLOTS
        if int(float(params.get(f"COMPASS_DEV_ID{slot}", 0))) != 0
    )

    detected_slots: list[tuple[int, int]] = [
        (instance.slot, instance.device_id)
        for instance in instances
        if instance.detected
    ] + list(extra_devices)
    detected_counts = Counter(device_id for _slot, device_id in detected_slots)
    for device_id, count in detected_counts.items():
        if count <= 1:
            continue
        slots = [
            slot for slot, candidate_id in detected_slots if candidate_id == device_id
        ]
        errors.append(
            f"detected compass device ID {device_id} appears in multiple slots: "
            f"{slots}; ArduPilot priority cannot distinguish duplicate full IDs"
        )

    if extra_devices:
        warnings.append(
            "additional unconfigured compass devices were detected in slots "
            f"{[slot for slot, _device_id in extra_devices]}"
        )

    if int(_number(params, "COMPASS_ENABLE", errors)) != 1:
        errors.append("COMPASS_ENABLE must be 1")

    calibration_fitness = _number(params, "COMPASS_CAL_FIT", errors)
    if not math.isclose(calibration_fitness, DEFAULT_COMPASS_CAL_FIT):
        errors.append(
            f"COMPASS_CAL_FIT={calibration_fitness:g}; "
            f"fleet policy requires {DEFAULT_COMPASS_CAL_FIT:g}"
        )

    disabled_driver_mask = int(_number(params, "COMPASS_DISBLMSK", errors))
    if disabled_driver_mask != 0:
        errors.append(
            f"COMPASS_DISBLMSK={disabled_driver_mask}; fleet policy resolves "
            "individual devices by ID and requires the driver mask to be 0"
        )

    external_offset_limit_mg = _number(params, "COMPASS_OFFS_MAX", errors)
    if not (
        MIN_CONFIGURED_OFFSET_LIMIT_MG
        <= external_offset_limit_mg
        <= MAX_CONFIGURED_OFFSET_LIMIT_MG
    ):
        errors.append(
            f"COMPASS_OFFS_MAX={external_offset_limit_mg:g}; expected an "
            f"ArduPilot limit from {MIN_CONFIGURED_OFFSET_LIMIT_MG:g} to "
            f"{MAX_CONFIGURED_OFFSET_LIMIT_MG:g} mG"
        )

    external_candidates = [
        instance
        for instance in instances
        if instance.device_id == expected_external_device_id and instance.external
    ]
    unexpected_external = [
        instance
        for instance in instances
        if instance.detected
        and instance.external
        and instance.device_id != expected_external_device_id
    ]
    external_compass = external_candidates[0] if len(external_candidates) == 1 else None
    if not external_candidates:
        errors.append(
            "expected external GPS compass "
            f"device ID {expected_external_device_id} was not detected as external"
        )
    elif len(external_candidates) > 1:
        slots = [instance.slot for instance in external_candidates]
        errors.append(
            "external GPS compass device ID "
            f"{expected_external_device_id} appears in multiple slots: {slots}"
        )
    if unexpected_external:
        errors.append(
            "unexpected compass slots are marked external: "
            f"{[(instance.slot, instance.device_id) for instance in unexpected_external]}"
        )

    for instance in instances:
        if not instance.detected and instance.used_for_yaw:
            errors.append(f"empty compass slot {instance.slot} is enabled for yaw")
        if (
            instance.detected
            and external_compass is not None
            and instance.slot != external_compass.slot
            and instance.used_for_yaw
        ):
            errors.append(
                f"non-primary compass slot {instance.slot} "
                f"(device ID {instance.device_id}) is enabled for yaw"
            )

    if external_compass is not None:
        if not external_compass.used_for_yaw:
            errors.append(
                f"external compass slot {external_compass.slot} is disabled for yaw"
            )
        if external_compass.offset_norm_mg <= 0:
            errors.append(
                f"external compass slot {external_compass.slot} has no calibration"
            )
        elif external_compass.offset_norm_mg > external_offset_limit_mg:
            errors.append(
                f"external compass slot {external_compass.slot} offset norm "
                f"{external_compass.offset_norm_mg:.1f} mG exceeds "
                f"configured COMPASS_OFFS_MAX={external_offset_limit_mg:.1f} mG"
            )
        elif (
            external_compass.offset_norm_mg
            > PREFERRED_MAX_EXTERNAL_OFFSET_NORM_MG
        ):
            warnings.append(
                f"external compass slot {external_compass.slot} offset norm "
                f"{external_compass.offset_norm_mg:.1f} mG exceeds the preferred "
                f"{PREFERRED_MAX_EXTERNAL_OFFSET_NORM_MG:.1f} mG limit but is "
                f"within configured COMPASS_OFFS_MAX={external_offset_limit_mg:.1f} mG"
            )
        elif external_compass.offset_norm_mg > PROJECT_TARGET_EXTERNAL_OFFSET_NORM_MG:
            warnings.append(
                f"external compass slot {external_compass.slot} offset norm "
                f"{external_compass.offset_norm_mg:.1f} mG exceeds the "
                f"project target of {PROJECT_TARGET_EXTERNAL_OFFSET_NORM_MG:g} mG"
            )

    priorities = tuple(
        int(_number(params, f"COMPASS_PRIO{priority}_ID", errors))
        for priority in CONFIGURED_COMPASS_SLOTS
    )
    nonzero_priorities = [device_id for device_id in priorities if device_id]
    if len(nonzero_priorities) != len(set(nonzero_priorities)):
        errors.append(f"compass priority IDs are not unique: {priorities}")

    if priorities[0] != expected_external_device_id:
        errors.append(
            f"compass priority 1 is device ID {priorities[0]}, expected "
            f"external device ID {expected_external_device_id}"
        )

    detected_ids = {device_id for _slot, device_id in detected_slots}
    stale_priorities = [
        device_id for device_id in nonzero_priorities if device_id not in detected_ids
    ]
    if stale_priorities:
        errors.append(
            f"compass priority list contains undetected device IDs: {stale_priorities}"
        )

    return CompassPolicyReport(
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        instances=instances,
        external_compass=external_compass,
        external_offset_limit_mg=external_offset_limit_mg,
        priorities=priorities,
        extra_devices=extra_devices,
    )


def format_compass_inventory(report: CompassPolicyReport) -> tuple[str, ...]:
    """Return compact lines suitable for both CLI and systemd journal logs."""
    lines = [
        "Compass priorities: "
        + " ".join(
            f"PRIO{index}={device_id}"
            for index, device_id in enumerate(report.priorities, start=1)
        )
    ]
    for instance in report.instances:
        decoded = decode_compass_device_id(instance.device_id)
        lines.append(
            f"Compass slot {instance.slot}: device_id={instance.device_id} "
            f"detected={instance.detected} external={instance.external} "
            f"use_for_yaw={instance.used_for_yaw} "
            f"bus_type={decoded['bus_type']} bus={decoded['bus']} "
            f"address={decoded['address']} device_type={decoded['device_type']}"
        )
    for slot, device_id in report.extra_devices:
        decoded = decode_compass_device_id(device_id)
        lines.append(
            f"Compass extra slot {slot}: device_id={device_id} detected=True "
            f"bus_type={decoded['bus_type']} bus={decoded['bus']} "
            f"address={decoded['address']} device_type={decoded['device_type']}"
        )
    return tuple(lines)


def plan_external_compass_repairs(
    params: Mapping[str, float],
    *,
    expected_external_device_id: int = EXPECTED_EXTERNAL_COMPASS_DEVICE_ID,
) -> dict[str, int]:
    """Plan only unambiguous priority and yaw-use repairs.

    Detection, external classification, and calibration are deliberately never
    guessed or rewritten. Exact duplicate device IDs are not repairable because
    ArduPilot's priority list is itself keyed by that full ID.
    """
    read_errors: list[str] = []
    instances = _read_instances(params, read_errors)
    if read_errors:
        raise UnsafeCompassRepairError("; ".join(dict.fromkeys(read_errors)))

    extra_devices = [
        (slot, int(float(params.get(f"COMPASS_DEV_ID{slot}", 0))))
        for slot in EXTRA_COMPASS_SLOTS
        if int(float(params.get(f"COMPASS_DEV_ID{slot}", 0))) != 0
    ]
    if extra_devices:
        raise UnsafeCompassRepairError(
            "additional unconfigured compass devices are present in slots "
            f"{[slot for slot, _device_id in extra_devices]}"
        )

    detected = [instance for instance in instances if instance.detected]
    counts = Counter(instance.device_id for instance in detected)
    duplicates = {
        device_id: [
            instance.slot for instance in detected if instance.device_id == device_id
        ]
        for device_id, count in counts.items()
        if count > 1
    }
    if duplicates:
        raise UnsafeCompassRepairError(
            f"duplicate detected compass device IDs cannot be repaired: {duplicates}"
        )

    external = [instance for instance in detected if instance.external]
    if len(external) != 1:
        raise UnsafeCompassRepairError(
            "expected exactly one detected compass marked external, found "
            f"{[(instance.slot, instance.device_id) for instance in external]}"
        )
    external_compass = external[0]
    if external_compass.device_id != expected_external_device_id:
        raise UnsafeCompassRepairError(
            f"external compass device ID is {external_compass.device_id}, expected "
            f"fleet device ID {expected_external_device_id}"
        )

    priority_errors: list[str] = []
    current_priorities = [
        int(_number(params, f"COMPASS_PRIO{index}_ID", priority_errors))
        for index in CONFIGURED_COMPASS_SLOTS
    ]
    if priority_errors:
        raise UnsafeCompassRepairError("; ".join(dict.fromkeys(priority_errors)))
    detected_ids = {instance.device_id for instance in detected}
    remaining_ids: list[int] = []
    for device_id in current_priorities + [instance.device_id for instance in detected]:
        if (
            device_id
            and device_id != external_compass.device_id
            and device_id in detected_ids
            and device_id not in remaining_ids
        ):
            remaining_ids.append(device_id)
    desired_priorities = [external_compass.device_id, *remaining_ids][:3]
    desired_priorities.extend([0] * (3 - len(desired_priorities)))

    desired: dict[str, int] = {
        f"COMPASS_PRIO{index}_ID": desired_priorities[index - 1]
        for index in CONFIGURED_COMPASS_SLOTS
    }
    # Enable the external source before disabling any other yaw source so a
    # mid-command transport failure cannot leave the vehicle with no compass.
    desired[f"COMPASS_USE{_slot_suffix(external_compass.slot)}"] = 1
    for instance in instances:
        if instance.slot != external_compass.slot:
            desired[f"COMPASS_USE{_slot_suffix(instance.slot)}"] = 0

    return {
        key: value
        for key, value in desired.items()
        if not math.isclose(float(params[key]), float(value))
    }


def parse_parameter_file(path: str | Path) -> dict[str, float]:
    """Parse MAVProxy whitespace or Mission Planner CSV parameter exports."""
    params: dict[str, float] = {}
    with Path(path).open(encoding="utf-8") as parameter_file:
        for line_number, raw_line in enumerate(parameter_file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [field for field in re.split(r"[\s,]+", line) if field]
            if len(fields) < 2:
                raise ValueError(
                    f"{path}:{line_number}: expected PARAMETER VALUE, got {line!r}"
                )
            try:
                params[fields[0]] = float(fields[1])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid value {fields[1]!r}"
                ) from error
    return params


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved ArduPilot parameter snapshot against the Rover "
            "fleet's external-compass policy. This command is read-only."
        )
    )
    parser.add_argument("parameter_file")
    parser.add_argument(
        "--json-output",
        help="Write the complete machine-readable report to this path.",
    )
    parser.add_argument(
        "--expected-device-id",
        type=int,
        default=EXPECTED_EXTERNAL_COMPASS_DEVICE_ID,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    try:
        params = parse_parameter_file(args.parameter_file)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        return 2

    report = evaluate_compass_policy(
        params,
        expected_external_device_id=args.expected_device_id,
    )
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if report.external_compass is not None:
        external = report.external_compass
        print(
            f"External compass: slot {external.slot}, "
            f"device ID {external.device_id}, "
            f"offset norm {external.offset_norm_mg:.1f} mG"
        )
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"FAIL: {error}")

    if report.ok:
        print("PASS: external GPS compass is the sole fleet-approved yaw source")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
